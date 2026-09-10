# [AI:BEGIN]
# PA_MCP - 数据修复回填脚本（一次性 / 可重复运行）
#
# 背景（2026-09-08）：调度器的 checkpoint 只有 (job, symbol) 两个维度，
# 跑过一轮后 pending 恒为空，增量更新形同虚设，导致：
#   · 85 只标的 K 线停在 08-19/08-14（含中国神华、工商银行等权重股）
#   · 自选股 5/7 完全没有 K 线
#   · index_daily / sentiment_daily 停在 08-14（pipeline 根本没有这两个 phase）
#   · fund_flow_daily 全空（东财缺 Referer 被断连 + 解析要求 7 段而接口只给 6 段）
# 代码层面已全部修好，本脚本负责「把历史欠账补回来」。
#
# 用法：
#   python scripts/repair_data_backfill.py --dry-run        # 只体检不写库
#   python scripts/repair_data_backfill.py --phases kline   # 只补 K 线
#   python scripts/repair_data_backfill.py                  # 全量补齐
# [AI:END]

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb  # noqa: E402

from pa_mcp.config import get_settings  # noqa: E402
from pa_mcp.data.scheduler import DataUpdateScheduler  # noqa: E402
from pa_mcp.data.source_factory import build_router  # noqa: E402
from pa_mcp.data.sources.akshare_adapter import AKShareAdapter  # noqa: E402
from pa_mcp.data.sources.sina_adapter import SinaAdapter  # noqa: E402
from pa_mcp.data.store import DuckDBStore  # noqa: E402

DB_PATH = r"D:\Project\AI\PA_MCP\PA_MCP\data\pa_mcp.duckdb"


def freshness() -> None:
    """打印各表数据新鲜度（只读）。"""
    con = duckdb.connect(DB_PATH, read_only=True)
    print("\n" + "=" * 62)
    print("数据新鲜度体检")
    print("=" * 62)
    try:
        mx = con.execute("SELECT MAX(date) FROM kline_daily").fetchone()[0]
        print(f"kline_daily    最新日期={mx}  标的数="
              f"{con.execute('SELECT COUNT(DISTINCT symbol) FROM kline_daily').fetchone()[0]}")
        print("  落后分布（max(date) → 标的数）：")
        for d, n in con.execute(
            "WITH m AS (SELECT symbol, MAX(date) d FROM kline_daily GROUP BY symbol) "
            "SELECT d, COUNT(*) FROM m GROUP BY d ORDER BY d DESC"
        ).fetchall():
            print(f"    {d}  {n}")
        missing = con.execute(
            "SELECT w.symbol FROM watchlist w "
            "WHERE NOT EXISTS (SELECT 1 FROM kline_daily k WHERE k.symbol = w.symbol)"
        ).fetchall()
        print(f"  自选股无K线: {[m[0] for m in missing] or '无'}")
    except Exception as e:
        print("kline_daily ERR", e)

    for table, col in [("index_daily", "date"), ("sentiment_daily", "date"),
                       ("fund_flow_daily", "trade_date")]:
        try:
            r = con.execute(f"SELECT MAX({col}), COUNT(*) FROM {table}").fetchone()
            print(f"{table:<16}最新={r[0]}  行数={r[1]}")
        except Exception as e:
            print(f"{table:<16}ERR {e}")
    con.close()


async def main() -> int:
    ap = argparse.ArgumentParser(description="PA_MCP 数据修复回填")
    ap.add_argument("--dry-run", action="store_true", help="只做体检，不写库")
    ap.add_argument("--phases", default="calendar,kline,index,sentiment,fundflow,indicators",
                    help="逗号分隔：calendar,kline,index,sentiment,fundflow,indicators")
    args = ap.parse_args()

    freshness()
    if args.dry_run:
        return 0

    wanted = {p.strip() for p in args.phases.split(",") if p.strip()}

    settings = get_settings()
    store = DuckDBStore()
    store.connect()
    router = build_router(
        settings,
        min_source_interval={"eastmoney": 1.2, "ths": 0.4},
        existing={"akshare": AKShareAdapter(), "sina": SinaAdapter()},
    )
    sched = DataUpdateScheduler(store, data_router=router)

    # 迁移：用现有 K 线回填 checkpoint.last_date，避免首轮把 1091 只全量重拉一遍
    if "kline" in wanted:
        n = sched.sync_checkpoint_from_data("kline_daily", "kline_daily")
        print(f"\n[迁移] kline checkpoint 回填 last_date：{n} 只")

    # 资金流旧断点全是「拉空」留下的脏点，直接重置
    if "fundflow" in wanted:
        n = sched.reset_checkpoint("fund_flow")
        print(f"[迁移] 清除 fund_flow 脏断点：{n} 条")

    # 指数脏行清理：历史上 astock/百度源把 sh000001 剥成 000001，返回
    # 平安银行 K 线并以指数 symbol 落库（点位 10~12 元）。先清再回填，
    # 否则增量逻辑会认为「已有数据」而跳过补齐。
    if "index" in wanted:
        n = sched.purge_corrupt_index_rows()
        print(f"[迁移] 清除 index_daily 脏行（点位像个股）：{n} 行")

    # 情绪脏行清理：pct_change 为 NULL 时期算出的「三计数全 0」缓存，
    # 不清掉会被「有行就跳过」的逻辑永久固化（情绪图恒为平线）。
    if "sentiment" in wanted:
        try:
            cur = store.execute(
                "SELECT COUNT(*) FROM sentiment_daily WHERE limit_up_count = 0 "
                "AND limit_down_count = 0 AND max_board_height = 0").fetchone()[0]
            if cur:
                store.execute(
                    "DELETE FROM sentiment_daily WHERE limit_up_count = 0 "
                    "AND limit_down_count = 0 AND max_board_height = 0")
            print(f"[迁移] 清除 sentiment_daily 全 0 脏行：{int(cur)} 行")
        except Exception as e:
            print(f"[迁移] sentiment 脏行清理失败：{str(e)[:160]}")

    # 派生列补算：源只给 OHLCV，pct_change/change/amplitude 从未写入，
    # 是涨停统计恒为 0、涨跌幅因子失效的根因。必须先补，再算情绪。
    if "kline" in wanted or "sentiment" in wanted:
        n = sched.recompute_kline_derived()
        print(f"[迁移] 补算 kline 派生列（pct_change/change/amplitude）：{n} 行")

    plan = [
        ("calendar", "1_calendar", lambda: sched._update_calendar(False)),
        ("kline", "3_daily_kline", lambda: sched._update_daily_kline(False)),
        ("fundflow", "6_capital_flow", lambda: sched._update_capital_flow(False)),
        ("index", "9_index_daily", lambda: sched._update_index_daily(False)),
        ("sentiment", "10_sentiment", lambda: sched._update_sentiment(False)),
        ("indicators", "8_indicators", lambda: sched._update_indicators(False)),
    ]

    print()
    for key, label, fn in plan:
        if key not in wanted:
            continue
        print(f"[{label}] 开始 ...", flush=True)
        try:
            rows = await fn()
            print(f"[{label}] 完成 rows={rows}", flush=True)
        except Exception as e:
            print(f"[{label}] 失败: {type(e).__name__}: {str(e)[:200]}", flush=True)

    store.close()
    freshness()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
