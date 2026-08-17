# [AI:BEGIN]
# PA_MCP - 种子股票池引导脚本
#
# AKShare(东财)全市场快照不可用时，用内置种子池装载 stock_basic，
# 保证日线/分钟线/财务/资金流 phase 可运行。
#
# 用法：
#   python scripts/bootstrap_universe.py          # 装载默认种子池
#   python scripts/bootstrap_universe.py --limit 500  # 只装载前500只
# [AI:END]

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pa_mcp.data.store import DuckDBStore

# 沪深核心股票种子池（(代码, 行业) 结构，行业用于板块轮动的合成板块兜底）
# 覆盖银行/白酒/新能源/医药/科技/基建/汽车家电等；已去重
SEED_SYMBOLS: list[tuple[str, str]] = [
    # 银行
    ("000001", "银行"), ("600036", "银行"), ("601398", "银行"), ("601288", "银行"),
    ("601988", "银行"), ("600016", "银行"), ("601166", "银行"), ("601328", "银行"),
    ("601818", "银行"), ("601939", "银行"), ("600000", "银行"),
    # 保险
    ("601601", "保险"), ("601318", "保险"), ("601628", "保险"), ("601336", "保险"),
    # 白酒
    ("600519", "白酒"), ("000858", "白酒"), ("000568", "白酒"), ("600809", "白酒"),
    # 食品饮料
    ("603288", "食品饮料"), ("000895", "食品饮料"), ("600887", "食品饮料"),
    # 新能源
    ("300750", "新能源"), ("002594", "新能源"), ("601012", "新能源"),
    ("600438", "新能源"), ("300274", "新能源"), ("002812", "新能源"),
    ("603799", "新能源"), ("601865", "新能源"),
    # 医药
    ("600276", "医药"), ("000538", "医药"), ("300760", "医药"), ("600196", "医药"),
    ("002001", "医药"), ("300015", "医药"), ("600085", "医药"),
    # 科技
    ("002415", "科技"), ("300059", "科技"), ("688981", "科技"), ("688001", "科技"),
    ("603986", "科技"), ("002230", "科技"), ("300308", "科技"), ("688036", "科技"),
    # 消费电子
    ("002475", "消费电子"), ("601138", "消费电子"), ("000725", "消费电子"),
    ("002241", "消费电子"),
    # 基建能源
    ("600900", "电力"), ("601857", "石油石化"), ("600028", "石油石化"),
    ("601088", "煤炭"), ("600019", "钢铁"), ("601390", "基建"), ("601668", "基建"),
    ("600585", "建材"),
    # 汽车家电
    ("601633", "汽车"), ("600104", "汽车"), ("000333", "家电"), ("000651", "家电"),
    ("600690", "家电"), ("002050", "家电"),
    # 其他白马
    ("600030", "证券"), ("601211", "证券"), ("600837", "证券"), ("601688", "证券"),
    ("600999", "证券"), ("300124", "电子"), ("002460", "有色"), ("600089", "军工"),
    ("600745", "电子"), ("603501", "电子"), ("688111", "软件"), ("688008", "电子"),
    ("688012", "电子"), ("688126", "电子"), ("688169", "电子"), ("300014", "电子"),
    ("300433", "电子"), ("002371", "电子"), ("002384", "电子"), ("300496", "软件"),
    ("300782", "电子"),
]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="种子股票池引导")
    parser.add_argument("--limit", type=int, default=0, help="只装载前N只")
    parser.add_argument("--db", default=None, help="DuckDB 路径")
    args = parser.parse_args()

    symbols = SEED_SYMBOLS
    if args.limit:
        symbols = symbols[: args.limit]

    store = DuckDBStore(args.db) if args.db else DuckDBStore()
    store.connect()

    # 持仓自动入池：portfolio 表的股票即使不在种子池也要保证可补数
    holdings: list[str] = []
    try:
        if store.table_exists("portfolio"):
            df = store.query_df("SELECT symbol FROM portfolio", [])
            holdings = [str(s) for s in df["symbol"].tolist()]
    except Exception:
        pass
    if holdings:
        symbols = list(symbols) + [(s, "其他") for s in holdings]

    # 仅补充缺失的 symbol（保留已有）
    existing = set()
    try:
        df = store.query_df("SELECT symbol FROM stock_basic")
        existing = set(df["symbol"].tolist())
    except Exception:
        pass

    # 去重（保留顺序）并过滤已存在
    seen: set[str] = set()
    new_symbols: list[tuple[str, str]] = []
    for sym, sector in symbols:
        if sym in existing or sym in seen:
            continue
        seen.add(sym)
        new_symbols.append((sym, sector))
    if not new_symbols:
        print(f"种子池已全部存在（{len(symbols)} 只）— 无需引导")
        store.close()
        return

    def _exchange(sym: str) -> str:
        if sym.startswith(("0", "3")):
            return "sz"
        if sym.startswith(("4", "8")) or sym.startswith("920"):
            return "bj"
        return "sh"

    basic = pd.DataFrame([{
        "symbol": sym, "name": f"SEED-{sym}", "industry": sector, "sector": sector,
        "market_cap": None, "list_date": None,
        "exchange": _exchange(sym),
        "board": "main", "is_st": False, "is_suspended": False,
    } for sym, sector in new_symbols])
    store.insert_df("stock_basic", basic, mode="insert")
    print(f"种子池引导完成：新增 {len(new_symbols)} 只，当前共 {len(existing) + len(new_symbols)} 只")
    store.close()


if __name__ == "__main__":
    main()
