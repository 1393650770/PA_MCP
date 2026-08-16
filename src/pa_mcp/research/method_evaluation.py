# [AI:BEGIN]
# PA_MCP - Research: 开源方法评价（组合管理 Tab）
#
# 组合管理 Tab「开源方法评价」两块内容，全部复用已有分析器（非必要不造轮子）：
#   上半部分：方法可信度 —— 量化策略用全策略事件研究对比（compare_all_strategies）
#            给出胜率/超额/有效判定；理财方法标注验证方式与当前评估状态。
#   下半部分：持仓×方法 —— 每只持仓逐一跑格雷厄姆/价值动量/CANSLIM/缠论/
#            综合信号，回答"我的持仓在每种方法眼里怎么样"。
#
# 设计要点：
# - METHOD_EVALUATORS 注册表：理财方法编目（新增方法只需注册一条）。
# - 每个 evaluator 独立 try/except，单个方法失败只降级该格，不拖垮整张表。
# - 数据从 store 读取（get_settings().database.path），与 PortfolioRiskDashboard
#   同一模式；klines/symbols/store 均可注入（测试隔离）。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

KLINE_LIMIT = 250  # 每票读取的日线根数（缠论/策略信号需要足够历史）

# 理财方法注册表（下半部分逐票评价 + 上半部分评估状态说明）
# id → {category, name_zh, verification(可信度说明), marketscope(全池/单票)}
METHOD_EVALUATORS: dict[str, dict] = {
    "graham": {
        "category": "理财方法",
        "name_zh": "格雷厄姆筛选",
        "verification": "确定性规则（7 条防御标准 + 安全边际）；无历史事件研究入口，"
                        "有效性看财务数据完整性与筛选稳定性",
    },
    "value_momentum": {
        "category": "理财方法",
        "name_zh": "价值×动量（Asness 2013）",
        "verification": "四象限复合分；可在「市场扫描」Tab 用价值×动量组合回测验证"
                        "（value_momentum_backtest）",
    },
    "canslim": {
        "category": "理财方法",
        "name_zh": "CANSLIM（欧奈尔）",
        "verification": "确定性七要素规则；有效性依赖财务数据完整性与池内 RS 排名",
    },
    "chan": {
        "category": "理财方法",
        "name_zh": "缠论结构",
        "verification": "结构/背驰信号有事件研究入口（chan_beichi_event_study），"
                        "背驰信号历史上经事件研究检验",
    },
    "consensus": {
        "category": "量化方法",
        "name_zh": "综合决策信号（5 源投票）",
        "verification": "共振/预测/策略/背驰/大盘加权投票；有事件研究入口"
                        "（consensus_event_study）",
    },
}

CATEGORY_ZH: dict[str, str] = {"理财方法": "理财", "量化方法": "量化"}


# ---- 数据读取（可注入，测试隔离） ----


def _store():
    from pa_mcp.config import get_settings
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(get_settings().database.path)
    store.connect()
    return store


def _load_holdings(store=None) -> list[str]:
    """读持仓代码列表（portfolio 表；无持仓返回空列表）。"""
    own = store is None
    store = store or _store()
    try:
        if not store.table_exists("portfolio"):
            return []
        df = store.query_df("SELECT symbol FROM portfolio", [])
        return [str(s) for s in df["symbol"].tolist()] if not df.empty else []
    except Exception as e:  # noqa: BLE001
        logger.debug("load holdings failed", error=str(e))
        return []
    finally:
        if own:
            store.close()


def _load_klines(store=None, symbols: Optional[list[str]] = None,
                 limit: int = KLINE_LIMIT) -> dict[str, pd.DataFrame]:
    """批量读日线（{symbol: DataFrame 升序}），单票失败跳过。"""
    own = store is None
    store = store or _store()
    try:
        klines: dict[str, pd.DataFrame] = {}
        for sym in symbols or []:
            try:
                df = store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT ?", [sym, limit])
                if df is None or df.empty:
                    continue
                klines[sym] = df.sort_values("date").reset_index(drop=True)
            except Exception as e:  # noqa: BLE001
                logger.debug("kline load failed", symbol=sym, error=str(e))
        return klines
    finally:
        if own:
            store.close()


# ---- 估值快照预取（async，避免分析器内部 asyncio.run 撞 running loop） ----


async def _fetch_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """预取估值快照 {symbol: {pe, pb, price}}。

    graham/value_momentum 缺省会在内部 asyncio.run 拉网络——在 async
    上下文中调用会失败（cannot be called from a running event loop）。
    这里统一 await 腾讯实时估值，网络失败降级为库内收盘价。
    """
    quotes: dict[str, dict[str, Any]] = {}
    try:
        from pa_mcp.data.sources.tencent_adapter import TencentAdapter
        adapter = TencentAdapter()
        for sym in symbols:
            try:
                q = await adapter.get_realtime_quote(sym)
                if q:
                    quotes[sym] = {
                        "pe": q.get("pe"), "pb": q.get("pb"),
                        "price": q.get("price") or q.get("last_close"),
                    }
            except Exception as e:  # noqa: BLE001
                logger.debug("quote fetch failed", symbol=sym, error=str(e))
    except Exception as e:  # noqa: BLE001
        logger.debug("quote fetch unavailable", error=str(e))
    return quotes


# ---- 上半部分：方法可信度 ----


def _evaluate_credibility_quant(klines: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """量化方法可信度：全策略事件研究对比（复用 compare_all_strategies）。"""
    if not klines:
        return {"error": "无行情数据，无法评估量化方法"}
    from pa_mcp.research.strategy_compare import compare_all_strategies
    try:
        result = compare_all_strategies(klines)
        if "error" in result:
            return {"error": result["error"]}
        rows = []
        for r in result["strategies"]:
            h5 = r.get("horizons", {}).get("5", {})
            rows.append({
                "strategy": r["strategy"],
                "total_events": r["total_events"],
                "n_stocks": r["n_stocks"],
                "excess_5d_pct": r["excess_5d_pct"],
                "win_rate_5d": h5.get("win_rate"),
                "edge_stock_fraction": r["edge_stock_fraction"],
                "useful": r["useful"],
            })
        return {
            "n_strategies": result["n_strategies"],
            "strategies": rows,
            "useful_strategies": result.get("useful_strategies", []),
        }
    except Exception as e:  # noqa: BLE001
        logger.error("quant credibility failed", error=str(e))
        return {"error": f"量化方法评估失败：{e}"}


def _evaluate_method_credibility(klines: dict[str, pd.DataFrame],
                                 quotes: Optional[dict] = None) -> list[dict[str, Any]]:
    """理财方法评估状态：用当前池跑各方法，报告样本与结论（数据充分性评价）。"""
    if not klines:
        return []
    symbols = list(klines.keys())
    rows: list[dict[str, Any]] = []
    # 格雷厄姆：筛选出 N 只 / 池规模
    try:
        from pa_mcp.research.graham import GrahamScreener
        results = GrahamScreener().screen(symbols, quotes=quotes)
        passed = sum(1 for r in results if r.score >= r.total_scored)
        rows.append({
            "method": "graham", "name_zh": "格雷厄姆筛选",
            "status": f"池内 {len(results)}/{len(symbols)} 只有财务数据可评",
            "conclusion": f"通过 {passed} 只（分数≥总条数）" if results
                          else "数据不足",
            "verification": METHOD_EVALUATORS["graham"]["verification"],
        })
    except Exception as e:  # noqa: BLE001
        rows.append({"method": "graham", "name_zh": "格雷厄姆筛选",
                     "status": "评估失败", "conclusion": str(e)[:60],
                     "verification": METHOD_EVALUATORS["graham"]["verification"]})
    # 价值动量：top 名单
    try:
        from pa_mcp.research.value_momentum import ValueMomentumScreen
        vm = ValueMomentumScreen().screen(symbols, top_n=5, quotes=quotes,
                                          klines=klines)
        best = "、".join(vm.get("best_candidates", []) or []) or "无"
        rows.append({
            "method": "value_momentum", "name_zh": "价值×动量",
            "status": f"池内 {len(vm.get('selection', []))} 只可评分",
            "conclusion": f"双高（便宜且走强）：{best}",
            "verification": METHOD_EVALUATORS["value_momentum"]["verification"],
        })
    except Exception as e:  # noqa: BLE001
        rows.append({"method": "value_momentum", "name_zh": "价值×动量",
                     "status": "评估失败", "conclusion": str(e)[:60],
                     "verification": METHOD_EVALUATORS["value_momentum"]["verification"]})
    # CANSLIM：池内要素通过数
    try:
        from pa_mcp.research.canslim import CanslimScanner
        hits = CanslimScanner().scan(pool=symbols, top_n=20)
        rows.append({
            "method": "canslim", "name_zh": "CANSLIM",
            "status": f"池内 {len(hits)} 只达标（需财务+行情）",
            "conclusion": "、".join(
                str(h.symbol) for h in hits[:5]) if hits else "无达标",
            "verification": METHOD_EVALUATORS["canslim"]["verification"],
        })
    except Exception as e:  # noqa: BLE001
        rows.append({"method": "canslim", "name_zh": "CANSLIM",
                     "status": "评估失败", "conclusion": str(e)[:60],
                     "verification": METHOD_EVALUATORS["canslim"]["verification"]})
    return rows


def evaluate_credibility(pool: Optional[list[str]] = None,
                         store=None,
                         quotes: Optional[dict] = None) -> dict[str, Any]:
    """方法可信度评价（上半部分）。

    量化方法 = 全策略事件研究对比；理财方法 = 当前池评估状态。
    pool 缺省从 store 有行情的股票取（持仓优先）。
    quotes 为可选估值快照（_fetch_quotes 预取，避免分析器内部 asyncio.run
    在 async 上下文中失败）。
    """
    own = store is None
    store = store or _store()
    try:
        symbols = pool or _load_holdings(store)
        if not symbols:
            # 无持仓时用库内最近有行情的股票（取前 6 只）
            try:
                df = store.query_df(
                    "SELECT DISTINCT symbol FROM kline_daily "
                    "ORDER BY symbol LIMIT 6", [])
                symbols = [str(s) for s in df["symbol"].tolist()]
            except Exception:  # noqa: BLE001
                symbols = []
        if not symbols:
            return {"error": "无行情数据，无法评估方法可信度"}
        klines = _load_klines(store, symbols)
        quant = _evaluate_credibility_quant(klines)
        methods = _evaluate_method_credibility(klines, quotes=quotes)
        return {"pool": symbols, "quant": quant, "methods": methods}
    finally:
        if own:
            store.close()


# ---- 下半部分：持仓×方法 ----


async def _eval_consensus(symbol: str,
                          kline_df: Optional[pd.DataFrame]) -> dict[str, Any]:
    """综合决策信号（5 源加权投票，异步）。"""
    from pa_mcp.research.consensus import ConsensusAnalyzer
    c = await ConsensusAnalyzer().analyze(symbol, kline_df=kline_df)
    if "error" in c:
        return {"label": "数据不足", "detail": "", "ok": False}
    return {"label": f"{c['signal']} {c['strength']:.0%}",
            "detail": f"{c['level']}·一致度{c.get('agreement', 0):.0%}",
            "ok": True}


def _eval_graham(symbol: str, klines: dict[str, pd.DataFrame],
                 quotes: Optional[dict] = None) -> dict[str, Any]:
    """格雷厄姆：得分 / 安全边际。"""
    from pa_mcp.research.graham import GrahamScreener
    results = GrahamScreener().screen([symbol], quotes=quotes)
    if not results:
        return {"label": "数据不足", "detail": "", "ok": False}
    r = results[0]
    mos = r.margin_of_safety_pct
    return {"label": f"{r.score}/{r.total_scored} 通过",
            "detail": f"安全边际 {mos:+.0f}%" if mos is not None else "安全边际—",
            "ok": r.score >= 1}


def _eval_value_momentum(symbol: str,
                         klines: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """价值×动量：象限 + 复合分。"""
    from pa_mcp.research.value_momentum import ValueMomentumScreen
    vm = ValueMomentumScreen().screen([symbol], top_n=5, klines=klines)
    sel = vm.get("selection") or {}
    if symbol not in sel:
        return {"label": "数据不足", "detail": "", "ok": False}
    s = sel[symbol]
    return {"label": s.get("quadrant", "—"),
            "detail": f"复合 {s.get('composite', 0):.2f}",
            "ok": "便宜" in str(s.get("quadrant", ""))}


def _eval_canslim(symbol: str,
                  klines: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """CANSLIM：单票要素判定。"""
    from pa_mcp.research.canslim import CanslimScanner
    hits = CanslimScanner().scan(pool=[symbol], top_n=1)
    if not hits:
        return {"label": "未达标/数据不足", "detail": "七要素或财务数据缺", "ok": False}
    h = hits[0]
    return {"label": "达标", "detail": f"评分 {h.score}/{h.total_scored}",
            "ok": True}


def _eval_chan(symbol: str,
               kline_df: Optional[pd.DataFrame]) -> dict[str, Any]:
    """缠论：结构位置 + 背驰信号。"""
    if kline_df is None or len(kline_df) < 60:
        return {"label": "数据不足", "detail": "需 ≥60 根日线", "ok": False}
    from pa_mcp.engine.indicators.chan import chan_analysis
    r = chan_analysis(kline_df)
    pos = r.get("position", "") if isinstance(r, dict) else ""
    beichi = r.get("beichi_signal", "") if isinstance(r, dict) else ""
    zh = {"上方": "偏多", "下方": "偏空", "内部": "震荡"}.get(pos, pos)
    return {"label": zh or "结构—",
            "detail": (f"背驰:{beichi}" if beichi else "无背驰"),
            "ok": zh == "偏多"}


def _evaluate_one(symbol: str, klines: dict[str, pd.DataFrame],
                  quotes: Optional[dict] = None) -> dict[str, Any]:
    """单票 × 全部理财/量化方法（逐个降级，不互相拖累）。"""
    kline_df = klines.get(symbol)
    row: dict[str, Any] = {"symbol": symbol}
    for mid, spec in METHOD_EVALUATORS.items():
        try:
            if mid == "graham":
                row["graham"] = _eval_graham(symbol, klines, quotes)
            elif mid == "value_momentum":
                row["value_momentum"] = _eval_value_momentum(symbol, klines)
            elif mid == "canslim":
                row["canslim"] = _eval_canslim(symbol, klines)
            elif mid == "chan":
                row["chan"] = _eval_chan(symbol, kline_df)
        except Exception as e:  # noqa: BLE001
            logger.debug("method eval failed", method=mid, symbol=symbol,
                         error=str(e))
            row[mid] = {"label": "评估失败", "detail": str(e)[:40], "ok": False}
    return row


async def evaluate_holdings(symbols: Optional[list[str]] = None,
                            store=None,
                            quotes: Optional[dict] = None) -> dict[str, Any]:
    """持仓×方法评价（下半部分）。

    symbols 缺省读 portfolio 表；klines/store 可注入（测试隔离）；
    quotes 为可选估值快照（_fetch_quotes 预取）。
    """
    own = store is None
    store = store or _store()
    try:
        syms = symbols or _load_holdings(store)
        if not syms:
            return {"error": "无持仓（先添加持仓），无法做持仓×方法评价"}
        klines = _load_klines(store, syms)
        rows: list[dict[str, Any]] = []
        for sym in syms:
            try:
                c = await _eval_consensus(sym, klines.get(sym))
            except Exception as e:  # noqa: BLE001
                logger.debug("consensus failed", symbol=sym, error=str(e))
                c = {"label": "评估失败", "detail": str(e)[:40], "ok": False}
            row = _evaluate_one(sym, klines, quotes)
            row["consensus"] = c
            rows.append(row)
        return {"holdings": syms, "rows": rows}
    finally:
        if own:
            store.close()


# ---- 组合报告（上半 + 下半） ----


def _fmt_cell(v: Optional[dict[str, Any]]) -> str:
    if not v or not v.get("label"):
        return "—"
    detail = v.get("detail") or ""
    mark = "🟢" if v.get("ok") else "⚪"
    return f"{mark} {v['label']}" + (f"（{detail}）" if detail else "")


def format_report(cred: dict[str, Any],
                  hold: dict[str, Any]) -> str:
    """上下两部分 → markdown。"""
    lines = ["## 🔍 开源方法评价", ""]
    # ---- 上半部分：方法可信度 ----
    lines.append("### ① 量化方法可信度（全策略事件研究对比）")
    quant = cred.get("quant") or {}
    if "error" in quant:
        lines.append(f"（{quant['error']}）")
    else:
        lines.append("| 策略 | 事件数 | 股票数 | 5日超额% | 胜率% | 有效 |")
        lines.append("|---|---|---|---|---|---|")
        for r in quant.get("strategies", []):
            wr = r.get("win_rate_5d")
            lines.append(f"| {r['strategy']} | {r['total_events']} | "
                         f"{r['n_stocks']} | {r['excess_5d_pct']:+.2f} | "
                         f"{wr if wr is not None else '—'} | "
                         f"{'✅' if r['useful'] else '❌'} |")
        if quant.get("useful_strategies"):
            lines.append("\n**可信（有效判定）**："
                         + "、".join(quant["useful_strategies"]))
        else:
            lines.append("\n**结论**：当前数据下无策略达到有效性门槛"
                         "（无稳定超额——真实市场常态，不是方法错了）")
    lines.append("")
    lines.append("### ② 理财方法评估状态")
    lines.append("| 方法 | 可评估范围 | 当前结论 | 可信度说明 |")
    lines.append("|---|---|---|---|")
    for m in cred.get("methods", []):
        lines.append(f"| {m['name_zh']} | {m['status']} | {m['conclusion']} | "
                     f"{m['verification'][:36]}… |")
    if not cred.get("methods"):
        lines.append("| — | 数据不足 | — | — |")
    lines.append("")
    # ---- 下半部分：持仓×方法 ----
    lines.append("### ③ 持仓×方法评价")
    if "error" in hold:
        lines.append(f"（{hold['error']}）")
    else:
        lines.append("| 代码 | 格雷厄姆 | 价值动量 | CANSLIM | 缠论 | 综合信号 |")
        lines.append("|---|---|---|---|---|---|")
        for row in hold.get("rows", []):
            lines.append(
                f"| {row['symbol']} | {_fmt_cell(row.get('graham'))} | "
                f"{_fmt_cell(row.get('value_momentum'))} | "
                f"{_fmt_cell(row.get('canslim'))} | {_fmt_cell(row.get('chan'))} | "
                f"{_fmt_cell(row.get('consensus'))} |")
        lines.append("\n🟢=偏多/达标，⚪=中性/数据不足。同一持仓多方法同向时可信度更高；"
                     "反向时先以风险控制为准。")
    lines.append("\n---")
    lines.append("*事件研究不含交易成本；理财方法为确定性规则。研究参考，非投资建议。*")
    return "\n".join(lines)


async def evaluate_methods_report(store=None,
                                  symbols: Optional[list[str]] = None) -> dict[str, Any]:
    """开源方法评价组合入口（UI / MCP 共用）。

    Args:
        store: 可注入数据仓库（测试隔离）
        symbols: 股票池（缺省读持仓）

    Returns:
        credibility: 方法可信度（上半部分）
        holdings: 持仓×方法（下半部分）
        report: markdown
    """
    own = store is None
    store = store or _store()
    try:
        syms = symbols or _load_holdings(store)
        quotes = await _fetch_quotes(syms) if syms else {}
        cred = evaluate_credibility(pool=symbols, store=store, quotes=quotes)
        hold = await evaluate_holdings(symbols=symbols, store=store,
                                       quotes=quotes)
        report = format_report(cred, hold)
        return {
            "credibility": cred,
            "holdings": hold,
            "report": report,
        }
    finally:
        if own:
            store.close()
