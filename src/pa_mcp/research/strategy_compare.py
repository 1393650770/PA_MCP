# [AI:BEGIN]
# PA_MCP - Research: 全策略事件研究对比（多策略同台检验）
#
# 复用现有能力（非必要不造轮子）：
#   - StrategyRegistry.auto_discover：全部注册策略（含 7 大方法）
#   - signal_forward_returns：信号前瞻收益 vs 无条件基准
#   - 多股票 × 多策略 → 聚合对比：哪策略真有预测力
#
# 判定规则（可追溯）：
#   - 有效 = 跨股票多数 has_edge 且聚合超额 > 0
#   - 输出：每策略胜率/超额/事件数/覆盖股票数/判定
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

HORIZONS = (5, 10, 20)
MIN_EVENTS = 10      # 每策略最少事件数（否则样本不足）
EDGE_MAJORITY = 0.5  # has_edge 股票占比门槛


def compare_all_strategies(
    klines: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = HORIZONS,
    min_events: int = MIN_EVENTS,
) -> dict[str, Any]:
    """全策略事件研究对比。

    Args:
        klines: {symbol: 日线 DataFrame}（各股 ≥ 120 根）
        horizons: 前瞻窗口
        min_events: 每策略最少事件数

    Returns:
        strategies: 每策略聚合结果（按 5 日超额排序）
        ranking: 有效策略清单
    """
    from pa_mcp.engine.strategies.base import StrategyRegistry
    from pa_mcp.research.event_study import signal_forward_returns

    if not klines:
        return {"error": "无行情数据"}

    registry = StrategyRegistry()
    registry.auto_discover()
    strategies = registry.list_all()

    rows: list[dict[str, Any]] = []
    for name in strategies:
        agg = {h: {"n_events": 0, "win_rate": 0.0, "excess": 0.0,
                   "has_edge": 0, "n_stocks": 0}
               for h in horizons}
        total_events = 0
        edge_stocks = 0
        stock_count = 0
        for sym, df in klines.items():
            try:
                inst = registry.get(name)
                signals = inst.generate_signals(df.copy())
            except Exception as e:  # noqa: BLE001
                logger.debug("strategy failed", name=name, symbol=sym,
                             error=str(e))
                continue
            if not signals:
                continue
            sig_df = pd.DataFrame([{
                "symbol": sym,
                "date": (getattr(s, "signal_time", None)
                         or str(getattr(s, "timestamp", ""))[:10]),
                "direction": getattr(s, "direction", "neutral").value
                            if hasattr(getattr(s, "direction", None), "value")
                            else str(getattr(s, "direction", "neutral")),
                "strategy_name": name,
            } for s in signals])
            if sig_df.empty:
                continue
            results = signal_forward_returns(df, sig_df, list(horizons))
            if not results:
                continue
            stock_count += 1
            total_events += len(signals)
            for r in results:
                a = agg[r.horizon]
                a["n_events"] += r.n_events
                a["win_rate"] += r.win_rate_pct
                a["excess"] += r.excess_return_pct
                a["has_edge"] += 1 if r.has_edge else 0
                a["n_stocks"] += 1

        if total_events < min_events or stock_count == 0:
            continue
        # 聚合（按股票数平均）
        for h in horizons:
            a = agg[h]
            if a["n_stocks"] > 0:
                a["win_rate"] = round(a["win_rate"] / a["n_stocks"], 1)
                a["excess"] = round(a["excess"] / a["n_stocks"], 2)
        edge_frac = round(edge_stocks / stock_count, 2) if stock_count else 0.0
        # 有效判定：5 日超额 > 0 且 多数股票 has_edge（任一 horizon）
        best_edge = max(a["has_edge"] for a in agg.values()) / stock_count \
            if stock_count else 0.0
        excess5 = agg[horizons[0]]["excess"] if horizons else 0.0
        useful = excess5 > 0 and best_edge >= EDGE_MAJORITY
        rows.append({
            "strategy": name,
            "description": (registry.get(name).description or "")[:40],
            "total_events": total_events,
            "n_stocks": stock_count,
            "edge_stock_fraction": best_edge,
            "horizons": {str(h): agg[h] for h in horizons},
            "excess_5d_pct": excess5,
            "useful": bool(useful),
        })

    if not rows:
        return {"error": "无策略产生足够信号（数据不足或样本过短）"}

    rows.sort(key=lambda r: r["excess_5d_pct"], reverse=True)
    useful = [r for r in rows if r["useful"]]
    return {
        "method": f"全策略事件研究对比（{len(rows)} 策略 × {len(klines)} 股票 × "
                  f"{len(horizons)} 个前瞻窗口）",
        "n_strategies": len(rows),
        "strategies": rows,
        "ranking": [r["strategy"] for r in rows],
        "useful_strategies": [r["strategy"] for r in useful],
        "n_useful": len(useful),
    }


def format_compare(result: dict[str, Any]) -> str:
    """对比结果 → markdown。"""
    if "error" in result:
        return f"策略对比不可用：{result['error']}"
    lines = [
        f"## 🏁 全策略事件研究对比",
        f"**{result['method']}**",
        "",
        "| 排名 | 策略 | 事件数 | 股票数 | 5日超额% | 10日超额% | 20日超额% | 有效 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(result["strategies"], 1):
        h5 = r["horizons"]["5"] if "5" in r["horizons"] else {}
        h10 = r["horizons"].get("10", {})
        h20 = r["horizons"].get("20", {})
        lines.append(
            f"| {i} | {r['strategy']} | {r['total_events']} | "
            f"{r['n_stocks']} | {h5.get('excess', 0):+.2f} | "
            f"{h10.get('excess', 0):+.2f} | {h20.get('excess', 0):+.2f} | "
            f"{'✅' if r['useful'] else '❌'} |")
    if result.get("useful_strategies"):
        lines.append("\n**有效策略**（5 日超额 > 0 且多数股票 has_edge）："
                     + "、".join(result["useful_strategies"]))
    else:
        lines.append("\n**结论**：当前数据下无策略达到有效性门槛"
                     "（多数方法无稳定超额——真实市场常态）")
    lines.append("\n*事件研究 = 信号后持有 N 日 vs 无条件基准（不含交易成本）。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)
