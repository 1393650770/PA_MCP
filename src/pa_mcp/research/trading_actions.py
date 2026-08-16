# [AI:BEGIN]
# PA_MCP - Research: 💰 今日操作面板（止盈止损 / 买入候选 / 操作建议）
#
# 核心需求落地——回答三个问题：
#   1. 我的持仓什么时候止损/止盈
#   2. 现在市场可以买什么
#   3. 现在适合做什么操作
#
# 规则（确定性、可追溯，全部复用已有信号）：
#   止盈：盈利≥15% 且综合信号看跌 → 分批止盈；盈利≥10% 且 5d 预测看跌 → 考虑止盈
#   止损：亏损≥10% → 纪律止损；亏损≥8% 且综合信号看跌 → 止损
#   持有：其他 → 持有观察（给触发价：跌破 MA20 止损 / 涨破压力止盈）
#   买入候选：综合信号看涨强 + 策略买入信号 交集优先，其次单强
#   操作基调：市场结构 bias + 情绪矩阵 → 总仓位增减建议
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

TAKE_PROFIT_1 = 15.0   # 强止盈阈值：盈利≥15% + 综合看跌
TAKE_PROFIT_2 = 10.0   # 弱止盈阈值：盈利≥10% + 预测看跌
STOP_LOSS_1 = -10.0    # 纪律止损：亏损≥10%
STOP_LOSS_2 = -8.0     # 止损：亏损≥8% + 综合看跌
DEFAULT_POOL = ["000001", "600036", "300750", "000858", "600519", "601318"]


async def trading_actions(symbols: Optional[list[str]] = None) -> dict[str, Any]:
    """今日操作面板。

    Args:
        symbols: 候选股票池（缺省内置 6 只）
    """
    pool = [s.strip() for s in (symbols or DEFAULT_POOL) if s.strip()][:10]
    out: dict[str, Any] = {"pool": pool}

    # ---- 1. 操作基调（市场结构 + 情绪矩阵） ----
    tone = {"bias": "未知", "market": "—", "position_advice": "—",
            "risk_level": "—"}
    try:
        from pa_mcp.research.market_structure import MarketStructureAnalyzer
        ms = await MarketStructureAnalyzer().analyze(use_network=True)
        if ms["index"]["rows"] > 0:
            j = ms["joint"]
            tone["bias"] = j["bias"]
            tone["market"] = j["verdict"][:80]
    except Exception:
        pass
    try:
        from pa_mcp.research.regime_matrix import RegimeMatrixAnalyzer
        m = RegimeMatrixAnalyzer().analyze()
        tone["market"] += f"；{m['matrix_cell']} → {m['regime_label']}"
        tone["position_advice"] = m["advice"]
        tone["risk_level"] = m["risk"]
    except Exception:
        pass
    # 基调 → 总仓位建议
    if tone["bias"] == "偏空" or "冰点" in tone["market"] \
            or "退潮" in tone["market"]:
        tone["total_position"] = "降低仓位（≤30%），以减仓/观望为主"
    elif tone["bias"] == "偏多" or "发酵" in tone["market"] \
            or "启动" in tone["market"]:
        tone["total_position"] = "可积极跟踪（≤70%），逢强参与"
    else:
        tone["total_position"] = "中性（≤50%），等待方向明确"
    out["tone"] = tone

    # ---- 2. 持仓操作（止盈止损） ----
    holdings_actions: list[dict] = []
    try:
        from pa_mcp.research.portfolio_risk import PortfolioRiskDashboard
        pr = await PortfolioRiskDashboard().analyze(use_llm=False)
        if "error" not in pr:
            for h in pr["holdings"]:
                action = _holding_action(h)
                holdings_actions.append(action)
    except Exception as e:  # noqa: BLE001
        logger.warning("持仓分析失败: %s", e)
    out["holdings"] = holdings_actions

    # ---- 3. 买入候选 ----
    buys = await _buy_candidates(pool)
    out["buy_candidates"] = buys

    # ---- 整合文本 ----
    out["report"] = _render(out)
    return out


def _holding_action(h: dict) -> dict[str, Any]:
    """单只持仓操作判定（可追溯规则）。"""
    sym = h["symbol"]
    pnl = h.get("pnl_pct", 0.0)
    price = h.get("price", 0.0)
    cost = h.get("cost", 0.0)
    pred = h.get("prediction") or {}
    res = h.get("resonance") or {}
    con = h.get("consensus") or {}
    pred_down = pred.get("direction") == "down"
    res_down = res.get("signal") == "down" and res.get("strength", 0) >= 0.7
    con_down = con.get("signal") == "down" and con.get("strength", 0) >= 0.6
    sell_signal = pred_down or res_down or con_down

    # 止盈止损触发价
    stop_level = round(price * 0.92, 2) if price else None      # -8% 参考
    take_level = round(cost * 1.15, 2) if cost else None        # +15% 参考

    if pnl >= TAKE_PROFIT_1 and con_down:
        action, reason = "分批止盈", (
            f"盈利 {pnl:+.1f}% 且综合信号看跌（{con.get('strength', 0):.0%}）——"
            f"落袋为安，可先减半")
    elif pnl >= TAKE_PROFIT_1 and res_down:
        action, reason = "考虑止盈", (
            f"盈利 {pnl:+.1f}% 且共振看跌（{res.get('strength', 0):.0%}）")
    elif pnl >= TAKE_PROFIT_2 and pred_down:
        action, reason = "考虑止盈", (
            f"盈利 {pnl:+.1f}% 且 5d 预测看跌——趋势可能转向")
    elif pnl <= STOP_LOSS_1:
        action, reason = "纪律止损", (
            f"亏损 {pnl:+.1f}% 已触发止损纪律（-10%），"
            f"按计划离场保护本金")
    elif pnl <= STOP_LOSS_2 and con_down:
        action, reason = "止损", (
            f"亏损 {pnl:+.1f}% 且综合信号看跌——风险信号叠加，止损离场")
    elif sell_signal:
        action, reason = "减仓", (
            f"信号转弱（预测/共振/综合有看跌信号），先减仓观察")
    else:
        action, reason = "持有观察", (
            f"无止损止盈信号；跌破 {stop_level} 止损 / 涨破 {take_level} 分批止盈"
            if stop_level and take_level else "等待信号")
    return {
        "symbol": sym, "name": h.get("name", ""),
        "pnl_pct": pnl, "price": price, "cost": cost,
        "action": action, "reason": reason,
        "stop_level": stop_level, "take_level": take_level,
        "signal_flags": {"pred_down": pred_down, "res_down": res_down,
                         "con_down": con_down},
    }


async def _buy_candidates(pool: list[str]) -> list[dict]:
    """买入候选：综合信号看涨（强）+ 策略买入信号 交集优先。"""
    candidates = []
    try:
        from pa_mcp.research.consensus import ConsensusAnalyzer
        for sym in pool:
            c = await ConsensusAnalyzer().analyze(sym)
            if "error" in c or c["signal"] != "up":
                continue
            strength = c["strength"]
            candidates.append({
                "symbol": sym,
                "consensus_strength": strength,
                "level": c["level"],
                "agreement": c["agreement"],
                "n_sources": len(c["sources"]),
                "reason": c.get("note", "")[:40],
            })
    except Exception:
        pass
    candidates.sort(key=lambda x: x["consensus_strength"], reverse=True)
    return candidates[:5]


def _render(out: dict) -> str:
    lines = [
        "## 💰 今日操作面板",
        "",
        "### 🧭 操作基调（市场环境）",
        f"- **市场结构**：{out['tone']['bias']}——{out['tone']['market'][:90]}",
        f"- **总仓位建议**：**{out['tone']['total_position']}**",
    ]
    if out["tone"]["position_advice"] and out["tone"]["position_advice"] != "—":
        lines.append(f"- 矩阵建议：{out['tone']['position_advice'][:60]}")
    if out["tone"]["risk_level"] and out["tone"]["risk_level"] != "—":
        lines.append(f"- 风险提示：{out['tone']['risk_level'][:60]}")

    lines.append("\n### 📌 你的持仓操作")
    if not out["holdings"]:
        lines.append("无持仓记录（先在组合管理添加持仓，这里会自动给出止损止盈建议）")
    else:
        lines.append("| 代码 | 盈亏% | 建议动作 | 触发价(止损/止盈) | 依据 |")
        lines.append("|---|---|---|---|---|")
        for h in out["holdings"]:
            lv = f"{h['stop_level']} / {h['take_level']}" \
                if h["stop_level"] and h["take_level"] else "—"
            lines.append(f"| {h['symbol']} | {h['pnl_pct']:+.1f}% | "
                         f"**{h['action']}** | {lv} | {h['reason'][:50]} |")

    lines.append("\n### 🎯 买入候选（综合信号看涨）")
    if not out["buy_candidates"]:
        lines.append("当前池内无综合信号看涨的标的——市场没有明确买入机会，观望为主")
    else:
        lines.append("| 代码 | 强度 | 一致度 | 信号源数 |")
        lines.append("|---|---|---|---|")
        for c in out["buy_candidates"]:
            lines.append(f"| {c['symbol']} | {c['consensus_strength']:.0%} | "
                         f"{c['agreement']:.0%} | {c['n_sources']} |")
        lines.append("\n*买入候选仅代表信号状态，需结合大盘基调（偏空时谨慎）。*")

    lines.append("\n---\n*规则可追溯（止盈：盈利≥15%+综合看跌 / 止损：亏损≥10%）。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)
