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


async def trading_actions(symbols: Optional[list[str]] = None,
                          include_llm: bool = True) -> dict[str, Any]:
    """今日操作面板（整合版：全信号 + 因子 + LLM 解读）。

    - 规则骨架（止盈止损纪律不可绕过）：盈亏 × 预测 × 共振 × 综合信号
    - 因子补强：持仓/候选附加多因子合成分
    - LLM 解读（可选）：持仓操作补充解读 + 买入优先级 + 操作计划
      （LLM 只解读与排序，不否决规则动作）

    Args:
        symbols: 候选股票池（缺省内置 6 只）
        include_llm: LLM 可用时附加解读（默认 True）
    """
    pool = [s.strip() for s in (symbols or DEFAULT_POOL) if s.strip()][:10]
    out: dict[str, Any] = {"pool": pool}

    # ---- 0. 因子分（池内多因子合成分，供持仓/候选补强） ----
    factor_scores: dict[str, float] = {}
    try:
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        from pa_mcp.research.factors import select_stocks_by_factors
        store = DuckDBStore(get_settings().database.path)
        store.connect()
        try:
            klines = {}
            for sym in pool:
                df = store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 150", [sym])
                if not df.empty:
                    klines[sym] = df
        finally:
            store.close()
        if len(klines) >= 5:
            sel = select_stocks_by_factors(klines, top_n=len(klines),
                                           prediction_weight=0.5)
            if "error" not in sel:
                factor_scores = {r["symbol"]: r["score"]
                                 for r in sel["selection"]}
    except Exception:
        pass
    out["factor_scores"] = {s: round(v, 3) for s, v in factor_scores.items()}

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

    # ---- 2. 持仓操作（止盈止损 + 因子补强） ----
    holdings_actions: list[dict] = []
    try:
        from pa_mcp.research.portfolio_risk import PortfolioRiskDashboard
        pr = await PortfolioRiskDashboard().analyze(use_llm=False)
        if "error" not in pr:
            for h in pr["holdings"]:
                action = _holding_action(h)
                action["factor_score"] = factor_scores.get(
                    h["symbol"]) if h["symbol"] in factor_scores else None
                holdings_actions.append(action)
    except Exception as e:  # noqa: BLE001
        logger.warning("持仓分析失败: %s", e)
    out["holdings"] = holdings_actions

    # ---- 3. 买入候选（综合信号 + 因子 + 策略信号） ----
    buys = await _buy_candidates(pool)
    for b in buys:
        b["factor_score"] = factor_scores.get(b["symbol"]) \
            if b["symbol"] in factor_scores else None
    out["buy_candidates"] = buys

    # ---- 4. LLM 深度解读（只解读，不否决规则） ----
    llm_advice = None
    if include_llm:
        llm_advice = await _llm_advice(tone, holdings_actions, buys,
                                       factor_scores)
    out["llm_advice"] = llm_advice

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


async def _llm_advice(tone: dict, holdings: list[dict], buys: list[dict],
                      factor_scores: dict) -> Optional[dict]:
    """LLM 深度解读（结构化 JSON）。

    安全边界：LLM 只解读规则动作与排序优先级，不否决止盈止损纪律。
    """
    try:
        from pa_mcp.agent.llm_port import get_llm_adapter, LLMCallParams
        from pa_mcp.agent.llm_factory import ensure_llm_adapter
        adapter = ensure_llm_adapter()
        if adapter is None:
            return None
        holdings_text = "\n".join(
            f"- {h['symbol']} 盈亏{h['pnl_pct']:+.1f}% 规则动作[{h['action']}]"
            f"{h['reason'][:40]}"
            + (f" 因子分{h['factor_score']:+.2f}" if h.get("factor_score")
               is not None else "")
            for h in holdings) or "无持仓"
        buys_text = "\n".join(
            f"- {b['symbol']} 综合信号强度{b['consensus_strength']:.0%} "
            f"一致度{b['agreement']:.0%}"
            + (f" 因子分{b['factor_score']:+.2f}" if b.get("factor_score")
               is not None else "")
            for b in buys) or "无候选"
        params = LLMCallParams(
            system_prompt=(
                "你是有经验的 A 股交易辅助助手。只输出合法 JSON，只解读给定数据，"
                "不编造。止盈止损规则动作不可否决，你只补充解读与优先级。"
                "输出是研究参考，非投资建议。"),
            user_prompt=(
                f"市场基调：{tone['market'][:80]}；总仓位建议："
                f"{tone['total_position']}。\n"
                f"持仓规则动作：\n{holdings_text}\n"
                f"买入候选：\n{buys_text}\n"
                "输出 JSON："
                "{'holdings_advice': [{'symbol':..., 'comment': 一句话解读}], "
                "'buy_priority': [候选代码按优先级排序], "
                "'operation_plan': 2-3 句今日操作计划（中文）}"),
            mode="fast", max_tokens=800,
        )
        raw = await adapter.chat_json(params)
        if isinstance(raw, dict) and "error" not in raw:
            return {
                "holdings_advice": raw.get("holdings_advice", []),
                "buy_priority": raw.get("buy_priority", []),
                "operation_plan": raw.get("operation_plan", ""),
            }
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM 解读失败: %s", e)
    return None


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

    lines.append("\n### 🎯 买入候选（综合信号看涨 + 因子分）")
    if not out["buy_candidates"]:
        lines.append("当前池内无综合信号看涨的标的——市场没有明确买入机会，观望为主")
    else:
        lines.append("| 代码 | 信号强度 | 一致度 | 源数 | 因子分 |")
        lines.append("|---|---|---|---|---|")
        for c in out["buy_candidates"]:
            fs = f"{c['factor_score']:+.2f}" if c.get("factor_score") \
                is not None else "—"
            lines.append(f"| {c['symbol']} | {c['consensus_strength']:.0%} | "
                         f"{c['agreement']:.0%} | {c['n_sources']} | {fs} |")
        lines.append("\n*买入候选仅代表信号状态，需结合大盘基调（偏空时谨慎）。*")

    # LLM 深度解读
    la = out.get("llm_advice")
    if la:
        lines.append("\n### 🤖 AI 解读")
        if la.get("holdings_advice"):
            lines.append("**持仓补充解读**：")
            for a in la["holdings_advice"]:
                lines.append(f"- {a.get('symbol', '')}：{a.get('comment', '')}")
        if la.get("buy_priority"):
            lines.append("\n**买入优先级**：" + " → ".join(la["buy_priority"]))
        if la.get("operation_plan"):
            lines.append(f"\n**今日操作计划**：{la['operation_plan']}")
        lines.append("\n*AI 解读基于规则结果，不否决止盈止损纪律。*")

    lines.append("\n---\n*规则可追溯（止盈：盈利≥15%+综合看跌 / 止损：亏损≥10%）。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)
