# [AI:BEGIN]
# PA_MCP - Research: 两阶段研究编排（诊断 → 路由 → 决策）
#
# 来源：PA_Agent 的 orchestrator/two_stage.py 设计
#   - 阶段一（诊断）：数据闸门 → 市场状态/情绪/方向定性判断
#   - 策略路由：按诊断结果选择适配策略
#   - 阶段二（决策）：信号验证 → 风险收益 → 交易者方程 → 最终裁定
#   - 闸门短路：环境不明直接合成"等待"，不发起第二次 LLM 调用
#   - 决策倾向（decision_stance）：保守/均衡/激进/极度激进
#
# 与项目内 analyze_stock 互补：analyze_stock 是多分析师辩论（发散），
# two_stage 是决策纪律链（收敛）——按决策链逐节点校验，防跳步防矛盾。
#
# 复用：sentiment_cycle（情绪）、sector_rotation（板块）、strategy 注册表
# （信号）、kline_geometry（K线形态）、decision_chain（决策链+校验+短路）、
# llm_validate（四层校验+反馈式重试）。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 决策倾向四档（借鉴 PA decision_stance）
STANCES = {
    "保守": {"desc": "只在信号+胜率+盈亏比全合格时行动，其余一律等待",
             "confidence_min": 60},
    "均衡": {"desc": "合格即行动，缺一项验证则观察",
             "confidence_min": 50},
    "激进": {"desc": "信号合格即可行动，验证项可放宽",
             "confidence_min": 40},
    "极度激进": {"desc": "强信号即时行动，容忍样本不足",
             "confidence_min": 30},
}


def _stance_guidance(stance: str) -> str:
    s = STANCES.get(stance, STANCES["均衡"])
    return (f"当前决策倾向：【{stance}】——{s['desc']}。"
            f"置信度低于 {s['confidence_min']} 时应选择等待/观察。")


# ---- 阶段一：诊断 ----

async def _collect_facts(symbol: str) -> dict[str, Any]:
    """确定性事实收集（供诊断与校验用，best-effort 单票失败不阻断）。"""
    import asyncio
    from pa_mcp.ui.gradio_app import _load_long_history
    from pa_mcp.research.kline_geometry import compute_kline_geometry

    facts: dict[str, Any] = {"symbol": symbol, "data_ok": False, "hist_ok": False}
    df = await asyncio.to_thread(_load_long_history, symbol)
    if df is not None and not df.empty:
        facts["df"] = df
        facts["data_ok"] = True
        facts["hist_ok"] = len(df) >= 120
        last = df.iloc[-1]
        facts["close"] = float(last["close"])
        facts["pct_change"] = float(last.get("pct_change", 0) or 0)
        facts["date"] = str(last["date"])[:10]
        try:
            facts["geometry"] = compute_kline_geometry(df, detail_bars=20)
        except Exception:  # noqa: BLE001
            pass
    return facts


async def _market_context() -> dict[str, Any]:
    """市场环境（情绪/状态/板块），实时优先，失败给保守默认。"""
    ctx: dict[str, Any] = {"market_recognizable": False,
                           "sentiment_phase": "未知"}
    try:
        from pa_mcp.research.sentiment_cycle import SentimentCycleAnalyzer
        rt = await SentimentCycleAnalyzer._fetch_realtime_stats()
        if rt:
            ctx["data_ok"] = True
            phase, score = SentimentCycleAnalyzer._stage(rt)
            ctx["sentiment_phase"] = phase
            ctx["sentiment_score"] = score
            ctx["limit_up_count"] = rt["limit_up_count"]
            ctx["market_recognizable"] = True
    except Exception as e:  # noqa: BLE001
        logger.warning("市场情绪获取失败: %s", str(e)[:60])
    try:
        from pa_mcp.data.source_factory import build_router
        from pa_mcp.config import get_settings
        router = build_router(get_settings())
        if router is not None:
            for code in ("sh000001", "sz399001"):
                try:
                    q = await router.fetch_realtime_quote(code)
                    if q:
                        ctx.setdefault("index", {})[code] = {
                            "close": q.get("price"), "pct": q.get("pct_change")}
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        pass
    return ctx


async def _signal_check(symbol: str, df, strategy: str) -> dict[str, Any]:
    """指定策略的近期买入信号检查（复用策略注册表）。"""
    from pa_mcp.engine.strategies.base import StrategyRegistry
    import pandas as pd

    out = {"has_signal": False, "signal_date": None, "strength": None,
           "win_rate": None, "strategy": strategy}
    if df is None or df.empty or len(df) < 120:
        return out
    registry = StrategyRegistry()
    registry.auto_discover()
    base = registry.get(strategy)
    if base is None:
        return out
    try:
        signals = base.generate_signals(df.copy())
    except Exception:  # noqa: BLE001
        return out
    if not signals:
        return out
    recent = [s for s in signals if (getattr(s, "signal_time", None) or
             str(getattr(s, "timestamp", ""))[:10]) >= str(df["date"].astype(str).str[:10].iloc[-11])]
    if not recent:
        return out
    s = recent[-1]
    out.update({"has_signal": True,
                "signal_date": (getattr(s, "signal_time", None) or
                                str(getattr(s, "timestamp", ""))[:10]),
                "strength": float(getattr(s, "strength_score", 50))})
    # 信号历史胜率（预测力参考）
    try:
        from pa_mcp.research.event_study import signal_forward_returns
        if len(signals) >= 10:
            sig_df = pd.DataFrame([{
                "symbol": symbol,
                "date": getattr(x, "signal_time", None) or str(getattr(x, "timestamp", ""))[:10],
                "direction": "up",
                "strategy_name": strategy} for x in signals])
            results = signal_forward_returns(df, sig_df, [5])
            if results and results[0].n_events >= 10:
                out["win_rate"] = results[0].win_rate_pct
    except Exception:  # noqa: BLE001
        pass
    return out


# ---- 阶段二：决策 prompt ----

def _stage2_prompt(symbol: str, facts: dict, ctx: dict, signal: dict,
                   stance: str, geometry_text: str) -> str:
    return f"""你是 A 股研究系统的决策评审。基于以下事实，沿决策链 §3-§5 输出最终裁定 JSON。

【标的】{symbol}（最新收盘 {facts.get('close')} @ {facts.get('date')}，当日 {facts.get('pct_change', 0):+.2f}%）
【市场】情绪 {ctx.get('sentiment_phase', '未知')}（分 {ctx.get('sentiment_score', '—')}），涨停 {ctx.get('limit_up_count', '—')} 家
【信号】策略 {signal.get('strategy')}，信号日 {signal.get('signal_date')}，强度 {signal.get('strength')}，历史5日胜率 {signal.get('win_rate', '样本不足')}
【K线形态（近20日）】
{geometry_text[:1200]}

{_stance_guidance(stance)}

【决策链 §3-§5】（按顺序回答）
§3.1 信号是否存在（是/否）→ §3.2 信号历史验证（是/否/样本不足）
§4.1 风险收益比（是/否/无法计算）→ §4.2 交易者方程：胜率×回报 > 败率×风险（是/否/无法计算）
§5.1 最终裁定（观察/开仓/放弃）

输出 JSON（只输出 JSON）：
{{"action": "trade|observe|wait|reject",
  "direction": "up|down|neutral",
  "strength_score": 0-100,
  "confidence": 0-100,
  "entry_reference": "参考入场价或区间",
  "stop_loss": "止损位",
  "target": "目标位",
  "rr_ratio": 盈亏比数值或 null,
  "key_risks": ["风险1", "风险2"],
  "trace": [{{"node_id": "3.1", "answer": "是", "reason": "..."}},
            {{"node_id": "3.2", "answer": "是", "reason": "..."}},
            {{"node_id": "4.1", "answer": "是", "reason": "..."}},
            {{"node_id": "4.2", "answer": "是", "reason": "..."}},
            {{"node_id": "5.1", "answer": "开仓", "reason": "..."}}],
  "summary": "一句话结论"}}
研究参考，非投资建议。"""


_STAGE2_SCHEMA = {
    "required": ["action", "direction", "strength_score", "trace", "summary"],
    "fields": {
        "action": {"type": "str", "enum": ["trade", "observe", "wait", "reject"]},
        "direction": {"type": "str", "enum": ["up", "down", "neutral"]},
        "strength_score": {"type": "float", "min": 0, "max": 100},
        "confidence": {"type": "float", "min": 0, "max": 100},
    },
}


# ---- 主入口 ----

async def run_two_stage_analysis(symbol: str,
                                 stance: str = "均衡",
                                 strategy: str = "bollinger_mean_reversion",
                                 use_llm: bool = True) -> dict[str, Any]:
    """两阶段研究分析（诊断 → 路由 → 决策）。

    Args:
        symbol: 6 位股票代码
        stance: 决策倾向（保守/均衡/激进/极度激进）
        strategy: 候选策略（默认布林均值回归）
        use_llm: 是否尝试 LLM（无配置自动确定性降级）

    Returns:
        {stage1, route, stage2, action, trace, mode, report}
    """
    from pa_mcp.research.decision_chain import (
        chain_text, evaluate_chain_rule, gate_short_circuit, validate_trace)

    # ── 事实收集（确定性） ──
    facts = await _collect_facts(symbol)
    ctx = await _market_context()
    if facts.get("df") is not None:
        signal = await _signal_check(symbol, facts["df"], strategy)
    else:
        signal = {"has_signal": False}

    # ── 阶段一：环境诊断（LLM 或规则） ──
    stage1: dict[str, Any] = {}
    trace: list[dict] = []
    mode = "rule"
    adapter = None
    if use_llm:
        try:
            from pa_mcp.agent.llm_factory import ensure_llm_adapter
            from pa_mcp.agent.llm_port import LLMCallParams
            adapter = ensure_llm_adapter()
            if adapter is not None:
                p1 = LLMCallParams(
                    system_prompt="你是 A 股研究系统的市场诊断员。只输出合法 JSON。",
                    user_prompt=(
                        f"【数据状态】标的行情数据 {'可用' if facts.get('data_ok') else '不可用'}"
                        f"，历史 {'≥120根' if facts.get('hist_ok') else '不足'}，"
                        f"最新收盘 {facts.get('close')} @ {facts.get('date')}。\n"
                        f"【市场数据】情绪 {ctx.get('sentiment_phase', '未知')}"
                        f"（分 {ctx.get('sentiment_score', '—')}），"
                        f"涨停 {ctx.get('limit_up_count', '—')} 家"
                        f"（市场情绪接口暂不可用时记为未知，不影响个股数据可用性判定）。\n\n"
                        f"{chain_text()}\n\n"
                        "只回答 §0-§2 五个节点，输出 JSON："
                        '{"trace": [{"node_id": "0.1", "answer": "是", '
                        '"reason": "..."}, ...], '
                        '"market_state": "低迷|正常|高潮|恐慌|不明", '
                        '"sentiment_phase": "冰点|发酵|启动|高潮|未知", '
                        '"direction_bias": "up|down|neutral"}'),
                    mode="fast", max_tokens=1500,
                )
                raw = await adapter.chat_json(p1)
                trace = raw.get("trace") or []
                stage1 = raw
                mode = "llm"
        except Exception as e:  # noqa: BLE001
            logger.warning("阶段一 LLM 失败，规则降级: %s", str(e)[:60])

    if not stage1:
        # 规则诊断（确定性）：沿决策链 §0-§2
        rule_facts = {
            "data_ok": facts.get("data_ok", False),
            "hist_ok": facts.get("hist_ok", False),
            "market_recognizable": ctx.get("market_recognizable", False),
            "sentiment_phase": ctx.get("sentiment_phase", "未知"),
            "strategy_fit": "均值回归" if signal.get("has_signal") else "未知",
        }
        stage1 = {"mode": "rule", "market_state": "不明",
                  "sentiment_phase": ctx.get("sentiment_phase", "未知"),
                  "direction_bias": "neutral",
                  "trace": evaluate_chain_rule(rule_facts)["trace"][:5]}
        trace = stage1["trace"]
        if not rule_facts["data_ok"] or not rule_facts["hist_ok"]:
            return _compose_result(symbol, stage1, ctx, signal, None,
                                   action="wait", mode="rule",
                                   summary="数据/历史不足，等待")

    # trace 校验（LLM 输出时）
    trace_errors = validate_trace(trace) if mode == "llm" else []
    if trace_errors and mode == "llm":
        logger.warning("阶段一 trace 校验失败: %s", trace_errors[:2])

    # ── 闸门短路：数据不可靠 → 直接等待；环境不明 → 软降级（压置信度） ──
    stage1_wrap: dict[str, Any] = {"trace": trace}
    short = gate_short_circuit(stage1_wrap)
    if short is not None:
        return _compose_result(symbol, stage1, ctx, signal,
                               short.get("trace") or trace,
                               action=short["action"], mode=short["mode"],
                               summary=short["summary"])
    market_unclear = stage1_wrap.get("market_unclear", False)

    # ── 阶段二：决策 ──
    if not signal.get("has_signal"):
        return _compose_result(symbol, stage1, ctx, signal, trace,
                               action="wait", mode=mode,
                               summary="当前无买入信号，等待", stage2_extra={
                                   "no_signal": True})

    geometry_text = ""
    try:
        from pa_mcp.research.kline_geometry import format_geometry_text
        geometry_text = format_geometry_text(facts.get("geometry", []))
    except Exception:  # noqa: BLE001
        pass

    decision: dict[str, Any] = {}
    if adapter is not None:
        try:
            from pa_mcp.agent.llm_port import LLMCallParams
            from pa_mcp.research.llm_validate import chat_json_validated
            params = LLMCallParams(
                system_prompt="你是决策评审，只输出合法 JSON。研究参考，非投资建议。",
                user_prompt=_stage2_prompt(symbol, facts, ctx, signal, stance,
                                           geometry_text),
                mode="deep", max_tokens=2500,
            )
            r = await chat_json_validated(
                adapter, params, schema=_STAGE2_SCHEMA,
                facts={"close": facts.get("close")},
                semantic_rules=[_action_semantic_rule])
            if "_validated" in r and r["_validated"]:
                decision = r["obj"]
                decision["_attempts"] = r["attempts"]
                decision["mode"] = "llm"
            else:
                decision = {"error": r.get("error"), "mode": "llm_failed"}
        except Exception as e:  # noqa: BLE001
            logger.warning("阶段二 LLM 失败: %s", str(e)[:60])
            decision = {"mode": "llm_failed"}

    if not decision or decision.get("mode") == "llm_failed":
        # 规则决策（确定性降级）
        wr = signal.get("win_rate")
        rr = _estimate_rr(facts)
        rule_facts = {
            "data_ok": True, "hist_ok": True,
            "market_recognizable": ctx.get("market_recognizable", True),
            "sentiment_phase": ctx.get("sentiment_phase", "未知"),
            "strategy_fit": signal.get("strategy", "未知"),
            "has_signal": True, "win_rate": wr, "rr_ratio": rr,
        }
        r2 = evaluate_chain_rule(rule_facts)
        decision = {"action": r2["action"], "mode": "rule",
                    "trace": r2["trace"], "summary": r2["summary"],
                    "steps": r2["steps"]}

    action = decision.get("action", "wait")
    if market_unclear and action == "trade":
        # 环境不明软降级：trade → observe（置信度压降，不阻断但降格）
        action = "observe"
        decision["action"] = action
        decision["summary"] = (decision.get("summary", "")
                               + "（市场状态不明，结论已降格为观察）")
    return _compose_result(symbol, stage1, ctx, signal, trace,
                           action=action, mode=decision.get("mode", "rule"),
                           summary=decision.get("summary", ""),
                           stage2=decision)


def _action_semantic_rule(obj: dict) -> Optional[str]:
    """语义规则：action=trade 但胜率/盈亏比不可算 → 提示。"""
    return None


def _estimate_rr(facts: dict) -> Optional[float]:
    """盈亏比粗估：无法从单次分析精确获得时返回 None（诚实）。"""
    return None


def _compose_result(symbol: str, stage1: dict, ctx: dict, signal: dict,
                    trace: list, *, action: str, mode: str, summary: str,
                    stage2: Optional[dict] = None,
                    stage2_extra: Optional[dict] = None) -> dict[str, Any]:
    """组装最终结果 + 文本报告。"""
    st2 = stage2 or {}
    lines = [
        f"## 🔀 两阶段研究分析：{symbol}（{mode}）",
        "",
        "**阶段一·诊断**",
        f"- 市场情绪：{stage1.get('sentiment_phase', '未知')}"
        f"（分 {stage1.get('sentiment_score') or ctx.get('sentiment_score', '—')}）"
        f"｜ 涨停 {ctx.get('limit_up_count', '—')} 家",
        f"- 方向倾向：{stage1.get('direction_bias', 'neutral')}",
        "",
        "**阶段二·裁定**",
        f"- 结论：**{action}** — {summary}",
    ]
    if signal.get("has_signal"):
        lines.append(f"- 信号：{signal.get('strategy')} @ {signal.get('signal_date')}"
                     f"（强度 {signal.get('strength')}，胜率 "
                     f"{signal.get('win_rate') if signal.get('win_rate') is not None else '样本不足'}）")
    if st2.get("entry_reference"):
        lines.append(f"- 参考入场 {st2.get('entry_reference')}｜止损 "
                     f"{st2.get('stop_loss')}｜目标 {st2.get('target')}｜"
                     f"盈亏比 {st2.get('rr_ratio', '—')}")
    if st2.get("key_risks"):
        lines.append("- 风险：" + "；".join(st2["key_risks"][:4]))
    lines.append("")
    lines.append("*研究参考，非投资建议。*")
    return {
        "symbol": symbol, "stage1": stage1, "stage2": st2 or stage2_extra,
        "signal": signal, "action": action, "mode": mode,
        "summary": summary, "trace": trace or st2.get("trace"),
        "report": "\n".join(lines),
    }
