# [AI:BEGIN]
# PA_MCP - 两阶段分析（市场诊断→路由→决策）+ JSON 校验重试 测试
# [AI:END]

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from pa_mcp.agent.orchestrator import (
    MARKET_STATE_ZH,
    AgentOrchestrator,
    route_strategy_by_market_state,
)


class MockAdapter:
    """可编程 mock LLM：按调用序号返回预设响应。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.provider_name = "mock"

    async def chat(self, params):
        self.calls.append(("chat", params.user_prompt))
        return self.responses.pop(0)

    async def chat_json(self, params):
        self.calls.append(("chat_json", params.user_prompt))
        return self.responses.pop(0)


def _kline():
    import numpy as np
    np.random.seed(3)
    close = 10.0
    rows = []
    for i in range(120):
        close *= 1 + np.random.normal(0.0005, 0.015)
        rows.append({
            "date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=i),
            "open": close * 0.995, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1e6,
        })
    return pd.DataFrame(rows)


def test_route_strategy_mapping():
    r = route_strategy_by_market_state("climax")
    assert "volume_price_momentum" in r["strategies"]
    assert "label" in r
    r_frozen = route_strategy_by_market_state("frozen")
    assert "roe_pb_value" in r_frozen["strategies"]
    r_unknown = route_strategy_by_market_state("???")
    assert "strategies" in r_unknown  # 回退 dull


def test_market_state_zh():
    assert MARKET_STATE_ZH["climax"] == "高潮期"


def test_diagnosis_deterministic():
    orch = AgentOrchestrator()
    d = orch._diagnosis_deterministic({
        "turnover_billion": 200, "limit_up_count": 5, "limit_down_count": 40})
    assert d["market_state"] == "frozen"
    d2 = orch._diagnosis_deterministic({
        "turnover_billion": 2000, "limit_up_count": 90, "limit_down_count": 2})
    assert d2["market_state"] == "climax"
    d3 = orch._diagnosis_deterministic(None)
    assert d3["market_state"] == "dull"


def test_validate_pm_json():
    orch = AgentOrchestrator()
    assert orch._validate_pm_json({"direction": "lol", "overall_strength_score": 100})
    assert orch._validate_pm_json({"direction": "neutral", "overall_strength_score": 50}) == []


def test_validate_analyst_json():
    orch = AgentOrchestrator()
    assert orch._validate_analyst_json({"strength_score": 150})
    assert orch._validate_analyst_json({"strength_score": 60}) == []


def test_chat_json_with_retry_succeeds_on_retry():
    orch = AgentOrchestrator()

    class P:
        system_prompt = "sys"
        user_prompt = "user"
        mode = "fast"
        max_tokens = 100

    # 第一次校验失败（strength_score 150），第二次成功
    adapter = MockAdapter([
        {"strength_score": 150, "risks": []},
        {"strength_score": 70, "risks": []},
    ])
    result = asyncio.run(orch._chat_json_with_retry(
        adapter, P(), orch._validate_analyst_json))
    assert result["strength_score"] == 70
    assert len(adapter.calls) == 2
    assert "校验失败" in adapter.calls[1][1]


def test_chat_json_with_retry_gives_up():
    orch = AgentOrchestrator()

    class P:
        system_prompt = "sys"
        user_prompt = "user"
        mode = "fast"
        max_tokens = 100

    adapter = MockAdapter([
        {"strength_score": 150}, {"strength_score": 150}, {"strength_score": 150},
    ])
    result = asyncio.run(orch._chat_json_with_retry(
        adapter, P(), orch._validate_analyst_json))
    assert result is None


def test_chat_json_with_retry_error_response():
    orch = AgentOrchestrator()

    class P:
        system_prompt = "s"
        user_prompt = "u"
        mode = "fast"
        max_tokens = 100

    adapter = MockAdapter([{"error": "rate limited"}])
    result = asyncio.run(orch._chat_json_with_retry(
        adapter, P(), orch._validate_analyst_json))
    assert result is None


def test_market_diagnosis_llm_valid():
    """LLM 返回合法诊断 → 返回带策略路由的诊断。"""
    orch = AgentOrchestrator()
    adapter = MockAdapter([{
        "market_state": "fermenting", "confidence": 80,
        "suggested_max_position_pct": 70, "risk_level": "medium",
        "key_observations": ["成交额放量"],
    }])
    # 临时注册 adapter（测试隔离：注册/注销）
    from pa_mcp.agent import llm_port
    orig = llm_port._adapter
    llm_port.register_adapter(adapter)
    try:
        d = asyncio.run(orch.market_diagnosis({"turnover_billion": 900}))
        assert d["market_state"] == "fermenting"
        assert d["market_state_zh"] == "发酵期"
        assert "ma_golden_cross" in d["strategy_routing"]["strategies"]
    finally:
        llm_port.register_adapter(orig)


def test_market_diagnosis_llm_invalid_falls_back():
    """LLM 返回非法状态 → 重试仍失败 → 确定性降级。"""
    orch = AgentOrchestrator()
    adapter = MockAdapter([
        {"market_state": "moon", "confidence": 50},
        {"market_state": "moon", "confidence": 50},
    ])
    from pa_mcp.agent import llm_port
    orig = llm_port._adapter
    llm_port.register_adapter(adapter)
    try:
        d = asyncio.run(orch.market_diagnosis(None))
        assert d["market_state"] in ("dull", "climax", "fermenting", "starting", "frozen")
        assert "strategy_routing" in d
    finally:
        llm_port.register_adapter(orig)


def _analyst_ok():
    return {"strength_score": 70, "risks": [], "key_findings": ["多头"],
            "dimension": "technical"}


def _pm_ok():
    return {
        "overall_strength_score": 68, "direction": "bullish",
        "dimension_scores": {"technical": 70, "capital": 60, "sentiment": 55,
                             "fundamental": 65, "event": 50},
        "key_evidence": [], "key_risks": [], "risk_reward_assessment": "favorable",
        "suggested_max_position_pct": 8,
    }


def _bull_ok():
    return {
        "bull_points": [{"point": "均线多头", "evidence": "MA5>MA20>MA60"},
                        {"point": "放量突破", "evidence": "量比1.8"},
                        {"point": "资金流入", "evidence": "主力净流入"}],
        "bear_rebuttals": [{"attack": "超买", "rebuttal": "强势不回补"}],
        "suggested_position_pct": 10,
    }


def _bear_ok():
    return {
        "bear_points": [{"point": "高位放量滞涨", "evidence": "量增价平"},
                        {"point": "大盘弱势", "evidence": "指数破位"},
                        {"point": "获利盘丰厚", "evidence": "累计涨幅大"}],
        "bull_rebuttals": [{"attack": "均线多头", "rebuttal": "乖离率过大"}],
        "biggest_missed_risk": "业绩不及预期",
    }


def _master_ok():
    return {
        "final_direction": "bearish", "final_strength_score": 45,
        "suggested_max_position_pct": 6,
        "master_style": "反身性",
        "verdict_reason": "高位风险大于收益",
        "key_evidence_used": ["量价背离"], "falsification_conditions": ["放量新高"],
        "final_risks": ["回调风险"],
    }


def test_deep_analyze_with_debate():
    """debate=True：PM 合成 → Bull → Bear → 大师裁定覆盖。"""
    orch = AgentOrchestrator()
    responses = [
        _analyst_ok(), _analyst_ok(), _analyst_ok(), _analyst_ok(), _analyst_ok(),
        _pm_ok(),
        _bull_ok(), _bear_ok(), _master_ok(),
    ]
    adapter = MockAdapter(responses)
    from pa_mcp.agent import llm_port
    orig = llm_port._adapter
    llm_port.register_adapter(adapter)
    try:
        result = asyncio.run(orch.deep_analyze(
            "000001", _kline(), debate=True))
        assert result.mode == "deep"
        # 大师裁定覆盖 PM：方向转空、分数 45、仓位 6%
        assert result.direction == "bearish"
        assert result.overall_strength_score == 45.0
        assert result.suggested_max_position_pct == 6.0
        assert result.master_verdict is not None
        assert result.master_verdict["master_style"] == "反身性"
        assert result.debate is not None
        assert len(result.debate["bull"]["bull_points"]) == 3
        assert len(result.debate["bear"]["bear_points"]) == 3
        assert "回调风险" in result.key_risks
    finally:
        llm_port.register_adapter(orig)


def test_deep_analyze_debate_master_fails_keeps_pm():
    """大师裁定失败 → 保留 PM 结论（不破坏主流程）。"""
    orch = AgentOrchestrator()
    responses = [
        _analyst_ok(), _analyst_ok(), _analyst_ok(), _analyst_ok(), _analyst_ok(),
        _pm_ok(),
        _bull_ok(), _bear_ok(),
        {"final_direction": "lol", "final_strength_score": -5},
    ]
    adapter = MockAdapter(responses)
    from pa_mcp.agent import llm_port
    orig = llm_port._adapter
    llm_port.register_adapter(adapter)
    try:
        result = asyncio.run(orch.deep_analyze(
            "000001", _kline(), debate=True))
        # PM 结论保留
        assert result.direction == "bullish"
        assert result.overall_strength_score == 68.0
        # 辩论部分仍记录（bull/bear 成功）
        assert result.debate is not None
        assert result.master_verdict is None
    finally:
        llm_port.register_adapter(orig)


def test_deep_analyze_debate_off_no_extra_calls():
    """debate=False：不调用辩论（调用次数 = 诊断? + 5 + PM）。"""
    orch = AgentOrchestrator()
    responses = [
        _analyst_ok(), _analyst_ok(), _analyst_ok(), _analyst_ok(), _analyst_ok(),
        _pm_ok(),
    ]
    adapter = MockAdapter(responses)
    from pa_mcp.agent import llm_port
    orig = llm_port._adapter
    llm_port.register_adapter(adapter)
    try:
        result = asyncio.run(orch.deep_analyze("000001", _kline()))
        assert result.master_verdict is None
        assert result.debate is None
        assert len(adapter.calls) == 6
    finally:
        llm_port.register_adapter(orig)


def test_deep_analyze_with_diagnosis_injection():
    """集成：诊断注入 + 5 分析师 + PM 合成，校验重试生效。"""
    orch = AgentOrchestrator()
    # 响应序列：诊断(1) + 5 分析师(5) + PM(1)
    analyst_ok = {
        "strength_score": 70, "risks": [], "key_findings": ["多头"],
        "dimension": "technical",
    }
    pm_ok = {
        "overall_strength_score": 68, "direction": "bullish",
        "dimension_scores": {"technical": 70, "capital": 60, "sentiment": 55,
                             "fundamental": 65, "event": 50},
        "key_evidence": [], "key_risks": [], "risk_reward_assessment": "favorable",
        "suggested_max_position_pct": 8,
    }
    responses = [
        {"market_state": "fermenting", "confidence": 70,
         "suggested_max_position_pct": 60, "risk_level": "medium",
         "key_observations": ["放量"]},
        analyst_ok, analyst_ok, analyst_ok, analyst_ok, analyst_ok,
        pm_ok,
    ]
    adapter = MockAdapter(responses)
    from pa_mcp.agent import llm_port
    orig = llm_port._adapter
    llm_port.register_adapter(adapter)
    try:
        result = asyncio.run(orch.analyze_with_diagnosis(
            "000001", _kline(),
            market_context={"turnover_billion": 900}))
        assert result.mode == "deep"
        assert result.market_diagnosis is not None
        assert result.market_diagnosis["market_state"] == "fermenting"
        assert result.direction == "bullish"
        assert result.suggested_max_position_pct <= 20  # RiskGuard 上限
        # 市场环境注入过分析师 prompt
        joined = " ".join(p for kind, p in adapter.calls if kind == "chat_json")
        assert "市场环境" in joined
        assert "发酵期" in joined
    finally:
        llm_port.register_adapter(orig)
