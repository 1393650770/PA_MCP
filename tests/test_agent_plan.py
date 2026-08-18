# Agent 自主规划（agent_plan）测试 — monkeypatch，零网络/LLM 依赖

from __future__ import annotations

import pytest

from pa_mcp.research import agent_plan as ap


async def test_rule_plan_routing():
    """无 LLM：目标关键词 → 预设流程。"""
    p = ap._rule_plan("帮我研究 601728 并给出操作建议")
    assert p["mode"] == "rule"
    assert len(p["steps"]) >= 3
    assert p["steps"][0]["action"] in ap.TOOLS_INDEX
    # 依赖链正确
    for i, s in enumerate(p["steps"]):
        if i > 0:
            assert s["depends_on"] == [p["steps"][i - 1]["id"]]


async def test_create_plan_rule_fallback(monkeypatch):
    """LLM 不可用 → 规则路由计划 + report。"""
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)
    r = await ap.create_plan("选 3 只股票")
    assert "error" not in r
    assert r["mode"] == "rule"
    assert "选股" in r["report"] or "扫描" in r["report"]
    assert r["steps"][0]["action"] in (
        "get_market_overview", "sentiment_cycle", "scan_canslim")


async def test_create_plan_empty_goal():
    r = await ap.create_plan("   ")
    assert "error" in r


async def test_create_plan_llm(monkeypatch):
    """LLM 规划：只保留已知工具、步骤上限。"""

    class _FakeAdapter:
        async def chat_json(self, params):
            return {
                "goal": "研究 601728",
                "steps": [
                    {"id": 1, "action": "get_kline",
                     "args": "symbol=601728", "purpose": "拉历史",
                     "depends_on": []},
                    {"id": 2, "action": "agent_analyze_stock",
                     "args": "symbol=601728", "purpose": "深度分析",
                     "depends_on": [1]},
                    {"id": 3, "action": "不存在的工具",
                     "args": "", "purpose": "x", "depends_on": []},
                ],
                "risks": ["注意"], "data_needs": ["行情"],
            }

    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter",
        lambda: _FakeAdapter())
    r = await ap.create_plan("研究 601728", max_steps=8)
    assert r["mode"] == "llm"
    assert [s["action"] for s in r["steps"]] == ["get_kline",
                                                 "agent_analyze_stock"]
    assert r["goal"] == "研究 601728"
    assert "🤖 LLM" in r["report"]


async def test_parse_plan_invalid():
    assert ap._parse_plan({"foo": "bar"}) is None
    assert ap._parse_plan(None) is None
    # 全部未知工具 → None
    assert ap._parse_plan({"steps": [
        {"id": 1, "action": "nope", "args": "", "purpose": "",
         "depends_on": []}]}) is None


async def test_update_plan_keep(monkeypatch):
    """评估通过 → 保持原计划。"""

    class _FakeAdapter:
        async def chat_json(self, params):
            return {"keep": True, "remaining": [], "reason": "没问题"}

    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter",
        lambda: _FakeAdapter())
    plan = ap._rule_plan("研究 601728")
    r = await ap.update_plan(plan, completed_id=1,
                             result_summary="行情正常，收盘 6.57")
    assert len(r["steps"]) == len(plan["steps"])  # 未变


async def test_update_plan_replan(monkeypatch):
    """数据缺口 → 替换剩余步骤。"""

    class _FakeAdapter:
        async def chat_json(self, params):
            return {"keep": False,
                    "remaining": [
                        {"id": 2, "action": "get_stock_capital_flow",
                         "args": "symbol=601728", "purpose": "补资金流",
                         "depends_on": [1]},
                        {"id": 3, "action": "agent_analyze_stock",
                         "args": "symbol=601728", "purpose": "深度分析",
                         "depends_on": [2]}],
                    "reason": "需补资金流"}

    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter",
        lambda: _FakeAdapter())
    plan = ap._rule_plan("研究 601728")
    r = await ap.update_plan(plan, completed_id=1,
                             result_summary="收盘 6.57，量能不足")
    actions = [s["action"] for s in r["steps"]]
    assert "get_stock_capital_flow" in actions
    assert actions[-1] == "agent_analyze_stock"
    assert r["steps"][0]["id"] == 1  # 已完成步骤保留
