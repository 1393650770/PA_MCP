# 选股多 Agent 辩论（debate_picks）测试
#
# 纯函数 + monkeypatch：orchestrator/_load_kline 可注入，不依赖真实 LLM/DB。

from __future__ import annotations

import pandas as pd
import pytest

from pa_mcp.research import debate_picks as dp


class FakeResult:
    """模拟 AnalysisResult（deep_analyze 返回值）。"""

    def __init__(self, direction="bullish", score=70, verdict=None,
                 mode="llm"):
        self.direction = direction
        self.overall_strength_score = score
        self.suggested_max_position_pct = 12.0
        self.mode = mode
        self.key_evidence = ["证据1", "证据2"]
        self.key_risks = ["风险1"]
        self.master_verdict = verdict


def _fake_kline():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    return pd.DataFrame({
        "symbol": ["000001"] * 120, "date": dates,
        "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
        "volume": 1_000_000, "amount": 10_000_000.0,
        "pct_change": 0.0,
    })


class FakeStore:
    """数据层隔离：不触碰真实 DuckDB（可能被其他进程占用）。"""

    def close(self) -> None:
        pass


def _fake_store(monkeypatch):
    monkeypatch.setattr(dp, "_store", lambda: FakeStore())


def _verdict(direction="bullish", score=75):
    return {
        "final_direction": direction,
        "final_strength_score": score,
        "suggested_max_position_pct": 10.0,
        "master_style": "大师团合议（3/3 位参与）",
        "verdict_reason": "投票：看多 2.0 / 看空 0.0 / 中性 0.0",
        "masters": [], "falsification_conditions": [], "final_risks": [],
    }


async def test_empty_symbols():
    result = await dp.debate_picks([], top_n=3)
    assert "error" in result
    assert "候选清单为空" in result["error"]


async def test_llm_debate_mode_ranking(monkeypatch):
    _fake_store(monkeypatch)
    """有 master_verdict 时按裁定分数降序排名，模式=llm。"""
    calls = []

    class _FakeOrch:
        async def deep_analyze(self, symbol, df, debate=True):
            calls.append(symbol)
            scores = {"300750": 80, "000858": 60, "600036": 70}
            return FakeResult(
                direction="bullish", score=scores.get(symbol, 50),
                verdict=_verdict("bullish", scores.get(symbol, 50)),
                mode="llm")

    monkeypatch.setattr(dp, "_load_kline",
                        lambda store, sym: _fake_kline())
    monkeypatch.setattr("pa_mcp.agent.orchestrator.get_orchestrator",
                        lambda: _FakeOrch())
    monkeypatch.setattr("pa_mcp.data.symbols.get_stock_name",
                        lambda sym: f"名称{sym}")

    result = await dp.debate_picks(["300750", "600036", "000858"], top_n=3)
    assert result["debate_mode"] == "llm"
    ranked = result["picks"]
    assert [p["symbol"] for p in ranked] == ["300750", "600036", "000858"]
    assert ranked[0]["verdict"]["final_strength_score"] == 80
    assert calls == ["300750", "600036", "000858"]  # debate=True 传入
    assert "大师合议" in result["report"]
    assert "投票" in result["report"]  # verdict_reason 输出
    assert "置信加权" in result["report"]


async def test_deterministic_fallback_without_verdict(monkeypatch):
    _fake_store(monkeypatch)
    """无 LLM：master_verdict=None → 模式=deterministic，报告明确标注。"""

    class _FakeOrch:
        async def deep_analyze(self, symbol, df, debate=True):
            return FakeResult(direction="neutral", score=55, verdict=None,
                              mode="rule_based")

    monkeypatch.setattr(dp, "_load_kline",
                        lambda store, sym: _fake_kline())
    monkeypatch.setattr("pa_mcp.agent.orchestrator.get_orchestrator",
                        lambda: _FakeOrch())

    result = await dp.debate_picks(["000001"], top_n=1)
    assert result["debate_mode"] == "deterministic"
    assert result["picks"][0]["verdict"] is None
    assert "未配置 LLM" in result["report"]
    assert "确定性规则分析" in result["report"]


async def test_single_pick_failure_does_not_abort(monkeypatch):
    _fake_store(monkeypatch)
    """单票分析失败只记入 errors，不拖垮整组。"""

    class _FakeOrch:
        async def deep_analyze(self, symbol, df, debate=True):
            if symbol == "600036":
                raise RuntimeError("模拟失败")
            return FakeResult(direction="bullish", score=70,
                              verdict=_verdict("bullish", 70), mode="llm")

    monkeypatch.setattr(dp, "_load_kline",
                        lambda store, sym: _fake_kline())
    monkeypatch.setattr("pa_mcp.agent.orchestrator.get_orchestrator",
                        lambda: _FakeOrch())

    result = await dp.debate_picks(["000001", "600036"], top_n=2)
    assert len(result["picks"]) == 1
    assert result["picks"][0]["symbol"] == "000001"
    assert len(result["errors"]) == 1
    assert "600036" in result["errors"][0]["symbol"]
    assert "模拟失败" in result["errors"][0]["error"]


async def test_missing_kline_flagged(monkeypatch):
    _fake_store(monkeypatch)
    """行情不足的候选标注跳过。"""

    class _FakeOrch:
        async def deep_analyze(self, symbol, df, debate=True):
            return FakeResult(score=60, verdict=_verdict("bullish", 60))

    monkeypatch.setattr(dp, "_load_kline",
                        lambda store, sym: None)  # 行情全缺
    monkeypatch.setattr("pa_mcp.agent.orchestrator.get_orchestrator",
                        lambda: _FakeOrch())

    result = await dp.debate_picks(["000001"], top_n=1)
    assert result["picks"] == []
    assert "行情数据不足" in result["errors"][0]["error"]
    assert "跳过" in result["report"]


async def test_top_n_capped(monkeypatch):
    _fake_store(monkeypatch)
    """top_n 超过 MAX_PICKS 被截断（成本保护）。"""
    calls = []

    class _FakeOrch:
        async def deep_analyze(self, symbol, df, debate=True):
            calls.append(symbol)
            return FakeResult(score=60,
                              verdict=_verdict("bullish", 60))

    monkeypatch.setattr(dp, "_load_kline",
                        lambda store, sym: _fake_kline())
    monkeypatch.setattr("pa_mcp.agent.orchestrator.get_orchestrator",
                        lambda: _FakeOrch())

    result = await dp.debate_picks(
        ["000001", "600036", "300750", "000858", "600519", "601318"],
        top_n=10)
    assert len(calls) <= dp.MAX_PICKS  # 5
    assert len(result["picks"]) <= dp.MAX_PICKS
