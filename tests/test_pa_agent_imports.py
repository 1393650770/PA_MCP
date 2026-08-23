# 从 PA_Agent/PA_Agent2 搬运能力测试：
# kline_geometry（K线几何特征）/ decision_chain（决策链+短路）/
# llm_validate（四层校验+反馈重试）/ two_stage（两阶段编排）

from __future__ import annotations

import asyncio
import pytest

import pandas as pd

from pa_mcp.research import decision_chain as dc
from pa_mcp.research import kline_geometry as kg
from pa_mcp.research import llm_validate as lv
from pa_mcp.research import two_stage as ts


# ---- K 线几何特征 ----

def _make_df(close_seq, n=100):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "open": [c - 0.1 for c in close_seq],
        "high": [c + 0.3 for c in close_seq],
        "low": [c - 0.3 for c in close_seq],
        "close": close_seq,
        "volume": [1e6] * n,
    })


def test_geometry_basic():
    df = _make_df([10 + i * 0.05 for i in range(100)])
    feats = kg.compute_kline_geometry(df, detail_bars=5)
    assert len(feats) == 5
    f = feats[0]
    # 趋势阳线：实体比 0.17 < 0.25 → doji（测试数据 open=close-0.1）
    assert f["bar_type"] in ("doji", "trend_bull", "other")
    assert 0 <= f["close_position"] <= 1
    assert f["date"].startswith("20")


def test_geometry_inside_outside():
    """内包/外包分类。"""
    # 手工构造：最后两根一根外包一根内包
    n = 100
    closes = [10.0] * n
    df = _make_df(closes)
    # 外包棒：high 高于前一根、low 低于前一根
    df.loc[98, ["high", "low", "open", "close"]] = [10.9, 9.1, 10.0, 10.5]
    df.loc[99, ["high", "low", "open", "close"]] = [10.5, 9.5, 10.0, 10.2]
    feats = kg.compute_kline_geometry(df, detail_bars=2)
    assert feats[1]["bar_type"] == "outside_bull"  # 外包（最新）
    assert feats[0]["bar_type"] == "inside"        # 内包


def test_geometry_empty():
    assert kg.compute_kline_geometry(pd.DataFrame(), 10) == []
    assert kg.format_geometry_text([]) == "无K线形态数据"


def test_geometry_text():
    df = _make_df([10 + i * 0.05 for i in range(100)])
    text = kg.format_geometry_text(kg.compute_kline_geometry(df, 3))
    assert "实体" in text


# ---- 决策链 ----

OK_TRACE = [{"node_id": n, "answer": a, "reason": "r"} for n, a in [
    ("0.1", "是"), ("0.2", "是"), ("1.1", "是"), ("1.2", "冰点"),
    ("2.1", "价值"), ("3.1", "是"), ("3.2", "是"), ("4.1", "是"),
    ("4.2", "是"), ("5.1", "开仓")]]


def test_chain_validate_ok():
    assert dc.validate_trace(OK_TRACE) == []


def test_chain_validate_bad():
    bad = [{"node_id": "9.9", "answer": "是", "reason": "不存在"},
           {"node_id": "5.1", "answer": "梭哈", "reason": "非法"}]
    errs = dc.validate_trace(bad)
    assert any("未知节点" in e for e in errs)
    assert any("非法" in e for e in errs)


def test_chain_order():
    bad = [{"node_id": "1.1", "answer": "是", "reason": "r"},
           {"node_id": "0.2", "answer": "是", "reason": "r"}]
    errs = dc.validate_trace(bad)
    assert any("顺序错误" in e for e in errs)


def test_gate_short_circuit_hard():
    r = dc.gate_short_circuit({"trace": [{"node_id": "0.1", "answer": "否",
                                          "reason": "断连"}]})
    assert r is not None and r["action"] == "wait"


def test_gate_soft_mark():
    wrap = {"trace": [{"node_id": "1.1", "answer": "否", "reason": "不明"}]}
    assert dc.gate_short_circuit(wrap) is None
    assert wrap.get("market_unclear") is True


def test_chain_rule_evaluate():
    r = dc.evaluate_chain_rule({"data_ok": True, "hist_ok": True,
                                "market_recognizable": True,
                                "sentiment_phase": "冰点",
                                "strategy_fit": "价值", "has_signal": True,
                                "win_rate": 55.0, "rr_ratio": 2.0})
    assert r["action"] == "trade"
    assert len(r["trace"]) == 10

    r2 = dc.evaluate_chain_rule({"data_ok": True, "hist_ok": True,
                                 "market_recognizable": True,
                                 "sentiment_phase": "高潮",
                                 "strategy_fit": "趋势", "has_signal": True,
                                 "win_rate": 40.0, "rr_ratio": 1.2})
    assert r2["action"] == "reject"


def test_chain_rule_insufficient():
    r = dc.evaluate_chain_rule({"data_ok": False, "hist_ok": False})
    assert r["action"] == "wait"


# ---- LLM 四层校验 ----

SCHEMA = {"required": ["symbol", "direction", "strength_score"],
          "fields": {"strength_score": {"type": "float", "min": 0, "max": 100},
                     "direction": {"type": "str",
                                   "enum": ["up", "down", "neutral"]}}}


def test_validate_ok():
    raw = '```json\n{"symbol": "000001", "direction": "up", "strength_score": 62}\n```'
    obj, errs = lv.validate_output(raw, SCHEMA, facts={"close": 11.27})
    assert not errs and obj["symbol"] == "000001"


def test_validate_schema_errors():
    obj, errs = lv.validate_output(
        '{"direction": "sideways", "strength_score": 150}', SCHEMA)
    assert any("必填" in e for e in errs)
    assert any("上限" in e for e in errs)
    assert any("枚举" in e for e in errs)


def test_validate_facts_mismatch():
    _, errs = lv.validate_output(
        '{"symbol": "000001", "direction": "up", "strength_score": 62, "close": 20}',
        SCHEMA, facts={"close": 11.27})
    assert any("不符" in e for e in errs)


def test_is_truncated():
    assert lv.is_truncated('{"symbol": "000001", "direc')
    assert not lv.is_truncated('{"symbol": "000001"}')


def test_semantics_contradiction():
    _, errs = lv.validate_output(
        '{"symbol": "000001", "direction": "up", "strength_score": 95, "action": "wait"}',
        SCHEMA)
    assert any("矛盾" in e for e in errs)


async def test_chat_json_validated_retry(monkeypatch):
    """反馈式重试：第一次错误，第二次修正。"""

    class FakeAdapter:
        def __init__(self):
            self.n = 0

        async def chat(self, params):
            self.n += 1
            from pa_mcp.agent.llm_port import LLMResponse
            if self.n == 1:
                return LLMResponse(content='{"direction": "up", "strength_score": 150}',
                                   model="f", provider="f")
            return LLMResponse(content='{"symbol": "000001", "direction": "up", "strength_score": 62}',
                               model="f", provider="f")

    from pa_mcp.agent.llm_port import LLMCallParams
    r = await lv.chat_json_validated(FakeAdapter(),
                                     LLMCallParams(system_prompt="s",
                                                   user_prompt="u"),
                                     schema=SCHEMA)
    assert r["_validated"] and r["attempts"] == 2
    assert r["obj"]["strength_score"] == 62


async def test_chat_json_validated_exhausted(monkeypatch):
    """重试耗尽 → 返回错误信息。"""

    class FakeAdapter:
        async def chat(self, params):
            from pa_mcp.agent.llm_port import LLMResponse
            return LLMResponse(content='{"direction": "up"}', model="f", provider="f")

    from pa_mcp.agent.llm_port import LLMCallParams
    r = await lv.chat_json_validated(FakeAdapter(),
                                     LLMCallParams(system_prompt="s",
                                                   user_prompt="u"),
                                     schema=SCHEMA, max_retries=1)
    assert not r.get("_validated")
    assert "未通过校验" in r.get("error", "")


# ---- 两阶段编排 ----

async def test_two_stage_rule_wait(monkeypatch):
    """规则模式：数据不足 → 等待。"""
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)
    async def _facts(symbol):
        return {"symbol": symbol, "data_ok": False, "hist_ok": False}

    async def _ctx():
        return {"market_recognizable": False, "sentiment_phase": "未知"}
    monkeypatch.setattr(ts, "_collect_facts", _facts)
    monkeypatch.setattr(ts, "_market_context", _ctx)
    r = await ts.run_two_stage_analysis("000001", use_llm=True)
    assert r["action"] == "wait"
    assert "数据" in r["summary"]


async def test_two_stage_rule_signal_wait(monkeypatch):
    """规则模式：数据 OK 但无信号 → 等待。"""

    async def _facts(symbol):
        return {"symbol": symbol, "data_ok": True, "hist_ok": True,
                "close": 11.27, "df": pd.DataFrame({"date": pd.date_range(
                    "2025-01-01", periods=130, freq="B"),
                    "open": [10.0] * 130, "high": [10.5] * 130,
                    "low": [9.5] * 130, "close": [10.0] * 130,
                    "volume": [1e6] * 130})}

    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)
    async def _ctx():
        return {"market_recognizable": True, "sentiment_phase": "冰点"}
    monkeypatch.setattr(ts, "_collect_facts", _facts)
    monkeypatch.setattr(ts, "_market_context", _ctx)
    async def _no_signal(s, df, st):
        return {"has_signal": False}
    monkeypatch.setattr(ts, "_signal_check", _no_signal)
    r = await ts.run_two_stage_analysis("000001", use_llm=True)
    assert r["action"] == "wait"
    assert "无买入信号" in r["summary"]


async def test_two_stage_rule_trade(monkeypatch):
    """规则模式：全链路通过 → 开仓。"""

    async def _facts(symbol):
        import numpy as np
        n = 130
        return {"symbol": symbol, "data_ok": True, "hist_ok": True,
                "close": 11.27, "df": pd.DataFrame({"date": pd.date_range(
                    "2025-01-01", periods=n, freq="B"),
                    "open": [10.0] * n, "high": [10.5] * n,
                    "low": [9.5] * n,
                    "close": np.linspace(10, 12, n).tolist(),
                    "volume": [1e6] * n})}

    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)
    async def _ctx():
        return {"market_recognizable": True, "sentiment_phase": "发酵"}
    monkeypatch.setattr(ts, "_collect_facts", _facts)
    monkeypatch.setattr(ts, "_market_context", _ctx)
    async def _yes_signal(s, df, st):
        return {"has_signal": True, "win_rate": 60.0, "strategy": st,
                "signal_date": "2026-08-20", "strength": 70}
    monkeypatch.setattr(ts, "_signal_check", _yes_signal)
    r = await ts.run_two_stage_analysis("000001", use_llm=True)
    assert r["action"] in ("trade", "observe")
    assert "报告" in r or r.get("report")


def _asyncio(coro):
    return asyncio.run(coro)


# 同步包装（pytest-asyncio 不可用时）
def test_two_stage_sync_wrappers(monkeypatch):
    _asyncio(test_two_stage_rule_wait(monkeypatch))
    _asyncio(test_two_stage_rule_signal_wait(monkeypatch))
    _asyncio(test_two_stage_rule_trade(monkeypatch))
