# 未来 K 线路径预测（future_path）测试
#
# 纯函数 + monkeypatch：LLM/DB 可注入，不依赖真实网络。

from __future__ import annotations

import pandas as pd
import pytest

from pa_mcp.research import future_path as fp


def _df():
    dates = pd.date_range("2025-01-01", periods=120, freq="B")
    close = 10.0 + pd.Series(range(120)) * 0.01
    return pd.DataFrame({
        "symbol": ["000001"] * 120, "date": dates,
        "open": close - 0.05, "high": close + 0.15,
        "low": close - 0.15, "close": close,
        "volume": 1_000_000, "amount": 10_000_000.0,
        "pct_change": 0.1,
    })


def _path_json_ok():
    return {
        "scenarios": [
            {"name": "base", "weight": 0.4, "logic": "震荡",
             "path": [{"date": "2026-08-18", "open": 10.5, "high": 10.8,
                       "low": 10.3, "close": 10.6}] * 10},
            {"name": "bull", "weight": 0.3, "logic": "突破",
             "path": [{"date": "2026-08-18", "open": 10.5, "high": 11.0,
                       "low": 10.4, "close": 10.9}] * 10},
        ],
        "base_direction": "up", "expected_return_pct": 1.2,
        "key_levels": {"support": [10.0], "resistance": [11.0]},
        "logic": "多头", "risks": ["风险"],
    }


def test_parse_path_json_valid():
    parsed = fp._parse_path_json(_path_json_ok())
    assert parsed is not None
    assert [s["name"] for s in parsed["scenarios"]] == ["base", "bull"]
    assert len(parsed["scenarios"][0]["path"]) == 10
    assert parsed["base_direction"] == "up"


def test_parse_path_json_too_short():
    """路径 <5 根的情景被拒绝；全拒则返回 None。"""
    raw = {"scenarios": [
        {"name": "base", "weight": 0.5, "logic": "x",
         "path": [{"date": "2026-08-18", "open": 1, "high": 1,
                   "low": 1, "close": 1}] * 3}]}
    assert fp._parse_path_json(raw) is None
    # 混合：一个有效一个无效
    raw2 = _path_json_ok()
    raw2["scenarios"].append({"name": "bad", "weight": 0.1, "logic": "x",
                              "path": [{"date": "2026-08-18", "open": 1,
                                        "high": 1, "low": 1, "close": 1}] * 2})
    parsed = fp._parse_path_json(raw2)
    assert parsed is not None
    assert len(parsed["scenarios"]) == 2


def test_parse_path_json_missing_keys():
    assert fp._parse_path_json({"foo": "bar"}) is None
    assert fp._parse_path_json(None) is None


def test_recent_summary_format():
    s = fp._recent_summary(_df())
    assert "近20日" in s and "收 11" in s


def test_deterministic_paths_structure():
    """统计模拟：3 情景、路径长度正确、OHLC 合理（high>=max(o,c)）。"""
    result = fp._deterministic_paths(_df(), horizon=10, drift=0.001)
    assert len(result["scenarios"]) == 3
    assert result["mode"] == "deterministic"
    for sc in result["scenarios"]:
        assert len(sc["path"]) == 10
        last_close = float(_df()["close"].iloc[-1])
        for p in sc["path"]:
            assert p["high"] >= max(p["open"], p["close"]) - 1e-9
            assert p["low"] <= min(p["open"], p["close"]) + 1e-9
            assert p["close"] > 0
        # 第一根与历史收盘衔接
        assert abs(sc["path"][0]["open"] / last_close - 1) < 0.2


async def test_predict_future_path_deterministic_fallback(monkeypatch):
    """无 LLM：统计模拟 + 历史拼接 + 报告。"""
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)

    async def _drift(df):
        return 0.001
    monkeypatch.setattr(fp, "_compute_drift", _drift)

    r = await fp.predict_future_path("000001", _df(), horizon=10,
                                     use_llm=True)
    assert r["mode"] == "deterministic"
    assert len(r["history"]) == 60
    assert len(r["scenarios"]) == 3
    assert r["key_levels"]["support"] and r["key_levels"]["resistance"]
    assert "统计模拟" in r["report"]


async def test_predict_future_path_llm(monkeypatch):
    """LLM 成功：mode=llm + 三情景。"""

    class _FakeAdapter:
        provider_name = "fake"

        async def chat_json(self, params):
            return _path_json_ok()

    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter",
        lambda: _FakeAdapter())
    monkeypatch.setattr(fp, "_market_context", lambda: "市场:低迷期")
    monkeypatch.setattr(fp, "_strategy_signals", lambda df: "无触发")

    r = await fp.predict_future_path("000001", _df(), horizon=10,
                                     use_llm=True)
    assert r["mode"] == "llm"
    assert r["base_direction"] == "up"
    assert len(r["scenarios"]) == 2
    assert "🤖 LLM" in r["report"]


def test_build_chart():
    """绘图：历史 K 线 + MA20 + 情景 = traces ≥ 3。"""
    r = fp._deterministic_paths(_df(), horizon=10, drift=0.001)
    data = {
        "symbol": "000001", "history": _df().tail(60).to_dict("records"),
        "scenarios": r["scenarios"], "base_direction": "up",
        "expected_return_pct": 1.0, "key_levels": {"support": [9.0],
                                                   "resistance": [11.0]},
        "mode": "deterministic",
    }
    fig = fp.build_chart(data)
    assert fig is not None
    assert len(fig.data) >= 3
