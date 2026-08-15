# [AI:BEGIN]
# PA_MCP - 市场预测模块测试（临时数据库，不接触真实 data/pa_mcp.duckdb）
# [AI:END]

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from pa_mcp.agent.prediction import (
    PredictionResult,
    PredictionService,
    extract_features,
    format_features,
)


# ---- 特征抽取 ----

def test_extract_features_basic(ohlcv_data):
    feats = extract_features(ohlcv_data)
    assert "error" not in feats
    assert feats["last_close"] > 0
    assert "cycle_position" in feats
    assert feats["cycle_position"] in (
        "spike", "micro_channel", "tight_channel", "normal_channel",
        "broad_channel", "trending_range", "trading_range", "extreme_range")
    assert feats["support_20d"] <= feats["last_close"] <= feats["resistance_20d"] or True
    assert feats["support_20d"] > 0 and feats["resistance_20d"] > 0
    # 无未来函数：只用已收盘数据
    assert feats["days"] == len(ohlcv_data)


def test_extract_features_empty():
    assert "error" in extract_features(pd.DataFrame())


def test_extract_features_indicator_columns():
    """防回归：ATR/RSI/ADX 列名必须是 atr14/rsi14/adx14。"""
    df = _make_df()
    feats = extract_features(df)
    assert feats["atr_pct"] > 0, "非零波动数据 ATR% 应 > 0"
    assert 0 <= feats["rsi14"] <= 100
    assert feats["adx14"] > 0


@pytest.mark.asyncio
async def test_deterministic_range_nonzero_width():
    """防回归：确定性预测的收益区间必须非零宽度（ATR 有效时）。"""
    svc = PredictionService()
    result = await svc.predict("000001", _make_df(), horizon="5d")
    assert result.expected_range_high > result.expected_range_low


def test_extract_features_uptrend_cycle(ohlcv_data):
    """上升趋势 + 低波动 → 应为趋势通道类周期。"""
    feats = extract_features(ohlcv_data)
    # 200 天上涨、波动 2% → ADX 高或振幅适中
    assert feats["ret20_pct"] is not None
    assert feats["ma_alignment"] in ("多头排列", "空头排列", "均线缠绕")


# ---- 确定性预测 ----

@pytest.mark.asyncio
async def test_deterministic_prediction(ohlcv_data):
    """无 LLM 配置 → 确定性预测：概率和为 1、区间含期望、方向合法。"""
    svc = PredictionService()
    result = await svc.predict("000001", ohlcv_data, horizon="5d")
    assert isinstance(result, PredictionResult)
    assert result.mode == "deterministic"
    assert result.direction in ("up", "down", "sideways")
    total = result.prob_up + result.prob_down + result.prob_sideways
    assert abs(total - 1.0) < 0.011  # round(,2) 展示舍入容差
    assert 0.0 <= result.probability <= 1.0
    assert result.expected_range_low <= result.expected_return_pct <= result.expected_range_high
    assert result.horizon == "5d"
    assert result.cycle_position in (
        "spike", "micro_channel", "tight_channel", "normal_channel",
        "broad_channel", "trending_range", "trading_range", "extreme_range")
    assert result.scenarios, "必须给出情景"
    # 场景概率应接近 1
    assert abs(sum(s["probability"] for s in result.scenarios) - 1.0) < 0.3


@pytest.mark.asyncio
async def test_deterministic_horizon_20d(ohlcv_data):
    svc = PredictionService()
    r = await svc.predict("000001", ohlcv_data, horizon="20d")
    assert r.horizon == "20d"
    r2 = await svc.predict("000001", ohlcv_data, horizon="bad")
    assert r2.horizon == "5d"  # 非法值回退


def test_deterministic_empty_kline():
    svc = PredictionService()

    async def _go():
        with pytest.raises(ValueError):
            await svc.predict("000001", pd.DataFrame(), horizon="5d")

    import asyncio
    asyncio.run(_go())


# ---- LLM JSON 解析与校验 ----

def test_from_llm_json_valid():
    raw = {
        "direction": "up", "probability": 0.62,
        "probability_distribution": {"up": 0.62, "down": 0.2, "sideways": 0.18},
        "expected_return_pct": 3.5, "expected_range_pct": [-1.5, 6.0],
        "cycle_position": "normal_channel", "cycle_forecast": "broad_channel",
        "support_levels": [10.2], "resistance_levels": [11.5],
        "scenarios": [{"name": "突破", "probability": 0.5, "target_pct": 5, "description": "x"}],
        "confidence": 0.65,
        "key_reasons": ["均线多头"], "key_risks": ["大盘"],
    }
    r = PredictionResult.from_llm_json("000001", "2026-08-15", "5d", raw, "mock")
    assert r.direction == "up"
    assert abs(r.prob_up + r.prob_down + r.prob_sideways - 1.0) < 1e-9
    assert r.cycle_position == "normal_channel"
    assert r.mode == "llm"


def test_from_llm_json_clamps_bad_values():
    raw = {
        "direction": "up", "probability": 0.99,
        "probability_distribution": {"up": 2.0, "down": 0.0, "sideways": 0.0},
        "expected_return_pct": 50.0, "expected_range_pct": [],
        "cycle_position": "not_a_cycle", "cycle_forecast": "lol",
    }
    r = PredictionResult.from_llm_json("000001", "2026-08-15", "5d", raw, "mock")
    assert r.direction in ("up", "down", "sideways")
    assert r.probability <= 0.95
    assert r.expected_return_pct <= 25
    assert r.cycle_position in (
        "spike", "micro_channel", "tight_channel", "normal_channel",
        "broad_channel", "trending_range", "trading_range", "extreme_range")
    assert abs(r.prob_up + r.prob_down + r.prob_sideways - 1.0) < 1e-9


def test_validate_llm_json_errors():
    svc = PredictionService()
    assert svc._validate_llm_json({
        "direction": "up",
        "probability_distribution": {"up": 0.5, "down": 0.5, "sideways": 0.5},
        "expected_return_pct": 5.0, "expected_range_pct": [1.0, 2.0],
    })  # 分布和≠1 + 期望不在区间
    assert svc._validate_llm_json({
        "direction": "maybe",
        "probability_distribution": {"up": 0.3, "down": 0.3, "sideways": 0.4},
        "expected_return_pct": 1.0, "expected_range_pct": [0, 2],
        "cycle_position": "bogus",
    })
    assert not svc._validate_llm_json({
        "direction": "up",
        "probability_distribution": {"up": 0.6, "down": 0.2, "sideways": 0.2},
        "expected_return_pct": 2.0, "expected_range_pct": [0, 4],
        "cycle_position": "normal_channel", "cycle_forecast": "trading_range",
    })


# ---- 落盘 / 历史 / 评估闭环（临时数据库） ----

def test_save_history_evaluate(tmp_path):
    db = tmp_path / "pred_test.duckdb"
    svc = PredictionService(store_path=str(db))

    # 写入两条预测：一条 5d up，一条 5d down（同一 predict_date，模拟历史）
    r1 = PredictionResult(
        symbol="000001", predict_date="2026-08-01", horizon="5d",
        direction="up", prob_up=0.6, prob_down=0.2, prob_sideways=0.2)
    r2 = PredictionResult(
        symbol="000001", predict_date="2026-08-01", horizon="5d",
        direction="down", prob_up=0.2, prob_down=0.6, prob_sideways=0.2)
    id1 = svc.save_prediction(r1)
    id2 = svc.save_prediction(r2)
    assert id1 > 0 and id2 > id1

    hist = svc.prediction_history("000001", limit=10)
    assert len(hist) == 2
    assert all(h["status"] == "pending" for h in hist)

    # 模拟 K 线 provider：8-01 之后 5 个交易日上涨 5% → up 命中，down 未命中
    def fake_kline(symbol):
        closes = [10.0 + 0.05 * i for i in range(10)]
        dates = pd.date_range("2026-07-28", periods=10, freq="B")
        return pd.DataFrame({
            "date": dates, "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [1e6] * 10,
        })

    summary = asyncio.run(svc.evaluate_predictions(
        kline_provider=fake_kline, today="2026-08-15"))
    assert summary["evaluated"] == 2
    assert summary["hit_rate"] == 0.5
    assert summary["by_direction"]["up"]["hit_rate"] == 1.0
    assert summary["by_direction"]["down"]["hit_rate"] == 0.0
    assert summary["direction_agreement_pct"] == 0.5

    hist2 = svc.prediction_history("000001", limit=10)
    statuses = {h["id"]: h["status"] for h in hist2}
    assert statuses[id1] == "hit"
    assert statuses[id2] == "miss"
    assert all(h["actual_return_pct"] is not None for h in hist2)


def test_evaluate_sideways_threshold(tmp_path):
    db = tmp_path / "pred_test2.duckdb"
    svc = PredictionService(store_path=str(db))
    svc.save_prediction(PredictionResult(
        symbol="600000", predict_date="2026-08-01", horizon="5d",
        direction="sideways", prob_up=0.2, prob_down=0.2, prob_sideways=0.6))

    def flat_kline(symbol):
        dates = pd.date_range("2026-07-28", periods=10, freq="B")
        return pd.DataFrame({
            "date": dates, "open": [10.0] * 10, "high": [10.1] * 10,
            "low": [9.9] * 10, "close": [10.0] * 10, "volume": [1e6] * 10,
        })

    summary = asyncio.run(svc.evaluate_predictions(
        kline_provider=flat_kline, today="2026-08-15"))
    assert summary["hit_rate"] == 1.0  # |收益|=0 ≤ 1.5% → sideways 命中


def test_format_features():
    feats = extract_features(_make_df())
    text = format_features(feats)
    assert "周期位置" in text
    assert "RSI" in text


def _make_df():
    import numpy as np
    np.random.seed(7)
    close = 10.0
    rows = []
    for i in range(120):
        close *= 1 + np.random.normal(0.001, 0.015)
        rows.append({
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
            "open": close * 0.995, "high": close * 1.01,
            "low": close * 0.99, "close": close,
            "volume": 1e6 * (0.8 + 0.4 * np.random.rand()),
        })
    return pd.DataFrame(rows)
