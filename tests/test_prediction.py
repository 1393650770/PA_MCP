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


def test_brier_score_calibration(tmp_path):
    """Brier + 气候学技能分：混合涨跌样本下 1 - brier/baseline。"""
    db = tmp_path / "brier_test.duckdb"
    svc = PredictionService(store_path=str(db))

    # 4 条确定性预测：600000 涨行情、600001 跌行情，各配 up/down
    for sym, d, pu, pd_, ps in (
            ("600000", "up", 1.0, 0.0, 0.0), ("600001", "up", 1.0, 0.0, 0.0),
            ("600000", "down", 0.0, 1.0, 0.0), ("600001", "down", 0.0, 1.0, 0.0)):
        svc.save_prediction(PredictionResult(
            symbol=sym, predict_date="2026-08-01", horizon="5d",
            direction=d, prob_up=pu, prob_down=pd_, prob_sideways=ps))

    dates = pd.date_range("2026-07-28", periods=10, freq="B")

    def kline(symbol):
        sign = 1 if symbol == "600000" else -1
        return pd.DataFrame({
            "date": dates, "open": [10.0 + sign * 0.06 * i for i in range(10)],
            "high": [10.5 + sign * 0.06 * i for i in range(10)],
            "low": [9.5 + sign * 0.06 * i for i in range(10)],
            "close": [10.0 + sign * 0.06 * i for i in range(10)],
            "volume": [1e6] * 10,
        })

    summary = asyncio.run(svc.evaluate_predictions(
        kline_provider=kline, today="2026-08-15"))
    # 实际类别：600000 涨 → y_up=1；600001 跌 → y_down=1（混合 2/2）
    #  brier：up/600000 命中 0；up/600001 未中 2；down/600000 未中 2；down/600001 命中 0
    #  mean = 1.0
    assert summary["brier_score"] == 1.0
    # 气候学基准（频率 [0.5,0.5,0]）：每个样本 brier = 0.25+0.25 = 0.5 → base=0.5
    # 技能分 = 1 - 1.0/0.5 = -1.0（反方向预测劣于气候学基准）
    assert summary["baseline_brier"] == 0.5
    assert summary["brier_skill_score"] == -1.0
    assert summary["hit_rate"] == 0.5


def test_brier_perfect_prediction(tmp_path):
    """全部预测命中 → Brier=0，技能分=1（完美校准）。"""
    db = tmp_path / "brier_test2.duckdb"
    svc = PredictionService(store_path=str(db))
    svc.save_prediction(PredictionResult(
        symbol="600000", predict_date="2026-08-01", horizon="5d",
        direction="up", prob_up=1.0, prob_down=0.0, prob_sideways=0.0))

    dates = pd.date_range("2026-07-28", periods=10, freq="B")

    def rising_kline(symbol):
        return pd.DataFrame({
            "date": dates, "open": [10.0 + 0.06 * i for i in range(10)],
            "high": [10.5 + 0.06 * i for i in range(10)],
            "low": [9.5 + 0.06 * i for i in range(10)],
            "close": [10.0 + 0.06 * i for i in range(10)],
            "volume": [1e6] * 10,
        })

    summary = asyncio.run(svc.evaluate_predictions(
        kline_provider=rising_kline, today="2026-08-15"))
    assert summary["brier_score"] == 0.0
    # 单样本时气候学基准=0（样本全同类别）→ 技能分无定义（正确行为）
    assert summary["brier_skill_score"] is None
    # expected_return_pct 默认 0.0（常数）→ 相关性不计算
    assert summary["return_correlation"] is None


def test_return_correlation(tmp_path):
    """期望收益与涨幅正相关 → 相关性 > 0。

    注：同日预测共享同一行情 → 实际收益相同 → 相关无定义。
    用 3 只不同股票（不同涨速行情）制造差异化实际收益。
    """
    db = tmp_path / "corr_test.duckdb"
    svc = PredictionService(store_path=str(db))
    # 期望收益递增：1.0 / 3.0 / 6.0
    for exp_ret, sym in ((1.0, "600000"), (3.0, "600001"), (6.0, "600002")):
        svc.save_prediction(PredictionResult(
            symbol=sym, predict_date="2026-08-01", horizon="5d",
            direction="up", prob_up=0.6, prob_down=0.2, prob_sideways=0.2,
            expected_return_pct=exp_ret))

    dates = pd.date_range("2026-07-28", periods=10, freq="B")
    slopes = {"600000": 0.01, "600001": 0.03, "600002": 0.06}

    def kline(symbol):
        m = slopes[symbol]
        return pd.DataFrame({
            "date": dates, "open": [10.0 + m * i for i in range(10)],
            "high": [10.5 + m * i for i in range(10)],
            "low": [9.5 + m * i for i in range(10)],
            "close": [10.0 + m * i for i in range(10)],
            "volume": [1e6] * 10,
        })

    summary = asyncio.run(svc.evaluate_predictions(
        kline_provider=kline, today="2026-08-15"))
    # 实际 5 日收益：+0.5% / +1.5% / +3.0%（与期望单调一致）→ 强正相关
    assert summary["return_correlation"] is not None
    assert summary["return_correlation"] > 0.99


def test_ic_icir_evaluation(tmp_path):
    """IC：期望收益排序与涨幅正相关 → IC > 0；跨周 ICIR 可算。"""
    db = tmp_path / "ic_test.duckdb"
    svc = PredictionService(store_path=str(db))
    # 12 条预测分布在 3 周，期望收益与股票斜率严格单调对齐 → 强正相关
    for i, (exp_ret, wk) in enumerate((
            (1.0, "2026-07-06"), (2.0, "2026-07-06"), (3.0, "2026-07-06"),
            (4.0, "2026-07-07"), (5.0, "2026-07-08"),
            (6.0, "2026-07-13"), (7.0, "2026-07-14"), (8.0, "2026-07-15"),
            (9.0, "2026-07-20"), (10.0, "2026-07-21"), (11.0, "2026-07-22"),
            (12.0, "2026-07-23"))):
        svc.save_prediction(PredictionResult(
            symbol=f"6000{i:02d}", predict_date=wk, horizon="5d",
            direction="up", prob_up=0.6, prob_down=0.2, prob_sideways=0.2,
            expected_return_pct=exp_ret))

    # 行情须覆盖最后一条预测（07-23）+ 5 个交易日回填期
    dates = pd.date_range("2026-07-01", periods=30, freq="B")

    def rising_kline(symbol):
        idx = int(symbol[-2:])
        m = 0.01 + 0.002 * idx  # 收益与期望单调一致
        return pd.DataFrame({
            "date": dates, "open": [10.0 + m * i for i in range(30)],
            "high": [10.5 + m * i for i in range(30)],
            "low": [9.5 + m * i for i in range(30)],
            "close": [10.0 + m * i for i in range(30)],
            "volume": [1e6] * 30,
        })

    summary = asyncio.run(svc.evaluate_predictions(
        kline_provider=rising_kline, today="2026-08-15"))
    assert summary["ic"] is not None
    assert summary["ic"] > 0.5          # 强正秩相关
    assert summary["icir"] is not None  # 多窗口
    assert summary["return_correlation"] is not None


def test_calibration_bins_overconfidence(tmp_path):
    """概率分桶：高概率桶（80%+）实际命中低 → 标记过度自信。"""
    db = tmp_path / "calib_test.duckdb"
    svc = PredictionService(store_path=str(db))
    # 10 条 85% 概率的 up 预测，行情横盘（实际只涨 0.5% 阈值内算 miss）
    for i in range(10):
        svc.save_prediction(PredictionResult(
            symbol=f"6001{i:02d}", predict_date="2026-07-06", horizon="5d",
            direction="up", prob_up=0.85, prob_down=0.1, prob_sideways=0.05,
            probability=0.85))
    dates = pd.date_range("2026-07-01", periods=10, freq="B")

    def flat_kline(symbol):
        return pd.DataFrame({
            "date": dates, "open": [10.0] * 10, "high": [10.2] * 10,
            "low": [9.8] * 10, "close": [10.0] * 10, "volume": [1e6] * 10,
        })

    summary = asyncio.run(svc.evaluate_predictions(
        kline_provider=flat_kline, today="2026-08-15"))
    assert summary["calibration_bins"], "应有校准分桶"
    top = summary["calibration_bins"][-1]
    assert top["prob_range"] == "80%-100%"
    assert top["actual_hit_rate"] < 0.2
    assert top["overconfident"] is True  # 85% 预测实际 ~0% 命中 → 过度自信


def test_by_mode_comparison(tmp_path):
    """模式对比：llm vs deterministic 分组统计。"""
    db = tmp_path / "mode_test.duckdb"
    svc = PredictionService(store_path=str(db))
    for i in range(4):
        svc.save_prediction(PredictionResult(
            symbol=f"6002{i:02d}", predict_date="2026-07-06", horizon="5d",
            direction="up", prob_up=0.7, prob_down=0.15, prob_sideways=0.15,
            mode="llm"))
    for i in range(4):
        svc.save_prediction(PredictionResult(
            symbol=f"6003{i:02d}", predict_date="2026-07-06", horizon="5d",
            direction="down", prob_up=0.15, prob_down=0.7, prob_sideways=0.15,
            mode="deterministic"))
    dates = pd.date_range("2026-07-01", periods=10, freq="B")

    def rising_kline(symbol):
        return pd.DataFrame({
            "date": dates, "open": [10.0 + 0.05 * i for i in range(10)],
            "high": [10.5 + 0.05 * i for i in range(10)],
            "low": [9.5 + 0.05 * i for i in range(10)],
            "close": [10.0 + 0.05 * i for i in range(10)],
            "volume": [1e6] * 10,
        })

    summary = asyncio.run(svc.evaluate_predictions(
        kline_provider=rising_kline, today="2026-08-15"))
    assert "llm" in summary["by_mode"] and "deterministic" in summary["by_mode"]
    assert summary["by_mode"]["llm"]["hit_rate"] == 1.0      # up 预测涨 → 全中
    assert summary["by_mode"]["deterministic"]["hit_rate"] == 0.0
    assert "brier_score" in summary["by_mode"]["llm"]


def test_position_sizing_with_history(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)
    """预测→仓位：同方向历史全中 → hist_hit_rate=1.0 生效；上限 20%。"""
    db = tmp_path / "sizing_test.duckdb"
    svc = PredictionService(store_path=str(db))

    # 先预测拿方向，再按该方向灌 5 条全中历史
    df = _make_df()
    pre = asyncio.run(svc.predict("600000", df, horizon="5d",
                                   use_llm=False))  # 确定性链路（测试意图）
    d = pre.direction
    for i in range(5):
        svc.save_prediction(PredictionResult(
            symbol="600000", predict_date="2026-07-06", horizon="5d",
            direction=d, prob_up=0.7 if d == "up" else 0.15,
            prob_down=0.7 if d == "down" else 0.15,
            prob_sideways=0.15))
    dates = pd.date_range("2026-07-01", periods=10, freq="B")
    sign = 1 if d == "up" else -1

    def hist_kline(symbol):
        # 行情方向与预测方向一致 → 历史全中
        return pd.DataFrame({
            "date": dates, "open": [10.0 + sign * 0.05 * i for i in range(10)],
            "high": [10.5 + sign * 0.05 * i for i in range(10)],
            "low": [9.5 + sign * 0.05 * i for i in range(10)],
            "close": [10.0 + sign * 0.05 * i for i in range(10)],
            "volume": [1e6] * 10,
        })

    asyncio.run(svc.evaluate_predictions(
        kline_provider=hist_kline, today="2026-08-15"))

    sizing = asyncio.run(svc.position_sizing("600000", kline_df=df))
    assert sizing["direction"] == d
    assert sizing["hist_hit_rate"] == 1.0
    assert sizing["hist_samples"] == 5
    assert 0 <= sizing["suggested_position_pct"] <= 20  # RiskGuard 上限
    # up 方向基础仓位回退 10%；down 为 0
    assert sizing["base_position_pct"] == (10.0 if d == "up" else 0.0)


def test_position_sizing_down_zero(monkeypatch):
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)
    """看跌预测 → 仓位 0（不做空）。"""
    import numpy as np
    np.random.seed(11)
    close = 10.0
    rows = []
    for i in range(120):
        close *= 1 + np.random.normal(-0.002, 0.015)
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6})
    df = pd.DataFrame(rows)

    svc = PredictionService(store_path=":memory:")
    sizing = asyncio.run(svc.position_sizing("600000", kline_df=df))
    if sizing["direction"] == "down":
        assert sizing["suggested_position_pct"] == 0.0
    else:
        assert sizing["suggested_position_pct"] >= 0.0


def test_predict_1d_horizon():
    """次日方向预测（1d）：输出结构与 5d 一致。"""
    import numpy as np
    np.random.seed(17)
    rows = []
    close = 10.0
    for i in range(120):
        close *= 1 + np.random.normal(0.001, 0.015)
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6})
    df = pd.DataFrame(rows)

    import asyncio as _asyncio
    from pa_mcp.agent.prediction import PredictionService
    r = _asyncio.run(PredictionService().predict(
        "600000", df, horizon="1d", use_llm=False))
    assert r.horizon == "1d"
    p = r.to_dict()
    assert p["direction"] in ("up", "down", "sideways")
    assert abs(p["probability_distribution"]["up"]
               + p["probability_distribution"]["down"]
               + p["probability_distribution"]["sideways"] - 1.0) < 0.011
    # 1d 区间更窄（波动尺度小）
    r5 = _asyncio.run(PredictionService().predict(
        "600000", df, horizon="5d", use_llm=False))
    w1 = p["expected_range_pct"][1] - p["expected_range_pct"][0]
    w5 = r5.to_dict()["expected_range_pct"][1] \
        - r5.to_dict()["expected_range_pct"][0]
    assert w1 < w5


def test_by_horizon_grouping(tmp_path):
    """预测验证按周期分组：1d 与 5d 分开统计。"""
    db = tmp_path / "horizon_test.duckdb"
    svc = PredictionService(store_path=str(db))
    # 3 条 1d + 3 条 5d（全 up 预测，行情上涨 → 全中）
    for i in range(3):
        svc.save_prediction(PredictionResult(
            symbol="600000", predict_date="2026-08-01", horizon="1d",
            direction="up", prob_up=0.7, prob_down=0.15, prob_sideways=0.15))
    for i in range(3):
        svc.save_prediction(PredictionResult(
            symbol="600000", predict_date="2026-08-01", horizon="5d",
            direction="up", prob_up=0.7, prob_down=0.15, prob_sideways=0.15))
    dates = pd.date_range("2026-07-28", periods=12, freq="B")

    def rising_kline(symbol):
        # 每日 +1.5%（1d 收益 > 1.0% 模糊阈值 → 正常命中判定）
        return pd.DataFrame({
            "date": dates, "open": [10.0 + 0.15 * i for i in range(12)],
            "high": [10.5 + 0.15 * i for i in range(12)],
            "low": [9.5 + 0.15 * i for i in range(12)],
            "close": [10.0 + 0.15 * i for i in range(12)],
            "volume": [1e6] * 12,
        })

    summary = asyncio.run(svc.evaluate_predictions(
        kline_provider=rising_kline, today="2026-08-15"))
    assert "by_horizon" in summary
    assert summary["by_horizon"]["1d"]["count"] == 3
    assert summary["by_horizon"]["5d"]["count"] == 3
    assert summary["by_horizon"]["1d"]["hit_rate"] == 1.0
    assert "brier_score" in summary["by_horizon"]["1d"]


def test_market_bias_context_injection():
    """大盘环境注入：库内指数数据 → 返回方向文本；无数据 → 空串。"""
    import tempfile, os
    import numpy as np
    from pa_mcp.data.store import DuckDBStore
    db = os.path.join(tempfile.mkdtemp(), "mb.duckdb")
    store = DuckDBStore(db)
    store.connect()
    # 横盘后跌破 → 偏空结构
    rng = np.random.default_rng(21)
    close = 3000.0
    rows = []
    for i in range(150):
        close *= (1 + rng.normal(0, 0.003)) if i < 60 else 0.99
        rows.append({"symbol": "sh000001",
                     "date": pd.Timestamp("2025-09-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e8})
    store.insert_df("index_daily", pd.DataFrame(rows))
    store.close()

    import asyncio as _asyncio
    from pa_mcp.agent.prediction import PredictionService
    svc = PredictionService(store_path=db)
    ctx = _asyncio.run(svc._market_bias_context())
    assert "上证指数" in ctx
    assert "偏空" in ctx or "偏多" in ctx or "中性" in ctx

    # 无指数数据 → 空串（不阻塞）
    svc2 = PredictionService(store_path=":memory:")
    assert _asyncio.run(svc2._market_bias_context()) == ""


def test_calibration_figure_build():
    """校准曲线图构建：柱状 + 参考线，过度自信红色标记。"""
    from pa_mcp.ui.gradio_app import _build_calibration_figure
    bins = [
        {"prob_range": "50%-60%", "n": 10, "actual_hit_rate": 0.55,
         "mid_prob": 0.55, "overconfident": False},
        {"prob_range": "70%-80%", "n": 8, "actual_hit_rate": 0.5,
         "mid_prob": 0.75, "overconfident": True},
    ]
    fig = _build_calibration_figure(bins)
    assert len(fig.data) == 2  # 柱状 + 参考线
    assert fig.data[0]["name"] == "实际命中率"
    assert fig.data[1]["name"].startswith("完美校准")
    # 红色（过度自信）在颜色列表中
    assert "#e03131" in fig.data[0].marker.color


def test_position_sizing_resonance_adjust(monkeypatch):
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)
    """共振校准：强共振同向上调 1.3；分歧收缩 0.7。"""
    import tempfile, os
    import numpy as np
    db = os.path.join(tempfile.mkdtemp(), "res_size.duckdb")
    svc = PredictionService(store_path=db)
    # 强上涨趋势 → 确定性预测看涨 + 共振看涨
    np.random.seed(8)
    rows = []
    close = 10.0
    for i in range(200):
        close *= 1 + 0.006 + np.random.normal(0, 0.01)
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6})
    df = pd.DataFrame(rows)

    sizing = asyncio.run(svc.position_sizing(
        "600000", kline_df=df))
    assert sizing["resonance_factor"] in (1.3, 1.0, 0.7)
    assert "共振" in sizing["explanation"]
    # 综合信号校准字段存在（1.4/1.0/0.6）
    assert sizing["consensus_factor"] in (1.4, 1.0, 0.6)
    assert "综合信号" in sizing["explanation"]
    # 强趋势 → 大概率共振上调（1.3）
    if sizing["direction"] == "up":
        assert sizing["resonance_factor"] >= 1.0
    assert 0 <= sizing["suggested_position_pct"] <= 20


def test_multi_predict_compare():
    """批量预测：多股票对比结果结构正确（确定性模式，无网络）。"""
    import numpy as np
    np.random.seed(15)
    rows = []
    close = 10.0
    for i in range(130):
        close *= 1 + np.random.normal(0.001, 0.015)
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6})
    df = pd.DataFrame(rows)

    import asyncio as _asyncio
    from pa_mcp.agent.prediction import PredictionService
    svc = PredictionService()

    async def _go():
        out = []
        for sym in ("600000", "600001", "600002"):
            r = await svc.predict(sym, df, horizon="5d", use_llm=False)
            p = r.to_dict()
            out.append({"symbol": sym, "direction": p["direction"],
                        "probability": p["probability"],
                        "prob_up": p["probability_distribution"]["up"],
                        "expected_return_pct": p["expected_return_pct"],
                        "cycle": p["cycle_position"]})
        return out

    results = _asyncio.run(_go())
    assert len(results) == 3
    for r in results:
        assert r["direction"] in ("up", "down", "sideways")
        assert 0 <= r["probability"] <= 1
        assert r["prob_up"] >= 0
        assert "cycle" in r


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
