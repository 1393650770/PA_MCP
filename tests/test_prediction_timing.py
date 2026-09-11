# [AI:BEGIN]
# PA_MCP - 预测时效性 / 反追高 测试
#
# 背景（真实事故）：用户反馈"看到预测结果的时候，它已经在猛猛大涨了"。
# 两个根因各写一组防回归测试：
#   1. 预测不暴露数据基准日（as_of），用户无法判断行情走没走完；
#   2. 对已经涨停/20日大涨的票照样给高置信看涨 → 引导追高。
# [AI:END]

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pa_mcp.agent.prediction import (
    GUARD_FACTORS,
    PredictionResult,
    PredictionService,
    entry_timing,
    extract_features,
)


def _df(n: int = 120, last_chg: float = 0.0, ret20: float = 0.0,
        as_of: str = "2026-09-10") -> pd.DataFrame:
    """构造日线：可指定末日涨幅与 20 日涨幅，用于触发追高判定。"""
    np.random.seed(3)
    rows = []
    # 倒推：先得到 20 日前的价格，再线性铺到"20 日涨幅"与"末日涨幅"
    close = 10.0
    for i in range(n):
        rows.append(close)
        close *= 1 + np.random.normal(0.0005, 0.008)
    closes = np.array(rows)
    # 叠加 20 日涨幅：只抬升最后 21 根（保证 close[-1]/close[-21] = 1+ret20）
    if ret20:
        ramp = np.linspace(1.0, 1.0 + ret20 / 100.0, 21)
        closes[-21:] = closes[-21:] * ramp
    if last_chg:
        closes[-1] = closes[-2] * (1 + last_chg / 100.0)
    dates = [datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=(n - 1 - i))
             for i in range(n)]
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": closes * 0.995,
        "high": np.maximum(closes * 1.01, closes * (1 + abs(last_chg) / 100.0)),
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(n, 1e6),
    })


# ---- 入场时机判定 ----

def test_entry_timing_ok_on_normal_stock():
    feats = extract_features(_df())
    et = entry_timing(feats)
    assert et["state"] == "ok"
    assert et["state_zh"]


def test_entry_timing_limit_up():
    """末日涨停 → limit_up（追高性价比最低，必须被识别）。"""
    feats = extract_features(_df(last_chg=10.0))
    et = entry_timing(feats)
    assert et["state"] == "limit_up"
    assert feats["last_chg_pct"] >= 9.8


def test_entry_timing_overheated_by_ret20():
    feats = extract_features(_df(ret20=35.0))
    assert entry_timing(feats)["state"] == "overheated"


def test_entry_timing_extended_by_ret5():
    df = _df(ret20=14.0)
    # 5 日内急拉 15%
    closes = df["close"].to_numpy(dtype=float).copy()
    closes[-5:] = closes[-6] * np.linspace(1.03, 1.15, 5)
    df["close"] = closes
    feats = extract_features(df)
    assert entry_timing(feats)["state"] in ("extended", "overheated")


def test_entry_timing_note_is_actionable():
    """告警文案必须能直接指导操作（含涨幅数字）。"""
    et = entry_timing(extract_features(_df(last_chg=10.0)))
    assert "涨停" in et["note"]
    assert "%" in et["note"]


# ---- 反追高校准（_finalize） ----

def _svc() -> PredictionService:
    return PredictionService()


def test_guard_shrinks_up_probability_on_limit_up():
    """涨停票的看涨预测必须被收缩，且给出追高预警。"""
    svc = _svc()
    feats = extract_features(_df(last_chg=10.0))
    r = PredictionResult(symbol="600000", predict_date="2026-09-11", horizon="5d")
    r.direction = "up"
    r.probability = 0.80
    r.prob_up, r.prob_down, r.prob_sideways = 0.80, 0.12, 0.08
    r.expected_return_pct = 5.0
    r.confidence = 0.7
    r.key_risks = ["原有风险"]

    before = r.probability
    svc._stale_trading_days = lambda a, b: 0      # 隔离 DB
    svc._calibrate = lambda res: res.probability  # 隔离 DB
    svc._finalize(r, feats)

    assert r.probability < before, "涨停票的看涨概率必须被收缩"
    assert r.probability >= 0.5, "收缩只向 0.5 靠拢，不反向"
    assert r.expected_return_pct < 5.0, "期望收益必须打折"
    assert any("追高预警" in x for x in r.key_risks)
    assert "原有风险" in r.key_risks, "原有风险不能被覆盖"
    assert r.entry_timing == "limit_up"


def test_guard_keeps_distribution_sum_to_one():
    """概率收缩后 up+down+sideways 仍须为 1。"""
    svc = _svc()
    feats = extract_features(_df(ret20=40.0))
    r = PredictionResult(symbol="600000", predict_date="2026-09-11", horizon="5d")
    r.direction = "up"
    r.probability, r.prob_up = 0.85, 0.85
    r.prob_down, r.prob_sideways = 0.10, 0.05
    svc._stale_trading_days = lambda a, b: 0
    svc._calibrate = lambda res: res.probability
    svc._finalize(r, feats)

    total = r.prob_up + r.prob_down + r.prob_sideways
    assert abs(total - 1.0) < 0.01, f"概率分布和应为 1，实际 {total}"
    assert abs(r.prob_up - r.probability) < 0.01


def test_guard_not_applied_to_down_direction():
    """看跌方向不做追高收缩（不存在'追高'问题）。"""
    svc = _svc()
    feats = extract_features(_df(last_chg=10.0))
    r = PredictionResult(symbol="600000", predict_date="2026-09-11", horizon="5d")
    r.direction = "down"
    r.probability, r.prob_up = 0.70, 0.15
    r.prob_down, r.prob_sideways = 0.70, 0.15
    r.expected_return_pct = -4.0
    svc._stale_trading_days = lambda a, b: 0
    svc._calibrate = lambda res: res.probability
    svc._finalize(r, feats)
    assert r.probability == 0.70
    assert not any("追高预警" in x for x in r.key_risks)


def test_guard_factor_table_is_monotonic():
    """越热收缩越狠。"""
    assert GUARD_FACTORS["limit_up"][0] < GUARD_FACTORS["overheated"][0]
    assert GUARD_FACTORS["overheated"][0] < GUARD_FACTORS["extended"][0]
    assert GUARD_FACTORS["ok"] == (1.0, 1.0)


# ---- 时效性 ----

def test_as_of_and_base_close_are_filled():
    feats = extract_features(_df(as_of="2026-09-10"))
    svc = _svc()
    r = PredictionResult(symbol="600000", predict_date="2026-09-11", horizon="5d")
    svc._stale_trading_days = lambda a, b: 0
    svc._calibrate = lambda res: res.probability
    svc._finalize(r, feats)
    assert r.as_of == "2026-09-10", "必须暴露数据基准日"
    assert r.base_close > 0, "必须暴露基准价供用户对照现价"


def test_stale_prediction_is_flagged_and_discounted():
    """数据滞后 1 个交易日 → 告警 + 置信度打折。"""
    svc = _svc()
    feats = extract_features(_df(as_of="2026-09-09"))
    r = PredictionResult(symbol="600000", predict_date="2026-09-10", horizon="5d")
    r.confidence = 0.8
    svc._stale_trading_days = lambda a, b: 1
    svc._calibrate = lambda res: res.probability
    svc._finalize(r, feats)
    assert r.stale_days == 1
    assert r.confidence < 0.8, "滞后数据的置信度必须打折"
    assert any("数据时效" in x for x in r.key_risks)


def test_freshness_text_in_payload():
    r = PredictionResult(symbol="600000", predict_date="2026-09-11", horizon="5d")
    r.stale_days = 2
    r.as_of = "2026-09-09"
    assert "滞后" in r.to_dict()["freshness"]
    r.stale_days = 0
    assert "滞后" not in r.to_dict()["freshness"]


def test_stale_days_fallback_weekday_count(monkeypatch):
    """日历不可用时的降级估算：只数工作日。

    强制让 _store() 抛错，保证走降级分支（否则依赖真实库里的日历，
    全量跑套件时若被换库/日历未覆盖，结果会随环境漂移）。
    """
    svc = _svc()

    def _boom():
        raise RuntimeError("no db")

    monkeypatch.setattr(svc, "_store", _boom)
    # 2026-09-04(周五) → 2026-09-07(周一) 跨周末，应计 1 个工作日
    assert svc._stale_trading_days("2026-09-04", "2026-09-07") == 1
    # 同日不滞后
    assert svc._stale_trading_days("2026-09-10", "2026-09-10") == 0
    # as_of 晚于 predict_date 视为不滞后
    assert svc._stale_trading_days("2026-09-11", "2026-09-10") == 0
    # 整周：09-07(周一) → 09-14(周一) 共 5 个工作日
    assert svc._stale_trading_days("2026-09-07", "2026-09-14") == 5


def test_stale_days_zero_when_calendar_covers_but_no_trading_day(monkeypatch):
    """日历覆盖该区间但区间内无交易日（假期周）→ 可信地返回 0。"""
    svc = _svc()

    class _FakeStore:
        def __init__(self):
            self.calls = 0

        def query_df(self, sql, params=None):
            self.calls += 1
            import pandas as pd
            # 第一次查询（区间内交易日数）= 0；第二次（覆盖范围检查）> 0
            return pd.DataFrame({"c": [0 if self.calls == 1 else 5]})

        def close(self):
            pass

    fake = _FakeStore()
    monkeypatch.setattr(svc, "_store", lambda: fake)
    assert svc._stale_trading_days("2026-09-04", "2026-09-07") == 0
    assert fake.calls == 2


def test_stale_days_falls_back_when_calendar_not_covering(monkeypatch):
    """日历表存在但没覆盖该区间（换库/新装）→ 必须退化估算，不能乐观返回 0。"""
    svc = _svc()

    class _EmptyStore:
        def query_df(self, sql, params=None):
            import pandas as pd
            return pd.DataFrame({"c": [0]})   # 区间内 0，覆盖范围也 0

        def close(self):
            pass

    monkeypatch.setattr(svc, "_store", lambda: _EmptyStore())
    # 日历不可信 → 走工作日估算：09-04(周五) → 09-07(周一) = 1
    assert svc._stale_trading_days("2026-09-04", "2026-09-07") >= 1


# ---- 端到端：确定性预测路径 ----

@pytest.mark.asyncio
async def test_predict_on_limit_up_stock_is_conservative():
    """端到端：涨停票的确定性预测不得给出高置信看涨。"""
    svc = _svc()
    svc._stale_trading_days = lambda a, b: 0
    df = _df(last_chg=10.0, ret20=25.0)
    # 让打分明确看多：末日大涨 + 均线多头
    res = await svc.predict("600000", df, horizon="5d", use_llm=False)
    assert res.as_of
    assert res.entry_timing == "limit_up"
    if res.direction == "up":
        assert res.probability <= 0.72, "涨停票看涨概率应被明显收缩"
        assert any("追高" in x for x in res.key_risks)


@pytest.mark.asyncio
async def test_predict_payload_exposes_timing_fields():
    res = await PredictionService().predict(
        "600000", _df(), horizon="5d", use_llm=False)
    d = res.to_dict()
    for k in ("as_of", "stale_days", "freshness", "entry_timing",
              "entry_timing_zh", "entry_note", "base_close",
              "calibrated_probability"):
        assert k in d, f"预测输出缺少时效性字段 {k}"
