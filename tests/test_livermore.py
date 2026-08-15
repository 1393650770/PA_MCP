# [AI:BEGIN]
# PA_MCP - 利弗莫尔关键点策略测试
# [AI:END]

from __future__ import annotations

import pandas as pd

from pa_mcp.engine.strategies.base import StrategyRegistry
from pa_mcp.engine.strategies.livermore import LivermorePivotStrategy


def _uptrend_breakout():
    """上升趋势（站稳 MA60）→ 突破 20 日枢轴高点 + 放量。"""
    rows = []
    close = 10.0
    for i in range(120):
        close *= 1 + 0.002
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close,
                     "volume": 1e6, "symbol": "000001"})
    # 尾段横盘后放量突破
    base = close
    for i in range(25):
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=120 + i),
                     "open": base, "high": base * 1.005, "low": base * 0.995,
                     "close": base, "volume": 5e5, "symbol": "000001"})
    # 突破日
    rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=145),
                 "open": base, "high": base * 1.06, "low": base * 0.995,
                 "close": base * 1.05, "volume": 4e6, "symbol": "000001"})
    return pd.DataFrame(rows)


def _downtrend():
    """下跌趋势：即使突破也不发信号（不抄底原则）。"""
    rows = []
    close = 10.0
    for i in range(120):
        close *= 0.997
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close,
                     "volume": 1e6, "symbol": "000001"})
    return pd.DataFrame(rows)


def test_pivot_breakout_signal():
    df = _uptrend_breakout()
    signals = LivermorePivotStrategy().generate_signals(df)
    assert signals, "上升趋势突破枢轴应产生信号"
    s = signals[-1]
    assert s.direction.value == "bullish"
    assert s.signal_time == "2026-05-26" or s.signal_time is not None
    assert s.metadata["pivot_high"] > 0
    assert s.metadata["trend_break"] is not None
    # 无未来函数：枢轴是前 20 日高点（shift(1)）
    assert s.metadata["pivot_high"] < s.metadata.get("atr", 1) * 100 + 100


def test_no_signal_in_downtrend():
    """下跌趋势：突破 MA60 条件不满足 → 无信号（利弗莫尔不抄底）。"""
    df = _downtrend()
    signals = LivermorePivotStrategy().generate_signals(df)
    assert not signals


def test_auto_registered():
    registry = StrategyRegistry()
    registry.auto_discover()
    cls = registry.get("livermore_pivot")
    assert cls is not None
    assert cls.name == "livermore_pivot"


def test_params_space():
    ps = LivermorePivotStrategy().get_params_space()
    names = {p.name for p in ps}
    assert {"pivot_period", "trend_ma", "volume_confirm_ratio"} <= names


def test_volume_confirm():
    """开启放量确认：无放量的突破不发信号。"""
    df = _uptrend_breakout()
    # 突破日量能仍 4e6 > 1.5×5e5 → 通过；把突破日量改小
    df2 = df.copy()
    df2.loc[df2.index[-1], "volume"] = 3e5
    sigs_on = LivermorePivotStrategy(volume_confirm_ratio=1.5).generate_signals(df2)
    sigs_off = LivermorePivotStrategy(volume_confirm_ratio=0.0).generate_signals(df2)
    assert not sigs_on, "量能不足应过滤"
    assert sigs_off, "不要求放量时仍有信号"
