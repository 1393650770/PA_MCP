# [AI:BEGIN]
# PA_MCP - 海龟交易策略测试
# [AI:END]

from __future__ import annotations

import pandas as pd

from pa_mcp.engine.strategies.base import StrategyRegistry
from pa_mcp.engine.strategies.turtle import TurtleBreakoutStrategy


def _flat_then_breakout():
    """前 60 日横盘（10±0.2）→ 后 20 日连续突破上涨。"""
    rows = []
    base = 10.0
    for i in range(60):
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": base, "high": base + 0.2, "low": base - 0.2,
                     "close": base, "volume": 1e6, "symbol": "000001"})
    c = 10.3
    for i in range(20):
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=60 + i),
                     "open": c, "high": c + 0.3, "low": c - 0.2,
                     "close": c, "volume": 3e6, "symbol": "000001"})
        c += 0.2
    return pd.DataFrame(rows)


def test_turtle_signals_on_breakout():
    df = _flat_then_breakout()
    strat = TurtleBreakoutStrategy()
    signals = strat.generate_signals(df)
    assert signals, "突破后应产生信号"
    s = signals[-1]
    assert s.direction.value == "bullish"
    assert s.signal_time is not None
    assert s.strategy_name == "turtle"
    # ATR 仓位：风险 1% / ATR%（约 2.5%）→ ~40% → 受 10% 上限
    assert 0 < s.suggested_max_position_pct <= 10
    assert s.metadata["entry_level"] > 0
    assert s.metadata["trend_end"] is not None
    # 无未来函数：entry_level 是信号日之前的通道
    assert s.signal_time[:10] >= "2026-01-01"


def test_turtle_no_signal_in_flat():
    """横盘无突破 → 无信号。"""
    rows = []
    for i in range(80):
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": 10.0, "high": 10.2, "low": 9.8,
                     "close": 10.0, "volume": 1e6, "symbol": "000001"})
    df = pd.DataFrame(rows)
    signals = TurtleBreakoutStrategy().generate_signals(df)
    assert not signals


def test_turtle_params_space():
    ps = TurtleBreakoutStrategy().get_params_space()
    names = {p.name for p in ps}
    assert {"entry_period", "exit_period", "atr_period", "risk_percent"} <= names


def test_turtle_auto_registered():
    """注册表自动发现：turtle 可被 scan_market/事件研究使用。"""
    registry = StrategyRegistry()
    registry.auto_discover()
    cls = registry.get("turtle")
    assert cls is not None
    assert cls.name == "turtle"


def test_turtle_signal_time_ordering():
    """信号时间必须按日期升序（供回测匹配）。"""
    df = _flat_then_breakout()
    signals = TurtleBreakoutStrategy().generate_signals(df)
    dates = [s.signal_time for s in signals]
    assert dates == sorted(dates)
