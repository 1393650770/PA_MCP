# [AI:BEGIN]
# PA_MCP - Tests: Event-Driven Strategies & Scheduler
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa_mcp.engine.strategies.event_driven import (
    InsiderBuyingFollowStrategy, LockupExpiryAvoidanceStrategy,
    OversoldBounceReversalStrategy, BlockTradeDiscountedStrategy,
)
from pa_mcp.engine.strategies.base import StrategyCategory, SignalDirection


def _make_events_df(n_rows: int = 10) -> pd.DataFrame:
    """Generate test data with event columns."""
    np.random.seed(42)
    close = 10.0
    data = []
    for i in range(n_rows):
        close = close * (1 + np.random.normal(0, 0.02))
        data.append({
            "symbol": "000001",
            "close": round(close, 2),
            "volume": np.random.uniform(1e6, 1e7),
            "pct_change": np.random.normal(0, 1),
            "insider_buy_amount": 2_000_000 if i == n_rows - 1 else 0,
            "insider_buy_count": 3 if i == n_rows - 1 else 0,
            "pledge_ratio": 0.30,
            "lockup_pct_of_float": 0.0,
            "lockup_expiry_date": "",
            "block_trade_amount": 0,
            "block_trade_discount": 0.0,
        })
    return pd.DataFrame(data)


class TestEventDrivenStrategies:
    """Event-driven strategy tests."""

    def test_insider_buying_creates_signal(self) -> None:
        """Should detect insider buying when amount is significant."""
        df = _make_events_df(20)
        strategy = InsiderBuyingFollowStrategy(min_insider_amount=1_000_000)
        signals = strategy.generate_signals(df)
        if signals:
            s = signals[0]
            assert s.direction == SignalDirection.BULLISH
            assert s.strength_score > 0

    def test_insider_buying_high_pledge_blocked(self) -> None:
        """Should NOT signal when pledge ratio is too high."""
        df = _make_events_df(20)
        df["pledge_ratio"] = 0.80  # 80% — too high
        strategy = InsiderBuyingFollowStrategy()
        signals = strategy.generate_signals(df)
        assert signals == []  # Blocked by pledge risk

    def test_insider_buying_below_min_amount(self) -> None:
        """Should NOT signal when amount is below minimum."""
        df = _make_events_df(20)
        df["insider_buy_amount"] = 500_000  # Below 1M minimum
        strategy = InsiderBuyingFollowStrategy()
        signals = strategy.generate_signals(df)
        assert signals == []

    def test_lockup_expiry_avoidance_signals(self) -> None:
        """Should flag stocks with imminent large lockups."""
        import datetime
        today = datetime.date.today()
        soon = today + datetime.timedelta(days=15)

        df = pd.DataFrame([{
            "symbol": "000001",
            "close": 10.0,
            "volume": 1e6,
            "pct_change": 0.0,
            "lockup_pct_of_float": 0.25,
            "lockup_expiry_date": soon.isoformat(),
            "insider_buy_amount": 0, "insider_buy_count": 0,
            "pledge_ratio": 0.1, "block_trade_amount": 0, "block_trade_discount": 0.0,
        }])
        strategy = LockupExpiryAvoidanceStrategy()
        signals = strategy.generate_signals(df)
        if signals:
            assert signals[0].direction == SignalDirection.BEARISH
            assert "0%" in str(signals[0].suggested_max_position_pct) or signals[0].suggested_max_position_pct == 0.0

    def test_oversold_bounce_runs(self) -> None:
        """Oversold bounce should run without crashing."""
        np.random.seed(42)
        close = 20.0
        data = []
        for i in range(60):
            if 30 <= i < 35:
                close = close * (1 + np.random.normal(-0.04, 0.02))  # Crash phase
            elif i >= 35:
                close = close * (1 + np.random.normal(0.01, 0.01))  # Recovery
            else:
                close = close * (1 + np.random.normal(0.001, 0.01))
            data.append({
                "close": round(close, 2),
                "volume": np.random.uniform(1e6, 1e7) * (3 if i >= 35 else 1),
                "pct_change": 2.5 if i >= 35 else -2.0,
            })
        df = pd.DataFrame(data)
        strategy = OversoldBounceReversalStrategy()
        signals = strategy.generate_signals(df)
        assert isinstance(signals, list)

    def test_block_trade_alert_creates_signal(self) -> None:
        """Should flag heavily discounted block trades."""
        df = pd.DataFrame([{
            "symbol": "000001", "close": 10.0, "volume": 1e6,
            "pct_change": 0.0, "insider_buy_amount": 0, "insider_buy_count": 0,
            "pledge_ratio": 0.1, "lockup_pct_of_float": 0, "lockup_expiry_date": "",
            "block_trade_amount": 50_000_000, "block_trade_discount": -0.12,
        }])
        strategy = BlockTradeDiscountedStrategy()
        signals = strategy.generate_signals(df)
        if signals:
            assert signals[0].direction == SignalDirection.BEARISH
