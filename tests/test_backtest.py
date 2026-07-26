# [AI:BEGIN]
# PA_MCP - Tests: Backtest Engine & A-Share Rules
# [AI:END]

from __future__ import annotations

import pytest
from datetime import date

from pa_mcp.backtest.calendar import AShareTradeCalendar, get_calendar
from pa_mcp.backtest.rules import (
    Board, FeeSchedule, compute_commission, compute_limit_price,
    compute_slippage, get_limit_pct, max_buyable_shares, round_lots,
    LimitUpBuyability, LimitDownSellability,
)


# ---- Trading Calendar Tests ----

class TestTradeCalendar:
    """A-share trading calendar tests."""

    def test_weekend_not_trading_day(self) -> None:
        """Saturday and Sunday are not trading days."""
        cal = AShareTradeCalendar()
        # 2026-07-25 is Saturday, 2026-07-26 is Sunday
        assert cal.is_trading_day(date(2026, 7, 25)) is False
        assert cal.is_trading_day(date(2026, 7, 26)) is False

    def test_weekday_is_trading_day(self) -> None:
        """Normal weekday (not holiday) IS a trading day."""
        cal = AShareTradeCalendar()
        # 2026-07-22 is Wednesday
        assert cal.is_trading_day(date(2026, 7, 22)) is True

    def test_holiday_not_trading_day(self) -> None:
        """National Day holiday is not a trading day."""
        cal = AShareTradeCalendar()
        # Oct 1 is National Day
        assert cal.is_trading_day(date(2026, 10, 1)) is False

    def test_next_trading_day_skips_weekend(self) -> None:
        """Next trading day after Friday should be Monday."""
        cal = AShareTradeCalendar()
        # 2026-07-24 is Friday
        next_day = cal.next_trading_day(date(2026, 7, 24))
        assert next_day.weekday() == 0  # Monday

    def test_trading_days_between(self) -> None:
        """Count trading days in a week."""
        cal = AShareTradeCalendar()
        days = cal.trading_days_between(date(2026, 7, 20), date(2026, 7, 24))
        assert len(days) == 5  # Mon-Fri


# ---- A-Share Rules Tests ----

class TestAShareRules:
    """Trading rules and constraints tests."""

    def test_main_board_limit_pct(self) -> None:
        """Main board limit is +/-10%."""
        assert get_limit_pct(Board.MAIN) == 0.10

    def test_chi_next_limit_pct(self) -> None:
        """ChiNext limit is +/-20%."""
        assert get_limit_pct(Board.CHI_NEXT) == 0.20

    def test_st_stock_limit_pct(self) -> None:
        """ST stocks limit is +/-5%."""
        assert get_limit_pct(Board.ST) == 0.05

    def test_limit_up_price_main(self) -> None:
        """Limit-up price for main board."""
        price = compute_limit_price(10.00, "up", Board.MAIN)
        assert price == 11.00  # 10.00 * 1.10

    def test_limit_down_price_main(self) -> None:
        """Limit-down price for main board."""
        price = compute_limit_price(10.00, "down", Board.MAIN)
        assert price == 9.00  # 10.00 * 0.90

    def test_commission_buy(self) -> None:
        """Buy commission calculation."""
        fee = compute_commission(10000.0, is_sell=False, exchange="SSE")
        # Commission: 10000 * 0.00025 = 2.5 (min 5) -> 5.0
        # Transfer fee: 10000 * 0.00001 = 0.1
        # Stamp duty: 0 (buy only)
        assert fee == pytest.approx(5.1, rel=0.1)

    def test_commission_sell(self) -> None:
        """Sell commission with stamp duty."""
        fee = compute_commission(10000.0, is_sell=True, exchange="SSE")
        # Commission: 5.0, Transfer: 0.1, Stamp: 10000 * 0.0005 = 5.0
        assert fee == pytest.approx(10.1, rel=0.2)

    def test_round_lots(self) -> None:
        """Shares should round down to 100."""
        assert round_lots(250) == 200
        assert round_lots(100) == 100
        assert round_lots(99) == 0

    def test_slippage_large_cap(self) -> None:
        """Large cap stocks have lower slippage."""
        slip = compute_slippage(10.0, market_cap=200.0)
        assert slip == 0.001  # 0.1%

    def test_slippage_small_cap(self) -> None:
        """Small cap stocks have higher slippage."""
        slip = compute_slippage(10.0, market_cap=5.0)
        assert slip == 0.003  # 0.3%

    def test_slippage_panic(self) -> None:
        """Panic premium adds 0.2%."""
        slip = compute_slippage(10.0, market_cap=200.0, is_panic=True)
        assert slip == 0.003  # 0.1% + 0.2%

    def test_max_buyable_shares(self) -> None:
        """Volume constraint limits position size."""
        shares = max_buyable_shares("000001", 10.0, 1_000_000.0)
        # Max 5% of volume = 50,000 CNY → 5,000 shares → round to 5000
        assert shares == 5000


# ---- Buyability Tests ----

class TestBuyability:
    """Limit-up buyability model tests."""

    def test_early_seal_unbuyable(self) -> None:
        """Pre-market seal: cannot buy."""
        model = LimitUpBuyability(seal_time="09:25", seal_amount=1e8, total_volume=1e7)
        can, reason = model.can_buy()
        assert can is False
        assert "Pre-market" in reason

    def test_strong_seal_unbuyable(self) -> None:
        """Seal amount > 5x volume: cannot buy."""
        model = LimitUpBuyability(seal_time="10:30", seal_amount=1e8, total_volume=1e7)
        can, reason = model.can_buy()
        assert can is False
        assert "Strong seal" in reason

    def test_broken_seal_buyable(self) -> None:
        """If board broke open, you could have bought."""
        model = LimitUpBuyability(seal_time="10:00", seal_amount=1e7, total_volume=1e7, broke_open=True)
        can, reason = model.can_buy()
        assert can is True


# ---- Limit-Down Sellability Tests ----

class TestSellability:
    """Limit-down sellability model tests."""

    def test_multi_day_limit_down_hard_to_sell(self) -> None:
        """Day 3+ limit-down: nearly impossible."""
        model = LimitDownSellability(consecutive_days=3)
        can, reason = model.can_sell()
        # With 0.3 * 0.3^2 = 2.7% prob, almost always False
        assert "nearly impossible" in reason or can is False
