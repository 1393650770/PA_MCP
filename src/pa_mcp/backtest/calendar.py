# [AI:BEGIN]
# PA_MCP - Backtest Engine: A-Share Trading Calendar
# Provides trading day queries, next/prev trading day navigation.
# [AI:END]

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd


class AShareTradeCalendar:
    """A-share trading calendar with holiday awareness.

    Core methods:
    - is_trading_day(d) -> bool
    - next_trading_day(d) -> date
    - prev_trading_day(d) -> date
    - trading_days_between(start, end) -> list[date]
    """

    # Major Chinese holidays (approximate, updated yearly)
    # In production, fetch from exchange or AKShare: ak.tool_trade_date_hist_sina()
    _FIXED_HOLIDAYS = {
        # New Year
        date(2026, 1, 1), date(2026, 1, 2),
        # Spring Festival (approximate)
        date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18),
        date(2026, 2, 19), date(2026, 2, 20),
        # Qingming
        date(2026, 4, 5), date(2026, 4, 6),
        # Labor Day
        date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3),
        # Dragon Boat
        date(2026, 6, 19), date(2026, 6, 20),
        # Mid-Autumn + National Day
        date(2026, 9, 25), date(2026, 10, 1), date(2026, 10, 2),
        date(2026, 10, 3), date(2026, 10, 4), date(2026, 10, 5),
        date(2026, 10, 6), date(2026, 10, 7),
    }

    def __init__(self, holidays: Optional[set[date]] = None) -> None:
        self._holidays = holidays or self._FIXED_HOLIDAYS
        self._cache: dict[str, bool] = {}

    def is_trading_day(self, d: date) -> bool:
        """Check if date is a trading day (Mon-Fri, not a holiday)."""
        key = d.isoformat()
        if key in self._cache:
            return self._cache[key]

        # Weekend check
        if d.weekday() >= 5:  # Saturday=5, Sunday=6
            self._cache[key] = False
            return False

        # Holiday check
        if d in self._holidays:
            self._cache[key] = False
            return False

        self._cache[key] = True
        return True

    def next_trading_day(self, d: Optional[date] = None) -> date:
        """Get next trading day after d (or today)."""
        if d is None:
            d = date.today()
        d = d + timedelta(days=1)
        while not self.is_trading_day(d):
            d = d + timedelta(days=1)
        return d

    def prev_trading_day(self, d: Optional[date] = None) -> date:
        """Get previous trading day before d (or today)."""
        if d is None:
            d = date.today()
        d = d - timedelta(days=1)
        while not self.is_trading_day(d):
            d = d - timedelta(days=1)
        return d

    def trading_days_between(self, start: date, end: date) -> list[date]:
        """Get all trading days between start and end (inclusive)."""
        days: list[date] = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days

    def trading_days_count(self, start: date, end: date) -> int:
        """Count trading days between start and end."""
        return len(self.trading_days_between(start, end))


# Global instance
_calendar: Optional[AShareTradeCalendar] = None


def get_calendar() -> AShareTradeCalendar:
    """Get or create the trading calendar singleton."""
    global _calendar
    if _calendar is None:
        _calendar = AShareTradeCalendar()
    return _calendar
