# [AI:BEGIN]
# PA_MCP - Data Source Base (Adapter Port)
#
# All data source adapters implement this protocol.
# Capability declaration + conformance test support.
# [AI:END]

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


class MarketDataSourcePort(ABC):
    """Abstract port for market data providers.

    Concrete adapters (AKShare, Sina, future vendors) implement this.
    Capabilities declare what the adapter CAN do; unsupported methods
    should raise NotImplementedError or return UNSUPPORTED sentinel.
    """

    # ---- Capability ----

    CAPABILITIES: dict[str, str] = {}

    @classmethod
    def supports(cls, capability: str) -> bool:
        """Check if this adapter supports a given capability."""
        return cls.CAPABILITIES.get(capability, "unavailable") != "unavailable"

    # ---- Market Data ----

    @abstractmethod
    async def get_daily_kline(
        self, symbol: str, start_date: str = "", end_date: str = "",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Get daily OHLCV bars. Must include symbol, source, price_adjust_mode cols."""

    @abstractmethod
    async def get_realtime_spot_all(self) -> pd.DataFrame:
        """Get all A-share stocks' real-time quotes."""


class DataSourceCapability(str):
    """Standardized capability names."""
    DAILY_BARS = "daily_bars"
    MINUTE_BARS = "minute_bars"
    SECURITY_STATUS = "security_status"
    CORPORATE_ACTIONS = "corporate_actions"
    FINANCIALS = "financials"
    INDEX_MEMBERSHIP = "index_membership"
    TRADE_CALENDAR = "trade_calendar"
    BENCHMARK_TOTAL_RETURN = "benchmark_total_return"
    EVENTS = "events"
    FUND_FLOW = "fund_flow"
    DRAGON_TIGER = "dragon_tiger"
    REALTIME_QUOTE = "realtime_quote"

    # Status values
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
