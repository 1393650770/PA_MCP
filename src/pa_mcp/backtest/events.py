# [AI:BEGIN]
# PA_MCP - Backtest: Event Model
#
# Event-driven simulation clock. The backtest loop processes
# events in chronological order. Each event carries a timestamp
# and triggers domain logic (signal generation, order routing,
# fill simulation, mark-to-market, etc.).
#
# Event order (daily):
#   SessionStart → CorporateAction* → BarOpen → BarClose →
#   SignalGenerated* → TargetPortfolioGenerated* →
#   OrderIntentCreated* → PreTradeRiskChecked* →
#   OrderSubmitted* → OrderAccepted/Rejected* →
#   Fill* → MarkToMarket → Settlement → SessionEnd
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CORPORATE_ACTION = "corporate_action"
    BAR_DATA = "bar_data"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_INTENT = "order_intent"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXPIRED = "order_expired"
    FILL = "fill"
    MARK_TO_MARKET = "mark_to_market"
    SETTLEMENT = "settlement"


@dataclass
class Event:
    """One simulation event."""
    event_type: EventType
    timestamp: datetime  # UTC
    trade_date: Date
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""


@dataclass
class MarketBar:
    """One bar of market data for a symbol."""
    symbol: str
    trade_date: Date
    open: float
    high: float
    low: float
    close: float
    volume: float   # shares
    amount: float   # yuan
    turnover: float = 0.0
    prev_close: Optional[float] = None
    is_suspended: bool = False
    is_limit_up: bool = False  # sealed (一字板)
    is_limit_down: bool = False
    board: str = "main"
