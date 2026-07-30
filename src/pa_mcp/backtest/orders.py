# [AI:BEGIN]
# PA_MCP - Backtest: Order Model
#
# T+1 A-share order lifecycle:
#   CREATED → SUBMITTED → ACCEPTED → PARTIALLY_FILLED → FILLED
#   ACCEPTED / PARTIALLY_FILLED → CANCELLED / EXPIRED / REJECTED
#
# Each Fill is an independent record with commission, tax, slippage.
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


VALID_TRANSITIONS = {
    OrderStatus.CREATED: {OrderStatus.SUBMITTED},
    OrderStatus.SUBMITTED: {OrderStatus.ACCEPTED, OrderStatus.REJECTED},
    OrderStatus.ACCEPTED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED},
    # Terminal states
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.EXPIRED: set(),
    OrderStatus.REJECTED: set(),
}


@dataclass
class Order:
    """A single order in the A-share market."""
    order_id: str = field(default_factory=lambda: uuid4().hex[:12])
    client_order_id: str = ""          # Idempotency key from strategy
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0                  # Desired shares (must be 100-share multiple for buy)
    filled_quantity: int = 0
    limit_price: Optional[float] = None  # None = market order
    status: OrderStatus = OrderStatus.CREATED
    created_at: Optional[datetime] = None
    trade_date: Optional[Date] = None
    strategy_name: str = ""
    signal_id: str = ""
    risk_decision_id: str = ""         # Set by RiskGuard before submission

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                                OrderStatus.EXPIRED, OrderStatus.REJECTED)

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.CREATED, OrderStatus.SUBMITTED,
                                OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)

    def transition(self, new_status: OrderStatus) -> None:
        """State machine: only valid transitions allowed."""
        valid = VALID_TRANSITIONS.get(self.status, set())
        if new_status not in valid:
            raise ValueError(
                f"Invalid order transition: {self.status.value} → {new_status.value}"
            )
        self.status = new_status


@dataclass
class Fill:
    """One fill (partial or full) of an order."""
    fill_id: str = field(default_factory=lambda: uuid4().hex[:12])
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0         # 佣金
    stamp_tax: float = 0.0          # 印花税 (sell only)
    transfer_fee: float = 0.0       # 过户费
    slippage_bps: float = 0.0       # 实际滑点 (bp)
    fill_time: Optional[datetime] = None
    trade_date: Optional[Date] = None

    @property
    def total_cost(self) -> float:
        """Total transaction cost (commission + tax + fees)."""
        return self.commission + self.stamp_tax + self.transfer_fee

    @property
    def fill_value(self) -> float:
        """Gross value of this fill."""
        return self.quantity * self.price
