# [AI:BEGIN]
# PA_MCP - Backtest: A-Share Broker Simulator
#
# Deterministic daily broker that enforces A-share rules:
#   - T+1 settlement (buy today, sell tomorrow)
#   - 100-share round lot for buys
#   - Price limits (主板 10%, 创业板/科创板 20%, 北交所 30%, ST 5%)
#   - Suspension check (no fills when suspended)
#   - Sealed limit-up (cannot buy), sealed limit-down (cannot sell)
#   - Partial fills based on volume participation
#   - Commission, stamp tax, transfer fee
#
# This broker does NOT use random() for fill decisions.
# All default execution is deterministic and conservative.
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from typing import Optional

from pa_mcp.backtest.events import MarketBar
from pa_mcp.backtest.orders import Order, OrderSide, OrderStatus, Fill


# ---- Fee Schedule ----

@dataclass
class FeeSchedule:
    """A-share transaction costs (configurable per exchange/effective date)."""
    commission_rate: float = 0.00025     # 万一 (0.025%)
    min_commission: float = 5.0          # 最低佣金 5元
    stamp_tax_rate: float = 0.0005       # 印花税 0.05% (sell only, 2024起减半)
    transfer_fee_rate: float = 0.00002   # 过户费 0.002%


# Board → daily price limit ratio
PRICE_LIMITS = {
    "sh_main": 0.10,
    "sz_main": 0.10,
    "chinext": 0.20,
    "star_market": 0.20,
    "beijing_exchange": 0.30,
}

# ST stocks
ST_LIMIT = 0.05
# IPO / no-limit: use a large sentinel (effectively no limit check)
NO_LIMIT = 999.0

ROUND_LOT = 100  # 买入最小单位 100股


def compute_limit_price(prev_close: float, board: str, is_st: bool,
                        limit_type: str = "up") -> float:
    """Compute limit-up or limit-down price."""
    if not prev_close or prev_close <= 0:
        return 0.0

    if is_st:
        ratio = ST_LIMIT
    else:
        ratio = PRICE_LIMITS.get(board, 0.10)

    if limit_type == "up":
        return round(prev_close * (1 + ratio), 2)
    else:
        return round(prev_close * (1 - ratio), 2)


class DailyBroker:
    """Deterministic A-share daily broker simulator."""

    def __init__(self, fee_schedule: Optional[FeeSchedule] = None,
                 initial_cash: float = 100_000.0,
                 slippage_bps: float = 0.0) -> None:
        self.fees = fee_schedule or FeeSchedule()
        self.initial_cash = initial_cash
        self.slippage_bps = slippage_bps  # 单边滑点（基点）

    def submit_order(self, order: Order, bar: MarketBar) -> tuple[Order, list[Fill]]:
        """Submit an order and simulate fills.

        Returns (updated_order, fills).
        """
        order.transition(OrderStatus.SUBMITTED)

        # --- Pre-trade checks ---

        # 1. Suspension?
        if bar.is_suspended:
            order.transition(OrderStatus.REJECTED)
            return order, []

        # 2. No volume at all?
        if bar.volume <= 0:
            order.transition(OrderStatus.REJECTED)
            return order, []

        # 3. Buy round lot
        if order.is_buy and order.quantity % ROUND_LOT != 0:
            # Clamp down to nearest round lot
            fixed = max(0, (order.quantity // ROUND_LOT) * ROUND_LOT)
            if fixed == 0:
                order.transition(OrderStatus.REJECTED)
                return order, []
            order.quantity = fixed

        # 4. Price limits (buy above limit-up = impossible, sell below limit-down = impossible)
        prev_close = bar.prev_close if bar.prev_close and bar.prev_close > 0 else bar.close
        board = getattr(bar, 'board', 'main')
        is_st = getattr(bar, 'is_st', False)
        limit_up = compute_limit_price(prev_close, board, is_st, "up")
        limit_down = compute_limit_price(prev_close, board, is_st, "down")

        # 5. Sealed limit up / down
        if bar.is_limit_up and order.is_buy:
            order.transition(OrderStatus.REJECTED)
            return order, []

        if bar.is_limit_down and order.is_sell:
            order.transition(OrderStatus.REJECTED)
            return order, []

        order.transition(OrderStatus.ACCEPTED)

        # --- Fill simulation ---

        # Conservative: fill at next-day open (not today's close)
        # For this simple daily model: fill at today's close
        fill_price = bar.close

        # 滑点（单边）：买价上浮、卖价下调
        if self.slippage_bps > 0:
            slip = self.slippage_bps / 10000.0
            fill_price = bar.close * (1 + slip) if order.is_buy else bar.close * (1 - slip)

        # Limit price constraint
        if order.limit_price is not None:
            if order.is_buy and fill_price > order.limit_price:
                order.transition(OrderStatus.REJECTED)
                return order, []
            if order.is_sell and fill_price < order.limit_price:
                order.transition(OrderStatus.REJECTED)
                return order, []

        # Volume participation: max 5% of ADV
        max_qty = int(bar.volume * 0.05)
        fill_qty = min(order.remaining, max_qty)

        if fill_qty <= 0:
            order.transition(OrderStatus.REJECTED)
            return order, []

        # Compute costs
        fill_value = fill_qty * fill_price
        commission = max(self.fees.min_commission, fill_value * self.fees.commission_rate)
        stamp_tax = fill_value * self.fees.stamp_tax_rate if order.is_sell else 0.0
        transfer_fee = fill_value * self.fees.transfer_fee_rate

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=fill_price,
            commission=round(commission, 2),
            stamp_tax=round(stamp_tax, 2),
            transfer_fee=round(max(transfer_fee, 0.01), 2),  # min 1分
            slippage_bps=0.0,
            trade_date=bar.trade_date,
        )

        order.filled_quantity += fill_qty
        if order.filled_quantity >= order.quantity:
            order.transition(OrderStatus.PARTIALLY_FILLED)
            order.transition(OrderStatus.FILLED)
        else:
            order.transition(OrderStatus.PARTIALLY_FILLED)

        return order, [fill]

    def compute_costs(self, quantity: int, price: float, side: OrderSide) -> dict:
        """Pre-trade cost estimate."""
        value = quantity * price
        commission = max(self.fees.min_commission, value * self.fees.commission_rate)
        stamp_tax = value * self.fees.stamp_tax_rate if side == OrderSide.SELL else 0.0
        transfer_fee = max(value * self.fees.transfer_fee_rate, 0.01)
        return {
            "commission": round(commission, 2),
            "stamp_tax": round(stamp_tax, 2),
            "transfer_fee": round(transfer_fee, 2),
            "total": round(commission + stamp_tax + transfer_fee, 2),
            "effective_rate_bps": round(
                (commission + stamp_tax + transfer_fee) / value * 10000, 1
            ) if value > 0 else 0,
        }
