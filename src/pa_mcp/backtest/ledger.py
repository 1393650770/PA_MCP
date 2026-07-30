# [AI:BEGIN]
# PA_MCP - Backtest: Ledger (Cash + Positions + NAV accounting)
#
# Maintains the single source of truth for:
#   - Cash balance (with frozen/unfrozen separation)
#   - Position lots (for T+1 sellable tracking)
#   - NAV = cash + Σ(position_qty × mark_price)
#
# Every Fill event updates the ledger. Every MarkToMarket event
# recalculates NAV. The ledger enforces no-negative-cash and
# no-negative-long-positions (no margin/short by default).
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Optional


@dataclass
class PositionLot:
    """One tax lot of a position (for T+1 tracking)."""
    symbol: str
    quantity: int          # shares held
    avg_cost: float        # weighted average cost per share
    buy_date: Date          # T (not sellable until T+1)
    sellable_date: Date     # T+1 or later


@dataclass
class Ledger:
    """Immutable-style ledger for a single backtest run."""

    cash: float = 0.0
    frozen_cash: float = 0.0          # Reserved for pending orders
    positions: dict[str, list[PositionLot]] = field(default_factory=dict)
    nav_history: list[dict] = field(default_factory=list)

    # ---- Position helpers ----

    def get_position(self, symbol: str) -> int:
        """Total shares held (all lots)."""
        lots = self.positions.get(symbol, [])
        return sum(lot.quantity for lot in lots)

    def get_sellable_quantity(self, symbol: str, as_of: Date) -> int:
        """Shares available to sell (T+1 settled)."""
        lots = self.positions.get(symbol, [])
        return sum(lot.quantity for lot in lots if lot.sellable_date <= as_of)

    def get_avg_cost(self, symbol: str) -> float:
        """Weighted average cost per share."""
        lots = self.positions.get(symbol, [])
        total_qty = sum(lot.quantity for lot in lots)
        if total_qty == 0:
            return 0.0
        return sum(lot.quantity * lot.avg_cost for lot in lots) / total_qty

    # ---- Cash ----

    @property
    def available_cash(self) -> float:
        """Cash available for new orders."""
        return self.cash - self.frozen_cash

    def add_cash(self, amount: float) -> None:
        """Deposit cash (e.g., from a sale)."""
        self.cash += amount

    def remove_cash(self, amount: float) -> None:
        """Withdraw cash (e.g., for a buy)."""
        if amount > self.available_cash:
            raise ValueError(
                f"Insufficient cash: need {amount:.2f}, available {self.available_cash:.2f}"
            )
        self.cash -= amount

    def freeze_cash(self, amount: float) -> None:
        """Reserve cash for a pending order."""
        if amount > self.available_cash:
            raise ValueError(f"Cannot freeze {amount:.2f} — insufficient funds")
        self.frozen_cash += amount

    def unfreeze_cash(self, amount: float) -> None:
        """Release reserved cash."""
        self.frozen_cash = max(0, self.frozen_cash - amount)

    # ---- Position mutation ----

    def add_lot(self, symbol: str, quantity: int, avg_cost: float, buy_date: Date) -> None:
        """Record a buy."""
        if symbol not in self.positions:
            self.positions[symbol] = []
        # Sellable on T+1
        from datetime import timedelta
        self.positions[symbol].append(PositionLot(
            symbol=symbol,
            quantity=quantity,
            avg_cost=avg_cost,
            buy_date=buy_date,
            sellable_date=buy_date + timedelta(days=1),
        ))

    def remove_shares(self, symbol: str, quantity: int, as_of: Date) -> float:
        """Remove shares (FIFO). Returns the cost basis removed."""
        sellable = self.get_sellable_quantity(symbol, as_of)
        if quantity > sellable:
            raise ValueError(
                f"Cannot sell {quantity} shares of {symbol} — only {sellable} sellable"
            )

        lots = self.positions.get(symbol, [])
        remaining = quantity
        cost_removed = 0.0

        # FIFO: sell oldest lots first
        for lot in sorted(lots, key=lambda l: l.buy_date):
            if remaining <= 0:
                break
            if lot.sellable_date > as_of:
                continue
            take = min(lot.quantity, remaining)
            cost_removed += take * lot.avg_cost
            lot.quantity -= take
            remaining -= take

        # Remove empty lots
        self.positions[symbol] = [lot for lot in lots if lot.quantity > 0]
        if not self.positions[symbol]:
            del self.positions[symbol]

        return cost_removed

    # ---- NAV ----

    def compute_nav(self, mark_prices: dict[str, float]) -> float:
        """NAV = cash + position value at mark prices."""
        position_value = sum(
            self.get_position(sym) * price
            for sym, price in mark_prices.items()
        )
        return self.cash + position_value

    def record_nav(self, trade_date: Date, mark_prices: dict[str, float]) -> dict:
        """Record one NAV snapshot and return it."""
        nav = self.compute_nav(mark_prices)
        snapshot = {
            "trade_date": trade_date.isoformat(),
            "nav": nav,
            "cash": self.cash,
            "positions": {sym: self.get_position(sym) for sym in self.positions},
        }
        self.nav_history.append(snapshot)
        return snapshot
