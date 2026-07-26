# [AI:BEGIN]
# PA_MCP - Backtest Engine: A-Share Trading Rules & Constraints
# Models real A-share trading constraints that backtrader/vectorbt don't handle.
# Key insight: "can buy" != "price reached limit" — modeling buyability is critical.
# [AI:END]

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class Board(str, Enum):
    """A-share trading boards with different price limit rules."""
    MAIN = "main"           # +/- 10%
    CHI_NEXT = "chi_next"   # +/- 20% (创业板)
    STAR = "star"           # +/- 20% (科创板)
    BEIJING = "beijing"     # +/- 30% (北交所)
    ST = "st"               # +/- 5% (ST stocks)
    IPO = "ipo"             # No limit (first 5 days post-registration reform)


class FeeSchedule:
    """A-share trading fees by exchange."""

    # Commission: 0.025% (min 5 CNY)
    COMMISSION_RATE = 0.00025
    MIN_COMMISSION = 5.0

    # Stamp duty: 0.05% on SELL only
    STAMP_DUTY_RATE = 0.0005

    # Transfer fee (过户费): SSE only, 0.001%
    TRANSFER_FEE_SSE = 0.00001
    TRANSFER_FEE_SZSE = 0.0  # SZSE exempt

    # ETF: no stamp duty
    ETF_STAMP_DUTY = 0.0


def get_limit_pct(board: Board) -> float:
    """Get price limit percentage for a board."""
    return {
        Board.MAIN: 0.10,
        Board.CHI_NEXT: 0.20,
        Board.STAR: 0.20,
        Board.BEIJING: 0.30,
        Board.ST: 0.05,
        Board.IPO: 999.0,  # effectively no limit
    }.get(board, 0.10)


def compute_limit_price(prev_close: float, direction: str, board: Board = Board.MAIN) -> float:
    """Compute limit-up or limit-down price for a stock.

    Args:
        prev_close: Previous close price
        direction: 'up' or 'down'
        board: Trading board
    """
    limit_pct = get_limit_pct(board)
    if limit_pct >= 999:
        return float("inf") if direction == "up" else 0.01

    if direction == "up":
        # Round to 2 decimal places, A-share convention
        return round(prev_close * (1 + limit_pct), 2)
    else:
        return round(prev_close * (1 - limit_pct), 2)


def compute_commission(amount: float, is_sell: bool = False, exchange: str = "SSE") -> float:
    """Compute total trading fees for a transaction.

    Args:
        amount: Trade amount in CNY
        is_sell: True for sell, False for buy
        exchange: 'SSE' or 'SZSE'
    """
    commission = max(amount * FeeSchedule.COMMISSION_RATE, FeeSchedule.MIN_COMMISSION)
    transfer_fee = amount * (FeeSchedule.TRANSFER_FEE_SSE if exchange == "SSE" else FeeSchedule.TRANSFER_FEE_SZSE)
    stamp_duty = amount * FeeSchedule.STAMP_DUTY_RATE if is_sell else 0.0
    return round(commission + transfer_fee + stamp_duty, 2)


def round_lots(shares: int) -> int:
    """Round down to nearest 100-share lot."""
    return (shares // 100) * 100


# ---- Buyability/Sellability Models ----

@dataclass
class LimitUpBuyability:
    """Models whether a limit-up stock is actually buyable.

    Core A-share truth: "the best limit-ups you CAN'T buy."

    Factors:
    - Seal time (封板时间): Earlier = harder to buy
    - Seal strength (封单强度): seal_amount/volume ratio
    - Board break (炸板): If it broke open, you could have bought
    - Time of day: Afternoon limit-ups are sketchier
    """

    seal_time: str = ""         # "09:35", "10:15", "14:30", etc.
    seal_amount: float = 0.0    # Seal order amount in CNY
    total_volume: float = 0.0   # Total trading volume
    broke_open: bool = False    # Did it break the limit board?
    re_sealed: bool = False     # Did it re-seal after breaking?

    def can_buy(self) -> tuple[bool, str]:
        """Determine if this limit-up was realistically buyable.

        Returns (can_buy, reason).
        """
        # Pre-market limit-up: nearly impossible to buy
        if self.seal_time <= "09:25":
            return False, "Pre-market seal — retail cannot buy"

        # Ultra-early seal (first 10 mins): very hard
        if self.seal_time <= "09:40" and not self.broke_open:
            return False, "Early seal <09:40 without breaking — near impossible"

        # Strong seal: seal amount > 5x volume = locked tight
        ratio = self.seal_amount / max(self.total_volume, 1)
        if ratio > 5 and not self.broke_open:
            return False, f"Strong seal (ratio {ratio:.1f}x) — locked"

        # Afternoon seal (>14:00): higher chance to buy, lower quality
        if self.seal_time >= "14:00" and not self.broke_open:
            # 40% chance for afternoon seals
            return random.random() < 0.4, "Afternoon seal — lower quality, possible entry"

        # Board broke open: you could have bought during the break
        if self.broke_open:
            return True, "Board broke open — opportunity existed"

        # Moderate seal (10:00-14:00, ratio 1-3x)
        return random.random() < 0.3, f"Moderate seal (ratio {ratio:.1f}x) — ~30% buy probability"


@dataclass
class LimitDownSellability:
    """Models whether you can actually sell during a limit-down.

    Consecutive limit-downs create exponentially worse sellability.
    """

    consecutive_days: int = 1
    limit_down_volume: float = 0.0
    float_market_cap: float = 0.0  # Circulating market cap

    def can_sell(self) -> tuple[bool, str]:
        """Determine if selling is realistically possible."""
        if self.consecutive_days == 1:
            # First limit-down: possible to sell if you're quick
            return random.random() < 0.3, "First limit-down — ~30% sell probability if quick"

        # Multi-day limit-down: probability decays exponentially
        prob = 0.3 * (0.3 ** (self.consecutive_days - 1))
        if prob < 0.01:
            return False, f"Day {self.consecutive_days} limit-down — nearly impossible to exit"
        return random.random() < prob, f"Day {self.consecutive_days} limit-down — {prob:.0%} sell probability"


# ---- Slippage Model ----

def compute_slippage(
    price: float, symbol: str = "",
    market_cap: Optional[float] = None,
    is_panic: bool = False,
) -> float:
    """Compute realistic slippage for a trade.

    Slippage factors by market cap:
    - Large cap (>100B): 0.1%
    - Mid cap (10-100B): 0.2%
    - Small cap (<10B): 0.3%
    - ChiNext/STAR: +0.2%
    - Panic selling: +0.2%

    Returns: slippage_pct (e.g., 0.002 = 0.2%)
    """
    base = 0.001  # 0.1% base

    if market_cap is not None:
        if market_cap < 10:
            base = 0.003  # Small cap
        elif market_cap < 100:
            base = 0.002  # Mid cap

    if is_panic:
        base += 0.002  # Panic premium

    return base


# ---- Volume / Liquidity Constraint ----

def max_buyable_shares(
    symbol: str, price: float, daily_volume: float,
    max_volume_pct: float = 0.05,
) -> int:
    """Maximum shares you can buy without excessive market impact.

    Single order should not exceed 5% of daily volume.
    """
    max_amount = daily_volume * max_volume_pct
    max_shares = int(max_amount / price) if price > 0 else 0
    return round_lots(max_shares)
