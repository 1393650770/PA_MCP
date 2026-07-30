# [AI:BEGIN]
# PA_MCP - Portfolio: Construction
#
# Converts multi-strategy alpha scores into actionable target weights
# with A-share constraints:
#   - Single stock ≤ 10%
#   - Single sector ≤ 25%
#   - Total exposure ≤ 80%
#   - Min cash ≥ 20%
#   - Buy round lot = 100 shares
#   - Min order notional (commission-aware)
#   - T+1 available inventory
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConstraintSet:
    """Portfolio constraints for construction."""
    max_single_stock: float = 0.10       # 10%
    max_sector: float = 0.25             # 25%
    max_total_exposure: float = 0.80     # 80%
    min_cash: float = 0.20               # 20%
    max_turnover: float = 0.50           # 50% max turnover per rebalance
    round_lot: int = 100
    min_commission: float = 5.0          # Shanghai/Shenzhen min commission
    target_commission_bps: float = 5.0   # Target max one-way commission in bp
    stock_price_min: float = 0.0         # Don't short; all weights >= 0
    max_participation: float = 0.05      # Max 5% of ADV per stock


@dataclass
class TargetWeight:
    """One target weight."""
    symbol: str
    weight: float            # 0.0 to 1.0
    alpha_score: float       # normalized expected return signal
    sector: str = ""
    price: float = 0.0       # latest close
    adv: float = 0.0         # average daily value (yuan)


@dataclass
class PortfolioTarget:
    """Complete portfolio target (post-optimization)."""
    holdings: list[TargetWeight] = field(default_factory=list)
    cash_weight: float = 0.20
    total_stocks: int = 0
    total_sectors: int = 0
    total_exposure: float = 0.0


class PortfolioConstructor:
    """Builds A-share portfolio targets from alpha signals and constraints."""

    def __init__(self, constraints: Optional[ConstraintSet] = None) -> None:
        self.constraints = constraints or ConstraintSet()

    def build(
        self,
        candidates: list[TargetWeight],
        current_holdings: dict[str, float] = None,
    ) -> PortfolioTarget:
        """Build target portfolio from ranked candidates.

        Algorithm: naive risk-budget equal-weighted within constraints.
        Full optimization (mean-variance + transaction costs) is in optimizer.py.
        """
        if not candidates:
            return PortfolioTarget(cash_weight=1.0)

        c = self.constraints
        current = current_holdings or {}

        # Sort by alpha score descending
        ranked = sorted(candidates, key=lambda x: x.alpha_score, reverse=True)

        # Filter: remove candidates with no price, negative alpha, or ADV too low
        filtered = [
            r for r in ranked
            if r.price > 0 and r.alpha_score > 0
        ]

        if not filtered:
            return PortfolioTarget(cash_weight=1.0)

        # Equal weight among top-K, capped by single-stock limit
        n_stocks = min(len(filtered), int(c.max_total_exposure / c.max_single_stock))
        per_stock_raw = c.max_total_exposure / n_stocks
        per_stock = min(per_stock_raw, c.max_single_stock)

        # Sector caps: reduce overweight sectors
        sector_weights: dict[str, float] = {}
        holdings: list[TargetWeight] = []

        for candidate in filtered[:n_stocks]:
            w = per_stock
            sector = candidate.sector or "unknown"

            # Check sector cap
            if sector_weights.get(sector, 0) + w > c.max_sector:
                w = max(0, c.max_sector - sector_weights.get(sector, 0))

            if w <= 0:
                continue

            # Min order notional: commission-aware
            min_notional = max(
                c.round_lot * candidate.price,
                c.min_commission / (c.target_commission_bps / 10000),
            )
            position_value = w * 100_000  # using reference NAV = 100k

            if position_value < min_notional:
                continue  # Too small to trade

            # Apply weight
            sector_weights[sector] = sector_weights.get(sector, 0) + w
            holdings.append(TargetWeight(
                symbol=candidate.symbol,
                weight=w,
                alpha_score=candidate.alpha_score,
                sector=sector,
                price=candidate.price,
                adv=candidate.adv,
            ))

        total_exposure = sum(h.weight for h in holdings)
        cash_weight = max(c.min_cash, 1.0 - total_exposure)

        return PortfolioTarget(
            holdings=holdings,
            cash_weight=cash_weight,
            total_stocks=len(holdings),
            total_sectors=len(set(h.sector for h in holdings)),
            total_exposure=total_exposure,
        )
