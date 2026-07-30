# [AI:BEGIN]
# PA_MCP - Portfolio: Simple Optimizer
#
# Naive optimizer: equal-weight within constraints + commission-aware.
# Full mean-variance + shrinkage + transaction costs deferred to Phase E2.
# This is the MINIMUM VIABLE optimizer for 10万元 account.
# [AI:END]

from __future__ import annotations

import numpy as np


class Optimizer:
    """Simplified constrained optimizer for A-share long-only portfolios."""

    def __init__(self, max_single: float = 0.10, max_total: float = 0.80,
                 min_cash: float = 0.20, round_lot: int = 100) -> None:
        self.max_single = max_single
        self.max_total = max_total
        self.min_cash = min_cash
        self.round_lot = round_lot

    def integer_shares(self, weights: dict[str, float],
                       prices: dict[str, float],
                       total_nav: float) -> dict[str, int]:
        """Convert target weights to integer share counts (100-share lots)."""
        result: dict[str, int] = {}

        for sym, weight in weights.items():
            if weight <= 0 or sym not in prices or prices[sym] <= 0:
                continue
            target_value = total_nav * weight
            raw_shares = int(target_value / prices[sym])
            # Round down to nearest 100
            lots = (raw_shares // self.round_lot) * self.round_lot
            if lots > 0:
                result[sym] = lots

        return result

    def check_constraints(self, shares: dict[str, int],
                          prices: dict[str, float],
                          total_nav: float) -> dict[str, bool]:
        """Verify constraints after integer repair."""
        checks = {}

        # Total exposure
        total_exposure = sum(
            shares[s] * prices[s] for s in shares
        ) / total_nav if total_nav > 0 else 0
        checks["total_exposure"] = total_exposure <= self.max_total + 0.001

        # Single stock
        for sym, qty in shares.items():
            weight = qty * prices[sym] / total_nav if total_nav > 0 else 0
            checks[f"single_{sym}"] = weight <= self.max_single + 0.001

        # Cash
        cash = 1.0 - total_exposure
        checks["min_cash"] = cash >= self.min_cash - 0.001

        return checks

    def repair_constraints(self, shares: dict[str, int],
                           prices: dict[str, float],
                           total_nav: float) -> dict[str, int]:
        """Scale down shares to satisfy all constraints."""
        if not shares or total_nav <= 0:
            return {}

        # Check total exposure first
        total_value = sum(shares[s] * prices[s] for s in shares)
        total_weight = total_value / total_nav

        if total_weight > self.max_total:
            scale = self.max_total / total_weight
            result: dict[str, int] = {}
            for sym, qty in shares.items():
                new_qty = max(0, int(qty * scale) // self.round_lot * self.round_lot)
                if new_qty > 0:
                    result[sym] = new_qty
            return result

        return dict(shares)
