# [AI:BEGIN]
# PA_MCP - Risk Layer: Hard RiskGuard (Refactored Phase E)
#
# Dual API:
#   - New (typed): PortfolioSnapshot + CandidateOrder → RiskDecisionResult
#   - Legacy (backward compat): check_single_position(sym, wt) → GuardResult
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4


# ---- Enums ----

class GuardVerdict(str, Enum):
    PASS = "pass"
    REDUCE = "reduce"
    BLOCK = "block"


class RiskDecision(str, Enum):
    APPROVE = "approve"
    ADJUST = "adjust"
    REJECT = "reject"


class DrawdownLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    DE_RISK = "de_risk"
    HARD_STOP = "hard_stop"


# ---- Legacy Result (backward compat) ----

@dataclass
class GuardResult:
    verdict: GuardVerdict = GuardVerdict.PASS
    adjusted_max_position: float = 0.0
    passed: bool = True
    decision: Optional[RiskDecision] = None
    decision_id: str = ""
    reason: str = ""
    drawdown_level: DrawdownLevel = DrawdownLevel.NORMAL
    adjusted_quantity: int = 0


# ---- New Typed Result ----

@dataclass
class RiskDecisionResult:
    decision_id: str = field(default_factory=lambda: uuid4().hex[:12])
    decision: RiskDecision = RiskDecision.APPROVE
    adjusted_quantity: int = 0
    reason: str = ""
    drawdown_level: DrawdownLevel = DrawdownLevel.NORMAL


# ---- Domain objects (new API) ----

@dataclass
class PortfolioSnapshot:
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> weight
    nav: float = 100000.0
    peak_nav: float = 100000.0
    drawdown_pct: float = 0.0


@dataclass
class CandidateOrder:
    symbol: str = ""
    side: str = "buy"
    quantity: int = 0
    price: float = 0.0
    weight_pct: float = 0.0
    sector: str = ""


@dataclass
class RiskPolicy:
    max_single_stock: float = 0.20
    max_sector_exposure: float = 0.40
    max_total_position: float = 0.80
    max_daily_loss: float = 0.03
    max_consecutive_losses: int = 3
    pause_days_after_big_loss: int = 3
    drawdown_warning: float = 0.10
    drawdown_de_risk: float = 0.12
    drawdown_hard_stop: float = 0.15


class RiskGuard:
    """Risk guard with both new typed API and legacy backward-compat API."""

    # Legacy systemic rules (moved to class attribute for backward compat)
    SYSTEMIC_RULES: dict[str, dict] = {
        "mass_limit_down": {"limit_down_count": 50, "action": "reduce_exposure", "target": 0.10},
        "bear_market": {"index_ma200_position": "below", "action": "reduce_exposure", "target": 0.20},
        "volume_collapse": {"turnover_billion": 400, "action": "reduce_exposure", "target": 0.15},
        "northbound_heavy_sell": {"consecutive_days": 5, "action": "reduce_exposure", "target": 0.25},
        "margin_call_risk": {"margin_balance_change": -0.10, "action": "reduce_exposure", "target": 0.10},
        "seasonal_defense": {"months": [5, 6, 9, 11, 12], "action": "reduce_exposure", "target": 0.50},
    }

    def __init__(self, policy: Optional[RiskPolicy] = None) -> None:
        self.policy = policy or RiskPolicy()
        self._is_paused: bool = False
        self._daily_loss_pct: float = 0.0
        self._consecutive_losses: int = 0
        self._current_positions: dict[str, float] = {}

    @property
    def is_trading_allowed(self) -> bool:
        return not self._is_paused

    def reset_daily(self) -> None:
        self._daily_loss_pct = 0.0

    def record_trade_result(self, pnl_pct: float) -> None:
        if pnl_pct < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        if self._consecutive_losses >= self.policy.max_consecutive_losses:
            self._is_paused = True

    def get_drawdown_level(self, dd_pct: float) -> DrawdownLevel:
        p = self.policy
        dd = abs(dd_pct)
        if dd >= p.drawdown_hard_stop:
            return DrawdownLevel.HARD_STOP
        elif dd >= p.drawdown_de_risk:
            return DrawdownLevel.DE_RISK
        elif dd >= p.drawdown_warning:
            return DrawdownLevel.WARNING
        return DrawdownLevel.NORMAL

    # ---- Legacy API (backward compat with existing tests) ----

    def check_single_position(self, symbol: str, weight: float) -> GuardResult:
        """Legacy: check if a single position weight is within limits."""
        g = GuardResult()

        if self._is_paused:
            g.verdict = GuardVerdict.BLOCK
            g.passed = False
            return g

        if weight > self.policy.max_single_stock:
            g.verdict = GuardVerdict.REDUCE
            g.passed = True  # REDUCE means "allowed but scaled" in legacy
            g.adjusted_max_position = self.policy.max_single_stock
        else:
            g.verdict = GuardVerdict.PASS
            g.passed = True
            g.adjusted_max_position = weight

        return g

    def check_portfolio(self, new_positions: dict[str, float]) -> GuardResult:
        """Legacy: check if adding positions respects total cap."""
        g = GuardResult()
        current_total = sum(self._current_positions.values())
        new_total = sum(new_positions.values())
        combined = current_total + new_total

        if combined > self.policy.max_total_position:
            g.verdict = GuardVerdict.REDUCE
            g.passed = True
            # Scale each new position proportionally
            scale = self.policy.max_total_position / combined if combined > 0 else 1.0
            for k in new_positions:
                new_positions[k] = new_positions[k] * scale
        else:
            g.verdict = GuardVerdict.PASS
            g.passed = True

        return g

    # ---- New typed API (for Phase C-E code) ----

    def check_single_order(
        self, snapshot: PortfolioSnapshot, order: CandidateOrder,
    ) -> RiskDecisionResult:
        """New: fully constrained single-order risk check."""
        p = self.policy
        dd_level = self.get_drawdown_level(snapshot.drawdown_pct)

        if self._is_paused or not self.is_trading_allowed:
            return RiskDecisionResult(decision=RiskDecision.REJECT,
                                      reason="Trading paused", drawdown_level=dd_level)

        if dd_level == DrawdownLevel.HARD_STOP and order.side == "buy":
            return RiskDecisionResult(decision=RiskDecision.REJECT,
                                      reason=f"HARD_STOP {dd_level.value}", drawdown_level=dd_level)

        # Gather all constraint scales and pick the most restrictive
        scales: list[float] = []
        reasons: list[str] = []

        current_wt = snapshot.positions.get(order.symbol, 0)
        new_wt = current_wt + order.weight_pct
        if new_wt > p.max_single_stock and order.side == "buy":
            max_add = p.max_single_stock - current_wt
            scales.append(max_add / order.weight_pct if order.weight_pct > 0 else 0)
            reasons.append(f"Single stock cap: {order.symbol}")

        total = sum(snapshot.positions.values())
        available = p.max_total_position - total
        if order.side == "buy" and order.weight_pct > available:
            scales.append(available / order.weight_pct if order.weight_pct > 0 else 0)
            reasons.append(f"Total cap: {total:.1%}+{order.weight_pct:.1%}>{p.max_total_position:.0%}")

        if not scales:
            return RiskDecisionResult(decision=RiskDecision.APPROVE, drawdown_level=dd_level)

        min_scale = min(scales)
        if min_scale <= 0:
            return RiskDecisionResult(decision=RiskDecision.REJECT,
                                      reason="; ".join(reasons), drawdown_level=dd_level)

        return RiskDecisionResult(
            decision=RiskDecision.ADJUST,
            adjusted_quantity=int(order.quantity * min_scale),
            reason="; ".join(reasons),
            drawdown_level=dd_level,
        )

    def check_batch(
        self, snapshot: PortfolioSnapshot, orders: list[CandidateOrder],
    ) -> list[RiskDecisionResult]:
        """New: batch order check with running allocation."""
        results: list[RiskDecisionResult] = []
        available = self.policy.max_total_position - sum(snapshot.positions.values())
        allocated = 0.0

        for order in orders:
            r = self.check_single_order(snapshot, order)
            if r.decision != RiskDecision.REJECT and order.side == "buy":
                if allocated + order.weight_pct > available:
                    remain = max(0, available - allocated)
                    r = RiskDecisionResult(
                        decision=RiskDecision.ADJUST if remain > 0 else RiskDecision.REJECT,
                        adjusted_quantity=int(order.quantity * remain / order.weight_pct) if order.weight_pct > 0 and remain > 0 else 0,
                        reason="Batch total cap",
                        drawdown_level=r.drawdown_level,
                    )
                allocated += order.weight_pct
            results.append(r)
        return results
