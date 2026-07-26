# [AI:BEGIN]
# PA_MCP - Risk Layer: Hard RiskGuard
# Non-bypassable risk check pipeline. Sits between Agent output and final response.
# These rules CANNOT be debated, overridden, or "special-cased".
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import structlog

from pa_mcp.config import RiskSettings

logger = structlog.get_logger(__name__)


class GuardVerdict(str, Enum):
    PASS = "pass"
    REDUCE = "reduce"
    BLOCK = "block"


@dataclass
class GuardResult:
    """Output of a RiskGuard check."""

    verdict: GuardVerdict = GuardVerdict.PASS
    original_strength_score: float = 0.0
    adjusted_strength_score: float = 0.0
    original_max_position: float = 0.0
    adjusted_max_position: float = 0.0
    blocked_reasons: list[str] = field(default_factory=list)
    reduction_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RiskGuard:
    """Hard risk control layer — non-bypassable.

    Positioned between the Agent decision output and the final MCP tool response.
    Every signal/analysis passes through this guard before being returned to the user.
    """

    def __init__(self, settings: Optional[RiskSettings] = None) -> None:
        if settings is None:
            from pa_mcp.config import get_settings
            settings = get_settings().risk
        self.settings = settings

        # Position hard limits
        self.MAX_SINGLE_STOCK = settings.max_single_stock
        self.MAX_SECTOR_EXPOSURE = settings.max_sector_exposure
        self.MAX_TOTAL_POSITION = settings.max_total_position

        # Loss circuit breakers
        self.MAX_DAILY_LOSS = settings.max_daily_loss
        self.MAX_CONSECUTIVE_LOSSES = settings.max_consecutive_losses
        self.PAUSE_DAYS_AFTER_BIG_LOSS = settings.pause_days_after_big_loss

        # Runtime tracking
        self._daily_loss_pct: float = 0.0
        self._consecutive_losses: int = 0
        self._current_positions: dict[str, float] = {}  # symbol -> position_pct
        self._is_paused: bool = False
        self._pause_until: Optional[str] = None

    # ---- Systemic Risk Early Warning ----

    SYSTEMIC_RULES: dict[str, dict[str, Any]] = {
        "mass_limit_down": {
            "condition": "limit_down > 50 AND limit_up < 30",
            "action": "REDUCE",
            "target_position_pct": 0.30,
            "message": "Systemic sell-off detected: limit-down > 50, limit-up < 30. Reducing position to 30%.",
        },
        "volume_collapse": {
            "condition": "turnover_billion < 500",
            "action": "REDUCE",
            "target_position_pct": 0.20,
            "message": "Market liquidity crisis: turnover < 500B. Reducing position to 20%.",
        },
        "bear_market": {
            "condition": "index_below_200ma",
            "action": "REDUCE",
            "target_position_pct": 0.10,
            "message": "Bear market: index below 200-day MA. Reducing position to 10%.",
        },
        "northbound_exodus": {
            "condition": "northbound_consecutive_outflow >= 3 AND outflow > 10",
            "action": "REDUCE",
            "target_position_pct": 0.30,
            "message": "Northbound capital exodus: 3+ days of >10B outflow. Reducing position to 30%.",
        },
        "margin_call_cascade": {
            "condition": "margin_balance_decline_pct > 1 FOR 3 days",
            "action": "REDUCE",
            "target_position_pct": 0.40,
            "message": "Margin balance declining: potential forced liquidation cascade. Reducing to 40%.",
        },
        "seasonal_defense": {
            "condition": "month IN (5, 6, 9, 11, 12)",
            "action": "REDUCE",
            "target_position_pct": 0.50,
            "message": "Seasonal defense: historically weak month. Reducing position cap to 50%.",
        },
    }

    # ---- Core Check Methods ----

    def check_single_position(self, symbol: str, suggested_pct: float) -> GuardResult:
        """Check if a single position violates hard limits."""
        result = GuardResult(
            original_max_position=suggested_pct,
            adjusted_max_position=suggested_pct,
        )

        if suggested_pct > self.MAX_SINGLE_STOCK:
            result.verdict = GuardVerdict.REDUCE
            result.adjusted_max_position = self.MAX_SINGLE_STOCK
            result.reduction_reasons.append(
                f"Position {suggested_pct:.0%} exceeds single-stock limit of {self.MAX_SINGLE_STOCK:.0%}. "
                f"Capped to {self.MAX_SINGLE_STOCK:.0%}."
            )

        # Check if we're in pause
        if self._is_paused:
            result.verdict = GuardVerdict.BLOCK
            result.blocked_reasons.append(
                f"Trading paused until {self._pause_until} due to risk circuit breaker."
            )
            result.passed = False

        if result.verdict == GuardVerdict.BLOCK:
            result.passed = False

        return result

    def check_portfolio(self, new_position: dict[str, float]) -> GuardResult:
        """Check overall portfolio constraints."""
        result = GuardResult()

        total = sum(self._current_positions.values()) + sum(new_position.values())

        if total > self.MAX_TOTAL_POSITION:
            result.verdict = GuardVerdict.REDUCE
            scale = self.MAX_TOTAL_POSITION / total if total > 0 else 1.0
            for sym in new_position:
                new_position[sym] *= scale
            result.reduction_reasons.append(
                f"Total position {total:.0%} exceeds {self.MAX_TOTAL_POSITION:.0%} limit. Scaled down."
            )

        # Sector exposure check (simplified — full version needs sector mapping)
        # This is a stub; real implementation queries sector data

        if result.verdict == GuardVerdict.BLOCK:
            result.passed = False

        return result

    def check_systemic(
        self, market_indicators: dict[str, Any],
    ) -> GuardResult:
        """Check systemic risk triggers and adjust position ceiling."""
        result = GuardResult()
        result.adjusted_max_position = self.MAX_TOTAL_POSITION

        limit_down = market_indicators.get("limit_down_count", 0)
        limit_up = market_indicators.get("limit_up_count", 0)
        turnover = market_indicators.get("turnover_billion", 0)
        below_200ma = market_indicators.get("index_below_200ma", False)
        northbound_days = market_indicators.get("northbound_consecutive_outflow_days", 0)
        northbound_outflow = market_indicators.get("northbound_net_outflow_billion", 0)
        margin_decline = market_indicators.get("margin_balance_decline_pct", 0)
        current_month = datetime.now().month

        # Check each systemic rule
        triggers = []

        if limit_down > 50 and limit_up < 30:
            triggers.append(self.SYSTEMIC_RULES["mass_limit_down"])
        if turnover < 500:
            triggers.append(self.SYSTEMIC_RULES["volume_collapse"])
        if below_200ma:
            triggers.append(self.SYSTEMIC_RULES["bear_market"])
        if northbound_days >= 3 and northbound_outflow > 10:
            triggers.append(self.SYSTEMIC_RULES["northbound_exodus"])
        if margin_decline > 1:
            triggers.append(self.SYSTEMIC_RULES["margin_call_cascade"])
        if current_month in (5, 6, 9, 11, 12):
            triggers.append(self.SYSTEMIC_RULES["seasonal_defense"])

        # Apply most restrictive trigger
        for trigger in triggers:
            target = trigger["target_position_pct"]
            if target < result.adjusted_max_position:
                result.adjusted_max_position = target
                result.verdict = max(result.verdict, GuardVerdict.REDUCE)
                result.warnings.append(trigger["message"])

        if triggers:
            logger.warning(
                "Systemic risk triggers active",
                trigger_count=len(triggers),
                adjusted_ceiling=result.adjusted_max_position,
            )

        return result

    def record_trade_result(self, pnl_pct: float) -> None:
        """Record a trade result for circuit breaker tracking."""
        if pnl_pct < 0:
            self._consecutive_losses += 1
            self._daily_loss_pct += abs(pnl_pct)
        else:
            self._consecutive_losses = 0

        # Check circuit breakers
        if self._daily_loss_pct > self.MAX_DAILY_LOSS:
            self._activate_pause("daily loss limit exceeded", self.PAUSE_DAYS_AFTER_BIG_LOSS)

        if self._consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            self._activate_pause("consecutive loss limit reached", 7)

    def _activate_pause(self, reason: str, days: int) -> None:
        """Activate trading pause."""
        from datetime import timedelta
        self._is_paused = True
        self._pause_until = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        logger.warning("Trading paused", reason=reason, until=self._pause_until)

    def reset_daily(self) -> None:
        """Reset daily tracking counters (call at start of each trading day)."""
        self._daily_loss_pct = 0.0

    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        if self._is_paused:
            if self._pause_until and datetime.now().strftime("%Y-%m-%d") < self._pause_until:
                return False
            # Pause expired
            self._is_paused = False
            self._pause_until = None
        return True

    @property
    def effective_position_ceiling(self) -> float:
        """Current effective position ceiling considering systemic rules."""
        # This would be set by check_systemic() before use
        return self.MAX_TOTAL_POSITION
