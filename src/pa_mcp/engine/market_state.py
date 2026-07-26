# [AI:BEGIN]
# PA_MCP - Engine Layer: Market State Detection
# Quantifies 5 market regimes using a scoring system with hysteresis-based transitions.
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from pa_mcp.engine.strategies.base import MarketState


@dataclass
class MarketIndicators:
    """Raw market indicators used for state detection."""

    limit_up_count: int = 0
    limit_down_count: int = 0
    up_count: int = 0
    down_count: int = 0
    total_stocks: int = 5000
    turnover_billion: float = 0.0  # Total market turnover in billions CNY
    margin_balance_change_pct: float = 0.0
    northbound_net_billion: float = 0.0  # Northbound net flow in billions
    breakout_rate: float = 0.0  # Limit-up break rate
    consecutive_up_height: int = 0  # Highest consecutive limit-up days
    index_ma_deviation: float = 0.0  # Index deviation from 200-day MA
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MarketStateDetector:
    """Detects current market regime from quantitative indicators.

    Uses sigmoid-based scoring functions for smooth membership,
    combined with hysteresis for stable state transitions.

    Each state has minimum duration constraints to prevent whipsawing.

    Valid transitions (hysteresis):
        CLIMAX -> FERMENTING -> STARTING -> DULL -> FROZEN
              ^________________________________________|
    """

    # Scoring thresholds
    LIMIT_UP_CLIMAX = 80
    LIMIT_UP_FERMENT = 40
    LIMIT_UP_START = 15
    TURNOVER_CLIMAX = 1500  # Billion CNY
    TURNOVER_FERMENT = 800
    TURNOVER_DULL = 400
    BREAKOUT_RATE_CLIMAX = 0.30
    NORTHBOUND_FERMENT = 5.0  # Billion CNY
    NORTHBOUND_DULL = -5.0
    INDEX_MA_CLIMAX = 0.15  # 15% above MA200
    INDEX_MA_FROZEN = -0.10  # 10% below MA200

    def __init__(self, min_state_duration_days: int = 3) -> None:
        self.min_state_duration = min_state_duration_days
        self._current_state: Optional[MarketState] = None
        self._state_start_date: Optional[str] = None
        self._days_in_state: int = 0

    @property
    def current_state(self) -> Optional[MarketState]:
        return self._current_state

    def detect(self, indicators: MarketIndicators, today: Optional[str] = None) -> MarketState:
        """Detect market state from indicators, with hysteresis."""
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")

        # Calculate raw scores for each state
        scores = self._score_states(indicators)

        # Determine raw state (highest score)
        raw_state = max(scores, key=scores.get)

        # Apply hysteresis
        new_state = self._apply_hysteresis(raw_state, today)

        # Update tracking
        if new_state != self._current_state:
            self._current_state = new_state
            self._state_start_date = today
            self._days_in_state = 1
        else:
            self._days_in_state += 1

        return self._current_state

    def _score_states(self, ind: MarketIndicators) -> dict[MarketState, float]:
        """Score each market state based on indicator values."""
        breadth = (ind.up_count - ind.down_count) / max(ind.total_stocks, 1)

        return {
            MarketState.CLIMAX: (
                self._sigmoid(ind.limit_up_count, self.LIMIT_UP_CLIMAX, 20) * 0.3
                + self._sigmoid(ind.turnover_billion, self.TURNOVER_CLIMAX, 300) * 0.2
                + self._sigmoid(ind.breakout_rate, self.BREAKOUT_RATE_CLIMAX, 0.1) * 0.2
                + self._sigmoid(ind.index_ma_deviation, self.INDEX_MA_CLIMAX, 0.05) * 0.2
                + self._sigmoid(ind.consecutive_up_height, 6, 2) * 0.1
            ),
            MarketState.FERMENTING: (
                self._gaussian(ind.limit_up_count, self.LIMIT_UP_FERMENT, 30) * 0.3
                + self._gaussian(ind.turnover_billion, self.TURNOVER_FERMENT, 300) * 0.25
                + self._sigmoid(ind.northbound_net_billion, self.NORTHBOUND_FERMENT, 3) * 0.25
                + self._gaussian(breadth, 0.1, 0.15) * 0.2
            ),
            MarketState.STARTING: (
                self._gaussian(ind.limit_up_count, self.LIMIT_UP_START, 15) * 0.3
                + self._gaussian(ind.turnover_billion, self.TURNOVER_DULL, 200) * 0.25
                + (1 - max(0, ind.northbound_net_billion / 10)) * 0.15
                + (ind.up_count / max(ind.total_stocks, 1)) * 0.3
            ),
            MarketState.DULL: (
                self._gaussian(ind.turnover_billion, self.TURNOVER_DULL, 150) * 0.4
                + (1 - abs(breadth)) * 0.3
                + self._sigmoid(-ind.northbound_net_billion, 3, 2) * 0.3
            ),
            MarketState.FROZEN: (
                (1 - self._sigmoid(ind.turnover_billion, self.TURNOVER_DULL, 100)) * 0.3
                + self._sigmoid(ind.limit_down_count, 30, 15) * 0.3
                + self._sigmoid(-ind.index_ma_deviation, -self.INDEX_MA_FROZEN, 0.03) * 0.2
                + (ind.down_count / max(ind.total_stocks, 1)) * 0.2
            ),
        }

    def _apply_hysteresis(self, raw_state: MarketState, today: str) -> MarketState:
        """Apply hysteresis: only switch state if minimum duration met.

        Transitions require the current state to have been held for
        min_state_duration days before a change is allowed.
        """
        if self._current_state is None:
            return raw_state

        if raw_state == self._current_state:
            return self._current_state

        # Allow transition only after minimum duration
        if self._days_in_state < self.min_state_duration:
            return self._current_state

        # Allow transition — but only to valid adjacent states
        valid_transitions = self._get_valid_transitions(self._current_state)
        if raw_state in valid_transitions:
            return raw_state

        return self._current_state

    @staticmethod
    def _get_valid_transitions(state: MarketState) -> set[MarketState]:
        """Get valid target states from current state."""
        transitions = {
            MarketState.CLIMAX: {MarketState.CLIMAX, MarketState.FERMENTING},
            MarketState.FERMENTING: {MarketState.FERMENTING, MarketState.CLIMAX, MarketState.STARTING, MarketState.DULL},
            MarketState.STARTING: {MarketState.STARTING, MarketState.FERMENTING, MarketState.DULL},
            MarketState.DULL: {MarketState.DULL, MarketState.STARTING, MarketState.FROZEN},
            MarketState.FROZEN: {MarketState.FROZEN, MarketState.DULL},
        }
        return transitions.get(state, set(MarketState))

    @staticmethod
    def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
        """Sigmoid function scaled to [0, 1]."""
        import math
        try:
            return 1.0 / (1.0 + math.exp(-(x - midpoint) / steepness))
        except OverflowError:
            return 1.0 if x > midpoint else 0.0

    @staticmethod
    def _gaussian(x: float, center: float, sigma: float) -> float:
        """Gaussian (bell curve) function scaled to [0, 1]."""
        import math
        try:
            return math.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        except OverflowError:
            return 0.0

    def get_strategy_mapping(self) -> dict[MarketState, list[str]]:
        """Map market states to recommended strategy categories."""
        return {
            MarketState.CLIMAX: ["trend", "momentum"],           # Ride the wave
            MarketState.FERMENTING: ["trend", "swing", "board"], # Active trading
            MarketState.STARTING: ["swing", "reversal", "value"],# Cautious entry
            MarketState.DULL: ["value", "grid"],                 # Defensive
            MarketState.FROZEN: [],                              # Stay out
        }
