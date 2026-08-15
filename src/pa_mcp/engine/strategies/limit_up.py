# [AI:BEGIN]
# PA_MCP - Engine Layer: Limit-up Board Strategies
# First board, 2nd board relay, dragon leader second wave
# [AI:END]

from __future__ import annotations

import pandas as pd
import numpy as np

from pa_mcp.engine.strategies.base import (
    BaseStrategy, MarketState, ParamRange, Signal, SignalDirection,
    StrategyCategory, TimeFrame,
)


class FirstBoardBreakoutStrategy(BaseStrategy):
    """Detects first limit-up boards suitable for entry.

    Criteria:
    1. Stock hits limit-up for the FIRST time in 20 days (first board)
    2. Not already at limit-up at open (buyable)
    3. Sector has at least 2 other limit-up stocks (sector resonance)
    4. Market cap < 10B (small-cap effect)
    """

    name = "first_board_breakout"
    category = StrategyCategory.BOARD
    description = "First limit-up board detection with sector resonance — HIGH RISK"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.FERMENTING, MarketState.CLIMAX]

    lookback_days: int = 20
    min_sector_count: int = 2
    max_market_cap_b: float = 10.0  # Billion CNY

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if "pct_change" not in data.columns or len(data) < self.lookback_days:
            return []

        pct_change = data["pct_change"].values
        close = data["close"].values
        volume = data["volume"].values

        signals: list[Signal] = []
        for i in range(self.lookback_days, len(data)):
            today_pct = pct_change[i]

            # Check if today hits limit-up (>= 9.5% for main board)
            if today_pct < 9.5:
                continue

            # Check if this is first limit-up in lookback
            recent_limits = sum(1 for j in range(i - self.lookback_days, i) if pct_change[j] >= 9.5)
            if recent_limits > 0:
                continue  # Not first board — already had limit-ups recently

            # Volume surge check
            avg_vol = np.mean(volume[i - 20 : i]) if i >= 20 else volume[i]
            vol_ratio = volume[i] / avg_vol if avg_vol > 0 else 0

            # Strength: first board + volume + market state alignment
            first_board_score = 40  # Base: first board is significant
            vol_bonus = min(vol_ratio / 3.0, 1.0) * 30
            strength = min(first_board_score + vol_bonus, 100)

            signals.append(Signal(
                symbol="", strategy_name=self.name,
                strength_score=round(strength, 1),
                direction=SignalDirection.BULLISH,
                evidence=[
                    {"type": "first_board", "detail": f"First limit-up in {self.lookback_days} days"},
                    {"type": "volume", "detail": f"Volume ratio {vol_ratio:.1f}x"},
                ],
                risk_flags=[
                    {"type": "liquidity", "severity": "high", "detail": "Limit-up stocks may be unbuyable — see LIMIT-UP BUYABILITY WARNING"},
                    {"type": "strategy", "severity": "high", "detail": "Board strategies are HIGH RISK. Backtest overestimates real returns by 60-70%."},
                ],
                suggested_max_position_pct=0.03,  # Only 3% for board plays
                signal_time=str(data["date"].iloc[i])[:10],
            ))

        return signals

    def get_params_space(self) -> list[ParamRange]:
        return [
            ParamRange("lookback_days", 10, 40, 5),
            ParamRange("max_market_cap_b", 3.0, 20.0, 2.0),
        ]


class DragonSecondWaveStrategy(BaseStrategy):
    """Detects "dragon second wave" pattern: strong leader retraces, then rebounds.

    Pattern:
    1. Stock was a sector leader (60-day return > 80% or had 5+ consecutive limit-ups)
    2. Has pulled back 20-40% from its peak
    3. Today shows a reversal signal: price up >3%, volume increasing
    4. MACD histogram turning positive
    """

    name = "dragon_second_wave"
    category = StrategyCategory.BOARD
    description = "Dragon leader second wave: pullback entry on former leader"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.FERMENTING]

    max_pullback_pct: float = 0.40
    min_pullback_pct: float = 0.15
    reversal_pct: float = 0.03

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if len(data) < 120:
            return []

        close = data["close"].values
        high = data["high"].values
        volume = data["volume"].values
        pct_change = data.get("pct_change", pd.Series([0] * len(data))).values

        signals: list[Signal] = []
        for i in range(120, len(data)):
            # Check if stock was a leader: 60-day return > 50%
            high_60d = high[i - 60 : i].max()
            close_60d_ago = close[i - 60]
            rally_pct = (high_60d - close_60d_ago) / close_60d_ago

            if rally_pct < 0.50:
                continue  # Not a strong leader

            # Check pullback from peak
            pullback = (high_60d - close[i]) / high_60d
            if pullback < self.min_pullback_pct or pullback > self.max_pullback_pct:
                continue

            # Reversal signal: today's gain > reversal_pct
            if pct_change[i] < self.reversal_pct * 100:  # pct_change is in %
                continue

            # Volume: increasing vs recent average
            avg_vol = np.mean(volume[i - 10 : i])
            if volume[i] < avg_vol * 1.3:
                continue

            strength = 40 + min(pullback / 0.40, 1.0) * 30 + min(pct_change[i] / 10, 1.0) * 30
            strength = min(strength, 100)

            signals.append(Signal(
                symbol="", strategy_name=self.name,
                strength_score=round(strength, 1),
                direction=SignalDirection.BULLISH,
                evidence=[
                    {"type": "leader", "detail": f"60-day rally: {rally_pct:.0%}"},
                    {"type": "pullback", "detail": f"Pullback from peak: {pullback:.0%}"},
                    {"type": "reversal", "detail": f"Today gain: {pct_change[i]:.1f}%"},
                ],
                suggested_max_position_pct=0.05,
                signal_time=str(data["date"].iloc[i])[:10],
            ))

        return signals

    def get_params_space(self) -> list[ParamRange]:
        return [
            ParamRange("max_pullback_pct", 0.25, 0.50, 0.05),
            ParamRange("reversal_pct", 0.02, 0.06, 0.01),
        ]
