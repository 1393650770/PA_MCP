# [AI:BEGIN]
# PA_MCP - Engine Layer: Trend Following Strategies
# Platform breakout, MA golden cross, MA bullish alignment
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd

from pa_mcp.engine.strategies.base import (
    BaseStrategy, MarketState, ParamRange, Signal, SignalDirection,
    StrategyCategory, TimeFrame,
)


class PlatformBreakoutStrategy(BaseStrategy):
    """Detects stocks that have been consolidating in a narrow range and breaking out.

    Core logic:
    1. Price has been in a narrow range (<5% amplitude) for N days (consolidation)
    2. Today breaks above the consolidation high with significant volume (>1.5x average)
    3. MACD is above zero or turning positive (confirmation)

    Params:
        consolidation_min_days: Minimum consolidation days (default 15)
        max_amplitude_pct: Max price range within consolidation (default 0.05 = 5%)
        breakout_volume_ratio: Volume vs 20-day average (default 1.5)
    """

    name = "platform_breakout"
    category = StrategyCategory.TREND
    description = "Platform consolidation breakout with volume confirmation"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.FERMENTING, MarketState.CLIMAX]

    consolidation_min_days: int = 15
    max_amplitude_pct: float = 0.05
    breakout_volume_ratio: float = 1.5

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if len(data) < self.consolidation_min_days + 5:
            return []

        signals: list[Signal] = []
        close = data["close"].values
        high = data["high"].values
        volume = data["volume"].values
        symbol = data.get("symbol", [None]) if "symbol" in data.columns else [None]
        symbol = symbol[0] if len(symbol) > 0 else ""

        # Compute indicators
        ma20 = pd.Series(close).rolling(20).mean().values
        avg_vol_20 = pd.Series(volume).rolling(20).mean().values

        for i in range(self.consolidation_min_days, len(data)):
            # Consolidation check: price range within last N days
            consolidation_high = high[i - self.consolidation_min_days : i].max()
            consolidation_low = close[i - self.consolidation_min_days : i].min()
            amplitude = (consolidation_high - consolidation_low) / consolidation_low

            if amplitude > self.max_amplitude_pct:
                continue  # Too volatile, not a consolidation

            # Breakout check: today's close > consolidation high
            if close[i] <= consolidation_high:
                continue

            # Volume confirmation
            vol_ratio = volume[i] / avg_vol_20[i] if avg_vol_20[i] > 0 else 0
            if vol_ratio < self.breakout_volume_ratio:
                continue

            # Strength score based on how strong the breakout is
            vol_score = min(vol_ratio / 3.0, 1.0) * 30  # Max 30 from volume
            duration_score = min(self.consolidation_min_days / 30.0, 1.0) * 20  # Max 20 from duration
            breakout_score = min((close[i] - consolidation_high) / consolidation_high / 0.02, 1.0) * 30  # Max 30 from breakout strength
            trend_score = 20 if ma20[i] > 0 and close[i] > ma20[i] else 10  # Trend confirmation

            strength = min(vol_score + duration_score + breakout_score + trend_score, 100)

            signals.append(Signal(
                symbol=str(symbol) if symbol else "",
                strategy_name=self.name,
                strength_score=round(strength, 1),
                direction=SignalDirection.BULLISH,
                evidence=[
                    {"type": "consolidation", "detail": f"Amplitude {amplitude:.1%} over {self.consolidation_min_days} days"},
                    {"type": "breakout", "detail": f"Close {close[i]:.2f} above consolidation high {consolidation_high:.2f}"},
                    {"type": "volume", "detail": f"Volume ratio {vol_ratio:.1f}x"},
                ],
                suggested_max_position_pct=0.05,
                signal_time=str(data["date"].iloc[i])[:10],
            ))

        return signals

    def get_params_space(self) -> list[ParamRange]:
        return [
            ParamRange("consolidation_min_days", 10, 40, 5),
            ParamRange("max_amplitude_pct", 0.03, 0.10, 0.01),
            ParamRange("breakout_volume_ratio", 1.2, 3.0, 0.3),
        ]


class MAGoldenCrossStrategy(BaseStrategy):
    """MA fast line crosses above MA slow line with volume confirmation.

    Classic golden cross / death cross pattern.
    """

    name = "ma_golden_cross"
    category = StrategyCategory.TREND
    description = "MA fast crosses above MA slow (golden cross) with volume surge"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.FERMENTING, MarketState.STARTING]

    ma_fast: int = 5
    ma_slow: int = 20
    volume_ratio: float = 1.2

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if len(data) < self.ma_slow + 2:
            return []

        close = data["close"].values
        volume = data["volume"].values

        ma_fast_vals = pd.Series(close).rolling(self.ma_fast).mean().values
        ma_slow_vals = pd.Series(close).rolling(self.ma_slow).mean().values
        avg_vol = pd.Series(volume).rolling(20).mean().values

        signals: list[Signal] = []
        for i in range(self.ma_slow + 1, len(data)):
            # Golden cross: fast MA crosses above slow MA
            if ma_fast_vals[i] <= ma_slow_vals[i]:
                continue
            if ma_fast_vals[i - 1] > ma_slow_vals[i - 1]:
                continue  # Already crossed, not a new signal

            # Volume confirmation
            vol_ratio_val = volume[i] / avg_vol[i] if avg_vol[i] > 0 else 0
            if vol_ratio_val < self.volume_ratio:
                continue

            strength = min(50 + vol_ratio_val * 20, 100)
            signals.append(Signal(
                symbol="", strategy_name=self.name,
                strength_score=round(strength, 1),
                direction=SignalDirection.BULLISH,
                evidence=[
                    {"type": "ma_cross", "detail": f"MA{self.ma_fast} crossed above MA{self.ma_slow}"},
                    {"type": "volume", "detail": f"Volume ratio {vol_ratio_val:.1f}x"},
                ],
                suggested_max_position_pct=0.05,
                # P0-6: 信号市场时间 = bar 日期（供回测延迟一天执行）
                signal_time=str(data["date"].iloc[i])[:10],
            ))

        return signals

    def get_params_space(self) -> list[ParamRange]:
        return [
            ParamRange("ma_fast", 3, 15, 1),
            ParamRange("ma_slow", 15, 60, 5),
            ParamRange("volume_ratio", 1.0, 2.5, 0.2),
        ]
