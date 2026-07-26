# [AI:BEGIN]
# PA_MCP - Engine Layer: Remaining Strategy Categories
# Swing, value, grid, reversal, momentum, event-driven strategies
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd

from pa_mcp.engine.strategies.base import (
    BaseStrategy, MarketState, ParamRange, Signal, SignalDirection,
    StrategyCategory, TimeFrame,
)


class MACDDivergenceSwingStrategy(BaseStrategy):
    """Swing trading: MACD divergence + volume confirmation.

    Detects bullish divergence: price makes lower low, but MACD makes higher low.
    """

    name = "macd_divergence_swing"
    category = StrategyCategory.SWING
    description = "MACD bullish divergence with volume confirmation"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.FERMENTING, MarketState.STARTING, MarketState.DULL]

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if len(data) < 40:
            return []

        close = data["close"].values

        # Compute MACD manually
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        hist = macd - signal_line

        signals: list[Signal] = []
        for i in range(40, len(data)):
            # Bullish divergence: price lower low, MACD higher low
            if close[i] >= close[i - 1]:
                continue  # Not a lower low

            # Find recent lows
            recent_close = close[i - 20 : i + 1]
            recent_hist = hist[i - 20 : i + 1]
            price_low_idx = np.argmin(recent_close)
            hist_low_idx = np.argmin(recent_hist)

            if price_low_idx != len(recent_close) - 1:
                continue  # Current is not the price low

            # Check if MACD histogram is rising while price is falling
            if hist[i] <= recent_hist[price_low_idx - 5] if price_low_idx >= 5 else hist[i - 5]:
                continue  # No divergence

            # MACD turning positive or histogram increasing
            if hist[i] <= hist[i - 1]:
                continue

            strength = 30 + min((hist[i] - hist[i - 5]) / max(abs(hist[i - 5]), 0.001) / 3, 1) * 30
            strength = min(strength + 20, 100)  # Base + divergence bonus

            signals.append(Signal(
                symbol="", strategy_name=self.name,
                strength_score=round(strength, 1),
                direction=SignalDirection.BULLISH,
                evidence=[
                    {"type": "divergence", "detail": "Price lower low, MACD higher low"},
                    {"type": "macd", "detail": f"MACD hist turning positive at {hist[i]:.4f}"},
                ],
                suggested_max_position_pct=0.05,
            ))

        return signals


class BollingerMeanReversionSwingStrategy(BaseStrategy):
    """Bollinger Band mean reversion: buy near lower band, sell near upper band."""

    name = "bollinger_mean_reversion"
    category = StrategyCategory.SWING
    description = "Bollinger Band mean reversion — buy at lower band, sell at middle"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.DULL, MarketState.STARTING, MarketState.FERMENTING]

    bb_period: int = 20
    bb_std: float = 2.0

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if len(data) < self.bb_period + 5:
            return []

        close = data["close"].values
        volume = data["volume"].values

        # Compute Bollinger Bands
        ma = pd.Series(close).rolling(self.bb_period).mean().values
        std = pd.Series(close).rolling(self.bb_period).std().values
        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std

        # Volume average
        avg_vol = pd.Series(volume).rolling(20).mean().values

        signals: list[Signal] = []
        for i in range(self.bb_period + 5, len(data)):
            # Price near or below lower band
            if close[i] > lower[i] * 1.02:
                continue

            # Bullish reversal: today's close > open (or > yesterday's close)
            if close[i] <= close[i - 1]:
                continue

            # Volume above average (confirmation)
            vol_ratio = volume[i] / avg_vol[i] if avg_vol[i] > 0 else 0
            if vol_ratio < 0.8:
                continue

            # How far below the band?
            band_distance = (lower[i] - close[i]) / lower[i]
            band_score = min(max(band_distance / 0.05, 0), 1) * 40  # Up to 40 from oversold

            strength = min(40 + band_score + min(vol_ratio, 1.0) * 20, 100)
            signals.append(Signal(
                symbol="", strategy_name=self.name,
                strength_score=round(strength, 1),
                direction=SignalDirection.BULLISH,
                evidence=[
                    {"type": "bollinger", "detail": f"Close {close[i]:.2f} near lower band {lower[i]:.2f}"},
                    {"type": "reversal", "detail": f"Bullish reversal candle detected"},
                ],
                suggested_max_position_pct=0.05,
            ))

        return signals


class ROEPBValueStrategy(BaseStrategy):
    """Value strategy: low PB + high ROE + reasonable PE."""

    name = "roe_pb_value"
    category = StrategyCategory.VALUE
    description = "Low PB, high ROE, reasonable PE — classic value screen"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.DULL, MarketState.STARTING, MarketState.FROZEN]

    max_pe: float = 25.0
    max_pb: float = 2.0
    min_roe: float = 0.10  # 10%

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if data.empty:
            return []

        signals: list[Signal] = []

        # This strategy requires fundamental data (PE, PB, ROE) in the DataFrame
        pe = data.get("pe", None)
        pb = data.get("pb", None)
        roe = data.get("roe", None)

        if pe is None or pb is None or roe is None:
            return []

        for i, row in data.iterrows():
            pe_val = float(row.get("pe", 999))
            pb_val = float(row.get("pb", 999))
            roe_val = float(row.get("roe", 0))

            if pe_val > self.max_pe or pe_val <= 0:
                continue
            if pb_val > self.max_pb or pb_val <= 0:
                continue
            if roe_val < self.min_roe:
                continue

            # Score: undervaluation + quality
            pe_score = max(0, (self.max_pe - pe_val) / self.max_pe) * 30
            pb_score = max(0, (self.max_pb - pb_val) / self.max_pb) * 30
            roe_score = min(roe_val / 0.20, 1.0) * 40  # 20% ROE = max score

            strength = min(pe_score + pb_score + roe_score, 100)
            if strength > 40:
                signals.append(Signal(
                    symbol=str(row.get("symbol", "")),
                    strategy_name=self.name,
                    strength_score=round(strength, 1),
                    direction=SignalDirection.BULLISH,
                    evidence=[
                        {"type": "valuation", "detail": f"PE={pe_val:.1f}, PB={pb_val:.2f}"},
                        {"type": "quality", "detail": f"ROE={roe_val:.1%}"},
                    ],
                    suggested_max_position_pct=0.08,  # Value positions can be larger
                ))

        return signals


class PriceRangeGridStrategy(BaseStrategy):
    """Grid trading: buy at grid lows, sell at grid highs.

    Uses ATR-based grid spacing for dynamic adjustment.
    """

    name = "range_grid"
    category = StrategyCategory.GRID
    description = "Price range grid trading with ATR-based spacing"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.DULL]  # ONLY in sideways markets!

    grid_levels: int = 5
    atr_multiplier: float = 2.0

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if len(data) < 50:
            return []

        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        # Compute ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                abs(high[1:] - close[:-1]),
                abs(low[1:] - close[:-1]),
            ),
        )
        atr = pd.Series(np.concatenate([[0], tr])).rolling(14).mean().values

        # Define grid range based on last 50 days
        range_high = high[-50:].max()
        range_low = low[-50:].min()
        grid_step = atr[-1] * self.atr_multiplier

        signals: list[Signal] = []
        price = close[-1]

        # Find nearest grid level below current price
        current_level = None
        for level in range(self.grid_levels):
            grid_price = range_low + level * grid_step
            if grid_price <= price:
                current_level = level

        if current_level is None or current_level >= self.grid_levels - 1:
            return []  # At top of grid — no buy

        # Check if price is near a grid line (within 0.5 ATR)
        nearest_grid = range_low + current_level * grid_step
        distance = (price - nearest_grid) / grid_step if grid_step > 0 else 999

        if distance > 0.3:
            return []  # Not close enough to grid line

        strength = max(0, min(60 - current_level * 10, 100))  # Lower grid = stronger
        signals.append(Signal(
            symbol="", strategy_name=self.name,
            strength_score=round(strength, 1),
            direction=SignalDirection.BULLISH,
            evidence=[
                {"type": "grid", "detail": f"Price {price:.2f} near grid line {nearest_grid:.2f} (level {current_level})"},
                {"type": "risk", "detail": "GRID TRADING WARNING: Can blow up in trending markets. Only use in confirmed range-bound conditions."},
            ],
            risk_flags=[
                {"type": "trend_risk", "severity": "high", "detail": "Grid strategies lose heavily in trending markets"},
            ],
            suggested_max_position_pct=0.03,  # Small for grid
        ))

        return signals


class VolumePriceBreakoutMomentumStrategy(BaseStrategy):
    """Momentum: high volume + price breakout with continuation."""

    name = "volume_price_momentum"
    category = StrategyCategory.MOMENTUM
    description = "Volume-price breakout momentum — strongest 20-day performers with volume confirmation"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.FERMENTING, MarketState.CLIMAX]

    momentum_period: int = 20
    top_pct: float = 0.10  # Top 10% performers

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if len(data) < self.momentum_period + 5:
            return []

        close = data["close"].values
        volume = data["volume"].values

        # 20-day momentum
        momentum = (close[-1] - close[-self.momentum_period]) / close[-self.momentum_period]

        # Volume confirmation: increasing
        recent_vol = volume[-10:].mean()
        older_vol = volume[-20:-10].mean()
        vol_trend = recent_vol / older_vol if older_vol > 0 else 0

        if momentum <= 0.05:
            return []  # Need at least 5% momentum

        if vol_trend < 1.0:
            return []  # Volume contracting, not confirming

        strength = min(40 + momentum * 100 + min(vol_trend, 2.0) * 15, 100)
        return [Signal(
            symbol="", strategy_name=self.name,
            strength_score=round(strength, 1),
            direction=SignalDirection.BULLISH,
            evidence=[
                {"type": "momentum", "detail": f"{self.momentum_period}-day momentum: {momentum:.1%}"},
                {"type": "volume_trend", "detail": f"Volume expanding {vol_trend:.1f}x"},
            ],
            suggested_max_position_pct=0.05,
        )]
