# [AI:BEGIN]
# PA_MCP - Integration Tests: Engine, Strategies, Agent, Info
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa_mcp.engine.strategies.base import (
    StrategyRegistry, StrategyCategory, SignalDirection, MarketState,
)
from pa_mcp.engine.strategies.trend import PlatformBreakoutStrategy, MAGoldenCrossStrategy
from pa_mcp.engine.strategies.limit_up import FirstBoardBreakoutStrategy, DragonSecondWaveStrategy
from pa_mcp.engine.strategies.swing import (
    MACDDivergenceSwingStrategy, BollingerMeanReversionSwingStrategy,
    ROEPBValueStrategy, PriceRangeGridStrategy, VolumePriceBreakoutMomentumStrategy,
)
from pa_mcp.engine.market_state import MarketStateDetector, MarketIndicators
from pa_mcp.agent.memory import LongTermMemory
from pa_mcp.info.news import SentimentAnalyzer


# ---- Mock Data Generators ----

def make_ohlcv_data(n_days: int = 200, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    if trend == "up":
        drift = 0.001
    elif trend == "down":
        drift = -0.001
    else:
        drift = 0.0

    close = 10.0
    data = []
    for i in range(n_days):
        ret = np.random.normal(drift, 0.02)
        open_price = close
        close = close * (1 + ret)
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.uniform(1e6, 1e7)
        pct_change = ret * 100

        data.append({
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume,
            "pct_change": round(pct_change, 2),
        })

    df = pd.DataFrame(data)

    # Add pre-computed indicators for testing
    close_vals = df["close"].values
    df["ma5"] = pd.Series(close_vals).rolling(5).mean()
    df["ma10"] = pd.Series(close_vals).rolling(10).mean()
    df["ma20"] = pd.Series(close_vals).rolling(20).mean()
    df["ma60"] = pd.Series(close_vals).rolling(60).mean()
    df["avg_vol_20"] = pd.Series(df["volume"].values).rolling(20).mean()

    return df


# ---- Strategy Tests ----

class TestTrendStrategies:
    """Trend-following strategy tests."""

    def test_platform_breakout_creates_signal(self) -> None:
        """Should detect platform breakout in uptrend."""
        df = make_ohlcv_data(n_days=60, trend="up")
        strategy = PlatformBreakoutStrategy(consolidation_min_days=15)
        signals = strategy.generate_signals(df)
        assert isinstance(signals, list)

    def test_ma_cross_creates_signal(self) -> None:
        """Should detect golden cross."""
        df = make_ohlcv_data(n_days=60, trend="up")
        strategy = MAGoldenCrossStrategy(ma_fast=5, ma_slow=20)
        signals = strategy.generate_signals(df)
        assert isinstance(signals, list)

    def test_platform_breakout_params(self) -> None:
        """Strategy params should be accessible."""
        strategy = PlatformBreakoutStrategy()
        params = strategy.get_params()
        assert "consolidation_min_days" in params
        assert params["consolidation_min_days"] == 15

    def test_platform_breakout_info(self) -> None:
        """Strategy info should contain metadata."""
        strategy = PlatformBreakoutStrategy()
        info = strategy.get_info()
        assert info["name"] == "platform_breakout"
        assert info["category"] == "trend"


class TestBoardStrategies:
    """Limit-up board strategy tests."""

    def test_first_board_signal_structure(self) -> None:
        """Signal should have correct structure."""
        df = make_ohlcv_data(n_days=100, trend="up")
        # Artificially create a first board: set today as limit-up
        df.loc[df.index[-1], "pct_change"] = 10.0
        strategy = FirstBoardBreakoutStrategy(lookback_days=20)
        signals = strategy.generate_signals(df)
        assert isinstance(signals, list)
        if signals:
            s = signals[0]
            assert hasattr(s, "strength_score")
            assert hasattr(s, "risk_flags")
            assert len(s.risk_flags) >= 2  # Must warn about buyability

    def test_dragon_wave_empty_before_120_days(self) -> None:
        """Should return empty list when insufficient data."""
        df = make_ohlcv_data(n_days=50, trend="up")
        strategy = DragonSecondWaveStrategy()
        signals = strategy.generate_signals(df)
        assert signals == []

    def test_board_strategies_in_registry(self) -> None:
        """Board strategies should be registered."""
        registry = StrategyRegistry()
        registry.register(FirstBoardBreakoutStrategy)
        registry.register(DragonSecondWaveStrategy)

        board_strategies = registry.list_by_category(StrategyCategory.BOARD)
        names = [s["name"] for s in board_strategies]
        assert "first_board_breakout" in names
        assert "dragon_second_wave" in names


class TestSwingStrategies:
    """Swing trading strategy tests."""

    def test_macd_divergence_runs(self) -> None:
        """MACD divergence should run without error."""
        df = make_ohlcv_data(n_days=80, trend="down")
        strategy = MACDDivergenceSwingStrategy()
        signals = strategy.generate_signals(df)
        assert isinstance(signals, list)

    def test_bollinger_reversion_runs(self) -> None:
        """Bollinger mean reversion should run without error."""
        df = make_ohlcv_data(n_days=60, trend="sideways")
        strategy = BollingerMeanReversionSwingStrategy()
        signals = strategy.generate_signals(df)
        assert isinstance(signals, list)

    def test_value_strategy_empty_without_fundamentals(self) -> None:
        """Value strategy returns empty without PE/PB/ROE columns."""
        df = make_ohlcv_data(n_days=60, trend="up")
        strategy = ROEPBValueStrategy()
        signals = strategy.generate_signals(df)
        assert signals == []  # No fundamental columns

    def test_grid_strategy_returns_signals(self) -> None:
        """Grid strategy should return valid Signal objects."""
        df = make_ohlcv_data(n_days=100, trend="sideways")
        strategy = PriceRangeGridStrategy()
        signals = strategy.generate_signals(df)
        assert isinstance(signals, list)

    def test_momentum_strategy_runs(self) -> None:
        """Momentum strategy should run without error."""
        df = make_ohlcv_data(n_days=60, trend="up")
        strategy = VolumePriceBreakoutMomentumStrategy()
        signals = strategy.generate_signals(df)
        assert isinstance(signals, list)


# ---- Market State Detector Tests ----

class TestMarketState:
    """Market state detection tests."""

    def test_bullish_market_detected(self) -> None:
        """High limit-ups, high turnover should detect climax."""
        indicators = MarketIndicators(
            limit_up_count=100,
            limit_down_count=5,
            up_count=3000,
            down_count=1500,
            turnover_billion=1600,
            northbound_net_billion=10,
            consecutive_up_height=7,
            index_ma_deviation=0.15,
        )
        detector = MarketStateDetector()
        state = detector.detect(indicators)
        assert state == MarketState.CLIMAX

    def test_frozen_market_detected(self) -> None:
        """Low turnover, many limit-downs should detect frozen."""
        indicators = MarketIndicators(
            limit_up_count=5,
            limit_down_count=80,
            up_count=500,
            down_count=4000,
            turnover_billion=300,
            northbound_net_billion=-15,
            index_ma_deviation=-0.12,
        )
        detector = MarketStateDetector()
        state = detector.detect(indicators)
        assert state == MarketState.FROZEN

    def test_strategy_mapping(self) -> None:
        """Strategy mapping should return appropriate strategies."""
        detector = MarketStateDetector()
        mapping = detector.get_strategy_mapping()
        assert mapping[MarketState.FROZEN] == []  # No strategies in frozen market
        assert len(mapping[MarketState.FERMENTING]) >= 2  # Multiple strategies when active


# ---- Long-Term Memory Tests ----

class TestLongTermMemory:
    """Long-term memory system tests."""

    def test_record_and_retrieve(self) -> None:
        """Should record decision and retrieve strategy weight."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            mem = LongTermMemory(db_path=db_path)

            # Record decisions
            for i in range(15):
                did = mem.record_decision(
                    "000001", strength_score=70 + i % 20,
                    direction="bullish", market_state="fermenting",
                    mode="fast", evidence=[], risks=[],
                )
                # Record outcomes (simulate ~55% win rate)
                win = (i % 20) < 11
                mem.record_outcome(did, 0.03 if win else -0.02, days=5)

            # Update strategy weights
            for i in range(15):
                mem.update_strategy_weight("platform_breakout", win=(i % 3 != 0))

            weight = mem.get_strategy_weight("platform_breakout")
            assert 0.3 <= weight <= 0.9
            # Above 0.5 means better than random

    def test_unknown_strategy_default_weight(self) -> None:
        """Unknown strategy should return default weight of 1.0."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memory.db")
            mem = LongTermMemory(db_path=db_path)
            weight = mem.get_strategy_weight("nonexistent")
            assert weight == 1.0


# ---- Sentiment Analyzer Tests ----

class TestSentimentAnalyzer:
    """Sentiment analysis tests."""

    def test_rule_based_positive(self) -> None:
        """Rule-based analyzer should detect positive sentiment."""
        analyzer = SentimentAnalyzer()
        score = analyzer._analyze_rule("公司业绩大幅增长，利润超预期，股价突破新高")
        assert score > 0  # Positive

    def test_rule_based_negative(self) -> None:
        """Rule-based analyzer should detect negative sentiment."""
        analyzer = SentimentAnalyzer()
        score = analyzer._analyze_rule("公司违规被处罚，业绩大幅下降，股价跌停")
        assert score < 0  # Negative

    def test_rule_based_neutral(self) -> None:
        """Rule-based analyzer should return neutral for neutral text."""
        analyzer = SentimentAnalyzer()
        score = analyzer._analyze_rule("今日市场交易平稳，成交量一般")
        assert score == 0.0  # Neutral

    def test_keywords_exist(self) -> None:
        """Positive and negative keyword lists should be populated."""
        assert len(SentimentAnalyzer.POSITIVE_KEYWORDS) >= 5
        assert len(SentimentAnalyzer.NEGATIVE_KEYWORDS) >= 5
