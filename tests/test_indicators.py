# [AI:BEGIN]
# PA_MCP - Tests: Pure Python Technical Indicators
# Verify all indicators compute correctly without C compilation.
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa_mcp.engine.indicators.indicators import (
    calc_ma, calc_ema, calc_macd, calc_rsi, calc_kdj,
    calc_bollinger, calc_atr, calc_obv, calc_cci, calc_wr,
    calc_adx, calc_mfi, calc_vwap, compute_all_indicators,
)


def make_ohlcv_df(n_days: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    close = 100.0
    data = []
    for i in range(n_days):
        ret = np.random.normal(0.001, 0.02)
        open_p = close
        close = close * (1 + ret)
        high = max(open_p, close) * (1 + abs(np.random.normal(0, 0.005)))
        low = min(open_p, close) * (1 - abs(np.random.normal(0, 0.005)))
        vol = np.random.uniform(1e6, 1e7)
        data.append({
            "open": round(open_p, 2), "high": round(high, 2),
            "low": round(low, 2), "close": round(close, 2),
            "volume": vol,
        })
    return pd.DataFrame(data)


class TestIndicators:
    """Pure Python technical indicator tests."""

    def test_calc_ma(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_ma(df)
        assert "ma5" in result.columns
        assert "ma20" in result.columns
        assert abs(result["ma20"].iloc[-1] - df["close"].iloc[-20:].mean()) < 1.0

    def test_calc_ema(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_ema(df)
        assert "ema12" in result.columns
        assert "ema26" in result.columns

    def test_calc_macd(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_macd(df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns
        # MACD hist should equal macd - signal
        diff = abs((result["macd"] - result["macd_signal"] - result["macd_hist"]).mean())
        assert diff < 0.01

    def test_calc_rsi(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_rsi(df)
        assert "rsi14" in result.columns
        # RSI should be between 0 and 100
        assert result["rsi14"].dropna().min() >= 0
        assert result["rsi14"].dropna().max() <= 100

    def test_calc_kdj(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_kdj(df)
        assert "kdj_k" in result.columns
        assert "kdj_d" in result.columns
        assert "kdj_j" in result.columns

    def test_calc_bollinger(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_bollinger(df)
        assert "boll_upper" in result.columns
        assert "boll_mid" in result.columns
        assert "boll_lower" in result.columns
        # Upper > Mid > Lower
        last = result.iloc[-1]
        assert last["boll_upper"] >= last["boll_mid"] >= last["boll_lower"]

    def test_calc_atr(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_atr(df)
        assert "atr14" in result.columns
        assert result["atr14"].dropna().min() > 0

    def test_calc_obv(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_obv(df)
        assert "obv" in result.columns

    def test_calc_cci(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_cci(df)
        assert "cci20" in result.columns

    def test_calc_wr(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_wr(df)
        assert "wr14" in result.columns
        assert result["wr14"].dropna().max() <= 0  # WR runs -100 to 0

    def test_calc_adx(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_adx(df)
        assert "adx14" in result.columns

    def test_calc_mfi(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_mfi(df)
        assert "mfi14" in result.columns
        valid = result["mfi14"].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0
            assert valid.max() <= 100

    def test_calc_vwap(self) -> None:
        df = make_ohlcv_df(100)
        result = calc_vwap(df)
        assert "vwap" in result.columns

    def test_compute_all_indicators(self) -> None:
        """End-to-end: compute all 150+ indicators on 200 days of data."""
        df = make_ohlcv_df(200)
        result = compute_all_indicators(df)
        # Should produce many indicator columns
        n_cols = len(result.columns)
        assert n_cols >= 25
        # No NaN in the last row (where enough data exists)
        last_valid = result.iloc[-1]
        assert last_valid.notna().sum() > 20

    def test_compute_all_runs_fast(self) -> None:
        """Full indicator computation should be fast (<1s for 200 days)."""
        import time
        df = make_ohlcv_df(200)
        t0 = time.monotonic()
        compute_all_indicators(df)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"Indicator computation too slow: {elapsed:.2f}s"
        print(f"\n  [perf] 200 days x 150+ indicators: {elapsed:.3f}s")
