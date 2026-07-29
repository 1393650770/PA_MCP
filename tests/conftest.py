# [AI:BEGIN]
# PA_MCP - Shared test fixtures (conftest.py)
# [AI:END]

import numpy as np
import pandas as pd
import pytest


# ---- Data Generators ----

@pytest.fixture
def ohlcv_data():
    """200-day synthetic OHLCV DataFrame in uptrend."""
    np.random.seed(42)
    close = 10.0
    data = []
    for i in range(200):
        ret = np.random.normal(0.001, 0.02)
        open_p = close
        close = close * (1 + ret)
        high = max(open_p, close) * (1 + abs(np.random.normal(0, 0.005)))
        low = min(open_p, close) * (1 - abs(np.random.normal(0, 0.005)))
        vol = np.random.uniform(1e6, 1e7)
        data.append({
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
            "symbol": "000001",
            "open": round(open_p, 2), "high": round(high, 2),
            "low": round(low, 2), "close": round(close, 2),
            "volume": vol, "pct_change": round(ret * 100, 2),
        })
    df = pd.DataFrame(data)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["avg_vol_20"] = df["volume"].rolling(20).mean()
    return df


@pytest.fixture
def ohlcv_data_down():
    """200-day synthetic OHLCV in downtrend."""
    np.random.seed(123)
    close = 30.0
    data = []
    for i in range(200):
        ret = np.random.normal(-0.001, 0.02)
        open_p = close
        close = close * (1 + ret)
        data.append({
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
            "symbol": "000001",
            "open": round(open_p, 2), "high": round(max(open_p, close) * 1.005, 2),
            "low": round(min(open_p, close) * 0.995, 2), "close": round(close, 2),
            "volume": np.random.uniform(5e5, 5e6), "pct_change": round(ret * 100, 2),
        })
    df = pd.DataFrame(data)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["avg_vol_20"] = df["volume"].rolling(20).mean()
    return df


@pytest.fixture
def ohlcv_data_sideways():
    """60-day synthetic OHLCV in sideways market."""
    np.random.seed(789)
    close = 10.0
    data = []
    for i in range(60):
        ret = np.random.normal(0, 0.015)
        open_p = close
        close = close * (1 + ret)
        data.append({
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
            "open": round(open_p, 2), "high": round(max(open_p, close) * 1.005, 2),
            "low": round(min(open_p, close) * 0.995, 2), "close": round(close, 2),
            "volume": np.random.uniform(1e6, 5e6), "pct_change": round(ret * 100, 2),
        })
    return pd.DataFrame(data)


@pytest.fixture
def events_data():
    """20-row event-rich test DataFrame."""
    np.random.seed(42)
    close = 10.0
    data = []
    for i in range(20):
        close = close * (1 + np.random.normal(0, 0.02))
        data.append({
            "symbol": "000001", "close": round(close, 2),
            "volume": np.random.uniform(1e6, 1e7),
            "pct_change": np.random.normal(0, 1),
            "insider_buy_amount": 2_000_000 if i == 19 else 0,
            "insider_buy_count": 3 if i == 19 else 0,
            "pledge_ratio": 0.30,
            "lockup_pct_of_float": 0.0, "lockup_expiry_date": "",
            "block_trade_amount": 0, "block_trade_discount": 0.0,
        })
    return pd.DataFrame(data)
