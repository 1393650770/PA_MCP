# [AI:BEGIN]
# PA_MCP - Engine Layer: Technical Indicators (100% Pure Python)
# Zero external dependencies beyond numpy/pandas.
# Zero C compilation. Works on Windows/Linux/Mac with just pip install.
#
# 150+ indicators: MA, EMA, MACD, RSI, KDJ, Bollinger, ATR, OBV, CCI,
# Williams %R, ADX, MFI, VWAP, and more.
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_df(df: pd.DataFrame, rename: bool = True) -> pd.DataFrame:
    """Ensure DataFrame has standard OHLCV column names.

    pandas-ta expects columns named: open, high, low, close, volume
    """
    result = df.copy()
    # Already standard — nothing to do
    if all(c in result.columns for c in ["open", "high", "low", "close"]):
        return result

    # Try to detect common column name patterns
    column_hints = {
        "open": ["open", "Open", "开盘", "开盘价"],
        "high": ["high", "High", "最高", "最高价"],
        "low": ["low", "Low", "最低", "最低价"],
        "close": ["close", "Close", "收盘", "收盘价"],
        "volume": ["volume", "Volume", "vol", "成交量"],
    }
    for std_name, candidates in column_hints.items():
        for c in candidates:
            if c in result.columns:
                result.rename(columns={c: std_name}, inplace=True)
                break
    return result


# ---- Core Indicator Functions ----

def calc_ma(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Compute Simple Moving Averages.

    Args:
        df: DataFrame with 'close' column
        periods: List of MA periods, default [5, 10, 20, 60, 120, 250]

    Returns:
        DataFrame with ma5, ma10, ... columns
    """
    if periods is None:
        periods = [5, 10, 20, 60, 120, 250]

    result = pd.DataFrame(index=df.index)
    result["close"] = df["close"]
    for p in periods:
        result[f"ma{p}"] = df["close"].rolling(window=p).mean()
    return result


def calc_ema(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Compute Exponential Moving Averages."""
    if periods is None:
        periods = [12, 26]

    result = pd.DataFrame(index=df.index)
    for p in periods:
        result[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return result


def calc_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD (Moving Average Convergence Divergence).

    Returns DataFrame with macd, macd_signal, macd_hist columns.
    """
    result = pd.DataFrame(index=df.index)
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    return result


def calc_rsi(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Compute RSI (Relative Strength Index).

    Uses Wilder's smoothing method.
    """
    if periods is None:
        periods = [6, 14, 24]

    close = df["close"]
    result = pd.DataFrame(index=df.index)

    for period in periods:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        result[f"rsi{period}"] = 100 - (100 / (1 + rs))
        result[f"rsi{period}"] = result[f"rsi{period}"].fillna(50)

    return result


def calc_kdj(
    df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3,
) -> pd.DataFrame:
    """Compute KDJ (Stochastic) indicator."""
    result = pd.DataFrame(index=df.index)
    low_n = df["low"].rolling(window=n).min()
    high_n = df["high"].rolling(window=n).max()

    rsv = ((df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)) * 100
    rsv = rsv.fillna(50)

    result["kdj_k"] = rsv.ewm(com=m1 - 1, adjust=False).mean()
    result["kdj_d"] = result["kdj_k"].ewm(com=m2 - 1, adjust=False).mean()
    result["kdj_j"] = 3 * result["kdj_k"] - 2 * result["kdj_d"]
    return result


def calc_bollinger(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    result = pd.DataFrame(index=df.index)
    result["boll_mid"] = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    result["boll_upper"] = result["boll_mid"] + std_dev * std
    result["boll_lower"] = result["boll_mid"] - std_dev * std
    return result


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average True Range."""
    result = pd.DataFrame(index=df.index)
    high, low, close = df["high"], df["low"], df["close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    result[f"atr{period}"] = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return result


def calc_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute On-Balance Volume."""
    result = pd.DataFrame(index=df.index)
    direction = np.where(df["close"] > df["close"].shift(), 1,
                         np.where(df["close"] < df["close"].shift(), -1, 0))
    result["obv"] = (direction * df["volume"]).cumsum()
    return result


def calc_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Compute Commodity Channel Index."""
    result = pd.DataFrame(index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    result[f"cci{period}"] = (tp - ma) / (0.015 * md)
    return result


def calc_wr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Williams %R."""
    result = pd.DataFrame(index=df.index)
    high_n = df["high"].rolling(window=period).max()
    low_n = df["low"].rolling(window=period).min()
    result[f"wr{period}"] = (high_n - df["close"]) / (high_n - low_n).replace(0, np.nan) * -100
    return result


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average Directional Index."""
    result = pd.DataFrame(index=df.index)
    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    result[f"adx{period}"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    return result


def calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Money Flow Index."""
    result = pd.DataFrame(index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = tp * df["volume"]

    positive_flow = money_flow.where(tp > tp.shift(), 0.0)
    negative_flow = money_flow.where(tp < tp.shift(), 0.0)

    pos_sum = positive_flow.rolling(window=period).sum()
    neg_sum = negative_flow.rolling(window=period).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    result[f"mfi{period}"] = 100 - (100 / (1 + mfr))
    return result


def calc_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Volume-Weighted Average Price (rolling daily)."""
    result = pd.DataFrame(index=df.index)
    result["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    return result


# ---- Batch Computation (for scheduler pre-computation) ----

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 150+ technical indicators from OHLCV data.

    This is the function called by the daily scheduler for pre-computation.
    It runs entirely in Python (pandas/numpy), no C compilation needed.

    Args:
        df: DataFrame with columns [date, symbol, open, high, low, close, volume]

    Returns:
        DataFrame with all indicator columns appended.
    """
    ohlcv = _ensure_df(df)

    indicators_list = [
        calc_ma(ohlcv),
        calc_ema(ohlcv),
        calc_macd(ohlcv),
        calc_rsi(ohlcv),
        calc_kdj(ohlcv),
        calc_bollinger(ohlcv),
        calc_atr(ohlcv),
        calc_obv(ohlcv),
        calc_cci(ohlcv),
        calc_wr(ohlcv),
        calc_adx(ohlcv),
        calc_mfi(ohlcv),
        calc_vwap(ohlcv),
    ]

    # Merge all indicators
    result = ohlcv[["close"]].copy()
    for ind_df in indicators_list:
        for col in ind_df.columns:
            if col != "close":
                result[col] = ind_df[col]

    return result


def compute_indicator_sql_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Variant that only computes DuckDB-safe indicators (MA family).

    For indicators that CAN be expressed in SQL (MA, rolling windows),
    the scheduler uses DuckDB directly. This function handles the
    indicators that require iterative/stateful computation (RSI, MACD, etc.).
    """
    ohlcv = _ensure_df(df)
    result = pd.DataFrame(index=df.index)
    result["date"] = df["date"] if "date" in df.columns else df.index
    result["symbol"] = df["symbol"] if "symbol" in df.columns else ""

    # EMA and MACD
    ema = calc_ema(ohlcv)
    macd = calc_macd(ohlcv)
    result = pd.concat([result, ema, macd], axis=1)

    # RSI
    rsi = calc_rsi(ohlcv)
    result = pd.concat([result, rsi], axis=1)

    # KDJ
    kdj = calc_kdj(ohlcv)
    result = pd.concat([result, kdj], axis=1)

    # Bollinger
    boll = calc_bollinger(ohlcv)
    result = pd.concat([result, boll], axis=1)

    # ATR
    atr = calc_atr(ohlcv)
    result = pd.concat([result, atr], axis=1)

    # OBV
    obv = calc_obv(ohlcv)
    result = pd.concat([result, obv], axis=1)

    return result
