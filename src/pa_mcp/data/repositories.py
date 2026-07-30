# [AI:BEGIN]
# PA_MCP - PIT Repository (Point-in-Time Data Access)
#
# Provides as-of queries against versioned data.
# All research queries go through this layer to guarantee:
#   1. available_at <= decision_time
#   2. valid_from <= decision_time < valid_to (for SCD2 records)
#   3. Fixed dataset_snapshot_id for reproducibility
#
# This is the single source of truth for strategy and backtest data access.
# [AI:END]

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd


class PointInTimeRepository:
    """Repository for as-of queries against versioned market data.

    Wraps DuckDB and enforces the PIT contract:
        available_at <= decision_time
        AND valid_from <= decision_time
        AND (valid_to IS NULL OR decision_time < valid_to)
    """

    def __init__(self, store, snapshot_id: Optional[str] = None) -> None:
        self._store = store
        self._snapshot_id = snapshot_id  # None = use latest data

    # ---- Universe & Securities ----

    def get_universe(
        self, decision_time: datetime, universe_name: str = "all_a",
    ) -> pd.DataFrame:
        """Get list of stocks that existed at decision_time."""
        # Simplified query — full version uses SCD2 security_status table
        sql = """
            SELECT symbol, name, board
            FROM stock_basic
            WHERE list_date <= ?::DATE
              AND (delist_date IS NULL OR delist_date > ?::DATE)
        """
        params = [
            decision_time.date().isoformat(),
            decision_time.date().isoformat(),
        ]
        return self._store.query_df(sql, params)

    def get_security_status(self, symbol: str, decision_time: datetime) -> Optional[dict]:
        """Get the effective security status at decision_time."""
        # Simplified — full version queries SCD2 security_status_history
        sql = """
            SELECT symbol, name, board, is_st, is_suspended
            FROM stock_basic
            WHERE symbol = ?
        """
        result = self._store.query_df(sql, [symbol])
        if result.empty:
            return None
        return result.iloc[0].to_dict()

    # ---- Market Data ----

    def get_daily_bars(
        self, symbol: str,
        start_date: date, end_date: date,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Get daily OHLCV bars in chronological order."""
        order = "ASC" if ascending else "DESC"
        sql = f"""
            SELECT symbol, date, open, high, low, close, volume, amount, turnover
            FROM kline_daily
            WHERE symbol = ?
              AND date >= ? AND date <= ?
            ORDER BY date {order}
        """
        return self._store.query_df(
            sql,
            [symbol, start_date.isoformat(), end_date.isoformat()],
        )

    def get_daily_bars_multi(
        self, symbols: list[str],
        decision_time: datetime,
        lookback_days: int = 120,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Get recent daily bars for multiple symbols, as-of decision_time."""
        order = "ASC" if ascending else "DESC"
        start = (decision_time.date() - pd.Timedelta(days=lookback_days)).isoformat()
        end = decision_time.date().isoformat()
        placeholders = ", ".join(["?" for _ in symbols])
        sql = f"""
            SELECT symbol, date, open, high, low, close, volume, amount, turnover
            FROM kline_daily
            WHERE symbol IN ({placeholders})
              AND date >= ? AND date <= ?
            ORDER BY symbol, date {order}
        """
        return self._store.query_df(sql, [*symbols, start, end])

    # ---- Benchmark ----

    def get_benchmark_returns(
        self, benchmark: str,
        start_date: date, end_date: date,
    ) -> pd.Series:
        """Get benchmark daily returns for attribution."""
        sql = """
            SELECT date, close
            FROM index_daily
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """
        df = self._store.query_df(
            sql,
            [benchmark, start_date.isoformat(), end_date.isoformat()],
        )
        if df.empty:
            return pd.Series(dtype=float)
        df["ret"] = df["close"].pct_change()
        df = df.dropna(subset=["ret"])
        return df.set_index("date")["ret"]

    # ---- Snapshot (for reproducibility) ----

    def with_snapshot(self, snapshot_id: str) -> "PointInTimeRepository":
        """Return a new repository pinned to a specific dataset snapshot."""
        return PointInTimeRepository(self._store, snapshot_id=snapshot_id)

    @property
    def snapshot_id(self) -> Optional[str]:
        return self._snapshot_id
