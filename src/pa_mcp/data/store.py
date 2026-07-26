# [AI:BEGIN]
# PA_MCP - Data Layer: DuckDB Store
# Manages database creation, migration, and CRUD operations.
# Uses shadow-table strategy for atomic data updates.
# [AI:END]

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---- SQL DDL Statements ----

TABLE_DEFINITIONS: dict[str, str] = {
    "kline_daily": """
        CREATE TABLE IF NOT EXISTS kline_daily (
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            amount DOUBLE,
            amplitude DOUBLE,
            pct_change DOUBLE,
            change DOUBLE,
            turnover DOUBLE,
            adjust_factor DOUBLE DEFAULT 1.0,
            PRIMARY KEY (symbol, date)
        )
    """,

    "kline_minute": """
        CREATE TABLE IF NOT EXISTS kline_minute (
            symbol VARCHAR(10) NOT NULL,
            datetime TIMESTAMP NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            amount DOUBLE,
            period VARCHAR(5) DEFAULT '5',
            PRIMARY KEY (symbol, datetime, period)
        )
    """,

    "index_daily": """
        CREATE TABLE IF NOT EXISTS index_daily (
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            amount DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """,

    "stock_basic": """
        CREATE TABLE IF NOT EXISTS stock_basic (
            symbol VARCHAR(10) PRIMARY KEY,
            name VARCHAR(50),
            industry VARCHAR(50),
            sector VARCHAR(100),
            market_cap DOUBLE,
            list_date DATE,
            exchange VARCHAR(10),
            board VARCHAR(20),
            is_st BOOLEAN DEFAULT FALSE,
            is_suspended BOOLEAN DEFAULT FALSE
        )
    """,

    "financials_income": """
        CREATE TABLE IF NOT EXISTS financials_income (
            symbol VARCHAR(10) NOT NULL,
            report_date DATE NOT NULL,
            pub_date DATE,
            revenue DOUBLE,
            operating_profit DOUBLE,
            net_profit DOUBLE,
            net_profit_parent DOUBLE,
            eps DOUBLE,
            roe DOUBLE,
            gross_margin DOUBLE,
            net_margin DOUBLE,
            revenue_yoy DOUBLE,
            profit_yoy DOUBLE,
            PRIMARY KEY (symbol, report_date)
        )
    """,

    "financials_balance": """
        CREATE TABLE IF NOT EXISTS financials_balance (
            symbol VARCHAR(10) NOT NULL,
            report_date DATE NOT NULL,
            pub_date DATE,
            total_assets DOUBLE,
            total_liabilities DOUBLE,
            equity_parent DOUBLE,
            current_assets DOUBLE,
            current_liabilities DOUBLE,
            debt_ratio DOUBLE,
            PRIMARY KEY (symbol, report_date)
        )
    """,

    "financials_cashflow": """
        CREATE TABLE IF NOT EXISTS financials_cashflow (
            symbol VARCHAR(10) NOT NULL,
            report_date DATE NOT NULL,
            pub_date DATE,
            cf_operations DOUBLE,
            cf_investing DOUBLE,
            cf_financing DOUBLE,
            free_cash_flow DOUBLE,
            PRIMARY KEY (symbol, report_date)
        )
    """,

    "capital_flow": """
        CREATE TABLE IF NOT EXISTS capital_flow (
            symbol VARCHAR(10) NOT NULL,
            trade_date DATE NOT NULL,
            main_net_inflow DOUBLE,
            retail_net_inflow DOUBLE,
            main_net_inflow_pct DOUBLE,
            northbound_net_inflow DOUBLE,
            northbound_hold_pct DOUBLE,
            PRIMARY KEY (symbol, trade_date)
        )
    """,

    "dragon_tiger": """
        CREATE TABLE IF NOT EXISTS dragon_tiger (
            trade_date DATE NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            seat_name VARCHAR(200),
            buy_amount DOUBLE DEFAULT 0,
            sell_amount DOUBLE DEFAULT 0,
            net_amount DOUBLE,
            seat_type VARCHAR(20),
            reason VARCHAR(200),
            PRIMARY KEY (trade_date, symbol, seat_name)
        )
    """,

    "major_events": """
        CREATE TABLE IF NOT EXISTS major_events (
            symbol VARCHAR(10) NOT NULL,
            event_date DATE NOT NULL,
            event_type VARCHAR(20) NOT NULL,
            detail JSON,
            PRIMARY KEY (symbol, event_date, event_type)
        )
    """,

    "trade_calendar": """
        CREATE TABLE IF NOT EXISTS trade_calendar (
            date DATE PRIMARY KEY,
            is_trading_day BOOLEAN DEFAULT TRUE,
            week_day INTEGER,
            holiday_name VARCHAR(50)
        )
    """,

    "indicator_cache": """
        CREATE TABLE IF NOT EXISTS indicator_cache (
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            ma5 DOUBLE, ma10 DOUBLE, ma20 DOUBLE, ma60 DOUBLE, ma120 DOUBLE, ma250 DOUBLE,
            ema12 DOUBLE, ema26 DOUBLE,
            macd DOUBLE, macd_signal DOUBLE, macd_hist DOUBLE,
            rsi6 DOUBLE, rsi14 DOUBLE, rsi24 DOUBLE,
            kdj_k DOUBLE, kdj_d DOUBLE, kdj_j DOUBLE,
            boll_upper DOUBLE, boll_mid DOUBLE, boll_lower DOUBLE,
            atr14 DOUBLE,
            obv DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """,

    "signal_cache": """
        CREATE TABLE IF NOT EXISTS signal_cache (
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            strategy_name VARCHAR(100) NOT NULL,
            strength_score DOUBLE,
            direction VARCHAR(10),
            details JSON,
            PRIMARY KEY (symbol, date, strategy_name)
        )
    """,
}


class DuckDBStore:
    """DuckDB-based data store for A-share market data.

    Features:
    - Auto-creates tables on first connection
    - Shadow-table strategy for atomic batch updates
    - Sorted storage for OLAP query optimization
    - Connection pooling for thread safety
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            from pa_mcp.config import get_settings
            db_path = get_settings().database.path

        self.db_path = str(db_path)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Get or create a database connection."""
        if self._conn is None:
            # Ensure parent directory exists
            db_dir = Path(self.db_path).parent
            os.makedirs(db_dir, exist_ok=True)

            self._conn = duckdb.connect(self.db_path)
            logger.info("DuckDB connected", path=self.db_path)
            self._init_tables()
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("DuckDB connection closed")

    def _init_tables(self) -> None:
        """Create all required tables if they don't exist."""
        conn = self.connect()
        for table_name, ddl in TABLE_DEFINITIONS.items():
            try:
                conn.execute(ddl)
                logger.debug("Table initialized", table=table_name)
            except Exception as e:
                logger.error("Failed to create table", table=table_name, error=str(e))
                raise

    # ---- CRUD Operations ----

    def execute(self, sql: str, params: Optional[list] = None) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query."""
        conn = self.connect()
        if params:
            return conn.execute(sql, params)
        return conn.execute(sql)

    def query_df(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """Execute SQL and return results as DataFrame."""
        return self.execute(sql, params).df()

    def insert_df(
        self, table_name: str, df: pd.DataFrame,
        mode: str = "append",
    ) -> None:
        """Insert a DataFrame into a table.

        Args:
            table_name: Target table name
            df: DataFrame to insert
            mode: 'append' (default) or 'replace'
        """
        conn = self.connect()
        if mode == "replace":
            conn.execute(f"DELETE FROM {table_name}")
        conn.register("_tmp_insert", df)
        conn.execute(f"INSERT OR REPLACE INTO {table_name} SELECT * FROM _tmp_insert")
        conn.unregister("_tmp_insert")

    def swap_table(self, table_name: str, df: pd.DataFrame) -> None:
        """Atomic table replacement using shadow table strategy.

        Creates _new table, inserts data, then renames for zero-downtime swap.
        This avoids the DuckDB write-lock issue during batch updates.
        """
        conn = self.connect()
        new_table = f"{table_name}_new"
        shadow_table = f"{table_name}_old"

        # Create shadow table with same schema
        conn.execute(f"DROP TABLE IF EXISTS {new_table}")
        conn.execute(f"DROP TABLE IF EXISTS {shadow_table}")

        # Get original DDL and create new table
        ddl_key = table_name.split(".")[-1]
        if ddl_key in TABLE_DEFINITIONS:
            ddl = TABLE_DEFINITIONS[ddl_key].replace(
                f"CREATE TABLE IF NOT EXISTS {ddl_key}",
                f"CREATE TABLE {new_table}",
            )
            conn.execute(ddl)

        # Insert data
        conn.register("_swap_data", df)
        conn.execute(f"INSERT INTO {new_table} SELECT * FROM _swap_data")
        conn.unregister("_swap_data")

        # Atomic swap
        conn.execute(f"ALTER TABLE IF EXISTS {table_name} RENAME TO {shadow_table}")
        conn.execute(f"ALTER TABLE {new_table} RENAME TO {table_name}")
        conn.execute(f"DROP TABLE IF EXISTS {shadow_table}")

        logger.info("Table swapped atomically", table=table_name, rows=len(df))

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        conn = self.connect()
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return result[0] > 0 if result else False

    def row_count(self, table_name: str) -> int:
        """Get approximate row count for a table."""
        conn = self.connect()
        result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0] if result else 0

    def get_latest_date(self, table_name: str, date_col: str = "date") -> Optional[str]:
        """Get the latest date in a table."""
        conn = self.connect()
        if not self.table_exists(table_name):
            return None
        result = conn.execute(
            f"SELECT MAX({date_col}) FROM {table_name}",
        ).fetchone()
        return result[0] if result and result[0] else None
