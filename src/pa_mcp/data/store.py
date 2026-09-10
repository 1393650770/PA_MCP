# [AI:BEGIN]
# PA_MCP - Data Layer: DuckDB Store
# Manages database creation, migration, and CRUD operations.
# Uses shadow-table strategy for atomic data updates.
# [AI:END]

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# DuckDB 排他锁重试间隔（秒）
_RETRY_INTERVAL_SECONDS = 2.0


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
            delist_date DATE,
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

    "fund_flow_daily": """
        CREATE TABLE IF NOT EXISTS fund_flow_daily (
            symbol VARCHAR(10) NOT NULL,
            trade_date DATE NOT NULL,
            main_net_inflow DOUBLE,
            small_net_inflow DOUBLE,
            mid_net_inflow DOUBLE,
            large_net_inflow DOUBLE,
            super_large_net_inflow DOUBLE,
            main_net_inflow_pct DOUBLE,
            PRIMARY KEY (symbol, trade_date)
        )
    """,

    # 游资情绪周期日统计（涨停梯队/连板高度/晋级率）
    "sentiment_daily": """
        CREATE TABLE IF NOT EXISTS sentiment_daily (
            date DATE PRIMARY KEY,
            limit_up_count INTEGER,
            limit_down_count INTEGER,
            max_board_height INTEGER,
            board2_count INTEGER,
            board3_count INTEGER,
            board4p_count INTEGER,
            first_board_count INTEGER,
            promotion_rate DOUBLE,
            sentiment_score DOUBLE,
            stage VARCHAR(20)
        )
    """,

    # 板块日线（东财 BK 板块指数，供板块轮动研究）
    "sector_daily": """
        CREATE TABLE IF NOT EXISTS sector_daily (
            sector_code VARCHAR(20) NOT NULL,
            name VARCHAR(50),
            date DATE NOT NULL,
            open DOUBLE,
            close DOUBLE,
            high DOUBLE,
            low DOUBLE,
            volume DOUBLE,
            amount DOUBLE,
            pct_change DOUBLE,
            turnover DOUBLE,
            PRIMARY KEY (sector_code, date)
        )
    """,

    # 板块轮动预测日志（LLM/确定性，供周度验证）
    "sector_prediction": """
        CREATE SEQUENCE IF NOT EXISTS sector_prediction_seq START 1;
        CREATE TABLE IF NOT EXISTS sector_prediction (
            id INTEGER DEFAULT nextval('sector_prediction_seq') PRIMARY KEY,
            predict_date DATE NOT NULL,
            mode VARCHAR(10) DEFAULT 'deterministic',
            top_sectors VARCHAR(500),
            avoid_sectors VARCHAR(300),
            rotation_logic VARCHAR(300),
            confidence DOUBLE,
            key_drivers VARCHAR(1000),
            risks VARCHAR(1000),
            status VARCHAR(10) DEFAULT 'pending',
            top3_avg_return_pct DOUBLE,
            market_avg_return_pct DOUBLE,
            excess_return_pct DOUBLE,
            evaluated_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    # 经验库：AI 分析历史（供检索增强注入 prompt）
    "analysis_history": """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            analyzed_date DATE NOT NULL,
            cycle_position VARCHAR(20) DEFAULT 'unknown',
            direction VARCHAR(10) DEFAULT 'neutral',
            strength_score DOUBLE DEFAULT 50,
            summary VARCHAR(1000),
            conclusion VARCHAR(500),
            dimension_scores VARCHAR(500),
            key_risks VARCHAR(1000),
            outcome VARCHAR(10) DEFAULT 'pending',
            actual_return_5d DOUBLE,
            actual_return_20d DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    # 市场预测日志（LLM/确定性预测落盘，供命中率验证）
    "prediction_log": """
        CREATE SEQUENCE IF NOT EXISTS prediction_log_seq START 1;
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER DEFAULT nextval('prediction_log_seq') PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            predict_date DATE NOT NULL,
            horizon VARCHAR(5) NOT NULL,
            direction VARCHAR(10) NOT NULL,
            probability DOUBLE,
            prob_up DOUBLE,
            prob_down DOUBLE,
            prob_sideways DOUBLE,
            expected_return_pct DOUBLE,
            expected_range_low DOUBLE,
            expected_range_high DOUBLE,
            cycle_position VARCHAR(20),
            cycle_forecast VARCHAR(20),
            support_levels VARCHAR(500),
            resistance_levels VARCHAR(500),
            scenarios VARCHAR(2000),
            confidence DOUBLE,
            key_reasons VARCHAR(2000),
            key_risks VARCHAR(2000),
            model VARCHAR(100),
            prompt_version VARCHAR(20),
            mode VARCHAR(10) DEFAULT 'deterministic',
            status VARCHAR(10) DEFAULT 'pending',
            actual_return_pct DOUBLE,
            evaluated_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    def __init__(
        self,
        db_path: Optional[str] = None,
        read_only: Optional[bool] = None,
    ) -> None:
        if db_path is None:
            from pa_mcp.config import get_settings
            db_path = get_settings().database.path

        self.db_path = str(db_path)
        if read_only is None:
            try:
                from pa_mcp.config import get_settings
                read_only = bool(get_settings().database.read_only)
            except Exception:  # pragma: no cover - 配置不可用时按读写处理
                read_only = False

        # 期望的打开模式（配置来源）
        self._requested_read_only = bool(read_only)
        # 实际生效的打开模式（connect() 成功之后才有意义）
        self.read_only: bool = self._requested_read_only
        # 连接重试预算（秒）：DuckDB 单文件排他锁，等一等通常就能拿到
        try:
            from pa_mcp.config import get_settings
            self._connect_timeout = float(
                get_settings().database.connect_timeout_seconds
            )
        except Exception:  # pragma: no cover - 配置不可用时用默认值
            self._connect_timeout = 30.0
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        # DuckDB 连接非线程安全：MCP/UI 工具经 asyncio.to_thread 并发访问
        # 同一实例时，C 层竞态会 segfault 直接杀死进程（表现为客户端
        # "Connection closed"）。进程内锁串行化所有访问（毫秒级开销）。
        self._lock = threading.RLock()

    def connect(self, timeout: Optional[float] = None) -> duckdb.DuckDBPyConnection:
        """Get or create a database connection (线程安全).

        DuckDB 单文件是**进程级排他锁**：UI / MCP Server / 数据调度器（或
        两个并发的 cron 会话）同时访问时，后到者会直接抛 IOException。
        实测（Windows）：文件被占用时读写与只读**都**打不开，所以这里按
        「排队重试 → 只读兜底」的顺序处理：

        1. 读写模式重试，直到 `database.connect_timeout_seconds`（默认 30s）
           —— 并发的 cron 会话大多只是几十秒的错峰问题，等一等就能拿到；
        2. 仍然拿不到 → 尝试只读模式（权限受限场景下有效）；
        3. 都失败 → 抛 RuntimeError，附可行动的排查建议。

        调用方应当在**使用点**调用本方法而不是在进程启动时抢锁，这样数据库
        暂时不可用时只影响单次调用，不会拖垮整个服务（MCP Server 启动失败
        等于该会话所有工具消失）。
        """
        with self._lock:
            if self._conn is not None:
                return self._conn

            # Ensure parent directory exists
            db_dir = Path(self.db_path).parent
            os.makedirs(db_dir, exist_ok=True)

            budget = self._connect_timeout if timeout is None else float(timeout)
            deadline = time.monotonic() + max(0.0, budget)
            attempt = 0

            while True:
                attempt += 1
                for read_only in self._open_modes():
                    try:
                        self._conn = duckdb.connect(
                            self.db_path, read_only=read_only,
                        )
                    except Exception as e:
                        if not self._is_lock_contention(e) and not read_only:
                            # 首选（读写）模式遇到非锁冲突错误（如文件损坏、
                            # 版本不兼容），重试没有意义，直接抛出原始错误
                            raise
                        logger.debug(
                            "DuckDB open blocked by another process",
                            path=self.db_path,
                            read_only=read_only,
                            attempt=attempt,
                            error=str(e)[:200],
                        )
                        continue

                    self.read_only = read_only
                    if read_only:
                        logger.warning(
                            "DuckDB connected in READ-ONLY mode (writes disabled)",
                            path=self.db_path,
                        )
                    else:
                        if attempt > 1:
                            logger.info(
                                "DuckDB connected after waiting for lock",
                                path=self.db_path, attempts=attempt,
                            )
                        else:
                            logger.info("DuckDB connected", path=self.db_path)
                        self._init_tables()
                    return self._conn

                if time.monotonic() >= deadline:
                    break
                time.sleep(min(_RETRY_INTERVAL_SECONDS, max(0.05, deadline - time.monotonic())))

            raise RuntimeError(
                f"无法打开 DuckDB 数据库（已被其他进程占用超过 "
                f"{budget:.0f}s）: {self.db_path}\n"
                f"原因: DuckDB 单文件排他锁，同一时间只允许一个进程访问"
                f"（UI / MCP Server / 数据调度器会互锁）。\n"
                f"解决: 1) 稍后重试；2) 关闭占用进程（UI、调度器或其他 "
                f"MCP Server）；3) 把并发任务错峰调度。"
            )

    def _open_modes(self) -> list[bool]:
        """Candidate open modes in priority order.

        只读模式不再尝试读写（避免抢锁）；读写模式失败后把只读作为兜底。
        """
        modes: list[bool] = [self._requested_read_only]
        if not self._requested_read_only:
            modes.append(True)
        return modes

    @staticmethod
    def _is_lock_contention(error: Exception) -> bool:
        """Whether the error means 'someone else holds the DuckDB file'."""
        msg = str(error)
        markers = (
            "另一个程序正在使用此文件",  # Windows 中文
            "being used by another process",  # Windows 英文
            "File is already open",
            "already open",
            "Could not set lock",
            "lock on file",
            "IO Error: Cannot open file",
        )
        return any(m in msg for m in markers)

    def is_writable(self) -> bool:
        """Whether this store can execute DDL/DML (线程安全)."""
        with self._lock:
            self.connect()
            return not self.read_only

    def ensure_writable(self, wait_seconds: float = 0.0) -> bool:
        """拿到写锁才返回 True；只读降级时关掉重开重试，直到预算用完。

        DuckDB 单文件排他锁：并发 cron 会话各起一个 MCP Server 时，后到者
        会被降级成只读。此时**所有**写操作都抛「只读降级模式」——调用方若
        不检查，就会把上千只标的逐个 no-op 一遍然后报「成功 0 行」，
        表现为「cron 状态 ok、数据却一天没动」的静默失败。
        """
        budget = max(0.0, float(wait_seconds))
        deadline = time.monotonic() + budget
        attempt = 0
        while True:
            attempt += 1
            with self._lock:
                try:
                    self.connect()
                    if not self.read_only:
                        return True
                except Exception:  # noqa: BLE001 - 打不开则重开重试
                    pass
                # 只读降级或打不开 → 关掉重开，重新抢写锁
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._conn = None
            if time.monotonic() >= deadline:
                logger.warning(
                    "DuckDB write lock unavailable",
                    path=self.db_path, attempts=attempt,
                    waited_seconds=round(budget, 1),
                )
                return False
            time.sleep(min(5.0, max(0.5, deadline - time.monotonic())))

    def _ensure_writable(self, action: str) -> None:
        """写操作前置检查：只读降级模式下给出可行动的明确报错。"""
        with self._lock:
            self.connect()
            if self.read_only:
                raise RuntimeError(
                    f"无法执行写操作（{action}）：当前为只读降级模式，"
                    f"数据库文件被其他进程占用: {self.db_path}\n"
                    f"原因: DuckDB 单文件排他锁，同一时间只允许一个写进程。\n"
                    f"解决: 关闭占用进程（UI / 调度器 / 其他 MCP Server）后重试。"
                )

    def close(self) -> None:
        """Close the database connection (线程安全)."""
        with self._lock:
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
        """Execute a SQL query (线程安全)."""
        with self._lock:
            conn = self.connect()
            if params:
                return conn.execute(sql, params)
            return conn.execute(sql)

    def query_df(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """Execute SQL and return results as DataFrame (线程安全)."""
        return self.execute(sql, params).df()

    def _get_table_columns(self, table_name: str) -> list[str]:
        """Get the column names of an existing table in DuckDB (线程安全)."""
        with self._lock:
            conn = self.connect()
            result = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table_name],
            ).fetchall()
            return [row[0] for row in result]

    def _validate_and_align_df(
        self, table_name: str, df: pd.DataFrame, *,
        fill_defaults: bool = True,
    ) -> pd.DataFrame:
        """Validate and align a DataFrame to match target table schema.

        Returns a DataFrame with columns matching the target table in order.
        Missing optional columns are filled with NaN defaults.
        Extra columns not in the target schema raise an error.
        """
        target_cols = self._get_table_columns(table_name)
        if not target_cols:
            raise ValueError(f"Table '{table_name}' does not exist or has no columns")

        df_cols = list(df.columns)
        missing_cols = set(target_cols) - set(df_cols)
        extra_cols = set(df_cols) - set(target_cols)

        if extra_cols:
            raise ValueError(
                f"DataFrame has columns not in target table '{table_name}': "
                f"{sorted(extra_cols)}. Target columns: {target_cols}"
            )

        df_aligned = df.copy()
        for col in missing_cols:
            if fill_defaults:
                # Fill missing optional columns with None/NaN
                df_aligned[col] = None
            else:
                raise ValueError(
                    f"Required column '{col}' missing in DataFrame for table '{table_name}'"
                )

        # Reorder to match target schema
        df_aligned = df_aligned[target_cols]

        # Type coercion: ensure numeric columns are float
        for col in target_cols:
            if col in df_aligned.columns and df_aligned[col].dtype == 'object':
                try:
                    df_aligned[col] = pd.to_numeric(df_aligned[col], errors='ignore')
                except (ValueError, TypeError):
                    pass

        return df_aligned

    def insert_df(
        self, table_name: str, df: pd.DataFrame,
        mode: str = "append",
    ) -> None:
        """Insert a DataFrame into a table with explicit column mapping.

        Args:
            table_name: Target table name (must be in TABLE_DEFINITIONS or already exist)
            df: DataFrame to insert
            mode: 'append' (default) or 'replace'
        """
        with self._lock:
            conn = self.connect()

            if df.empty:
                logger.debug("insert_df skipped: empty DataFrame", table=table_name)
                return

            self._ensure_writable(f"insert_df → {table_name}")

            # Validate table exists
            if not self.table_exists(table_name):
                raise ValueError(
                    f"Table '{table_name}' does not exist. Cannot insert into non-existent table."
                )

            # Align DataFrame columns to target schema
            df_aligned = self._validate_and_align_df(table_name, df)

            if mode == "replace":
                conn.execute(f"DELETE FROM {table_name}")

            # Use explicit column list for safety
            cols = df_aligned.columns.tolist()
            col_list = ", ".join(cols)

            # Register temp table and insert with explicit columns
            verb = "INSERT OR REPLACE" if mode != "insert" else "INSERT"
            conn.register("__tmp_insert", df_aligned)
            try:
                conn.execute(
                    f"{verb} INTO {table_name} ({col_list}) "
                    f"SELECT {col_list} FROM __tmp_insert"
                )
            finally:
                conn.unregister("__tmp_insert")

        # (锁外读取路径已在 execute/query_df 加锁)
    def swap_table(self, table_name: str, df: pd.DataFrame) -> None:
        """Atomic table replacement using shadow table strategy.

        Creates _new table, inserts data with explicit column mapping,
        then renames for zero-downtime swap.
        """
        with self._lock:
            conn = self.connect()
            self._ensure_writable(f"swap_table → {table_name}")
            new_table = f"{table_name}_new"
            shadow_table = f"{table_name}_old"

            # Clean up any leftover shadow tables
            try:
                conn.execute(f"DROP TABLE IF EXISTS {new_table}")
                conn.execute(f"DROP TABLE IF EXISTS {shadow_table}")
            except Exception:
                pass

            # Get original DDL and create new table
            ddl_key = table_name.split(".")[-1]
            if ddl_key not in TABLE_DEFINITIONS:
                raise ValueError(f"No DDL defined for table '{ddl_key}'")
            ddl = TABLE_DEFINITIONS[ddl_key].replace(
                f"CREATE TABLE IF NOT EXISTS {ddl_key}",
                f"CREATE TABLE {new_table}",
            )
            conn.execute(ddl)

            if df.empty:
                logger.warning("swap_table called with empty DataFrame", table=table_name)
                # Still create the new table, then swap
                conn.execute(f"ALTER TABLE IF EXISTS {table_name} RENAME TO {shadow_table}")
                conn.execute(f"ALTER TABLE {new_table} RENAME TO {table_name}")
                conn.execute(f"DROP TABLE IF EXISTS {shadow_table}")
                return

            # Align and insert with explicit columns
            new_cols = self._get_table_columns(new_table)
            df_aligned = self._validate_and_align_df(ddl_key, df, fill_defaults=True)
            col_list = ", ".join(new_cols)

            conn.register("__swap_data", df_aligned)
            try:
                conn.execute(
                    f"INSERT INTO {new_table} ({col_list}) "
                    f"SELECT {col_list} FROM __swap_data"
                )
            finally:
                conn.unregister("__swap_data")

            # Atomic swap
            conn.execute(f"ALTER TABLE IF EXISTS {table_name} RENAME TO {shadow_table}")
            conn.execute(f"ALTER TABLE {new_table} RENAME TO {table_name}")
            conn.execute(f"DROP TABLE IF EXISTS {shadow_table}")

            logger.info("Table swapped atomically", table=table_name, rows=len(df_aligned))

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        with self._lock:
            conn = self.connect()
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return result[0] > 0 if result else False

    def row_count(self, table_name: str) -> int:
        """Get approximate row count for a table."""
        with self._lock:
            conn = self.connect()
        result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0] if result else 0

    def get_latest_date(self, table_name: str, date_col: str = "date") -> Optional[str]:
        """Get the latest date in a table."""
        with self._lock:
            conn = self.connect()
        if not self.table_exists(table_name):
            return None
        result = conn.execute(
            f"SELECT MAX({date_col}) FROM {table_name}",
        ).fetchone()
        return result[0] if result and result[0] else None
