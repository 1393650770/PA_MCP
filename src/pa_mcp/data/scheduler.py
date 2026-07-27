# [AI:BEGIN]
# PA_MCP - Data Layer: Daily Update Scheduler Pipeline
# 8-phase pipeline: calendar → basic info → daily kline → minute kline →
# financials → capital flow → dragon tiger → technical indicators
# [AI:END]

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from pa_mcp.data.quality import DataValidator, ValidationReport

logger = structlog.get_logger(__name__)


@dataclass
class PhaseResult:
    """Result of one pipeline phase."""
    phase_name: str
    success: bool
    rows_updated: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class PipelineReport:
    """Aggregated pipeline run report."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    phases: list[PhaseResult] = field(default_factory=list)
    total_elapsed: float = 0.0
    validation: Optional[ValidationReport] = None

    @property
    def all_success(self) -> bool:
        return all(p.success for p in self.phases)

    @property
    def failed_phases(self) -> list[str]:
        return [p.phase_name for p in self.phases if not p.success]


class DataUpdateScheduler:
    """Orchestrates the daily 17:30 data update pipeline.

    Phase order matters — earlier phases populate tables later phases depend on.

    Pipeline:
    1. Trading calendar
    2. Stock basic info (list of all stocks)
    3. Daily kline (price/volume data)
    4. Minute kline (intraday data)
    5. Financial statements (balance sheet, income, cash flow)
    6. Capital flow (main force, northbound)
    7. Dragon-tiger board
    8. Technical indicators (pre-computation)
    9. Validation
    """

    def __init__(self, store, akshare_adapter=None, retry_count: int = 3) -> None:
        self._store = store
        self._akshare = akshare_adapter
        self._retry_count = retry_count

    async def run(self, force_full: bool = False) -> PipelineReport:
        """Run the complete daily update pipeline.

        Args:
            force_full: If True, do full refresh instead of incremental update.
        """
        t0 = time.monotonic()
        report = PipelineReport()

        phases = [
            ("1_calendar", self._update_calendar),
            ("2_stock_basic", self._update_stock_basic),
            ("3_daily_kline", self._update_daily_kline),
            ("4_minute_kline", self._update_minute_kline),
            ("5_financials", self._update_financials),
            ("6_capital_flow", self._update_capital_flow),
            ("7_dragon_tiger", self._update_dragon_tiger),
            ("8_indicators", self._update_indicators),
        ]

        for phase_name, phase_func in phases:
            try:
                phase_result = await self._run_with_retry(phase_name, phase_func, force_full)
                report.phases.append(phase_result)
                if not phase_result.success:
                    logger.error(
                        "Pipeline phase failed",
                        phase=phase_name,
                        error=phase_result.error,
                    )
            except Exception as e:
                report.phases.append(PhaseResult(
                    phase_name=phase_name, success=False, error=str(e),
                ))
                logger.error("Pipeline phase crashed", phase=phase_name, error=str(e))

        # Validation
        validator = DataValidator()
        report.validation = validator.validate_all(self._store)

        report.total_elapsed = round(time.monotonic() - t0, 2)
        logger.info(
            "Data update pipeline complete",
            total_seconds=report.total_elapsed,
            all_success=report.all_success,
            failed=report.failed_phases,
        )
        return report

    async def _run_with_retry(
        self, name: str, func, force_full: bool,
    ) -> PhaseResult:
        """Execute a phase with retry logic."""
        last_error: Optional[str] = None
        for attempt in range(self._retry_count + 1):
            t0 = time.monotonic()
            try:
                rows = await func(force_full)
                elapsed = time.monotonic() - t0
                return PhaseResult(
                    phase_name=name, success=True,
                    rows_updated=rows,
                    elapsed_seconds=round(elapsed, 2),
                )
            except Exception as e:
                last_error = str(e)
                if attempt < self._retry_count:
                    backoff = 2 ** attempt + 1
                    logger.warning(
                        f"Phase {name} failed, retrying",
                        attempt=attempt + 1, backoff_seconds=backoff,
                    )
                    await asyncio.sleep(backoff)

        return PhaseResult(
            phase_name=name, success=False, error=last_error,
        )

    # ---- Phase Implementations ----
    # Each returns the number of rows updated.

    async def _update_calendar(self, force_full: bool) -> int:
        """Update trading calendar."""
        # For now, use the built-in calendar which has hardcoded holidays.
        # In production, fetch from AKShare: ak.tool_trade_date_hist_sina()
        from pa_mcp.backtest.calendar import get_calendar
        cal = get_calendar()
        today = datetime.now().date()
        start = today - timedelta(days=365)

        import pandas as pd
        rows = []
        current = start
        while current <= today:
            rows.append({
                "date": current.isoformat(),
                "is_trading_day": cal.is_trading_day(current),
                "week_day": current.weekday(),
                "holiday_name": "",
            })
            current += timedelta(days=1)

        df = pd.DataFrame(rows)
        self._store.insert_df("trade_calendar", df, mode="replace")
        logger.info("Calendar updated", rows=len(df))
        return len(df)

    async def _update_stock_basic(self, force_full: bool) -> int:
        """Update stock basic info from AKShare."""
        if self._akshare is None:
            return 0

        try:
            df = await self._akshare.get_realtime_spot_all()
            if df is None or df.empty:
                return 0

            # Extract basic info
            basic = pd.DataFrame()
            basic["symbol"] = df.get("代码", pd.Series(dtype=str))
            basic["name"] = df.get("名称", pd.Series(dtype=str))
            basic["market_cap"] = df.get("总市值", pd.Series(dtype=float))
            basic["board"] = "main"

            self._store.insert_df("stock_basic", basic, mode="replace")
            return len(basic)
        except Exception as e:
            logger.warning("Stock basic update skipped (AKShare may need network)", error=str(e))
            return 0

    async def _update_daily_kline(self, force_full: bool) -> int:
        """Update daily kline for all stocks (incremental)."""
        store = self._store
        akshare = self._akshare
        if akshare is None:
            return 0

        today = datetime.now().strftime("%Y%m%d")

        # Get list of all stocks
        try:
            basic_df = store.query_df("SELECT symbol FROM stock_basic LIMIT 5200")
            symbols = basic_df["symbol"].tolist() if not basic_df.empty else []
        except Exception:
            symbols = []

        if not symbols:
            logger.warning("No stocks in stock_basic, skipping kline update")
            return 0

        updated = 0
        batch_size = 50

        for batch_start in range(0, len(symbols), batch_size):
            batch = symbols[batch_start:batch_start + batch_size]
            for sym in batch[:10]:  # Limit to 10 stocks per run to avoid rate limiting
                try:
                    # Incremental: fetch last 30 days
                    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
                    df = await akshare.get_daily_kline(
                        symbol=sym, period="daily",
                        start_date=start_date, end_date=today,
                        adjust="qfq",
                    )
                    if not df.empty:
                        store.insert_df("kline_daily", df, mode="append")
                        updated += len(df)
                except Exception as e:
                    logger.debug("Kline fetch failed for symbol", symbol=sym, error=str(e))

            await asyncio.sleep(1)  # Rate limiting

        logger.info("Daily kline updated", stocks_updated=updated)
        return updated

    async def _update_minute_kline(self, force_full: bool) -> int:
        """Update intraday minute kline (only for current day)."""
        # Minute kline is large and expensive — only fetch if explicitly requested
        if not force_full:
            logger.debug("Skipping minute kline (incremental mode, use force_full for intraday)")
            return 0
        return 0  # Stub — implement when needed

    async def _update_financials(self, force_full: bool) -> int:
        """Update financial statements (quarterly, incremental)."""
        # Financial data updates quarterly, not daily.
        # Check if we need to refresh (last update > 30 days ago)
        latest = self._store.get_latest_date("financials_income", "pub_date")
        if latest and (datetime.now() - datetime.fromisoformat(str(latest))).days < 30:
            logger.debug("Financials up to date, skipping")
            return 0

        logger.info("Financial data refresh needed (last: %s)", latest)
        # Stub — AKShare financial endpoints to be called here
        return 0

    async def _update_capital_flow(self, force_full: bool) -> int:
        """Update capital flow data for current day."""
        if self._akshare is None:
            return 0
        # Stub — requires AKShare fund flow endpoints
        return 0

    async def _update_dragon_tiger(self, force_full: bool) -> int:
        """Update dragon-tiger board data for current day."""
        if self._akshare is None:
            return 0
        try:
            df = await self._akshare.get_dragon_tiger_detail()
            if df is not None and not df.empty:
                self._store.insert_df("dragon_tiger", df, mode="append")
                return len(df)
        except Exception as e:
            logger.debug("Dragon-tiger not available for today", error=str(e))
        return 0

    async def _update_indicators(self, force_full: bool) -> int:
        """Pre-compute technical indicators for all stocks."""
        store = self._store

        try:
            # DuckDB SQL for batch indicator computation
            result = store.execute("""
                INSERT OR REPLACE INTO indicator_cache
                SELECT
                    symbol, date,
                    AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ma5,
                    AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as ma10,
                    AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20,
                    AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) as ma60,
                    AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW) as ma120,
                    AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) as ma250,
                    NULL as ema12, NULL as ema26,
                    NULL as macd, NULL as macd_signal, NULL as macd_hist,
                    NULL as rsi6, NULL as rsi14, NULL as rsi24,
                    NULL as kdj_k, NULL as kdj_d, NULL as kdj_j,
                    NULL as boll_upper, NULL as boll_mid, NULL as boll_lower,
                    NULL as atr14, NULL as obv
                FROM kline_daily
                WHERE date >= CURRENT_DATE - INTERVAL '270 days'
            """)
            row_count = result.fetchone()[0] if result else 0
            logger.info("Indicators pre-computed", rows=row_count)
            return row_count or 0
        except Exception as e:
            logger.warning("Indicator pre-computation skipped", error=str(e))
            return 0
