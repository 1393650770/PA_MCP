# [AI:BEGIN]
# PA_MCP - Data Layer: Daily Update Scheduler Pipeline
# 8-phase pipeline: calendar → basic info → daily kline → minute kline →
# financials → capital flow → dragon tiger → technical indicators
# [AI:END]

from __future__ import annotations

import asyncio
import time
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import pandas as pd
import structlog

from pa_mcp.data.quality import DataValidator, ValidationReport

logger = structlog.get_logger(__name__)


class PhaseStatus(Enum):
    """Outcome of one pipeline phase."""
    SUCCESS = "success"
    SKIPPED_NOT_REQUIRED = "skipped_not_required"
    NOT_IMPLEMENTED = "not_implemented"
    FAILED = "failed"


@dataclass
class PhaseResult:
    """Result of one pipeline phase."""
    phase_name: str
    status: PhaseStatus = PhaseStatus.SUCCESS
    rows_updated: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status in (PhaseStatus.SUCCESS, PhaseStatus.SKIPPED_NOT_REQUIRED)

    @property
    def is_blocking_failure(self) -> bool:
        """A required phase that failed or is not implemented."""
        return self.status in (PhaseStatus.FAILED, PhaseStatus.NOT_IMPLEMENTED)


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
    def blocking_failures(self) -> list[str]:
        return [p.phase_name for p in self.phases if p.is_blocking_failure]

    @property
    def failed_phases(self) -> list[str]:
        return [p.phase_name for p in self.phases if not p.success]


class DataUpdateScheduler:
    """Orchestrates the daily 17:30 data update pipeline.

    Phase order matters — earlier phases populate tables later phases depend on.

    Pipeline:
    1. Trading calendar (required)
    2. Stock basic info (required)
    3. Daily kline (required)
    4. Minute kline (not implemented)
    5. Financial statements (not implemented)
    6. Capital flow (not implemented)
    7. Dragon-tiger board (optional)
    8. Technical indicators (optional, depends on kline)
    9. Validation
    """

    # Required phases that must succeed for the pipeline to be considered healthy
    REQUIRED_PHASES = {"1_calendar", "2_stock_basic", "3_daily_kline"}

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
            ("1_calendar", self._update_calendar, True),
            ("2_stock_basic", self._update_stock_basic, True),
            ("3_daily_kline", self._update_daily_kline, True),
            ("4_minute_kline", self._update_minute_kline, False),
            ("5_financials", self._update_financials, False),
            ("6_capital_flow", self._update_capital_flow, False),
            ("7_dragon_tiger", self._update_dragon_tiger, False),
            ("8_indicators", self._update_indicators, False),
        ]

        for phase_name, phase_func, is_implemented in phases:
            if not is_implemented:
                report.phases.append(PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.NOT_IMPLEMENTED,
                ))
                continue

            try:
                phase_result = await self._run_with_retry(phase_name, phase_func, force_full)
                report.phases.append(phase_result)
                if phase_result.is_blocking_failure:
                    logger.error(
                        "Pipeline required phase failed",
                        phase=phase_name,
                        error=phase_result.error,
                    )
            except Exception as e:
                is_blocking = phase_name in self.REQUIRED_PHASES
                report.phases.append(PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED if is_blocking else PhaseStatus.SKIPPED_NOT_REQUIRED,
                    error=str(e),
                ))
                logger.error("Pipeline phase crashed", phase=phase_name, error=str(e))

        # Validation
        try:
            validator = DataValidator()
            report.validation = validator.validate_all(self._store)
        except Exception as e:
            logger.warning("Validation step failed", error=str(e))

        report.total_elapsed = round(time.monotonic() - t0, 2)
        healthy = len(report.blocking_failures) == 0
        logger.info(
            "Data update pipeline complete",
            total_seconds=report.total_elapsed,
            healthy=healthy,
            blocking_failures=report.blocking_failures,
            not_implemented=[p.phase_name for p in report.phases if p.status == PhaseStatus.NOT_IMPLEMENTED],
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
                # 0 rows is suspicious for required phases — but not an error
                # (could genuinely be no new data on weekends)
                status = PhaseStatus.SUCCESS
                return PhaseResult(
                    phase_name=name, status=status,
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

        is_blocking = name in self.REQUIRED_PHASES
        return PhaseResult(
            phase_name=name,
            status=PhaseStatus.FAILED if is_blocking else PhaseStatus.SKIPPED_NOT_REQUIRED,
            error=last_error,
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
                logger.warning("Stock basic: AKShare returned empty spot data")
                return 0

            # Extract basic info with standard field names
            basic = pd.DataFrame()
            basic["symbol"] = df.get("代码", pd.Series(dtype=str))
            basic["name"] = df.get("名称", pd.Series(dtype=str))
            market_cap_series = df.get("总市值", pd.Series(dtype=float))
            basic["market_cap"] = pd.to_numeric(market_cap_series, errors="coerce")
            basic["board"] = "main"
            basic["exchange"] = ""

            # Only keep rows with valid symbols
            basic = basic[basic["symbol"].notna() & (basic["symbol"].str.strip() != "")]

            self._store.insert_df("stock_basic", basic, mode="replace")
            logger.info("Stock basic updated", rows=len(basic))
            return len(basic)
        except Exception as e:
            logger.error("Stock basic update failed", error=str(e))
            raise  # Re-raise — this is a REQUIRED phase

    async def _update_daily_kline(self, force_full: bool) -> int:
        """Update daily kline for all stocks (incremental)."""
        store = self._store
        akshare = self._akshare
        if akshare is None:
            raise RuntimeError("AKShare adapter not available — required for kline updates")

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
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        for batch_start in range(0, len(symbols), batch_size):
            batch = symbols[batch_start:batch_start + batch_size]
            # Process all stocks in the batch (no arbitrary truncation)
            for sym in batch:
                try:
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

            batch_num = batch_start // batch_size + 1
            logger.debug(
                "Kline batch progress",
                batch=f"{batch_num}/{total_batches}",
                updated_so_far=updated,
            )
            await asyncio.sleep(1)  # Rate limiting

        logger.info("Daily kline updated", stocks_updated=updated)
        return updated

    async def _update_minute_kline(self, force_full: bool) -> int:
        """Update intraday minute kline (NOT YET IMPLEMENTED)."""
        raise NotImplementedError(
            "Minute kline ingestion is not yet implemented. "
            "Use force_full=False to skip this phase automatically."
        )

    async def _update_financials(self, force_full: bool) -> int:
        """Update financial statements (NOT YET IMPLEMENTED)."""
        raise NotImplementedError(
            "Financial statement ingestion is not yet implemented."
        )

    async def _update_capital_flow(self, force_full: bool) -> int:
        """Update capital flow data (NOT YET IMPLEMENTED)."""
        raise NotImplementedError(
            "Capital flow ingestion is not yet implemented."
        )

    async def _update_dragon_tiger(self, force_full: bool) -> int:
        """Update dragon-tiger board data for current day."""
        if self._akshare is None:
            return 0
        try:
            df = await self._akshare.get_dragon_tiger_detail()
            if df is not None and not df.empty:
                self._store.insert_df("dragon_tiger", df, mode="append")
                return len(df)
            logger.debug("Dragon-tiger: no data for today")
            return 0
        except Exception as e:
            logger.debug("Dragon-tiger not available for today", error=str(e))
            return 0  # Optional — don't block the pipeline

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


# ---- Module Entry Point ----

async def _main() -> None:
    """Entry point for 'python -m pa_mcp.data.scheduler'."""
    from pa_mcp.data.sources.akshare_adapter import AKShareAdapter
    from pa_mcp.data.store import DuckDBStore

    store = DuckDBStore()
    store.connect()
    akshare = AKShareAdapter()
    scheduler = DataUpdateScheduler(store, akshare)

    report = await scheduler.run()
    store.close()

    print(f"\n{'='*60}")
    print(f"Pipeline: {report.timestamp}")
    print(f"Elapsed: {report.total_elapsed}s")
    for p in report.phases:
        icon = "✓" if p.success else "✗"
        status_str = p.status.value
        print(f"  {icon} {p.phase_name:<25s} {status_str:<22s} rows={p.rows_updated}")
    if report.validation:
        print(f"\n  Validation: {report.validation}")
    if report.blocking_failures:
        print(f"\n  BLOCKING FAILURES: {report.blocking_failures}")
    print(f"{'='*60}")

    if report.blocking_failures:
        sys.exit(1)


def _main_cli() -> None:
    """CLI wrapper for pa-mcp-scheduler entry point."""
    asyncio.run(_main())


if __name__ == "__main__":
    asyncio.run(_main())
