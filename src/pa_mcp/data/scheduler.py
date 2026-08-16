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


class CapabilityMissingError(RuntimeError):
    """当前源链缺少某 phase 所需的数据能力（配置问题，非数据错误）。

    调度器将其标记为 SKIPPED_NOT_REQUIRED（不阻塞 pipeline），
    但会在日志中明确提示需要在 data_sources.sources 中加入对应源。
    """


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

    def __init__(self, store, akshare_adapter=None, retry_count: int = 3,
                 data_router=None) -> None:
        """Initialize scheduler.

        Args:
            store: DuckDBStore instance.
            akshare_adapter: Legacy direct adapter (kept for backward compat;
                             ignored when data_router is provided).
            retry_count: Retries per phase.
            data_router: DataSourceRouter for multi-source failover. When
                         provided, kline/stock-basic fetches go through the
                         router instead of the raw akshare adapter.
        """
        self._store = store
        self._akshare = akshare_adapter
        self._retry_count = retry_count
        self._router = data_router

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
            ("4_minute_kline", self._update_minute_kline, True),
            ("5_financials", self._update_financials, True),
            ("6_capital_flow", self._update_capital_flow, True),
            ("7_dragon_tiger", self._update_dragon_tiger, True),
            ("8_indicators", self._update_indicators, True),
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
            except CapabilityMissingError as e:
                # 源链缺能力 → SKIPPED（不阻塞），日志明确提示
                report.phases.append(PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED_NOT_REQUIRED,
                    error=str(e),
                ))
                logger.warning("Phase skipped: capability missing", phase=phase_name, error=str(e))
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
            except CapabilityMissingError:
                raise  # 配置缺能力 — 不重试，直接上抛给 run() 标记 SKIPPED
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
        """Update stock basic info — via router when available, else AKShare.

        Raises:
            AllSourcesFailedError: 所有源都失败（含"无能力"）
            RuntimeError: 无任何源可用
        """
        from pa_mcp.data.router import AllSourcesFailedError

        try:
            if self._router is not None:
                df, source_name = await self._router.fetch_realtime_spot_all()
            elif self._akshare is not None:
                df = await self._akshare.get_realtime_spot_all()
                source_name = "akshare"
            else:
                raise RuntimeError("No data source available for stock basic")

            if df is None or df.empty:
                logger.warning("Stock basic: source returned empty spot data", source=source_name)
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
            logger.info("Stock basic updated", rows=len(basic), source=source_name)
            return len(basic)
        except AllSourcesFailedError as e:
            # 链中所有源都不支持全市场快照（如仅 tencent/sina）→ 需 AKShare
            if all(v == "capability unsupported" for v in e.source_errors.values()):
                logger.warning(
                    "Stock basic: 当前源链无全市场快照能力，需在链中加入 AKShare",
                    sources=list(e.source_errors.keys()),
                )
                raise CapabilityMissingError(
                    "当前源链无全市场快照能力（需 AKShare）。"
                    "请将 akshare 加入 data_sources.sources 配置，"
                    "或手动维护 stock_basic。"
                ) from e
            raise  # 真实失败 — 这是 REQUIRED phase
        except Exception as e:
            logger.error("Stock basic update failed", error=str(e))
            raise  # Re-raise — this is a REQUIRED phase

    async def _update_daily_kline(self, force_full: bool) -> int:
        """Update daily kline for all stocks (incremental) — multi-source."""
        store = self._store
        if self._router is None and self._akshare is None:
            raise RuntimeError("No data source available — required for kline updates")

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
        source_stats: dict[str, int] = {}  # source -> rows served

        # 断点续传：跳过已完成的 symbol
        done = self._get_checkpoint("kline_daily")
        pending = [s for s in symbols if s not in done]
        if done:
            logger.info(
                "Checkpoint resume",
                done=len(done), pending=len(pending), total=len(symbols),
            )

        skipped = 0
        failed: list[str] = []
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            for sym in batch:
                try:
                    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
                    if self._router is not None:
                        df, source_name = await self._router.fetch_daily_kline(
                            symbol=sym, period="daily",
                            start_date=start_date, end_date=today,
                            adjust="qfq",
                        )
                        source_stats[source_name] = source_stats.get(source_name, 0) + len(df)
                    else:
                        df = await self._akshare.get_daily_kline(
                            symbol=sym, period="daily",
                            start_date=start_date, end_date=today,
                            adjust="qfq",
                        )
                    if not df.empty:
                        store.insert_df("kline_daily", self._to_table_df(df, "kline_daily"), mode="append")
                        updated += len(df)
                    # 记录 checkpoint（成功即记，允许断点续传）
                    self._set_checkpoint("kline_daily", sym)
                except Exception as e:
                    failed.append(sym)
                    logger.debug("Kline fetch failed for symbol", symbol=sym, error=str(e))

            batch_num = batch_start // batch_size + 1
            done_now = len(done) + batch_start + len(batch)
            pct = done_now / len(symbols) * 100 if symbols else 100
            logger.info(
                "Kline batch progress",
                batch=f"{batch_num}/{total_batches}",
                progress_pct=round(pct, 1),
                updated_so_far=updated,
                failed_so_far=len(failed),
            )
            await asyncio.sleep(1)  # Rate limiting

        coverage = round((len(symbols) - len(failed)) / len(symbols) * 100, 1) if symbols else 0
        logger.info(
            "Daily kline updated",
            stocks_updated=updated, source_stats=source_stats,
            coverage_pct=coverage, failed=len(failed),
        )
        return updated

    def _to_table_df(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Strip adapter metadata columns (source, price_adjust_mode, etc.)
        that are not part of the target table schema.

        Keeps DuckDBStore's strict extra-column rejection intact while
        allowing adapters to carry provenance metadata.
        """
        try:
            table_cols = set(self._store._get_table_columns(table_name))
        except Exception:
            return df
        keep = [c for c in df.columns if c in table_cols]
        return df[keep]

    # ---- 断点续传 Checkpoint ----

    def _ensure_checkpoint_table(self) -> None:
        self._store.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_checkpoint (
                job VARCHAR(50) NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job, symbol)
            )
        """)

    def _get_checkpoint(self, job: str) -> set[str]:
        try:
            self._ensure_checkpoint_table()
            df = self._store.query_df(
                "SELECT symbol FROM ingestion_checkpoint WHERE job = ?",
                [job],
            )
            return set(df["symbol"].tolist()) if not df.empty else set()
        except Exception:
            return set()

    def _set_checkpoint(self, job: str, symbol: str) -> None:
        try:
            self._ensure_checkpoint_table()
            self._store.execute(
                "INSERT OR REPLACE INTO ingestion_checkpoint (job, symbol) VALUES (?, ?)",
                [job, symbol],
            )
        except Exception:
            pass  # checkpoint 失败不阻断主流程

    def reset_checkpoint(self, job: str) -> int:
        """清空某任务的断点（全量重跑时调用）。返回删除行数。"""
        try:
            self._ensure_checkpoint_table()
            result = self._store.execute(
                "DELETE FROM ingestion_checkpoint WHERE job = ?", [job],
            )
            return result.fetchone()[0] if result else 0
        except Exception:
            return 0

    async def _update_minute_kline(self, force_full: bool) -> int:
        """Update intraday minute kline (5分钟) via Tencent.

        腾讯 m5 接口免费可用。仅更新近几个交易日（数据量大）。
        分钟数据入库 kline_minute 表（period='5'）。
        """
        from pa_mcp.data.sources.tencent_adapter import TencentAdapter

        if self._router is None:
            raise RuntimeError("Router required for minute kline update")

        try:
            basic_df = self._store.query_df(
                "SELECT symbol FROM stock_basic ORDER BY symbol LIMIT 100",
            )
            symbols = basic_df["symbol"].tolist() if not basic_df.empty else []
        except Exception:
            symbols = []

        if not symbols:
            logger.warning("No stocks in stock_basic, skipping minute kline")
            return 0

        done = self._get_checkpoint("kline_minute")
        pending = [s for s in symbols if s not in done]
        logger.info(
            "Minute kline start", total=len(symbols),
            done=len(done), pending=len(pending),
        )

        adapter = TencentAdapter()
        updated = 0
        try:
            for sym in pending:
                try:
                    df = await adapter.get_daily_kline(
                        symbol=sym, period="5", adjust="qfq",
                    )
                    if not df.empty:
                        # 标准化为 kline_minute 表
                        out = df.rename(columns={"date": "datetime"})
                        out["period"] = "5"
                        out = out.tail(240)  # 近2个交易日（5分钟×48×2≈240）
                        self._store.insert_df(
                            "kline_minute",
                            self._to_table_df(out, "kline_minute"),
                            mode="insert",
                        )
                        updated += len(out)
                    self._set_checkpoint("kline_minute", sym)
                except Exception as e:
                    logger.debug("Minute kline failed", symbol=sym, error=str(e)[:120])
                if len(pending) > 10:
                    await asyncio.sleep(0.5)
        finally:
            await adapter.close()

        logger.info("Minute kline updated", stocks_updated=updated)
        return updated

    # AKShare 财务摘要指标名 → financials_income 列
    FINANCIAL_INDICATOR_MAP = {
        "营业收入": "revenue",
        "营业总收入": "revenue",
        "净利润": "net_profit",
        "归母净利润": "net_profit_parent",
        "基本每股收益": "eps",
        "净资产收益率": "roe",
        "销售毛利率": "gross_margin",
        "销售净利率": "net_margin",
        "资产负债率": "debt_ratio",
    }

    async def _update_financials(self, force_full: bool) -> int:
        """Update financial statements via AKShare 财务摘要（宽表转置）。

        Only processes a capped universe per run (AKShare is slow);
        skip if refreshed within 30 days unless force_full.
        """
        from pa_mcp.data.sources.akshare_adapter import AKShareAdapter

        # 30 天内已刷新则跳过（除非 force_full）
        if not force_full:
            try:
                latest = self._store.get_latest_date("financials_income", "report_date")
                if latest:
                    age_days = (datetime.now().date() -
                                datetime.fromisoformat(str(latest)).date()).days
                    if age_days < 30:
                        logger.info("Financials fresh, skipping", age_days=age_days)
                        return 0
            except Exception:
                pass

        akshare = AKShareAdapter()
        try:
            basic_df = self._store.query_df(
                "SELECT symbol FROM stock_basic ORDER BY symbol LIMIT 50",
            )
            symbols = basic_df["symbol"].tolist() if not basic_df.empty else []
        except Exception:
            symbols = []

        if not symbols:
            logger.warning("No stocks in stock_basic, skipping financials")
            return 0

        updated = 0
        failed = 0
        for sym in symbols:
            try:
                raw = await akshare._call_with_retry(
                    akshare._get_ak().stock_financial_abstract, symbol=sym,
                )
                records = self._abstract_to_records(raw, sym)
                if records:
                    out = pd.DataFrame(records)
                    # 去重后入库（主键 symbol+report_date）
                    out = out.drop_duplicates(subset=["symbol", "report_date"])
                    self._store.insert_df(
                        "financials_income",
                        self._to_table_df(out, "financials_income"),
                        mode="insert",
                    )
                    updated += len(out)
            except Exception as e:
                failed += 1
                logger.debug("Financials failed", symbol=sym, error=str(e)[:120])
            await asyncio.sleep(0.3)

        logger.info("Financials updated", stocks_updated=updated, failed=failed)
        return updated

    @classmethod
    def _abstract_to_records(cls, raw: pd.DataFrame, symbol: str) -> list[dict]:
        """转置 AKShare 财务摘要宽表（指标×报告期）→ 每报告期一条记录。"""
        if raw is None or raw.empty:
            return []

        periods = [c for c in raw.columns if c not in ("选项", "指标")]
        periods = [p for p in periods if str(p).isdigit() or "-" in str(p)]
        if not periods:
            return []

        # AKShare abstract 列序：最新报告期在左 → 取前 8 期
        records = []
        for period in periods[:8]:
            p = str(period).strip()
            # 20260630 -> 2026-06-30（DATE 列格式）
            if len(p) == 8 and p.isdigit():
                p = f"{p[:4]}-{p[4:6]}-{p[6:]}"
            rec: dict = {"symbol": symbol, "report_date": p}
            for _, row in raw.iterrows():
                ind = str(row.get("指标", ""))
                col = cls.FINANCIAL_INDICATOR_MAP.get(ind)
                if col:
                    val = row.get(period)
                    try:
                        rec[col] = float(val) if val is not None and str(val) not in ("--", "nan", "") else None
                    except (ValueError, TypeError):
                        rec[col] = None
            records.append(rec)

        return records

    async def _update_capital_flow(self, force_full: bool) -> int:
        """Update daily fund flow (个股资金流) via EastMoney (限流 1.2s)."""
        from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter

        if self._router is None:
            raise RuntimeError("Router required for fund flow update")

        # 取股票池（最多处理全部，东财限流下会慢 — 分批+断点）
        try:
            basic_df = self._store.query_df(
                "SELECT symbol FROM stock_basic ORDER BY symbol",
            )
            symbols = basic_df["symbol"].tolist() if not basic_df.empty else []
        except Exception:
            symbols = []

        if not symbols:
            logger.warning("No stocks in stock_basic, skipping fund flow")
            return 0

        done = self._get_checkpoint("fund_flow")
        pending = [s for s in symbols if s not in done]
        logger.info(
            "Fund flow start",
            total=len(symbols), done=len(done), pending=len(pending),
        )

        adapter = EastMoneyAdapter()
        updated = 0
        failed = 0
        try:
            for i, sym in enumerate(pending):
                try:
                    df = await adapter.get_stock_fund_flow(sym, days=1)
                    if not df.empty:
                        # 只保留最新一行入库
                        df = df.tail(1)
                        self._store.insert_df(
                            "fund_flow_daily",
                            self._to_table_df(df, "fund_flow_daily"),
                            mode="insert",
                        )
                        updated += 1
                    self._set_checkpoint("fund_flow", sym)
                except Exception:
                    failed += 1
                if (i + 1) % 50 == 0:
                    logger.info(
                        "Fund flow progress",
                        done=len(done) + i + 1, total=len(symbols),
                        updated=updated, failed=failed,
                    )
                await asyncio.sleep(1.2)  # 东财限流
        finally:
            await adapter.close()

        logger.info(
            "Fund flow updated",
            stocks_updated=updated, failed=failed,
        )
        return updated

    async def _update_dragon_tiger(self, force_full: bool) -> int:
        """Update dragon-tiger board (龙虎榜) via EastMoney datacenter.

        Optional phase: EastMoney may be rate-limited; failure is tolerated.
        """
        from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter

        adapter = EastMoneyAdapter()
        try:
            await asyncio.sleep(1.2)  # 东财限流
            df = await adapter.get_dragon_tiger(page_size=50)
            if df is None or df.empty:
                logger.debug("Dragon-tiger: no data")
                return 0

            # 标准化为 dragon_tiger 表结构（seat 明细）
            rows = []
            for _, r in df.iterrows():
                reason = str(r.get("reason", ""))
                rows.append({
                    "trade_date": str(r.get("trade_date", ""))[:10],
                    "symbol": str(r.get("symbol", "")),
                    "seat_name": f"东财汇总-{reason[:20]}",
                    "buy_amount": float(r.get("buy_amount", 0) or 0),
                    "sell_amount": float(r.get("sell_amount", 0) or 0),
                    "net_amount": float(r.get("net_amount", 0) or 0),
                    "seat_type": "summary",
                    "reason": reason,
                })

            if not rows:
                return 0
            out = pd.DataFrame(rows)
            # 去重：同一 (trade_date, symbol, seat_name) 只保留一条（东财多原因上榜会重复）
            out = out.drop_duplicates(subset=["trade_date", "symbol", "seat_name"])
            self._store.insert_df("dragon_tiger", self._to_table_df(out, "dragon_tiger"),
                                  mode="insert")
            logger.info("Dragon-tiger updated", rows=len(out))
            return len(out)
        except Exception as e:
            logger.debug("Dragon-tiger not available (EastMoney波动)", error=str(e)[:120])
            return 0  # Optional — don't block the pipeline
        finally:
            await adapter.close()

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
    """Entry point for 'python -m pa_mcp.data.scheduler'.

    Usage:
        python -m pa_mcp.data.scheduler            # 增量（断点续传）
        python -m pa_mcp.data.scheduler --full     # 全量重跑（重置断点）
    """
    import argparse

    parser = argparse.ArgumentParser(description="PA_MCP 数据调度")
    parser.add_argument("--full", action="store_true",
                        help="全量重跑（重置断点续传）")
    args = parser.parse_args()

    from pa_mcp.config import get_settings
    from pa_mcp.data.source_factory import build_router
    from pa_mcp.data.sources.akshare_adapter import AKShareAdapter
    from pa_mcp.data.sources.sina_adapter import SinaAdapter
    from pa_mcp.data.store import DuckDBStore

    store = DuckDBStore()
    store.connect()
    settings = get_settings()

    # 多源容灾路由（配置驱动；eastmoney 1.2s 防封、ths 0.4s 限流）
    router = build_router(
        settings,
        min_source_interval={"eastmoney": 1.2, "ths": 0.4},
        existing={"akshare": AKShareAdapter(), "sina": SinaAdapter()},
    )

    scheduler = DataUpdateScheduler(store, data_router=router)

    if args.full:
        n = scheduler.reset_checkpoint("kline_daily")
        print(f"已重置断点（{n} 条）— 全量重跑")

    report = await scheduler.run(force_full=args.full)
    store.close()

    # Print source health for observability
    print("数据源健康:")
    for h in router.health_report():
        print(f"  {h['name']:<12s} {h['state']:<10s} ok={h['success_count']} fail={h['failure_count']} trips={h['trip_count']}")

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
