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
from datetime import date, datetime, timedelta
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
            ("9_index_daily", self._update_index_daily, True),
            ("10_sentiment", self._update_sentiment, True),
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

            # 保留已有行业映射（sector 是合成板块/部分选股依赖；快照源如
            # 新浪不提供行业字段，replace 会清空 → 先 merge 再写入）
            try:
                old = self._store.query_df(
                    "SELECT symbol, sector FROM stock_basic WHERE sector != ''", [])
                if not old.empty:
                    sector_map = dict(zip(old["symbol"], old["sector"]))
                    basic["sector"] = basic["symbol"].map(
                        lambda s: sector_map.get(s, ""))
            except Exception:
                basic["sector"] = ""

            basic = self._enrich_stock_basic(basic, df)

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

    # 快照源（新浪/腾讯等）通常只给 代码/名称/市值，不带行业、上市日期、
    # ST 标记、停牌标记。而 insert_df(mode="replace") 会把未提供的列对齐成
    # NULL —— 于是每次日更都会把 industry/market_cap/list_date/is_st/
    # is_suspended 清空（生产库曾 5554 行 100% 为 NULL），直接导致：
    #   · ST/退市股无法识别（下单无拦截）
    #   · 停牌股无法识别
    #   · 市值筛选（scan_volume_surge.market_cap_min）失效
    #   · 行业敞口风控（max_sector_exposure）无法计算
    # 这里做两件事：1) 从 名称/代码 派生可零成本得到的字段；
    #             2) 源没给、但库里已有的值一律保留，不被日更抹掉。
    _ST_MARKERS = ("ST", "退")

    def _enrich_stock_basic(self, basic: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        """补齐/保留 stock_basic 的关键字段，避免每日 replace 造成的数据丢失。"""
        if basic.empty:
            return basic

        sym = basic["symbol"].astype(str).str.strip()
        name = basic.get("name", pd.Series([""] * len(basic))).astype(str).fillna("")

        # ---- 1) 由代码/名称零成本派生 ----
        # is_st：A 股 ST/*ST/退市整理 在股票简称里体现，可用于风控拦截
        upper = name.str.upper()
        basic["is_st"] = upper.str.contains("ST", regex=False) | upper.str.contains("退", regex=False)

        # exchange / board：由代码前缀判定（6/9→SH，0/3→SZ，8/4→BJ）
        def _exchange(s: str) -> str:
            if s.startswith(("6", "9")):
                return "SH"
            if s.startswith(("0", "3")):
                return "SZ"
            if s.startswith(("8", "4")):
                return "BJ"
            return ""

        def _board(s: str) -> str:
            if s.startswith("688") or s.startswith("689"):
                return "star"      # 科创板
            if s.startswith("30"):
                return "chinext"   # 创业板
            if s.startswith(("8", "4")):
                return "bse"       # 北交所
            return "main"

        basic["exchange"] = sym.map(_exchange)
        basic["board"] = sym.map(_board)

        # 停牌：快照无成交（volume=0 / 最新价为 0）通常意味着停牌
        suspended = pd.Series(False, index=basic.index)
        for col in ("成交量", "volume", "最新价", "trade"):
            if raw is not None and col in getattr(raw, "columns", []):
                vals = pd.to_numeric(raw[col], errors="coerce")
                vals = vals.reindex(basic.index) if len(vals) == len(basic) else vals
                suspended = suspended | (vals.fillna(0) == 0)
                break
        basic["is_suspended"] = suspended

        # ---- 2) 源未提供但库里已有的字段：保留旧值，绝不抹成 NULL ----
        preserve_cols = ("industry", "list_date", "market_cap", "delist_date")
        need_preserve = [
            c for c in preserve_cols
            if c not in basic.columns or basic[c].isna().all()
        ]
        if need_preserve:
            try:
                cols_sql = ", ".join(["symbol"] + need_preserve)
                old = self._store.query_df(
                    f"SELECT {cols_sql} FROM stock_basic", [])
                if not old.empty:
                    for c in need_preserve:
                        old_c = old[["symbol", c]].dropna()
                        old_c = old_c[old_c[c].astype(str).str.strip() != ""]
                        if old_c.empty:
                            continue
                        mapping = dict(zip(old_c["symbol"].astype(str), old_c[c]))
                        cur = basic[c] if c in basic.columns else pd.Series(
                            [None] * len(basic), index=basic.index)
                        basic[c] = [
                            (v if (v is not None and not pd.isna(v)) else mapping.get(s))
                            for s, v in zip(sym, cur)
                        ]
            except Exception as e:
                logger.debug("stock_basic preserve skipped", error=str(e))

        return basic

    def _incremental_universe(self, store, cap: int = 1200) -> list[str]:
        """增量更新的标的池。

        只取 kline_daily 已有标的会形成「鸡生蛋」：新加自选/持仓的标的、
        以及全新安装的库，因为没有历史 K 线就永远不会被拉取。

        因此取并集（并保持优先级顺序，保证限量时研究标的优先）：
          1. 库内已有 K 线的标的（继续增量续拉）
          2. 自选股 watchlist
          3. 持仓 portfolio
          4. stock_basic 中的研究标的（受 cap 限制，避免拖垮 cron）
        """
        symbols: list[str] = []
        seen: set[str] = set()

        def _add(rows) -> None:
            for s in rows:
                s = str(s).strip()
                if s and s not in seen:
                    seen.add(s)
                    symbols.append(s)

        queries = [
            "SELECT DISTINCT symbol FROM kline_daily",
            "SELECT DISTINCT symbol FROM watchlist",
            "SELECT DISTINCT symbol FROM portfolio",
        ]
        for q in queries:
            try:
                df = store.query_df(q)
                if not df.empty:
                    _add(df["symbol"].tolist())
            except Exception:
                continue  # 表不存在时跳过

        # stock_basic 兜底：仅在池子很小（如全新安装）时补充，避免全市场拖死
        if len(symbols) < cap:
            try:
                df = store.query_df(
                    f"SELECT symbol FROM stock_basic LIMIT {cap - len(symbols)}")
                if not df.empty:
                    _add(df["symbol"].tolist())
            except Exception:
                pass

        if symbols:
            logger.info("Incremental kline universe", count=len(symbols))
        return symbols

    def _kline_latest_dates(self, table: str = "kline_daily") -> dict[str, date]:
        """{symbol: 库内最新交易日}。

        用于动态计算增量起点：固定 30 天窗口只能补「近一个月内」的缺口，
        一旦某只标的落后更久（源抖动/停牌/新加自选），就永远补不回来。
        """
        try:
            df = self._store.query_df(
                f"SELECT symbol, MAX(date) AS d FROM {table} GROUP BY symbol")
            if df.empty:
                return {}
            return {str(s): self._as_date(d) for s, d in zip(df["symbol"], df["d"])}
        except Exception:
            return {}

    @classmethod
    def _df_max_date(cls, df: Optional[pd.DataFrame], col: str = "date") -> Optional[date]:
        """取 DataFrame 某日期列的最大值（用于写 checkpoint 的 last_date）。"""
        if df is None or df.empty or col not in getattr(df, "columns", []):
            return None
        try:
            return cls._as_date(pd.to_datetime(df[col], errors="coerce").max())
        except Exception:
            return None

    async def _update_daily_kline(self, force_full: bool) -> int:
        """Update daily kline for all stocks (incremental) — multi-source."""
        store = self._store
        if self._router is None and self._akshare is None:
            raise RuntimeError("No data source available — required for kline updates")

        # 写锁预检：拿不到写锁就明确失败，绝不静默空跑整个标的池。
        # 机器休眠后 OpenClaw 会把错过的任务一次性补跑，多个 isolated 会话
        # 各起一个 MCP Server 抢同一把 DuckDB 排他锁，后到者降级只读 ——
        # 于是 1096 只标的被逐个 no-op 一遍，phase 报 success/0 行，
        # 表现为「cron 状态 ok，数据却一天没动」。
        if not store.ensure_writable(self.WRITE_LOCK_WAIT_SECONDS):
            raise RuntimeError(
                f"DuckDB 写锁被其他进程占用超过 "
                f"{self.WRITE_LOCK_WAIT_SECONDS:.0f}s，本次放弃（数据未更新，"
                f"请关闭占用进程或错峰重试）"
            )

        today = datetime.now().strftime("%Y%m%d")

        # 标的清单：
        #   force_full=True  → stock_basic 全市场（~5500 只，全量重跑用 CLI）
        #   增量（默认）     → 见 _incremental_universe()：
        #     kline 已有 ∪ 自选 ∪ 持仓 ∪ stock_basic 里的研究标的。
        #     原因：全市场 5544 只增量拉取 >9 分钟，MCP 工具/OpenClaw cron
        #     必然超时中断（曾导致数据停在 08-24 而 cron 状态 ok 的静默失败）。
        #     注意：早期实现只取 kline_daily 已有标的，形成鸡生蛋问题——
        #     新加入自选/组合的标的、以及全新安装的库永远拿不到 K 线。
        try:
            if force_full:
                basic_df = store.query_df(
                    "SELECT symbol FROM stock_basic LIMIT 5200")
                symbols = basic_df["symbol"].tolist() if not basic_df.empty else []
            else:
                symbols = self._incremental_universe(store)
        except Exception:
            symbols = []

        if not symbols:
            logger.warning("No symbols to update, skipping kline update")
            return 0
        logger.info("Kline update targets", count=len(symbols),
                    mode="full" if force_full else "incremental(db)")

        # ---- 按 checkpoint(日期维度) 拆分待拉清单 ----
        # 旧逻辑「跑过就跳过」导致一轮之后 pending 恒为空、增量形同虚设；
        # 现在只有 last_date >= 目标交易日 才跳过。
        target = self._target_trade_date()
        latest = self._kline_latest_dates()
        # 库内完全无 K 线的标的（新加自选/持仓、或曾拉空留下断点）强制重拉
        always = {s for s in symbols if s not in latest}
        pending, skipped = self._split_pending(
            "kline_daily", symbols, target, always=always)
        logger.info(
            "Kline checkpoint",
            target=target.isoformat(), pending=len(pending),
            skipped=skipped, no_data_forced=len(always),
        )

        updated = 0
        empty = 0
        batch_size = 50
        total_batches = (len(pending) + batch_size - 1) // batch_size or 1
        source_stats: dict[str, int] = {}  # source -> rows served
        failed: list[str] = []
        consecutive_failed = 0
        read_only_abort = False

        if not pending:
            logger.info("Kline already up to date", target=target.isoformat())
            return 0

        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            for sym in batch:
                try:
                    # 增量起点按库内最新日期动态计算：落后多久都能补回
                    # （固定 30 天窗口曾让落后标的永远补不上）
                    base = latest.get(sym)
                    if base is None:
                        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
                    else:
                        # 回退 7 天做重叠：qfq 复权因子重算会改写近期行
                        start_date = (base - timedelta(days=7)).strftime("%Y%m%d")

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

                    max_d = self._df_max_date(df)
                    if df is not None and not df.empty:
                        store.insert_df("kline_daily", self._to_table_df(df, "kline_daily"), mode="append")
                        updated += len(df)
                        if max_d:
                            latest[sym] = max(latest[sym], max_d) if latest.get(sym) else max_d
                    else:
                        # 拉空（停牌/退市/源无此标的）：记 last_date=目标日，
                        # 当天不重试；次日目标日推进后会自然重试。
                        empty += 1
                    # 只有「没抛异常」才写点 —— 失败必须留待下次重试
                    self._set_checkpoint("kline_daily", sym, max_d or target)
                    consecutive_failed = 0
                except Exception as e:
                    failed.append(sym)
                    consecutive_failed += 1
                    err = str(e)
                    if "只读降级" in err or "READ-ONLY" in err.upper():
                        # 运行中掉回只读：再跑下去全是 no-op，直接停
                        logger.error(
                            "Kline aborted: store degraded to read-only",
                            symbol=sym, err=err[:160],
                        )
                        read_only_abort = True
                        break
                    logger.debug("Kline fetch failed for symbol",
                                 symbol=sym, error=str(e))
                    if consecutive_failed >= self.MAX_CONSECUTIVE_FAILURES:
                        logger.warning(
                            "Kline source down, aborting phase",
                            consecutive_failed=consecutive_failed,
                            processed=batch_start + batch.index(sym) + 1,
                        )
                        break

            if read_only_abort:
                break

            if consecutive_failed >= self.MAX_CONSECUTIVE_FAILURES:
                break

            batch_num = batch_start // batch_size + 1
            pct = (batch_start + len(batch)) / len(pending) * 100
            logger.info(
                "Kline batch progress",
                batch=f"{batch_num}/{total_batches}",
                progress_pct=round(pct, 1),
                updated_so_far=updated, empty_so_far=empty,
                failed_so_far=len(failed),
            )
            await asyncio.sleep(1)  # Rate limiting

        coverage = round((len(pending) - len(failed)) / len(pending) * 100, 1)
        logger.info(
            "Daily kline updated",
            target=target.isoformat(), pending=len(pending), skipped=skipped,
            rows=updated, empty=empty, source_stats=source_stats,
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
    #
    # 语义（2026-09-08 修正）：checkpoint 记录「该 job 下每只标的已拉到哪个
    # 交易日 (last_date)」，而不是「这只标的曾经成功过」。
    #
    # 旧实现只存 (job, symbol)，两个致命后果：
    #   · 跑完一轮后 pending 永远为空 → 增量更新形同虚设，数据永久停更
    #     （生产库曾出现：85 只权重股 K 线停在 08-19，而 checkpoint 显示
    #       全部「09-03 已完成」）；
    #   · fetch 返回空（源抖动/停牌）也照样写点 → 静默丢失且次日不重试。
    #
    # 新语义：
    #   · 仅当 last_date >= 目标交易日 才跳过该标的；
    #   · 拉空时写 last_date=目标日（当天不重试，次日目标日推进后自然重试）；
    #   · 拉失败（抛异常）绝不写点 —— 下次必须重试。

    # 连续失败熔断阈值（源整体挂掉时避免全池空转拖死 pipeline）
    MAX_CONSECUTIVE_FAILURES = 20

    # 写锁等待预算（秒）。DuckDB 单文件排他锁：多个 cron 会话（尤其是
    # 机器休眠后所有错过任务被同时补跑）各起一个 MCP Server 时，后到者会
    # 降级成只读 —— 此时继续跑只会把上千只标的逐个空跑。宁可等，拿不到
    # 就明确失败并让调用方错峰重试。
    WRITE_LOCK_WAIT_SECONDS = 300.0

    # 东财资金流每次往回拉几个交易日（顺带回填近期缺口）
    FUND_FLOW_LOOKBACK_DAYS = 5

    # 指数日线：pipeline 会持续维护的主要指数
    INDEX_SYMBOLS = ("sh000001", "sz399001", "sz399006")

    # 指数最低合理点位：低于此值视为「个股价格冒充指数」（如 sh000001
    # 被写成平安银行的 11.8 元），写入前丢弃。
    INDEX_MIN_CLOSE = 100.0

    def _ensure_checkpoint_table(self) -> None:
        self._store.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_checkpoint (
                job VARCHAR(50) NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_date DATE,
                PRIMARY KEY (job, symbol)
            )
        """)
        # 老库迁移：补 last_date 列（DuckDB 支持 IF NOT EXISTS）
        try:
            self._store.execute(
                "ALTER TABLE ingestion_checkpoint "
                "ADD COLUMN IF NOT EXISTS last_date DATE"
            )
        except Exception:
            pass

    @staticmethod
    def _as_date(value: Any) -> Optional[date]:
        """把 DuckDB/ pandas 返回的日期值统一成 datetime.date。"""
        if value is None:
            return None
        try:
            # NaT / NaN 必须先判——pd.NaT 是 datetime 子类，先走 isinstance
            # 会拿到一个非法 date
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    def _get_checkpoint(self, job: str) -> dict[str, Optional[date]]:
        """返回 {symbol: 已拉到的最新交易日}（值为 None = 拉过但无数据）。"""
        try:
            self._ensure_checkpoint_table()
            df = self._store.query_df(
                "SELECT symbol, last_date FROM ingestion_checkpoint WHERE job = ?",
                [job],
            )
            if df.empty:
                return {}
            return {
                str(s): self._as_date(d)
                for s, d in zip(df["symbol"], df["last_date"])
            }
        except Exception:
            return {}

    def _set_checkpoint(self, job: str, symbol: str,
                        last_date: Optional[date] = None) -> None:
        """记录某标的已拉到 last_date。失败路径请勿调用。"""
        try:
            self._ensure_checkpoint_table()
            self._store.execute(
                "INSERT OR REPLACE INTO ingestion_checkpoint "
                "(job, symbol, last_date) VALUES (?, ?, ?)",
                [job, symbol, last_date],
            )
        except Exception:
            pass  # checkpoint 失败不阻断主流程（退化为「无断点」，可重跑）

    def _target_trade_date(self) -> date:
        """本轮要拉到的目标交易日 = 最近一个「已收盘」的交易日。

        收盘前（15:00 前）不把今天算作目标日——当日行情尚未定稿，
        否则会把 last_date 记成今天而漏掉真正需要补的数据。
        """
        today = datetime.now().date()
        target: Optional[date] = None
        try:
            df = self._store.query_df(
                "SELECT MAX(date) FROM trade_calendar "
                "WHERE is_trading_day AND date <= ?",
                [today],
            )
            if not df.empty:
                target = self._as_date(df.iloc[0, 0])
        except Exception:
            target = None
        if target is None:
            target = today

        if target >= today and datetime.now().hour < 15:
            try:
                prev = self._store.query_df(
                    "SELECT MAX(date) FROM trade_calendar "
                    "WHERE is_trading_day AND date < ?",
                    [today],
                )
                if not prev.empty:
                    p = self._as_date(prev.iloc[0, 0])
                    if p is not None:
                        target = p
            except Exception:
                pass
        return target

    def _split_pending(self, job: str, symbols: list[str], target: date,
                       always: Optional[set[str]] = None,
                       ) -> tuple[list[str], int]:
        """按 checkpoint 拆分待拉清单。

        Args:
            always: 无视 checkpoint 强制拉取的标的集（如库内完全无 K 线的
                    自选/持仓股——它们的 checkpoint 可能是「拉空」留下的）。

        Returns:
            (待拉清单, 已跳过数量)
        """
        done = self._get_checkpoint(job)
        always = always or set()
        pending: list[str] = []
        skipped = 0
        for s in symbols:
            last = done.get(s)
            if s in always or last is None or last < target:
                pending.append(s)
            else:
                skipped += 1
        return pending, skipped

    def sync_checkpoint_from_data(self, job: str, table: str,
                                  date_col: str = "date") -> int:
        """用目标表现有数据回填 checkpoint.last_date。

        用途：升级到「日期维度 checkpoint」后的一次性迁移 —— 否则老库的
        last_date 全为 NULL，会导致下一轮把全池重拉一遍（正确但极慢）。

        返回被更新的行数。
        """
        try:
            self._ensure_checkpoint_table()
            self._store.execute(
                f"UPDATE ingestion_checkpoint c SET last_date = "
                f"(SELECT MAX(t.{date_col}) FROM {table} t WHERE t.symbol = c.symbol) "
                f"WHERE c.job = ?",
                [job],
            )
            # 有数据但没断点的标的（如新加的自选股）：补一条断点，避免首轮全拉
            self._store.execute(
                f"INSERT OR REPLACE INTO ingestion_checkpoint (job, symbol, last_date) "
                f"SELECT ?, symbol, MAX({date_col}) FROM {table} GROUP BY symbol",
                [job],
            )
            df = self._store.query_df(
                "SELECT COUNT(*) FROM ingestion_checkpoint WHERE job = ? AND last_date IS NOT NULL",
                [job],
            )
            return int(df.iloc[0, 0]) if not df.empty else 0
        except Exception as e:
            logger.warning("Checkpoint sync failed", job=job, error=str(e))
            return 0

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

    def purge_corrupt_index_rows(
        self, min_close: Optional[float] = None,
    ) -> int:
        """清理 index_daily 里「点位像个股价格」的脏行，返回删除行数。

        历史污染：astock/百度源把 sh000001 剥前缀成 000001，返回平安银行
        K 线并以 sh000001 落库（2026-08 起）。这些行的 close 只有 10~12 元，
        与真实指数（3000+）差两个数量级，用点位下限即可安全识别。
        """
        floor = self.INDEX_MIN_CLOSE if min_close is None else float(min_close)
        try:
            before = self._store.execute(
                "SELECT COUNT(*) FROM index_daily WHERE close < ?", [floor],
            ).fetchone()[0]
            if not before:
                return 0
            self._store.execute(
                "DELETE FROM index_daily WHERE close < ?", [floor])
            logger.warning("Purged corrupt index rows",
                           rows=int(before), min_close=floor)
            return int(before)
        except Exception as e:
            logger.warning("purge_corrupt_index_rows failed", error=str(e)[:160])
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

        # 分钟线是当日盘口数据：目标日就是今天，last_date < 今天即重拉
        target = datetime.now().date()
        pending, skipped = self._split_pending("kline_minute", symbols, target)
        logger.info(
            "Minute kline start", total=len(symbols),
            skipped=skipped, pending=len(pending),
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
                            mode="append"  # 幂等 upsert（主键冲突自动覆盖）,
                        )
                        updated += len(out)
                    self._set_checkpoint("kline_minute", sym, target)
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
                        mode="append"  # 幂等 upsert（主键冲突自动覆盖）,
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

        # 取股票池：force_full → 全市场；增量 → 库内已有（东财限流 1.2s/只，
        # 全市场 5554 只 ≈ 2 小时必然拖死 cron；库内 ~900 只 ≈ 18 分钟可完成）
        try:
            if force_full:
                basic_df = self._store.query_df(
                    "SELECT symbol FROM stock_basic ORDER BY symbol")
                symbols = basic_df["symbol"].tolist() if not basic_df.empty else []
            else:
                symbols = self._incremental_universe(self._store)
        except Exception:
            symbols = []

        if not symbols:
            logger.warning("No symbols to update, skipping fund flow")
            return 0

        target = self._target_trade_date()
        pending, skipped = self._split_pending("fund_flow", symbols, target)
        logger.info(
            "Fund flow start", target=target.isoformat(),
            total=len(symbols), skipped=skipped, pending=len(pending),
        )

        adapter = EastMoneyAdapter()
        updated = 0
        empty = 0
        failed = 0
        consecutive_failed = 0
        try:
            for i, sym in enumerate(pending):
                try:
                    # 多取几天：既补当日，也顺带回填前几天可能的缺口
                    df = await adapter.get_stock_fund_flow(sym, days=self.FUND_FLOW_LOOKBACK_DAYS)
                    max_d = self._df_max_date(df, "trade_date")
                    if not df.empty:
                        self._store.insert_df(
                            "fund_flow_daily",
                            self._to_table_df(df, "fund_flow_daily"),
                            mode="append",  # 幂等 upsert（主键冲突自动覆盖）
                        )
                        updated += len(df)
                    else:
                        empty += 1
                    consecutive_failed = 0
                    # 与 kline 同款语义：只有没抛异常才写点；拉空记目标日，
                    # 当日不重试、次日目标日推进后自然重试。
                    self._set_checkpoint("fund_flow", sym, max_d or target)
                except Exception:
                    failed += 1
                    consecutive_failed += 1
                    if consecutive_failed >= 15:
                        # 东财断连熔断：连续 15 只失败即停（避免 5554 只
                        # 空转 2 小时拖死 pipeline——之前静默失败的元凶）
                        logger.warning(
                            "Fund flow source down, aborting phase",
                            consecutive_failed=consecutive_failed,
                            processed=i + 1,
                        )
                        break
                if (i + 1) % 50 == 0:
                    logger.info(
                        "Fund flow progress",
                        done=i + 1, total=len(pending),
                        updated=updated, empty=empty, failed=failed,
                    )
                await asyncio.sleep(1.2)  # 东财限流
        finally:
            await adapter.close()

        logger.info(
            "Fund flow updated",
            stocks_updated=updated, empty=empty, failed=failed,
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
                                  mode="append")  # 幂等 upsert（主键冲突自动覆盖）
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

    async def _update_index_daily(self, force_full: bool) -> int:
        """维护主要指数日线（index_daily）。

        此前 pipeline 根本没有这一 phase —— index_daily 只在 readiness
        被研究工具触发时「缺数据才拉一次」，于是长期停在 08-14。
        纳入日常调度后按库内最新日期增量续拉。
        """
        if self._router is None:
            raise RuntimeError("Router required for index daily update")

        target = self._target_trade_date()
        latest: dict[str, Optional[date]] = {}
        try:
            df = self._store.query_df(
                "SELECT symbol, MAX(date) AS d FROM index_daily GROUP BY symbol")
            if not df.empty:
                latest = {str(s): self._as_date(d)
                          for s, d in zip(df["symbol"], df["d"])}
        except Exception:
            latest = {}

        today = datetime.now().strftime("%Y%m%d")
        updated = 0
        for sym in self.INDEX_SYMBOLS:
            base = latest.get(sym)
            if base is not None and base >= target:
                continue
            start_date = (
                (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
                if base is None
                else (base - timedelta(days=7)).strftime("%Y%m%d")
            )
            try:
                df, _src = await self._router.fetch_daily_kline(
                    symbol=sym, period="daily",
                    start_date=start_date, end_date=today, adjust="qfq",
                )
                if df is None or df.empty:
                    continue
                df = df.copy()
                # 护栏：A 股指数点位不可能低于 100。历史上 astock/百度源把
                # sh000001 剥成 000001 去请求，返回平安银行 K 线并以指数
                # 的 symbol 落库，把 index_daily 悄悄写坏成 11.8 元。这里
                # 对明显像个股价格的“指数”数据直接丢弃，宁可缺也不写错。
                if "close" in df.columns:
                    close_num = pd.to_numeric(df["close"], errors="coerce")
                    bad = close_num < self.INDEX_MIN_CLOSE
                    if bool(bad.any()):
                        logger.warning(
                            "Index data rejected (price looks like a stock)",
                            symbol=sym, rows_dropped=int(bad.sum()))
                        df = df[~bad]
                    if df.empty:
                        continue
                df["symbol"] = sym
                self._store.insert_df(
                    "index_daily",
                    self._to_table_df(df, "index_daily"),
                    mode="append",  # 幂等 upsert（主键 symbol+date）
                )
                updated += len(df)
            except Exception as e:
                logger.warning("Index daily failed", symbol=sym, error=str(e)[:120])
            await asyncio.sleep(0.5)

        logger.info("Index daily updated", rows=updated,
                    target=target.isoformat(), symbols=len(self.INDEX_SYMBOLS))
        return updated

    async def _update_sentiment(self, force_full: bool) -> int:
        """维护游资情绪日统计（sentiment_daily）。

        此前同样没有调度入口，只有 readiness 临时算一次 —— 表停在 08-14。
        注意：SentimentCycleAnalyzer.analyze() 的实时分支内部用 asyncio.run，
        在本 pipeline 的事件循环里会抛 RuntimeError，故强制走库内计算分支
        （use_realtime=False）。
        """
        from pa_mcp.research.sentiment_cycle import SentimentCycleAnalyzer

        try:
            result = await asyncio.to_thread(
                SentimentCycleAnalyzer().analyze, None, 5, False)
        except Exception as e:
            logger.warning("Sentiment computation failed", error=str(e)[:200])
            return 0

        if not isinstance(result, dict) or "error" in result:
            logger.warning("Sentiment unavailable", detail=str(result)[:200])
            return 0

        logger.info("Sentiment updated", date=result.get("date"),
                    stage=result.get("stage"))
        return 1


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
    from pa_mcp.logging_setup import setup_logging
    setup_logging()
    asyncio.run(_main())


if __name__ == "__main__":
    asyncio.run(_main())
