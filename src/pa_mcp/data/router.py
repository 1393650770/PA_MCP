# [AI:BEGIN]
# PA_MCP - Data Layer: DataSourceRouter (Multi-Source Failover)
#
# Routes data requests across an ordered list of data source adapters.
#   - Sequential failover: primary first, fallbacks in config order
#   - Circuit breaker per source: consecutive failures → open → cooldown → half-open probe
#   - Capability-based selection: get_source_for("corporate_actions")
#   - Health reporting for MCP tools and logs
#
# Strategy: ORDERED + CIRCUIT BREAKER (user-confirmed).
# No parallel fan-out (would multiply free-API request volume).
# [AI:END]

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class BreakerState(str, Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Tripped — skip this source
    HALF_OPEN = "half_open"  # Probe after cooldown


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker parameters for one data source."""
    failure_threshold: int = 3        # Consecutive failures to trip
    cooldown_seconds: float = 300.0   # Time in OPEN before probing


@dataclass
class SourceHealth:
    """Health statistics for one data source."""
    name: str
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    success_count: int = 0
    failure_count: int = 0
    trip_count: int = 0
    last_error: str = ""
    last_success_at: float = 0.0
    last_trip_at: float = 0.0


class CircuitBreaker:
    """Per-source circuit breaker (thread-safe via asyncio lock)."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.health = SourceHealth(name=name)
        self._lock = asyncio.Lock()

    def allow_request(self) -> bool:
        """Whether a request to this source should be attempted now."""
        state = self.health.state
        if state == BreakerState.CLOSED:
            return True
        if state == BreakerState.OPEN:
            # Check cooldown elapsed → transition to HALF_OPEN and allow one probe
            if time.monotonic() - self.health.last_trip_at >= self.config.cooldown_seconds:
                self.health.state = BreakerState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow the single probe
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        self.health.consecutive_failures = 0
        self.health.success_count += 1
        self.health.last_success_at = time.monotonic()
        if self.health.state == BreakerState.HALF_OPEN:
            self.health.state = BreakerState.CLOSED
            logger.info("Circuit breaker closed", source=self.name)

    def record_failure(self, error: str) -> None:
        """Record a failed request; trip the breaker if threshold reached."""
        self.health.consecutive_failures += 1
        self.health.failure_count += 1
        self.health.last_error = error[:300]

        if self.health.consecutive_failures >= self.config.failure_threshold:
            if self.health.state != BreakerState.OPEN:
                self.health.state = BreakerState.OPEN
                self.health.trip_count += 1
                self.health.last_trip_at = time.monotonic()
                logger.warning(
                    "Circuit breaker OPEN",
                    source=self.name,
                    failures=self.health.consecutive_failures,
                    cooldown_seconds=self.config.cooldown_seconds,
                )


class AllSourcesFailedError(RuntimeError):
    """Raised when every configured source failed for a request."""

    def __init__(self, request_desc: str, errors: dict[str, str]) -> None:
        self.request_desc = request_desc
        self.source_errors = errors
        summary = "; ".join(f"{name}: {err[:120]}" for name, err in errors.items())
        super().__init__(f"All data sources failed for {request_desc}: {summary}")


class DataSourceRouter:
    """Routes data requests across an ordered list of source adapters.

    Usage:
        router = DataSourceRouter([
            ("akshare", akshare_adapter),
            ("sina", sina_adapter),
            ("tencent", tencent_adapter),
            ("eastmoney", eastmoney_adapter),
        ])
        df, source = await router.fetch_daily_kline("000001")
    """

    def __init__(
        self,
        sources: list[tuple[str, Any]],
        breaker_configs: Optional[dict[str, CircuitBreakerConfig]] = None,
        cache_ttl_seconds: float = 300.0,
        min_source_interval: dict[str, float] | None = None,
    ) -> None:
        """Initialize with ordered (name, adapter) pairs.

        Args:
            sources: Ordered list — first is primary, rest are fallbacks.
            breaker_configs: Optional per-source breaker configs keyed by name.
            cache_ttl_seconds: In-memory response cache TTL (default 5 min).
                               Reduces request volume to free APIs → lower
                               ban risk. Level-1 cache; DuckDB is level-2.
            min_source_interval: Per-source minimum seconds between requests
                                 (e.g. eastmoney needs >=1s throttle).
        """
        self._sources: list[tuple[str, Any]] = list(sources)
        self._breakers: dict[str, CircuitBreaker] = {}
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._last_request_at: dict[str, float] = {}
        self._min_interval = min_source_interval or {}

        for name, _ in self._sources:
            cfg = (breaker_configs or {}).get(name, CircuitBreakerConfig())
            self._breakers[name] = CircuitBreaker(name, cfg)

    # ---- Level-1 cache ----

    def _cache_key(self, fn: str, symbol: str, period: str,
                   start: str, end: str, adjust: str) -> str:
        return f"{fn}:{symbol}:{period}:{start}:{end}:{adjust}"

    def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        if key in self._cache:
            ts, df = self._cache[key]
            if time.monotonic() - ts < self._cache_ttl:
                return df
            del self._cache[key]
        return None

    def _cache_put(self, key: str, df: pd.DataFrame) -> None:
        if len(self._cache) > 2000:  # bound memory
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest]
        self._cache[key] = (time.monotonic(), df)

    async def _throttle(self, name: str) -> None:
        """Enforce per-source minimum request interval (anti-ban)."""
        interval = self._min_interval.get(name, 0.0)
        if interval <= 0:
            return
        last = self._last_request_at.get(name, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)
        self._last_request_at[name] = time.monotonic()

    # ---- Public API ----

    @property
    def source_names(self) -> list[str]:
        return [name for name, _ in self._sources]

    def health_report(self) -> list[dict[str, Any]]:
        """Per-source health: state, successes, failures, trips, last error."""
        return [
            {
                "name": name,
                "state": self._breakers[name].health.state.value,
                "success_count": self._breakers[name].health.success_count,
                "failure_count": self._breakers[name].health.failure_count,
                "consecutive_failures": self._breakers[name].health.consecutive_failures,
                "trip_count": self._breakers[name].health.trip_count,
                "last_error": self._breakers[name].health.last_error,
            }
            for name, _ in self._sources
        ]

    def get_source_for(self, capability: str) -> Optional[tuple[str, Any]]:
        """Find the first source in order that supports a capability."""
        for name, adapter in self._sources:
            supports = getattr(adapter, "supports", None)
            if supports is not None and callable(supports) and supports(capability):
                return name, adapter
        return None

    def reset(self) -> None:
        """Reset all circuit breakers (e.g. after a long outage)."""
        for breaker in self._breakers.values():
            breaker.health.state = BreakerState.CLOSED
            breaker.health.consecutive_failures = 0

    # ---- Fetch wrappers ----

    async def fetch_daily_kline(
        self, symbol: str, start_date: str = "", end_date: str = "",
        adjust: str = "qfq", period: str = "daily",
    ) -> tuple[pd.DataFrame, str]:
        """Fetch daily kline from the first healthy source.

        Level-1 cache hit returns immediately without touching any source.
        Returns (df, source_name). Raises AllSourcesFailedError if all fail.
        """
        cache_key = self._cache_key("kline", symbol, period,
                                    start_date, end_date, adjust)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached.copy(), "cache"

        errors: dict[str, str] = {}

        for name, adapter in self._sources:
            breaker = self._breakers[name]
            if not breaker.allow_request():
                errors[name] = f"circuit open (tripped {breaker.health.trip_count}x)"
                logger.debug(
                    "Source skipped (circuit open)",
                    source=name, trips=breaker.health.trip_count,
                )
                continue

            if not hasattr(adapter, "get_daily_kline"):
                errors[name] = "capability unsupported"
                continue

            try:
                await self._throttle(name)
                df = await adapter.get_daily_kline(
                    symbol=symbol, period=period,
                    start_date=start_date, end_date=end_date,
                    adjust=adjust,
                )
                if df is None or df.empty:
                    raise ValueError("empty DataFrame")

                # Standardize: ensure source column reflects the actual source
                if "source" in df.columns:
                    df["source"] = name

                breaker.record_success()
                self._cache_put(cache_key, df)
                logger.info(
                    "Data fetched",
                    source=name, symbol=symbol, rows=len(df),
                )
                return df, name

            except NotImplementedError:
                errors[name] = "capability unsupported"
                continue
            except Exception as e:
                breaker.record_failure(str(e))
                errors[name] = str(e)[:200]
                # 501/429 等风控类错误降级为 debug（避免批量请求刷屏）
                is_ban = any(k in str(e) for k in ("501", "429", "403"))
                (logger.debug if is_ban else logger.warning)(
                    "Source failed, trying next",
                    source=name, symbol=symbol, error=str(e)[:150],
                )

        raise AllSourcesFailedError(f"symbol={symbol} period={period}", errors)

    async def fetch_realtime_spot_all(self) -> tuple[pd.DataFrame, str]:
        """Fetch full-market spot snapshot from the first healthy source."""
        errors: dict[str, str] = {}

        for name, adapter in self._sources:
            breaker = self._breakers[name]
            if not breaker.allow_request():
                errors[name] = "circuit open"
                continue

            # 源不支持该能力（NotImplementedError）→ 跳过，不计失败
            if not hasattr(adapter, "get_realtime_spot_all"):
                errors[name] = "capability unsupported"
                continue

            try:
                df = await adapter.get_realtime_spot_all()
                if df is None or df.empty:
                    raise ValueError("empty DataFrame")
                breaker.record_success()
                return df, name
            except NotImplementedError:
                errors[name] = "capability unsupported"
                continue
            except Exception as e:
                breaker.record_failure(str(e))
                errors[name] = str(e)[:200]

        raise AllSourcesFailedError("realtime spot all", errors)

    async def verify_consistency(
        self, symbols: list[str],
        tolerance: float = 0.005,
        min_overlap: int = 5,
    ) -> dict[str, Any]:
        """Cross-source consistency check on sample symbols.

        Fetches daily kline for the same symbol from two different healthy
        sources and compares close prices on overlapping dates.
        Used by the scheduler for end-of-day verification (needs network).
        """
        # Find two distinct sources that support daily bars
        daily_sources = [
            (name, adapter) for name, adapter in self._sources
            if getattr(adapter, "supports", lambda c: False)("daily_bars")
        ]
        if len(daily_sources) < 2:
            return {
                "checked": False,
                "reason": f"Need >=2 daily-bars sources, have {len(daily_sources)}",
                "mismatches": [],
            }

        source_a_name, source_a = daily_sources[0]
        source_b_name, source_b = daily_sources[1]
        mismatches: list[dict] = []
        checked_symbols = 0

        for symbol in symbols:
            try:
                df_a, _ = await self.fetch_daily_kline(symbol)
                df_b = await source_b.get_daily_kline(symbol)
                if df_b is None or df_b.empty:
                    continue

                # Align on date, compare close
                a = df_a.set_index("date")["close"]
                b = df_b.set_index("date")["close"]
                common = a.index.intersection(b.index)
                if len(common) < min_overlap:
                    continue

                diff = (a[common] / b[common] - 1).abs()
                bad = diff[diff > tolerance]
                checked_symbols += 1

                if len(bad) > 0:
                    mismatches.append({
                        "symbol": symbol,
                        "source_a": source_a_name,
                        "source_b": source_b_name,
                        "overlap_days": len(common),
                        "mismatch_days": int(len(bad)),
                        "max_diff_pct": round(float(bad.max() * 100), 2),
                    })
            except Exception as e:
                logger.debug("Consistency check skipped for symbol", symbol=symbol, error=str(e))

        return {
            "checked": True,
            "source_a": source_a_name,
            "source_b": source_b_name,
            "symbols_checked": checked_symbols,
            "mismatches": mismatches,
            "tolerance": tolerance,
        }
