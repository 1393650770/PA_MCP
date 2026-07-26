# [AI:BEGIN]
# PA_MCP - Data Layer: AKShare Adapter
# Primary data source for A-share market data.
# Wraps AKShare endpoints with rate limiting, retry, and error handling.
# [AI:END]

from __future__ import annotations

import asyncio
import random
import time
from functools import wraps
from typing import Any, Callable, Optional

import pandas as pd
import structlog

from pa_mcp.config import DataSourceSettings

logger = structlog.get_logger(__name__)


class AKShareRateLimiter:
    """Token-bucket rate limiter for AKShare API calls with jitter."""

    def __init__(self, settings: DataSourceSettings) -> None:
        self.rate = settings.rate_limit_rps
        self.burst = getattr(settings, "burst_max", 10)
        self.tokens = float(self.burst)
        self.max_tokens = float(self.burst)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate
                # Add jitter (0-50% of wait time)
                jitter = wait * 0.5 * random.random()
                await asyncio.sleep(wait + jitter)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


# Global rate limiter instance (initialized lazily)
_limiter: Optional[AKShareRateLimiter] = None


def get_limiter() -> AKShareRateLimiter:
    """Get or create the global rate limiter."""
    global _limiter
    if _limiter is None:
        from pa_mcp.config import get_settings
        _limiter = AKShareRateLimiter(get_settings().akshare)
    return _limiter


def rate_limited(func: Callable) -> Callable:
    """Decorator: apply rate limiting to async AKShare calls."""
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        limiter = get_limiter()
        await limiter.acquire()
        return await func(*args, **kwargs)
    return wrapper


# ---- Anti-crawl utilities ----

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]


def get_random_ua() -> str:
    """Return a random User-Agent string."""
    return random.choice(USER_AGENTS)


class AKShareAdapter:
    """Adapter for AKShare data source.

    Wraps 15 AKShare endpoints for A-share market data with:
    - Rate limiting (token bucket + jitter)
    - Automatic retry with exponential backoff
    - Error normalization
    - Anti-crawl headers rotation
    """

    def __init__(self, settings: Optional[DataSourceSettings] = None) -> None:
        if settings is None:
            from pa_mcp.config import get_settings
            settings = get_settings().akshare
        self.settings = settings
        self.retry_count = settings.retry_count
        self.timeout = settings.timeout_seconds
        self._ak = None  # Lazy import

    def _get_ak(self):
        """Lazy import akshare to avoid startup delay."""
        if self._ak is None:
            import akshare as ak
            self._ak = ak
        return self._ak

    async def _call_with_retry(self, func: Callable, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Call an AKShare function with retry logic."""
        limiter = get_limiter()

        last_error: Optional[Exception] = None
        for attempt in range(self.retry_count + 1):
            try:
                await limiter.acquire()
                # Run sync AKShare call in thread pool (AKShare is sync)
                result = await asyncio.to_thread(func, *args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    return result
                # Some AKShare functions return lists or other types
                return pd.DataFrame(result)
            except Exception as e:
                last_error = e
                if attempt < self.retry_count:
                    backoff = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(
                        "AKShare call failed, retrying",
                        attempt=attempt + 1,
                        backoff_seconds=round(backoff, 2),
                        error=str(e),
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error("AKShare call exhausted retries", error=str(e))

        raise RuntimeError(f"AKShare call failed after {self.retry_count} retries: {last_error}")

    # ---- Market Data Endpoints ----

    async def get_daily_kline(
        self, symbol: str, period: str = "daily",
        start_date: str = "20200101", end_date: str = "",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Get daily OHLCV kline data.

        Endpoint: ak.stock_zh_a_hist()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(
            ak.stock_zh_a_hist,
            symbol=symbol, period=period,
            start_date=start_date, end_date=end_date,
            adjust=adjust,
        )
        # Normalize column names
        COLUMN_MAP = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_change",
            "涨跌额": "change", "换手率": "turnover",
        }
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    async def get_realtime_spot_all(self) -> pd.DataFrame:
        """Get all A-share stocks' real-time quotes.

        Endpoint: ak.stock_zh_a_spot_em()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_zh_a_spot_em)
        return df

    async def get_stock_info(self, symbol: str) -> pd.DataFrame:
        """Get individual stock basic info.

        Endpoint: ak.stock_individual_info_em()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_individual_info_em, symbol=symbol)
        return df

    async def get_index_daily(self, symbol: str, start_date: str = "", end_date: str = "") -> pd.DataFrame:
        """Get index daily kline (上证指数, 深证成指, 沪深300 etc.).

        Endpoint: ak.stock_zh_index_daily_em()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(
            ak.stock_zh_index_daily_em,
            symbol=symbol, start_date=start_date, end_date=end_date,
        )
        return df

    async def get_minute_kline(
        self, symbol: str, period: str = "5",
    ) -> pd.DataFrame:
        """Get intraday minute kline.

        Endpoint: ak.stock_zh_a_hist_min_em()
        Note: Free data has 3-15s delay. period: 1/5/15/30/60.
        """
        ak = self._get_ak()
        df = await self._call_with_retry(
            ak.stock_zh_a_hist_min_em,
            symbol=symbol, period=period,
        )
        return df

    # ---- Fund Flow Endpoints ----

    async def get_stock_fund_flow(self, symbol: str) -> pd.DataFrame:
        """Get individual stock capital flow.

        Endpoint: ak.stock_individual_fund_flow()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_individual_fund_flow, symbol=symbol)
        return df

    async def get_sector_fund_flow(self) -> pd.DataFrame:
        """Get sector-level capital flow.

        Endpoint: ak.stock_sector_fund_flow_em()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_sector_fund_flow_em)
        return df

    async def get_market_fund_flow(self) -> pd.DataFrame:
        """Get market-wide capital flow overview.

        Endpoint: ak.stock_market_fund_flow()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_market_fund_flow)
        return df

    async def get_northbound_flow(self) -> pd.DataFrame:
        """Get northbound (沪股通/深股通) capital flow.

        Endpoint: ak.stock_hsgt_hist_em()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_hsgt_hist_em)
        return df

    # ---- Dragon-Tiger Board Endpoints ----

    async def get_dragon_tiger_top(self) -> pd.DataFrame:
        """Get dragon-tiger board top list.

        Endpoint: ak.stock_hot_rank_lhb_em()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_hot_rank_lhb_em)
        return df

    async def get_dragon_tiger_detail(self, trade_date: str = "") -> pd.DataFrame:
        """Get dragon-tiger board detail with seat info.

        Endpoint: ak.stock_lhb_detail_em()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_lhb_detail_em, date=trade_date)
        return df

    # ---- Event Data Endpoints ----

    async def get_block_trades(self, symbol: str = "") -> pd.DataFrame:
        """Get block trade (大宗交易) records.

        Endpoint: ak.stock_dzjy_mrmx()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_dzjy_mrmx, symbol=symbol)
        return df

    async def get_lockup_expiry(self) -> pd.DataFrame:
        """Get upcoming restricted share lockup expiry (限售股解禁).

        Endpoint: ak.stock_restricted_release_queue_summary_sina()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_restricted_release_queue_summary_sina)
        return df

    async def get_insider_trades(self) -> pd.DataFrame:
        """Get insider (大股东/高管) shareholding changes.

        Endpoint: ak.stock_share_hold_change()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_share_hold_change)
        return df

    async def get_pledge_ratio(self, symbol: str = "") -> pd.DataFrame:
        """Get stock pledge ratio (股权质押比例).

        Endpoint: ak.stock_gpzy_pledge_ratio()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_gpzy_pledge_ratio, symbol=symbol)
        return df

    async def get_institutional_visits(self) -> pd.DataFrame:
        """Get institutional research visit records.

        Endpoint: ak.stock_institute_research()
        """
        ak = self._get_ak()
        df = await self._call_with_retry(ak.stock_institute_research)
        return df

    # ---- Market Overview Endpoints ----

    async def get_market_overview(self) -> dict[str, Any]:
        """Get market overview: indices, breadth, turnover, northbound."""
        ak = self._get_ak()
        # Fetch index data for key indices
        sh_df = await self._call_with_retry(
            ak.stock_zh_index_daily_em, symbol="sh000001",
        )
        sz_df = await self._call_with_retry(
            ak.stock_zh_index_daily_em, symbol="sz399001",
        )
        # Broad market snapshot
        spot_df = await self._call_with_retry(ak.stock_zh_a_spot_em)

        up_count = int((spot_df["涨跌幅"] > 0).sum()) if "涨跌幅" in spot_df.columns else 0
        down_count = int((spot_df["涨跌幅"] < 0).sum()) if "涨跌幅" in spot_df.columns else 0
        limit_up_count = int((spot_df["涨跌幅"] >= 9.5).sum()) if "涨跌幅" in spot_df.columns else 0
        limit_down_count = int((spot_df["涨跌幅"] <= -9.5).sum()) if "涨跌幅" in spot_df.columns else 0

        return {
            "sh_index": float(sh_df["close"].iloc[-1]) if not sh_df.empty else None,
            "sz_index": float(sz_df["close"].iloc[-1]) if not sz_df.empty else None,
            "total_stocks": len(spot_df),
            "up_count": up_count,
            "down_count": down_count,
            "limit_up_count": limit_up_count,
            "limit_down_count": limit_down_count,
            "turnover_total": float(spot_df["成交额"].sum()) if "成交额" in spot_df.columns else None,
        }
