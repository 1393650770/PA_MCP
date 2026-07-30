# [AI:BEGIN]
# PA_MCP - Data Layer: Sina Finance Adapter (Fallback Data Source)
# Sina provides free HTTP APIs for A-share kline, real-time quotes.
# Used as fallback when AKShare is unavailable.
# All endpoints are public, no API key needed.
# [AI:END]

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# User-Agent rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 Version/17.4 Safari/605.1.15",
]


class SinaAdapter:
    """Sina Finance data adapter — free, public HTTP API.

    Endpoints:
    - Daily K-line: https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData
    - Real-time quotes: http://hq.sinajs.cn/list=...

    Benefits over AKShare:
    - Extremely stable API (unchanged for 10+ years)
    - No rate limiting on the standard endpoints
    - Works without akshare package (pure httpx)
    """

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self._client: Any = None

    async def _get_client(self):
        """Lazy import httpx client."""
        if self._client is None:
            import httpx
            import random
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": random.choice(USER_AGENTS)},
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _to_sina_code(symbol: str) -> str:
        """Convert 000001 -> sh600001 or sz000001 for Sina API.

        Supported prefixes:
          sh: 600/601/603/605 (上海主板), 688 (科创板)
          sz: 000/001/002/003 (深圳主板+中小板), 300/301 (创业板)
          bj: 8xx/4xx (北京证券交易所 — mapped to Sina as sh for API compatibility)
        """
        code = symbol.strip()
        prefixes_sh = ["600", "601", "603", "605", "688"]
        prefixes_sz = ["000", "001", "002", "003", "300", "301"]
        prefixes_bj = ["4", "8"]

        if any(code.startswith(p) for p in prefixes_sh):
            return f"sh{code}"
        elif any(code.startswith(p) for p in prefixes_sz):
            return f"sz{code}"
        elif any(code.startswith(p) for p in prefixes_bj):
            # Sina does not natively support BJ exchange — fallback to sh
            logger.warning(
                "Beijing Exchange stock mapped to sh (Sina may not have data)",
                symbol=symbol,
            )
            return f"sh{code}"
        else:
            # Unknown code range — try sh, but warn
            logger.warning("Unknown code prefix, defaulting to sh", symbol=symbol)
            return f"sh{code}"

    async def get_daily_kline(
        self, symbol: str, period: str = "daily",
        start_date: str = "", end_date: str = "",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Get daily kline from Sina Finance.

        Args:
            symbol: Stock code (e.g., '600036')
            period: Only 'daily' supported by Sina
            start_date: YYYYMMDD
            end_date: YYYYMMDD
            adjust: 'qfq', 'hfq', 'bfq'
        """
        sina_code = self._to_sina_code(symbol)

        # Determine scale based on adjust mode
        # qfq=0 (forward adjustment), hfq=1 (backward), bfq=2 (no adjust)
        scale_map = {"qfq": 0, "hfq": 1, "bfq": 2}
        scale = scale_map.get(adjust, 0)

        # Build URL
        # Sina API: KLINE format with datalen parameter
        url = (
            f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            f"CN_MarketData.getKLineData"
            f"?symbol={sina_code}&scale={scale}&ma=no&datalen=2000"
        )

        client = await self._get_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("Sina kline failed", symbol=symbol, error=str(e))
            raise

        if not data or not isinstance(data, list):
            raise RuntimeError(f"Sina returned invalid data for {symbol}")

        # Parse Sina response: each item is a dict with string values
        rows = []
        for item in data:
            try:
                row = {
                    "date": item.get("day", ""),
                    "open": float(item.get("open", 0)),
                    "high": float(item.get("high", 0)),
                    "low": float(item.get("low", 0)),
                    "close": float(item.get("close", 0)),
                    "volume": float(item.get("volume", 0)),
                    "amount": 0.0,  # Sina doesn't provide amount in this endpoint
                }
                rows.append(row)
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])

        # Filter by date range
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df["date"] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df["date"] <= end_dt]

        # Add pct_change
        df["pct_change"] = df["close"].pct_change() * 100

        # Add source tracking and symbol
        df["symbol"] = symbol
        df["source"] = "sina"
        df["price_adjust_mode"] = adjust
        # Note: Sina does not provide 'amount' (成交额) in this endpoint
        if "amount" not in df.columns:
            df["amount"] = 0.0

        return df.sort_values("date").reset_index(drop=True)

    # ---- Capability Declaration ----

    CAPABILITIES = {
        "daily_bars": "available",           # 日K线（免费、稳定，但无成交额）
        "minute_bars": "unavailable",        # Sina HTTP API 不提供分钟线
        "security_status": "unavailable",    # 无历史ST/停牌/退市
        "corporate_actions": "unavailable",  # 无分红送转
        "financials": "unavailable",
        "index_membership": "unavailable",
        "trade_calendar": "unavailable",
        "benchmark_total_return": "unavailable",
        "events": "unavailable",
        "fund_flow": "unavailable",
        "dragon_tiger": "unavailable",
        "realtime_quote": "available",       # 实时快照（免费，延迟3-15s）
    }

    @classmethod
    def supports(cls, capability: str) -> bool:
        """Check if this adapter supports a given capability."""
        return cls.CAPABILITIES.get(capability, "unavailable") != "unavailable"

    async def get_realtime_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote from Sina live API.

        Sina live API format:
        var hq_str_sh600036="名称,今开,昨收,现价,最高,最低,买一,卖一,成交量,成交额,..."
        """
        sina_code = self._to_sina_code(symbol)
        url = f"http://hq.sinajs.cn/list={sina_code}"

        client = await self._get_client()
        try:
            # Sina real-time needs Referer header
            import httpx
            import random
            temp_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "User-Agent": random.choice(USER_AGENTS),
                    "Referer": "https://finance.sina.com.cn",
                },
            )
            response = await temp_client.get(url)
            await temp_client.aclose()
            response.raise_for_status()
            text = response.text
        except Exception as e:
            logger.error("Sina real-time quote failed", symbol=symbol, error=str(e))
            raise

        # Parse: var hq_str_sh600036="data1,data2,..."
        match = re.search(r'"(.+)"', text)
        if not match:
            raise RuntimeError(f"Sina real-time parse failed for {symbol}")

        fields = match.group(1).split(",")
        if len(fields) < 32:
            raise RuntimeError(f"Sina real-time insufficient fields for {symbol}")

        return {
            "name": fields[0],
            "open": float(fields[1]) if fields[1] else 0.0,
            "prev_close": float(fields[2]) if fields[2] else 0.0,
            "price": float(fields[3]) if fields[3] else 0.0,
            "high": float(fields[4]) if fields[4] else 0.0,
            "low": float(fields[5]) if fields[5] else 0.0,
            "volume": float(fields[8]) if len(fields) > 8 and fields[8] else 0.0,
            "amount": float(fields[9]) if len(fields) > 9 and fields[9] else 0.0,
            "bid1": float(fields[11]) if len(fields) > 11 and fields[11] else 0.0,
            "ask1": float(fields[21]) if len(fields) > 21 and fields[21] else 0.0,
            "date": fields[30] if len(fields) > 30 else "",
            "time": fields[31] if len(fields) > 31 else "",
        }
