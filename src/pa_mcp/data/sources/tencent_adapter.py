# [AI:BEGIN]
# PA_MCP - Data Layer: Tencent Finance Adapter (备用数据源)
#
# Free public Tencent quote API — no key required, very stable.
#   Daily/period kline: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
#   Real-time quote:    https://qt.gtimg.cn/q={code}
#
# Volume unit: Tencent returns 手 (lots) — this adapter converts to 股 (shares).
# [AI:END]

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class TencentAdapter:
    """Tencent Finance data adapter — free public HTTP API, no key.

    Benefits:
    - Very stable public endpoint (decade-old API)
    - No rate limiting on standard endpoints
    - No API key required
    - Provides daily, weekly, monthly, and minute klines
    """

    CAPABILITIES = {
        "daily_bars": "available",        # 日/周/月K线
        "minute_bars": "available",       # 1/5/15/30/60分钟线
        "realtime_quote": "available",    # 实时快照
        "security_status": "unavailable",
        "corporate_actions": "unavailable",
        "financials": "unavailable",
        "index_membership": "unavailable",
        "trade_calendar": "unavailable",
        "benchmark_total_return": "unavailable",
        "events": "unavailable",
        "fund_flow": "unavailable",
        "dragon_tiger": "unavailable",
    }

    volume_unit = "shares"  # 本 adapter 输出统一为“股”

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self._client: Any = None

    @classmethod
    def supports(cls, capability: str) -> bool:
        return cls.CAPABILITIES.get(capability, "unavailable") != "unavailable"

    async def _get_client(self):
        """Lazy-import httpx client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _to_tencent_code(symbol: str) -> str:
        """Convert 000001 -> sz000001, 600036 -> sh600036, sh000300 -> sh000300.

        Already-prefixed codes (sh/sz/bj) pass through unchanged — used for
        indices (sh000300, sz399001) and BSE new segments.
        """
        code = symbol.strip()
        if code[:2].lower() in ("sh", "sz", "bj"):
            return code

        prefixes_sh = ["600", "601", "603", "605", "688", "689"]
        prefixes_sz = ["000", "001", "002", "003", "300", "301"]
        prefixes_bj_new = ["920"]          # 北交所新号段 (920xxx)
        prefixes_bj_old = ["4", "8"]       # 北交所老号段 (43x/83x/87x)

        if any(code.startswith(p) for p in prefixes_sh):
            return f"sh{code}"
        elif any(code.startswith(p) for p in prefixes_sz):
            return f"sz{code}"
        elif any(code.startswith(p) for p in prefixes_bj_new):
            return f"bj{code}"
        elif any(code.startswith(p) for p in prefixes_bj_old):
            # 老号段已迁移至920 — 保留映射但标记可能返回僵尸报价
            return f"bj{code}"
        # Default: sz
        return f"sz{code}"

    async def get_daily_kline(
        self, symbol: str, period: str = "daily",
        start_date: str = "", end_date: str = "",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Get kline data from Tencent Finance.

        Args:
            symbol: Stock code (e.g. '000001')
            period: 'daily', 'weekly', 'monthly', '1', '5', '15', '30', '60'
            start_date: YYYYMMDD
            end_date: YYYYMMDD
            adjust: 'qfq' (forward), 'hfq' (backward), 'bfq' (raw)
        """
        tencent_code = self._to_tencent_code(symbol)

        # Tencent fq param: qfq=前复权, hfq=后复权, '' or none=不复权
        fq_map = {"qfq": "qfq", "hfq": "hfq", "bfq": ""}
        fq = fq_map.get(adjust, "qfq")

        # Map period
        period_map = {
            "daily": "day", "weekly": "week", "monthly": "month",
            "1": "m1", "5": "m5", "15": "m15", "30": "m30", "60": "m60",
        }
        tencent_period = period_map.get(period, "day")

        # Tencent requires YYYY-MM-DD format (with dashes); convert if YYYYMMDD
        def _fmt(d: str) -> str:
            d = d.strip()
            if len(d) == 8 and d.isdigit():
                return f"{d[:4]}-{d[4:6]}-{d[6:]}"
            return d

        start_fmt = _fmt(start_date) if start_date else ""
        end_fmt = _fmt(end_date) if end_date else ""

        # Count: try to cover from start_date; Tencent caps at 640 bars per call.
        count = 640

        client = await self._get_client()

        # 分钟线走 mkline 接口（fqkline 不支持 m5/m15/m30/m60）
        if tencent_period.startswith("m"):
            url = (
                f"https://ifzq.gtimg.cn/appstock/app/kline/mkline"
                f"?param={tencent_code},{tencent_period},,{count}"
            )
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error("Tencent mkline failed", symbol=symbol, error=str(e))
                raise
            rows = self._parse_mkline(data, tencent_code, tencent_period)
            if not rows:
                logger.warning("Tencent returned no minute data", symbol=symbol)
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["symbol"] = symbol
            df["source"] = "tencent"
            df["price_adjust_mode"] = "bfq"  # 分钟线不复权
            return df.sort_values("datetime").reset_index(drop=True)

        url = (
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={tencent_code},{tencent_period},{start_fmt},{end_fmt},{count},{fq}"
        )

        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("Tencent kline failed", symbol=symbol, error=str(e))
            raise

        rows = self._parse_kline(data, tencent_code, period, fq=fq)
        if not rows:
            logger.warning("Tencent returned no kline data", symbol=symbol)
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        df["source"] = "tencent"
        df["price_adjust_mode"] = adjust
        return df.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _parse_mkline(data: dict, tencent_code: str, period: str) -> list[dict]:
        """Parse Tencent mkline (分钟线) response.

        Response: data -> {code: {"m5": [["202608061055","open","close","high","low","vol","{}","pct"], ...]}}
        Date format: YYYYMMDDHHMM (no dashes). Volume in 手 (lots).
        """
        rows: list[dict] = []
        code_data = data.get("data", {}).get(tencent_code, {})
        if not isinstance(code_data, dict):
            return rows
        bars = code_data.get(period)
        if not isinstance(bars, list):
            return rows

        for item in bars:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                continue
            try:
                raw_dt = str(item[0])
                # YYYYMMDDHHMM -> datetime
                dt = pd.Timestamp(
                    f"{raw_dt[:4]}-{raw_dt[4:6]}-{raw_dt[6:8]} "
                    f"{raw_dt[8:10]}:{raw_dt[10:12]}:00"
                )
                rows.append({
                    "datetime": dt,
                    "open": float(item[1]),
                    "close": float(item[2]),
                    "high": float(item[3]),
                    "low": float(item[4]),
                    "volume": float(item[5]) * 100,  # 手 → 股
                    "amount": 0.0,
                })
            except (ValueError, TypeError, IndexError):
                continue

        return rows

    @staticmethod
    def _parse_kline(data: dict, tencent_code: str, period: str,
                     fq: str = "qfq") -> list[dict]:
        """Parse Tencent kline JSON into row dicts.

        Response shape (current live API):
            {"code": 0, "data": {"sh600036": {
                "qfqday": [["2026-07-30","10.0","10.5","10.8","9.9","123456",...], ...],
                "qt": {...}, "mx_price": ..., "prec": 2, "version": ...}}}
        Key pattern: {fq}{period_suffix} — qfqday / hfqday / day / qfqweek /
                     qfqm5 / m5 etc. Older responses used "qfqkline".
        """
        rows: list[dict] = []
        code_data = data.get("data", {}).get(tencent_code, {})
        if not code_data:
            return rows

        # Period suffix used by Tencent in the data key
        suffix_map = {
            "daily": "day", "weekly": "week", "monthly": "month",
            "1": "m1", "5": "m5", "15": "m15", "30": "m30", "60": "m60",
        }
        suffix = suffix_map.get(period, "day")

        # Candidate keys: current {fq}{suffix}, legacy {fq}kline, raw {suffix}
        candidates = []
        if fq:
            candidates.append(f"{fq}{suffix}")          # qfqday
            candidates.append(f"{fq}kline")              # legacy qfqkline
        candidates.append(suffix)                        # day / m5 (raw)
        candidates.append(f"hfq{suffix}")

        bars = None
        for key in candidates:
            val = code_data.get(key)
            if isinstance(val, list):
                bars = val
                break

        # Some responses flatten data -> {code: [ [..], .. ]} as a direct list
        if bars is None and isinstance(code_data, list):
            bars = code_data

        if not bars:
            return rows

        for item in bars:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                continue
            try:
                rows.append({
                    "date": item[0],
                    "open": float(item[1]),
                    "close": float(item[2]),
                    "high": float(item[3]),
                    "low": float(item[4]),
                    # Tencent volume is in 手 (lots) — convert to 股 (shares)
                    "volume": float(item[5]) * 100,
                    "amount": 0.0,  # not provided in this endpoint
                })
            except (ValueError, TypeError):
                continue

        return rows

    async def get_realtime_quote(self, symbol: str) -> dict[str, Any]:
        """Get rich real-time quote from Tencent (qt.gtimg.cn).

        Returns 50+ fields including valuation: PE, PB, market cap,
        turnover rate, volume ratio, limit prices, etc.

        Field layout (split by ~):
            0=unknown 1=name 2=code 3=price 4=prev_close 5=open
            6=volume(lots) 7=outer 8=inner 9=bid1 ... 19=ask1
            31=change 32=change_pct 33=high 34=low
            35=price/vol/amount 36=volume 37=amount 38=turnover_pct
            39=PE 43=amplitude 44=float_mktcap(亿) 45=total_mktcap(亿)
            46=PB 47=limit_up 48=limit_down 49=volume_ratio 50=avg_price
        """
        tencent_code = self._to_tencent_code(symbol)
        url = f"https://qt.gtimg.cn/q={tencent_code}"

        client = await self._get_client()
        try:
            response = await client.get(url, headers={"Referer": "https://gu.qq.com/"})
            response.raise_for_status()
            # Tencent realtime returns GBK-encoded text
            text = response.content.decode("gbk", errors="replace")
        except Exception as e:
            logger.error("Tencent realtime quote failed", symbol=symbol, error=str(e))
            raise

        for line in text.strip().split(";"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("~")
            if len(parts) < 50 or parts[2] != symbol:
                continue

            def _f(idx: int, default: float = 0.0) -> float:
                try:
                    return float(parts[idx])
                except (ValueError, IndexError):
                    return default

            return {
                "symbol": symbol,
                "name": parts[1],
                "price": _f(3),
                "prev_close": _f(4),
                "open": _f(5),
                "volume_lots": _f(6),
                "bid1": _f(9),
                "ask1": _f(19),
                "change": _f(31),
                "change_pct": _f(32),
                "high": _f(33),
                "low": _f(34),
                "amount": _f(37),
                "turnover_pct": _f(38),
                "pe": _f(39),
                "amplitude_pct": _f(43),
                "float_market_cap_billion": _f(44),
                "total_market_cap_billion": _f(45),
                "pb": _f(46),
                "limit_up_price": _f(47),
                "limit_down_price": _f(48),
                "volume_ratio": _f(49),
                "avg_price": _f(50),
                "source": "tencent",
            }

        raise RuntimeError(f"Tencent realtime parse failed for {symbol}")

    async def get_realtime_spot_all(self) -> pd.DataFrame:
        """Full-market real-time snapshot (NOT supported by the simple Tencent endpoint).

        Tencent's free endpoint is per-symbol; there is no public all-market
        snapshot endpoint without a key. Raise NotImplementedError so the
        router treats this capability as unavailable.
        """
        raise NotImplementedError(
            "Tencent free API does not provide full-market spot snapshot"
        )
