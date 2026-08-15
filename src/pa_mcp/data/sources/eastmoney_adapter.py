# [AI:BEGIN]
# PA_MCP - Data Layer: EastMoney Adapter (备用数据源)
#
# Free public EastMoney (东方财富) quote API — no key required.
#   Daily kline: https://push2his.eastmoney.com/api/qt/stock/kline/get
#
# Volume unit: EastMoney returns 股 (shares) directly.
# [AI:END]

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class EastMoneyAdapter:
    """EastMoney (东方财富) data adapter — free public HTTP API, no key.

    Benefits:
    - Free public endpoint, very stable
    - Provides daily/weekly/monthly/minute klines
    - Volume directly in shares (股)
    """

    CAPABILITIES = {
        "daily_bars": "available",        # 日/周/月K线 (klt=101/102/103)
        "minute_bars": "available",       # 1/5/15/30/60分钟线 (klt=1/5/15/30/60)
        "realtime_quote": "available",    # 实时快照（个股）
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

    volume_unit = "shares"  # 东财 volume 单位就是股

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
    def _to_secid(symbol: str) -> str:
        """Convert stock code to EastMoney secid.

        secid format: {market}.{code}
          market=1 for Shanghai (6xx/68x), 0 for Shenzhen (0xx/3xx),
          market=0 also used for Beijing Exchange codes (4xx/8xx) on the
          push2his kline endpoint.
        """
        code = symbol.strip()
        if code.startswith(("6", "9")):
            # 6xx = Shanghai main/STAR; 9xx = Shanghai B-shares
            return f"1.{code}"
        elif code.startswith(("0", "3", "2")):
            return f"0.{code}"
        elif code.startswith("920"):
            # 北交所新号段 (since ~2024): 920xxx — EastMoney market=0 on kline API
            return f"0.{code}"
        elif code.startswith(("4", "8")):
            # 北交所老号段 (43x/83x/87x — 已迁移至920，保留映射以兼容历史)
            return f"0.{code}"
        # Default: Shenzhen
        return f"0.{code}"

    async def get_daily_kline(
        self, symbol: str, period: str = "daily",
        start_date: str = "", end_date: str = "",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Get daily kline from EastMoney.

        Args:
            symbol: Stock code (e.g. '000001')
            period: 'daily', 'weekly', 'monthly', '1', '5', '15', '30', '60'
            start_date: YYYYMMDD
            end_date: YYYYMMDD
            adjust: 'qfq' (forward), 'hfq' (backward), 'bfq' (raw)
        """
        secid = self._to_secid(symbol)

        # EastMoney klt: 101=daily, 102=weekly, 103=monthly, 1/5/15/30/60=minute
        klt_map = {
            "daily": 101, "weekly": 102, "monthly": 103,
            "1": 1, "5": 5, "15": 15, "30": 30, "60": 60,
        }
        klt = klt_map.get(period, 101)

        # fqt: 0=不复权, 1=前复权, 2=后复权
        fqt_map = {"bfq": 0, "qfq": 1, "hfq": 2}
        fqt = fqt_map.get(adjust, 1)

        # Date range: EastMoney needs YYYYMMDD; default to last ~3 years
        beg = start_date or "20230101"
        end = end_date or "20500101"

        url = (
            f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
            f"?secid={secid}"
            f"&fields1=f1,f2,f3,f4,f5,f6"
            f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
            f"&klt={klt}&fqt={fqt}"
            f"&beg={beg}&end={end}"
        )

        client = await self._get_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("EastMoney kline failed", symbol=symbol, error=str(e))
            raise

        rows = self._parse_klines(data)
        if not rows:
            logger.warning("EastMoney returned no kline data", symbol=symbol)
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        df["source"] = "eastmoney"
        df["price_adjust_mode"] = adjust
        return df.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _parse_klines(data: dict) -> list[dict]:
        """Parse EastMoney kline JSON.

        Response shape:
            {"data": {"klines": ["2026-07-30,10.00,10.80,10.90,9.90,1234567,12345678.00,...", ...]}}
        kline string: date,open,close,high,low,volume,amount,amplitude,pct_change,change,turnover
        """
        rows: list[dict] = []
        klines = data.get("data", {}).get("klines") or []
        if not klines:
            return rows

        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    "date": parts[0],
                    "open": float(parts[1]),
                    "close": float(parts[2]),
                    "high": float(parts[3]),
                    "low": float(parts[4]),
                    # EastMoney kline volume is in 手 (lots) — convert to 股
                    "volume": float(parts[5]) * 100,
                    "amount": float(parts[6]) if len(parts) > 6 else 0.0,
                    "amplitude": float(parts[7]) if len(parts) > 7 else 0.0,
                    "pct_change": float(parts[8]) if len(parts) > 8 else 0.0,
                    "change": float(parts[9]) if len(parts) > 9 else 0.0,
                    "turnover": float(parts[10]) if len(parts) > 10 else 0.0,
                })
            except (ValueError, TypeError):
                continue

        return rows

    async def get_realtime_spot_all(self) -> pd.DataFrame:
        """Full-market real-time snapshot (NOT supported by the simple kline API).

        EastMoney has a spot-list endpoint but it requires paging params;
        raise NotImplementedError so the router treats this as unavailable.
        """
        raise NotImplementedError(
            "EastMoney kline endpoint does not provide full-market spot snapshot"
        )

    # ---- 东财独有数据（龙虎榜/资金流）— 调用前必须限流 ----

    async def get_dragon_tiger(self, trade_date: str = "",
                               page_size: int = 20) -> pd.DataFrame:
        """Get dragon-tiger board (龙虎榜) daily details from EastMoney.

        Exclusive to EastMoney's datacenter. Must be rate-limited.

        Args:
            trade_date: YYYY-MM-DD, empty = latest trading day
            page_size: Number of records
        """
        url = (
            "https://datacenter-web.eastmoney.com/api/data/v1/get"
            "?reportName=RPT_DAILYBILLBOARD_DETAILSNEW"
            "&columns=ALL"
            f"&pageNumber=1&pageSize={page_size}"
            "&sortColumns=TRADE_DATE&sortTypes=-1"
        )
        if trade_date:
            url += f"&filter=(TRADE_DATE%3D%27{trade_date}%27)"

        client = await self._get_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("EastMoney dragon-tiger failed", error=str(e))
            raise

        rows = data.get("result", {}).get("data") or []
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # 标准化字段
        rename = {
            "TRADE_DATE": "trade_date", "SECURITY_CODE": "symbol",
            "SECURITY_NAME_ABBR": "name", "EXPLAIN": "reason",
            "CLOSE_PRICE": "close", "CHANGE_RATE": "pct_change",
            "BILLBOARD_NET_AMT": "net_amount",
            "BILLBOARD_BUY_AMT": "buy_amount",
            "BILLBOARD_SELL_AMT": "sell_amount",
            "TURNOVERRATE": "turnover",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ("trade_date",):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
        df["source"] = "eastmoney"
        return df

    async def get_stock_fund_flow(self, symbol: str, days: int = 20) -> pd.DataFrame:
        """Get individual stock main-force fund flow (个股资金流) from EastMoney.

        klines format: date, main_net, small_net, mid_net, large_net, super_large_net
        (units: 元)

        Args:
            symbol: Stock code
            days: Number of days (lmt param)
        """
        secid = self._to_secid(symbol)
        url = (
            "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
            f"?secid={secid}"
            "&fields1=f1,f2,f3,f7"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
            f"&klt=101&lmt={days}"
        )

        client = await self._get_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("EastMoney fund flow failed", symbol=symbol, error=str(e))
            raise

        klines = data.get("data", {}).get("klines") or []
        if not klines:
            return pd.DataFrame()

        rows = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 7:
                continue
            try:
                rows.append({
                    "symbol": symbol,
                    "trade_date": parts[0],
                    # parts: date, main_net, small_net, mid_net, large_net, super_large_net, main_net_pct
                    "main_net_inflow": float(parts[1]),
                    "small_net_inflow": float(parts[2]),
                    "mid_net_inflow": float(parts[3]),
                    "large_net_inflow": float(parts[4]),
                    "super_large_net_inflow": float(parts[5]),
                    "main_net_inflow_pct": float(parts[6]) if len(parts) > 6 else 0.0,
                    "source": "eastmoney",
                })
            except (ValueError, TypeError):
                continue

        return pd.DataFrame(rows)

    # ---- 板块行情/资金流（东财独有，供板块轮动研究） ----

    async def get_sector_boards(self, board_type: str = "industry",
                                top_n: int = 100) -> pd.DataFrame:
        """板块列表与实时快照（push2 clist）。

        Args:
            board_type: 'industry'（行业板块，m:90+t:2）或 'concept'（概念，m:90+t:3）
            top_n: 返回数量
        """
        fs = "m:90+t:2" if board_type == "industry" else "m:90+t:3"
        url = (
            "https://push2.eastmoney.com/api/qt/clist/get"
            f"?pn=1&pz={top_n}&po=1&np=1"
            f"&fltt=2&invt=2&fid=f3&fs={fs}"
            "&fields=f2,f3,f8,f12,f14,f62,f66,f72"
        )
        client = await self._get_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("EastMoney sector boards failed", error=str(e))
            raise

        rows = data.get("data", {}).get("diff") or []
        out = []
        for r in rows:
            try:
                out.append({
                    "sector_code": str(r.get("f12", "")),
                    "name": str(r.get("f14", "")),
                    "pct_change": float(r.get("f3") or 0),
                    "turnover": float(r.get("f8") or 0),
                    "main_net_inflow": float(r.get("f62") or 0),
                    "main_net_inflow_pct": float(r.get("f66") or 0),
                    "up_stock_count": int(r.get("f72") or 0),
                    "board_type": board_type,
                })
            except (ValueError, TypeError):
                continue
        return pd.DataFrame(out)

    async def get_sector_kline(self, sector_code: str,
                               days: int = 120) -> pd.DataFrame:
        """板块指数历史日线（push2his，secid=90.BKxxxx）。

        Args:
            sector_code: 板块代码（如 BK0475 银行），须带 BK 前缀
        """
        if not sector_code.upper().startswith("BK"):
            raise ValueError(f"板块代码须为 BK 前缀：{sector_code}")
        url = (
            "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            f"?secid=90.{sector_code.upper()}"
            "&fields1=f1,f2,f3,f4,f5,f6"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
            f"&klt=101&fqt=1&lmt={days}"
        )
        client = await self._get_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("EastMoney sector kline failed",
                         sector=sector_code, error=str(e))
            raise

        klines = data.get("data", {}).get("klines") or []
        rows = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 7:
                continue
            try:
                rows.append({
                    "sector_code": sector_code.upper(),
                    "name": str(data.get("data", {}).get("name", "")),
                    "date": parts[0],
                    "open": float(parts[1]), "close": float(parts[2]),
                    "high": float(parts[3]), "low": float(parts[4]),
                    "volume": float(parts[5]), "amount": float(parts[6]),
                    # 部分字段可选
                    "amplitude": float(parts[8]) if len(parts) > 8 else 0.0,
                    "pct_change": float(parts[9]) if len(parts) > 9 else 0.0,
                    "turnover": float(parts[10]) if len(parts) > 10 else 0.0,
                    "source": "eastmoney",
                })
            except (ValueError, TypeError, IndexError):
                continue
        return pd.DataFrame(rows)

    async def get_sector_fund_flow(self, sector_code: str,
                                   days: int = 20) -> pd.DataFrame:
        """板块主力资金流（push2 fflow，secid=90.BKxxxx）。

        字段：trade_date, main_net_inflow, small/mid/large/super_large, main_net_pct
        """
        url = (
            "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
            f"?secid=90.{sector_code.upper()}"
            "&fields1=f1,f2,f3,f7"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
            f"&klt=101&lmt={days}"
        )
        client = await self._get_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("EastMoney sector fund flow failed",
                         sector=sector_code, error=str(e))
            raise

        klines = data.get("data", {}).get("klines") or []
        rows = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 7:
                continue
            try:
                rows.append({
                    "sector_code": sector_code.upper(),
                    "trade_date": parts[0],
                    "main_net_inflow": float(parts[1]),
                    "small_net_inflow": float(parts[2]),
                    "mid_net_inflow": float(parts[3]),
                    "large_net_inflow": float(parts[4]),
                    "super_large_net_inflow": float(parts[5]),
                    "main_net_inflow_pct": float(parts[6]) if len(parts) > 6 else 0.0,
                    "source": "eastmoney",
                })
            except (ValueError, TypeError):
                continue
        return pd.DataFrame(rows)
