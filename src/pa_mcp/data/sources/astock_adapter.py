# [AI:BEGIN]
# PA_MCP - Data Layer: a-stock-data 适配器（源链首选，质量增强版）
#
# 移植自开源项目 simonlin1212/a-stock-data（Apache-2.0，已注明出处，
# 完整 SKILL.md 见工程 skills/a-stock-data/）：
#   1. 腾讯实时报价质量版：前缀路由白名单（沪指数/ETF/北交所 920 段）、
#      is_stale 僵尸报价检测（停牌/北交所老码返回定格报价不报错——实测
#      bj832982 报 112.60 vs 真实 920982 为 131.74，差值 17%）、
#      44/45 市值字段修正（曾标反）。
#   2. 百度股市通 K 线：返回自带 ma5/ma10/ma20（独有能力），正确请求头
#      （Accept/Origin/Referer）绕开 403。
#
# 定位：源链首选（config sources 首位）。失败自动熔断回退腾讯/新浪等。
# [AI:END]

from __future__ import annotations

import logging
import urllib.request
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 沪指数白名单（000300 等不能落到 sz，否则返回空或错票）
SH_INDEX = {"000300", "000905", "000016", "000688", "000852", "000010"}

_BAIDU_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/vnd.finance-web.v1+json",
    "Origin": "https://gushitong.baidu.com",
    "Referer": "https://gushitong.baidu.com/",
}


class AstockAdapter:
    """a-stock-data 数据适配器（腾讯质量版实时 + 百度 K 线带 MA）。"""

    CAPABILITIES = {
        "daily_bars": "available",        # 百度 K 线（带 MA5/10/20）
        "realtime_quote": "available",    # 腾讯质量版（is_stale 检测）
        "minute_bars": "unavailable",
        "security_status": "unavailable",
        "corporate_actions": "unavailable",
        "financials": "unavailable",
        "index_membership": "unavailable",
        "trade_calendar": "unavailable",
        "benchmark_total_return": "unavailable",
        "events": "unavailable",
        "fund_flow": "unavailable",
        "dragon_tiger": "unavailable",
        "spot_all": "unavailable",
    }

    volume_unit = "shares"

    def __init__(self, timeout: int = 12) -> None:
        self.timeout = timeout

    @classmethod
    def supports(cls, capability: str) -> bool:
        return cls.CAPABILITIES.get(capability, "unavailable") != "unavailable"

    # ---- 前缀路由（与 a-stock-data 一致） ----

    @staticmethod
    def _prefix(code: str) -> str:
        low = code.lower()
        if low.startswith(("sh", "sz", "bj")):
            return low
        if code.startswith("92"):
            return f"bj{code}"
        if code in SH_INDEX or code.startswith(("5", "6", "9")):
            return f"sh{code}"
        if code.startswith(("4", "8")):
            return f"bj{code}"
        return f"sz{code}"

    # ---- 腾讯实时报价（质量版） ----

    async def get_realtime_quote(self, symbol: str) -> Optional[dict[str, Any]]:
        """腾讯实时报价（a-stock-data 质量版）。

        含 is_stale 僵尸报价检测（停牌/北交所老码返回定格报价时标记，
        不当作真实行情）与市值字段修正。
        """
        p = self._prefix(symbol)
        url = f"https://qt.gtimg.cn/q={p}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            data = resp.read().decode("gbk", errors="replace")
        except Exception as e:  # noqa: BLE001
            logger.warning("astock quote failed: %s (%s)", symbol, str(e)[:80])
            return None

        for line in data.strip().split(";"):
            line = line.strip()
            if not line or "=" not in line or '"' not in line:
                continue
            key = line.split("=")[0].split("_")[-1]
            vals = line.split('"')[1].split("~")
            if len(vals) < 53 or vals[2] != symbol[-6:]:
                continue

            def _f(i: int, d: float = 0.0) -> float:
                try:
                    return float(vals[i]) if vals[i] else d
                except (TypeError, ValueError):
                    return d

            price = _f(3)
            last_close = _f(4)
            amount_wan = _f(37)
            stale = (amount_wan == 0 and price == last_close and price > 0)
            q = {
                "symbol": symbol,
                "name": vals[1],
                "price": price,
                "prev_close": last_close,
                "open": _f(5),
                "change_pct": _f(32),
                "high": _f(33),
                "low": _f(34),
                "amount_wan": amount_wan,
                "turnover_pct": _f(38),
                "pe": _f(39),
                "pb": _f(46),
                "float_mcap_yi": _f(44),
                "mcap_yi": _f(45),
                "limit_up_price": _f(47),
                "limit_down_price": _f(48),
                "vol_ratio": _f(49),
                "is_stale": stale,
                "source": "astock",
            }
            if stale:
                logger.warning("astock stale quote detected: %s @ %s",
                               symbol, price)
            return q
        return None

    # ---- 百度 K 线（带 MA） ----

    async def get_daily_kline(self, symbol: str, period: str = "daily",
                              start_date: str = "", end_date: str = "",
                              adjust: str = "qfq") -> pd.DataFrame:
        """百度股市通 K 线（自带 MA5/10/20；不复权）。

        正确请求头（Accept/Origin/Referer）绕开 403。返回标准化列：
        symbol/date/open/close/high/low/volume/amount/ma5/ma10/ma20/
        source/price_adjust_mode。
        """
        if period != "daily":
            raise NotImplementedError(
                f"astock adapter only supports daily bars, got {period!r}")
        code = symbol[-6:]
        url = ("https://finance.pae.baidu.com/selfselect/getstockquotation"
               "?all=1&isIndex=false&isBk=false&isBlock=false"
               "&isFutures=false&isStock=true&newFormat=1"
               f"&group=quotation_kline_ab&finClientType=pc&code={code}"
               f"&start_time={start_date}&ktype=1")
        req = urllib.request.Request(url, headers=_BAIDU_HEADERS)
        try:
            import json
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            d = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except Exception as e:  # noqa: BLE001
            logger.warning("astock kline failed: %s (%s)", symbol,
                           str(e)[:80])
            return pd.DataFrame()

        md = (d.get("Result") or {}).get("newMarketData") or {}
        keys = md.get("keys", [])
        raw_rows = md.get("marketData", "")
        if not keys or not raw_rows:
            return pd.DataFrame()
        rows = []
        for line in raw_rows.split(";"):
            if not line.strip():
                continue
            vals = line.split(",")
            if len(vals) != len(keys):
                continue
            row = {}
            for k, v in zip(keys, vals):
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df.get("time"), unit="s", errors="coerce")
        df = df.dropna(subset=["date"])
        # 日期过滤（字符串级）
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        if df.empty:
            return pd.DataFrame()
        df["symbol"] = symbol
        df["source"] = "astock"
        df["price_adjust_mode"] = "bfq"
        keep = ["symbol", "date", "open", "close", "high", "low",
                "volume", "amount", "source", "price_adjust_mode"]
        for ma in ("ma5avgprice", "ma10avgprice", "ma20avgprice"):
            if ma in df.columns:
                keep.append(ma)
        df = df[[c for c in keep if c in df.columns]]
        return df.sort_values("date").reset_index(drop=True)

    async def close(self) -> None:
        pass
