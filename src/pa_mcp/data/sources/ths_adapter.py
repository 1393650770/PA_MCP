# [AI:BEGIN]
# PA_MCP - Data Layer: 同花顺（10jqka）Adapter（历史补数源）
#
# Free public 10jqka API — no key required, 全历史日线（1991 至今 ≈8472 根）。
#   Daily kline by year: http://d.10jqka.com.cn/v6/line/hs_{code}/01/{year}.js
#   Recent 140 bars:     http://d.10jqka.com.cn/v6/line/hs_{code}/01/last.js
# 必需 header: Referer: http://stockpage.10jqka.com.cn/
#
# 实测结论（勿踩坑）：
#   1. all.js 返回**分数组结构**（dates 无年份 / price 季度分组 / priceFactor），
#      解析复杂且无 OHLC —— 弃用，一律用按年请求 {year}.js。
#   2. ?type=hfq 参数实测**不生效**（返回与默认相同的数据）—— 仅支持不复权
#      （bfq），price_adjust_mode 恒为 "bfq"。
#   3. 按年拼接全历史 = 每票最多 ~36 个逐年请求，必须限流（本 adapter 内部
#      0.4s 间隔 + router 级 min_source_interval={"ths": 0.4} 双重保障）。
#
# 定位：链尾历史补数源（长历史回测/事件研究数据不足时启用），
#       非日常实时行情源。
# [AI:END]

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

_BASE_URL = "http://d.10jqka.com.cn/v6/line/{prefix}/01/{year}.js"  # 01=日线段
_HEADERS = {
    "Referer": "http://stockpage.10jqka.com.cn/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}
_MIN_YEAR = 1991          # A 股最早数据年份
_THROTTLE_SECONDS = 0.4   # 单次全历史请求内部逐年请求的最小间隔
_ROW_FIELDS = 7           # date,open,high,low,close,volume,amount


class ThsAdapter:
    """同花顺（10jqka）数据适配器 — 免费公开接口，无 key，仅不复权日线。

    Benefits:
    - 全历史日线（1991 至今），按年请求拼接，任意区间可拉
    - 无 API key、无登录；请求需带 Referer header
    """

    CAPABILITIES = {
        "daily_bars": "available",        # 仅「不复权」日线（?type=hfq 实测无效）
        "minute_bars": "unavailable",     # 未实现（01/ 段仅日线）
        "realtime_quote": "unavailable",  # 未实现（有 hq 接口但未实现，不虚标）
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

    volume_unit = "shares"  # 同花顺日线 volume 单位就是股，无需换算

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self._client: Any = None
        self._last_request_at: float = 0.0

    @classmethod
    def supports(cls, capability: str) -> bool:
        return cls.CAPABILITIES.get(capability, "unavailable") != "unavailable"

    async def _get_client(self):
        """Lazy-import httpx client（与 tencent/eastmoney 同一范式）。"""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=self.timeout, headers=_HEADERS)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ---- 纯函数（可单测） ----

    @staticmethod
    def _to_ths_code(symbol: str) -> str:
        """6 位代码 → 同花顺前缀段。

        规则：6/0/3 开头 → hs_（沪深共用 hs_ 段）；4/8/920 开头 → bj_（北交所）。
        已带 sh/sz 前缀透传为 hs_，带 bj 透传为 bj_；其余默认 hs_。
        """
        code = symbol.strip()
        prefix = code[:2].lower()
        if prefix in ("sh", "sz"):
            return "hs_" + code[2:]
        if prefix == "bj":
            return "bj_" + code[2:]
        if code.startswith("bj"):
            return "bj_" + code[2:]
        if code[:1] in ("4", "8") or code.startswith("920"):
            return "bj_" + code
        return "hs_" + code

    @staticmethod
    def _strip_jsonp(text: str) -> dict[str, Any]:
        """剥 JSONP 包装：quotebridge_...({...}) → dict。

        无括号抛 ValueError（解析失败由 router 记录并熔断）。
        """
        start = text.find("(")
        end = text.rfind(")")
        if start < 0 or end <= start:
            raise ValueError(f"Invalid JSONP payload: {text[:80]!r}")
        return json.loads(text[start + 1:end])

    @staticmethod
    def _parse_year_data(data_str: str) -> list[tuple]:
        """解析单年 data 串 → [(date, open, high, low, close, volume, amount)]。

        格式：`YYYYMMDD,open,high,low,close,volume,amount,turnover,,,0;...`。
        字段数 <7 的行跳过（容错，与 tencent/eastmoney 风格一致）。
        """
        if not data_str:
            return []
        rows: list[tuple] = []
        for line in data_str.split(";"):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < _ROW_FIELDS:
                continue
            try:
                row = (
                    parts[0].strip(),                      # date YYYYMMDD
                    float(parts[1]), float(parts[2]),      # open high
                    float(parts[3]), float(parts[4]),      # low close
                    float(parts[5]), float(parts[6]),      # volume amount
                )
            except (ValueError, IndexError):
                continue
            rows.append(row)
        return rows

    async def _fetch_url(self, client, ths_code: str, year: int) -> list[tuple]:
        """单次请求 → rows；某年无数据（未上市/停牌）返回 []（不抛错）。"""
        url = _BASE_URL.format(prefix=ths_code, year=year)
        resp = await client.get(url)
        resp.raise_for_status()
        text = resp.content.decode("utf-8", errors="ignore") or \
            resp.content.decode("gbk", errors="ignore")
        try:
            payload = self._strip_jsonp(text)
        except ValueError:
            return []
        return self._parse_year_data(payload.get("data") or "")

    async def _fetch_year(self, client, ths_code: str, year: int) -> list[tuple]:
        """单年请求；404 = 代码/年份不存在 → 合法空态 []（不抛错）。

        北交所 bj_ 段实测无对应接口（404），回退 hs_ 重试一次；
        两段都 404 视为该代码无数据（router 会切下一源）。
        其他网络错误透传 raise（router 熔断）。
        """
        import httpx
        try:
            return await self._fetch_url(client, ths_code, year)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            if ths_code.startswith("bj_"):
                logger.warning(
                    "ths bj_ prefix 404, retrying as hs_",
                    code=ths_code, year=year)
                try:
                    return await self._fetch_url(
                        client, "hs_" + ths_code[3:], year)
                except httpx.HTTPStatusError as e2:
                    if e2.response.status_code == 404:
                        return []  # 两段都查不到 → 合法空态
                    raise
            return []

    async def _throttle(self) -> None:
        """实例内最小请求间隔（全历史 36 个逐年请求的热点保护）。"""
        now = time.monotonic()
        wait = _THROTTLE_SECONDS - (now - self._last_request_at)
        if wait > 0:
            import asyncio
            await asyncio.sleep(wait)
        self._last_request_at = time.monotonic()

    # ---- 主入口 ----

    async def get_daily_kline(self, symbol: str, period: str = "daily",
                              start_date: str = "", end_date: str = "",
                              adjust: str = "qfq") -> pd.DataFrame:
        """日 K 线（不复权）— 按年请求拼接 + 日期过滤 + 字段标准化。

        Args:
            symbol: 6 位代码（000001 / sh600036）
            period: 仅 "daily"（其余抛 NotImplementedError）
            start_date/end_date: YYYYMMDD；空则全历史（1991 至今）
            adjust: 任何值均返回不复权数据（?type=hfq 实测不生效），
                请求值非 "bfq" 时 warning 一次

        Returns:
            标准化列：symbol/date/open/close/high/low/volume/amount/
                     source/price_adjust_mode；全空返回空 DataFrame（合法态）。
        """
        if period != "daily":
            raise NotImplementedError(
                f"ths adapter only supports daily bars, got {period!r}")
        if adjust and adjust != "bfq":
            logger.warning(
                "ths returns unadjusted (bfq) data only; "
                "adjust=%r ignored", adjust)

        ths_code = self._to_ths_code(symbol)
        today = datetime.now().strftime("%Y%m%d")
        start = start_date or f"{_MIN_YEAR}0101"
        end = end_date or today
        start_year = max(int(start[:4]), _MIN_YEAR)
        end_year = int(end[:4])

        client = await self._get_client()
        rows: list[tuple] = []
        for year in range(start_year, end_year + 1):
            await self._throttle()
            rows.extend(await self._fetch_year(client, ths_code, year))
        if not rows:
            logger.warning("ths returned no data", symbol=symbol,
                           start=start, end=end)
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "date", "open", "high", "low", "close", "volume", "amount"])
        # 字符串级日期过滤（YYYYMMDD 字典序 == 时间序）
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        if df.empty:
            return pd.DataFrame()

        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["source"] = "ths"
        df["price_adjust_mode"] = "bfq"
        df = df.sort_values("date").reset_index(drop=True)
        return df[
            ["symbol", "date", "open", "close", "high", "low",
             "volume", "amount", "source", "price_adjust_mode"]]
