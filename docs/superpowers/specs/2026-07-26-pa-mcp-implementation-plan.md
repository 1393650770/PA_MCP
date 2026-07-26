# PA_MCP — Data Layer, Strategy Engine & Backtest Engine: Detailed Implementation Plan

**Date:** 2026-07-26
**Version:** 2.0
**Scope:** Phase 1 (Data Layer) + Phase 3 (Strategy + Backtest Engine)
**Based on:** `docs/superpowers/specs/2026-07-26-pa-mcp-design.md`

---

## Table of Contents

1. [Data Layer — Data Source Adapters](#1-data-layer--data-source-adapters)
   - [1.1 AKShare Adapter](#11-akshare-adapter)
   - [1.2 BaoStock Adapter](#12-baostock-adapter)
   - [1.3 efinance Adapter](#13-efinance-adapter)
2. [Data Layer — Multi-Source Aggregator](#2-data-layer--multi-source-aggregator)
3. [Data Layer — DuckDB Table Design](#3-data-layer--duckdb-table-design)
4. [Data Layer — Redis Cache Strategy](#4-data-layer--redis-cache-strategy)
5. [Data Layer — Cron Update Jobs](#5-data-layer--cron-update-jobs)
6. [Data Layer — Data Quality Validation](#6-data-layer--data-quality-validation)
7. [Strategy Engine — Base Class & Registry](#7-strategy-engine--base-class--registry)
8. [Strategy Engine — 40+ Strategies (8 Categories)](#8-strategy-engine--40-strategies-8-categories)
9. [Strategy Engine — Signal Format (JSON Schema)](#9-strategy-engine--signal-format-json-schema)
10. [Strategy Engine — Parameter Optimization](#10-strategy-engine--parameter-optimization)
11. [Strategy Engine — Signal Aggregation](#11-strategy-engine--signal-aggregation)
12. [Backtest Engine — backtrader Wrapper](#12-backtest-engine--backtrader-wrapper)
13. [Backtest Engine — A-Share Rules Implementation](#13-backtest-engine--a-share-rules-implementation)
14. [Backtest Engine — API & Report Format](#14-backtest-engine--api--report-format)

---

## 1. Data Layer — Data Source Adapters

### 1.1 AKShare Adapter

#### 1.1.1 API Endpoints to Encapsulate

AKShare version: `>=1.16.0` (pin to `1.16.70` for stability)

| # | AKShare Function | Our Wrapper Method | Purpose |
|---|-----------------|-------------------|---------|
| 1 | `ak.stock_zh_a_hist()` | `get_daily_kline()` | Daily OHLCV history, supports qfq/hfq/bfq |
| 2 | `ak.stock_zh_a_hist_min_em()` | `get_minute_kline()` | 1/5/15/30/60 min intraday |
| 3 | `ak.stock_zh_a_spot_em()` | `get_realtime_spot_all()` | All stocks real-time quote (5000+ rows) |
| 4 | `ak.stock_individual_info_em()` | `get_stock_info()` | Single stock basic info |
| 5 | `ak.stock_zh_a_hist_tx()` | `get_daily_kline_tx()` | Fallback daily kline from Tencent |
| 6 | `ak.stock_zh_index_daily_em()` | `get_index_daily()` | Index daily kline (000300, 000001, etc.) |
| 7 | `ak.stock_individual_fund_flow()` | `get_stock_fund_flow()` | Individual stock capital flow |
| 8 | `ak.stock_sector_fund_flow_em()` | `get_sector_fund_flow()` | Sector capital flow |
| 9 | `ak.stock_hot_rank_lhb_em()` | `get_lhb_top_list()` | Top dragon-tiger board stocks |
| 10 | `ak.stock_lhb_detail_em()` | `get_lhb_detail()` | Dragon-tiger board detail (daily) |
| 11 | `ak.stock_hot_rank_em()` | `get_hot_rank()` | Hot stock ranking by popularity |
| 12 | `ak.stock_zh_a_st_em()` | `get_st_stocks()` | ST stock list |
| 13 | `ak.stock_zh_a_new_em()` | `get_new_stocks()` | New IPO list |
| 14 | `ak.stock_info_a_code_name()` | `get_all_stock_codes()` | All stock code-name mapping |
| 15 | `ak.stock_financial_abstract_ths()` | `get_financial_summary()` | Financial report summary (balance/income/cash) |

#### 1.1.2 Anti-Crawl Strategy (AKShare Proxy Patch Pattern)

AKShare's core issue: requests to East Money (Dongfang Caifu) and Sina APIs will rate-limit by IP. We implement a layered defense based on `akshare-proxy-patch` patterns:

```python
# -- [AI:BEGIN]
# src/pa_mcp/data/sources/akshare_adapter.py

import time
import random
import logging
from typing import Callable
from functools import wraps

logger = logging.getLogger(__name__)

class AKShareRateLimiter:
    """
    Rate limiter with exponential backoff for AKShare API calls.
    Implements: request queuing, random jitter, IP rotation awareness,
    and automatic slowdown when rate-limit headers detected.
    """

    def __init__(
        self,
        min_interval: float = 0.5,       # min seconds between calls
        max_interval: float = 5.0,       # max backoff seconds
        burst_limit: int = 3,            # max calls before throttling
        burst_window: float = 2.0,       # seconds for burst window
        backoff_factor: float = 2.0,     # exponential backoff multiplier
        global_slowdown: float = 1.0,    # applied after rate-limit detection
    ):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.burst_limit = burst_limit
        self.burst_window = burst_window
        self.backoff_factor = backoff_factor
        self.global_slowdown = global_slowdown
        self._call_times: list[float] = []
        self._consecutive_errors = 0
        self._last_error_time = 0.0

    def wait(self):
        """Block until it's safe to make the next call."""
        now = time.time()
        # Prune old timestamps outside burst window
        self._call_times = [
            t for t in self._call_times
            if now - t < self.burst_window
        ]
        if len(self._call_times) >= self.burst_limit:
            sleep_time = self.burst_window - (now - self._call_times[0])
            sleep_time *= self.global_slowdown
            # Add jitter: +/- 20%
            jitter = random.uniform(0.8, 1.2)
            sleep_time = min(sleep_time * jitter, self.max_interval)
            if sleep_time > 0:
                time.sleep(sleep_time)
        # Enforce min interval with global slowdown
        base_wait = self.min_interval * self.global_slowdown
        time.sleep(base_wait + random.uniform(0, 0.3))
        self._call_times.append(time.time())

    def report_error(self):
        """Report an API error; increases global slowdown."""
        self._consecutive_errors += 1
        self._last_error_time = time.time()
        # Exponential increase in slowdown
        self.global_slowdown = min(
            self.global_slowdown * self.backoff_factor,
            30.0  # cap at 30x slowdown
        )

    def report_success(self):
        """Report successful API call; gradually reduce slowdown."""
        if self._consecutive_errors > 0 and time.time() - self._last_error_time > 60:
            self._consecutive_errors = 0
            self.global_slowdown = max(self.global_slowdown * 0.5, 1.0)


def rate_limited(retries: int = 3, backoff_base: float = 2.0):
    """Decorator: retry on failure with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    self._limiter.wait()
                    result = func(self, *args, **kwargs)
                    self._limiter.report_success()
                    return result
                except Exception as e:
                    last_error = e
                    self._limiter.report_error()
                    wait_s = backoff_base ** attempt + random.uniform(0, 1)
                    logger.warning(
                        f"[AKShare] {func.__name__} attempt {attempt+1}/{retries} "
                        f"failed: {e}. Retrying in {wait_s:.1f}s"
                    )
                    time.sleep(wait_s)
            raise RuntimeError(
                f"[AKShare] {func.__name__} failed after {retries} retries"
            ) from last_error
        return wrapper
    return decorator
# -- [AI:END]
```

Additional anti-crawl measures:

1. **Session Reuse**: Use a single `requests.Session` with persistent cookies.
2. **User-Agent Rotation**: Rotate among 5-8 common browser User-Agent strings.
3. **Referer Header**: Always set `Referer: https://data.eastmoney.com/` for East Money endpoints.
4. **Batch Delay**: When fetching full market data (5000+ stocks), insert 0.3-0.5s delay per batch of 50 stocks.
5. **Proxy Pool (Optional)**: Support configurable HTTP/HTTPS proxy list; rotate on rate-limit detection.
6. **Cache-Lock**: Use Redis distributed lock to prevent concurrent AKShare calls from the same process.

#### 1.1.3 Adapter Class Skeleton

```python
# -- [AI:BEGIN]
# src/pa_mcp/data/sources/akshare_adapter.py

import pandas as pd
import akshare as ak
from typing import Optional

class AKShareAdapter:
    """Adapter for AKShare data source. All methods return pd.DataFrame."""

    def __init__(self):
        self._limiter = AKShareRateLimiter()

    @rate_limited(retries=3)
    def get_daily_kline(
        self,
        symbol: str,
        start_date: str = "19900101",
        end_date: str = "20500101",
        adjust: str = "qfq",  # "qfq" | "hfq" | "" (bfq)
    ) -> pd.DataFrame:
        """
        symbol: 6-digit code like "000001" (no exchange prefix)
        Returns columns: date, open, close, high, low, volume, amount,
                        amplitude, pct_change, change, turnover, adjust
        """
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        # Normalize column names to English
        df = df.rename(columns={
            "date": "date",
            "kai": "open",
            "shou": "close",
            "gao": "high",
            "di": "low",
            "chengjiaoliang": "volume",
            "chengjiaoe": "amount",
            "zhenfu": "amplitude",
            "zhangdiefu": "pct_change",
            "zhangdiee": "change",
            "huanshoulv": "turnover",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        df["source"] = "akshare"
        return df

    @rate_limited(retries=3)
    def get_realtime_spot_all(self) -> pd.DataFrame:
        """Returns all ~5000 A-share stocks with real-time quote."""
        df = ak.stock_zh_a_spot_em()
        # <rename columns>
        return df

    @rate_limited(retries=3)
    def get_index_daily(
        self,
        symbol: str,
        start_date: str = "19900101",
        end_date: str = "20500101",
    ) -> pd.DataFrame:
        """symbol: like "000300" for HS300, "000001" for SSE Composite."""
        df = ak.stock_zh_index_daily_em(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        return df

    @rate_limited(retries=3)
    def get_lhb_detail(self, date_str: str = "") -> pd.DataFrame:
        """Dragon-tiger board detail for a specific date."""
        df = ak.stock_lhb_detail_em(date=date_str)
        return df

    @rate_limited(retries=3)
    def get_financial_summary(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Returns dict with keys: balance_sheet, income_stmt, cash_flow."""
        df = ak.stock_financial_abstract_ths(symbol=symbol)
        return self._parse_financials(df)

    def _parse_financials(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Parse THS financial abstract into separate financial statements."""
        # -- [AI] implementation depends on THS output format
        return {"balance_sheet": pd.DataFrame(), "income_stmt": pd.DataFrame(), "cash_flow": pd.DataFrame()}
# -- [AI:END]
```

---

### 1.2 BaoStock Adapter

#### 1.2.1 API Endpoints

BaoStock requires login (`bs.login()`) before any query and logout (`bs.logout()`) after. It is a _pull_-based system: call `query_xxx()` then iterate results.

| BaoStock Function | Our Wrapper Method | Purpose |
|-------------------|-------------------|---------|
| `bs.query_stock_basic()` | `get_stock_basic()` | All stock code/name/ipoDate/type |
| `bs.query_history_k_data_plus()` | `get_daily_kline()` | Daily OHLCV with adjust flag |
| `bs.query_history_k_data_plus()` | `get_minute_kline()` | 5/15/30/60 min kline |
| `bs.query_dividend_data()` | `get_dividend()` | Dividend history for adjust factor calc |
| `bs.query_stock_industry()` | `get_industry()` | Stock-industry mapping |
| `bs.query_balance_sheet_data()` | `get_balance_sheet()` | Balance sheet financials |
| `bs.query_income_statement_data()` | `get_income_stmt()` | Income statement |
| `bs.query_cash_flow_data()` | `get_cash_flow()` | Cash flow statement |

#### 1.2.2 K-Line Field Mapping Table

BaoStock uses Chinese field names. We normalize to English:

```python
# -- [AI]
# BaoStock -> Internal standard field mapping
BAOSTOCK_KLINE_FIELDS = {
    "date":         "date",          # YYYY-MM-DD
    "code":         "symbol",        # sh.000001 format -> convert to 000001
    "open":         "open",
    "high":         "high",
    "low":          "low",
    "close":        "close",
    "preclose":     "pre_close",
    "volume":       "volume",        # shares (not lots!)
    "amount":       "amount",        # CNY
    "adjustflag":   "adjust_flag",   # "1"=qfq, "2"=hfq, "3"=bfq
    "turn":         "turnover",      # turnover rate %
    "tradestatus":  "trade_status",  # "1"=normal, "0"=halted
    "pctChg":       "pct_change",    # daily change %
    "peTTM":        "pe_ttm",
    "pbMRQ":        "pb_mrq",
    "psTTM":        "ps_ttm",
    "pcfNcfTTM":    "pcf_ncf_ttm",
    "isST":         "is_st",         # "1"=ST, "0"=normal
}

# BaoStock requires field list as comma-separated string
BAOSTOCK_FIELDS_STR = ",".join(BAOSTOCK_KLINE_FIELDS.keys())
```

#### 1.2.3 Adjust Factor (Fuquan Yinzi) Handling

BaoStock provides three adjust modes via `adjustflag`:
- `"1"`: forward-adjusted (qian fuquan qfq) -- adjust all historical prices down
- `"2"`: backward-adjusted (hou fuquan hfq) -- adjust all historical prices up
- `"3"`: no adjustment (bu fuquan bfq)

For our aggregator: always fetch `qfq` (adjustflag=1) from BaoStock. Forward-adjustment is the industry standard for backtesting because it preserves the most recent price as-is.

To compute custom adjust factors (needed if we want to switch between adjust modes ourselves):

```python
# -- [AI:BEGIN]
def compute_adjust_factor(dividend_df: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative forward-adjust factor from dividend data.
    adjust_factor[i] = product of all later adjustments applied to bar i.
    Used to convert bfq <-> qfq locally without re-fetching.
    """
    # dividend_df columns: date, dividend_per_share (cash), bonus_share_ratio,
    #                       rights_issue_price, rights_issue_ratio
    # Formula: factor_t = (close_before - cash_dividend + rights_issue_price * rights_ratio)
    #                    / (close_before * (1 + bonus_ratio + rights_ratio))
    # Pre-computed by BaoStock; we just store for reference/cross-check.
    pass
# -- [AI:END]
```

#### 1.2.4 Adapter Class Skeleton

```python
# -- [AI:BEGIN]
# src/pa_mcp/data/sources/baostock_adapter.py

import baostock as bs
import pandas as pd
import logging
from contextlib import contextmanager
from typing import Optional, Generator

logger = logging.getLogger(__name__)

class BaoStockAdapter:
    """Adapter for BaoStock data source."""

    def __init__(self):
        self._logged_in = False

    @contextmanager
    def _session(self) -> Generator[None, None, None]:
        """Context manager: auto login/logout for BaoStock."""
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")
            self._logged_in = True
        try:
            yield
        finally:
            bs.logout()
            self._logged_in = False

    def get_daily_kline(
        self,
        symbol: str,
        start_date: str = "1990-01-01",
        end_date: str = "2050-01-01",
        adjust: str = "qfq",
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        symbol: "000001" (6-digit) -> converted to "sh.000001" internally
        adjust: "qfq" (1), "hfq" (2), "bfq" (3)
        """
        adjust_map = {"qfq": "1", "hfq": "2", "bfq": "3"}
        bs_code = self._to_bs_code(symbol)
        fields_str = fields or BAOSTOCK_FIELDS_STR

        with self._session():
            rs = bs.query_history_k_data_plus(
                code=bs_code,
                fields=fields_str,
                start_date=start_date,
                end_date=end_date,
                frequency="d",            # "d"|"w"|"m"|"5"|"15"|"30"|"60"
                adjustflag=adjust_map.get(adjust, "1"),
            )
            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

        if not data_list:
            logger.warning(f"[BaoStock] No data for {symbol} ({bs_code})")
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={k: v for k, v in BAOSTOCK_KLINE_FIELDS.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_change"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["symbol"] = symbol
        df["source"] = "baostock"
        return df

    def get_stock_basic(self) -> pd.DataFrame:
        """Get all stock basic info: code, name, ipoDate, type, status."""
        with self._session():
            rs = bs.query_stock_basic()
            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={
            "code": "symbol_raw",
            "code_name": "name",
            "ipoDate": "ipo_date",
            "outDate": "delist_date",
            "type": "stock_type",
            "status": "status",
        })
        # strip "sh." / "sz." prefix
        df["symbol"] = df["symbol_raw"].str.replace(r"^(sh|sz)\.", "", regex=True)
        return df

    def _to_bs_code(self, symbol: str) -> str:
        """Convert 6-digit symbol to BaoStock code format (sh.000001 / sz.000001)."""
        if symbol.startswith(("sh.", "sz.")):
            return symbol
        if len(symbol) != 6:
            raise ValueError(f"Invalid symbol: {symbol}")
        if symbol.startswith(("6", "9")):
            return f"sh.{symbol}"
        else:
            return f"sz.{symbol}"
# -- [AI:END]
```

---

### 1.3 efinance Adapter

#### 1.3.1 API Endpoints

efinance sources all data from East Money (Dongfang Caifu). Its API covers the broadest real-time data surface.

| efinance Function | Our Wrapper Method | Purpose |
|-------------------|-------------------|---------|
| `ef.stock.get_quote_history()` | `get_daily_kline()` | Daily OHLCV, multi-period, multi-adjust |
| `ef.stock.get_realtime_quotes()` | `get_realtime_spot_all()` | Full market real-time quotes |
| `ef.stock.get_latest_quote()` | `get_latest_quote()` | Single/multi stock latest snap |
| `ef.stock.get_base_info()` | `get_stock_info()` | Stock basic info (PE, market cap, etc.) |
| `ef.stock.get_quote_snapshot()` | `get_quote_snapshot()` | Single stock full snapshot (5-level depth) |
| `ef.stock.get_history_bill()` | `get_fund_flow_history()` | Historical capital flow (main/retail/institution) |
| `ef.stock.get_today_bill()` | `get_fund_flow_today()` | Intraday capital flow bar-by-bar |
| `ef.stock.get_daily_billboard()` | `get_lhb_daily()` | Dragon-tiger board daily detail |
| `ef.stock.get_all_company_performance()` | `get_financial_performance()` | Quarterly financial performance |
| `ef.stock.get_top10_stock_holder_info()` | `get_top_holders()` | Top 10 shareholders by period |
| `ef.stock.get_belong_board()` | `get_sector_membership()` | Sector/industry board membership |
| `ef.stock.get_members()` | `get_index_members()` | Index constituent stocks |

#### 1.3.2 Anti-Crawl Strategy

efinance has built-in retry (`@retry(tries=3)` on some functions already). East Money's API is relatively generous but still needs:

1. **Session-level Cookie Persistence**: East Money web API requires cookies from initial page load.
2. **Request Batching**: When pulling history for multiple stocks, batch 10-20 per call using efinance's native list support (which uses `multitasking` for concurrent requests).
3. **Exponential Backoff**: Wrap all calls with 3-retry exponential backoff (2^0, 2^1, 2^2 seconds).
4. **Auto-throttle Detection**: If a `json.decoder.JSONDecodeError` or `ConnectionError` occurs, increase global delay.
5. **Market Hours Avoidance**: During 9:30-15:00, increase delay for real-time endpoints to 2-3s between calls.

```python
# -- [AI:BEGIN]
# src/pa_mcp/data/sources/efinance_adapter.py

import efinance as ef  # efinance.stock
import pandas as pd
import time
import random
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

class EFinanceAdapter:
    """Adapter for efinance (East Money) data source."""

    def __init__(self):
        self._last_call = 0.0
        self._min_interval = 0.3  # 300ms min between calls

    def _throttle(self):
        """Enforce minimum inter-call interval with jitter."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed + random.uniform(0, 0.2)
            time.sleep(wait)
        self._last_call = time.time()

    def get_daily_kline(
        self,
        symbol: str,
        start_date: str = "19000101",
        end_date: str = "20500101",
        klt: int = 101,          # 101=daily, 102=weekly, 103=monthly
        fqt: int = 1,            # 0=bfq, 1=qfq, 2=hfq
    ) -> pd.DataFrame:
        """Fetch daily kline from efinance (East Money source)."""
        self._throttle()
        df = ef.stock.get_quote_history(
            stock_codes=symbol,
            beg=start_date,
            end=end_date,
            klt=klt,
            fqt=fqt,
        )
        # Standardize column names
        df = df.rename(columns={
            "stock_code": "symbol",
            "stock_name": "name",
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
            "turnover_rate": "turnover",
            "amplitude": "amplitude",
            "pct_change": "pct_change",
            "change_amount": "change",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["source"] = "efinance"
        return df

    def get_realtime_spot_all(self, market: str = "all") -> pd.DataFrame:
        """Full market real-time quotes."""
        self._throttle()
        df = ef.stock.get_realtime_quotes()
        return df

    def get_latest_quote(self, symbols: Union[str, list[str]]) -> pd.DataFrame:
        """Latest snapshot for one or more stocks."""
        self._throttle()
        return ef.stock.get_latest_quote(stock_codes=symbols)

    def get_fund_flow_history(self, symbol: str) -> pd.DataFrame:
        """Historical capital flow: main force, retail, institution net flows."""
        self._throttle()
        df = ef.stock.get_history_bill(stock_code=symbol)
        return df

    def get_daily_billboard(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Daily dragon-tiger board data. Dates in 'YYYY-MM-DD' format."""
        self._throttle()
        df = ef.stock.get_daily_billboard(start_date=start_date, end_date=end_date)
        return df

    def get_all_company_performance(self, report_date: Optional[str] = None) -> pd.DataFrame:
        """Full market financial performance for a quarter."""
        self._throttle()
        df = ef.stock.get_all_company_performance(date=report_date)
        return df

    def get_top_holders(self, symbol: str, top: int = 4) -> pd.DataFrame:
        """Top 10 shareholders for last `top` reporting periods."""
        self._throttle()
        return ef.stock.get_top10_stock_holder_info(stock_code=symbol, top=top)
# -- [AI:END]
```

---

## 2. Data Layer — Multi-Source Aggregator

### 2.1 Priority & Fallback Chain

```
AKShare (primary) --> efinance (secondary) --> BaoStock (tertiary)
```

| Data Type | Primary | Secondary | Tertiary |
|-----------|---------|-----------|----------|
| Daily K-Line | AKShare (East Money) | efinance (East Money) | BaoStock |
| Real-Time Quote | efinance (East Money) | AKShare (East Money) | -- |
| Minute K-Line | AKShare | efinance | BaoStock |
| Financial Reports | AKShare (THS) | efinance (East Money) | BaoStock |
| Capital Flow | efinance (East Money) | AKShare | -- |
| Dragon-Tiger Board | efinance | AKShare | -- |
| Stock Basic Info | BaoStock | AKShare | efinance |
| Index Data | AKShare | efinance | -- |
| Sector Mapping | efinance | BaoStock | -- |

### 2.2 Consistency Check Logic

After fetching data from all three sources for the same stock/date range:

```python
# -- [AI:BEGIN]
# src/pa_mcp/data/aggregator.py

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataAggregator:
    """Multi-source data aggregator with fallback, consistency check, and canonical output."""

    # Deviation threshold for alerting (2% relative)
    CONSISTENCY_THRESHOLD = 0.02

    # Field mapping from each source to canonical schema
    CANONICAL_FIELDS = [
        "date", "symbol", "name", "open", "high", "low", "close",
        "volume", "amount", "turnover", "pct_change", "amplitude",
        "pre_close", "adjust_flag", "source",
    ]

    def __init__(self, akshare=None, efinance=None, baostock=None):
        self.ak = akshare
        self.ef = efinance
        self.bs = baostock

    def get_daily_kline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        Fetch daily kline with multi-source fallback.
        Priority: AKShare -> efinance -> BaoStock.
        If >=2 sources succeed, run consistency check.
        """
        results: dict[str, Optional[pd.DataFrame]] = {
            "akshare": None,
            "efinance": None,
            "baostock": None,
        }
        errors: dict[str, str] = {}

        # -- Try AKShare first
        try:
            results["akshare"] = self.ak.get_daily_kline(
                symbol, start_date, end_date, adjust
            )
        except Exception as e:
            errors["akshare"] = str(e)
            logger.warning(f"[Aggregator] AKShare failed for {symbol}: {e}")

        # -- Try efinance
        try:
            results["efinance"] = self.ef.get_daily_kline(
                symbol, start_date, end_date, fqt=1 if adjust == "qfq" else 0
            )
        except Exception as e:
            errors["efinance"] = str(e)
            logger.warning(f"[Aggregator] efinance failed for {symbol}: {e}")

        # -- Try BaoStock as last resort
        try:
            results["baostock"] = self.bs.get_daily_kline(
                symbol, start_date, end_date, adjust
            )
        except Exception as e:
            errors["baostock"] = str(e)
            logger.warning(f"[Aggregator] BaoStock failed for {symbol}: {e}")

        # -- Determine which results to use
        successful = {k: v for k, v in results.items() if v is not None and not v.empty}

        if not successful:
            raise RuntimeError(
                f"[Aggregator] All sources failed for {symbol}. Errors: {errors}"
            )

        # -- Consistency check: use the first 2 successful sources
        sources = list(successful.keys())
        if len(sources) >= 2:
            self._check_consistency(
                symbol, successful[sources[0]], successful[sources[1]],
                sources[0], sources[1],
            )

        # -- Return primary source data, canonicalized
        primary = sources[0]
        df = successful[primary]
        return self._canonicalize(df, primary)

    def _check_consistency(
        self,
        symbol: str,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        source1: str,
        source2: str,
    ):
        """Check if close prices from two sources deviate by >2% on overlapping dates."""
        close1 = df1.set_index("date")["close"]
        close2 = df2.set_index("date")["close"]
        common_dates = close1.index.intersection(close2.index)

        if len(common_dates) == 0:
            logger.warning(
                f"[Aggregator] No overlapping dates for {symbol} between "
                f"{source1} and {source2}"
            )
            return

        deviations = (close1[common_dates] - close2[common_dates]).abs() / close1[common_dates]
        bad_dates = deviations[deviations > self.CONSISTENCY_THRESHOLD]

        if len(bad_dates) > 0:
            # -- [AI] Alert: data inconsistency detected
            logger.error(
                f"[Aggregator] CONSISTENCY ALERT: {symbol} has {len(bad_dates)} "
                f"dates with close price deviation > 2% between {source1} and {source2}. "
                f"Dates: {list(bad_dates.index[:5])} (showing first 5)"
            )
            self._send_alert(
                level="WARNING",
                message=f"Data inconsistency: {symbol} close prices deviate >2% "
                        f"on {len(bad_dates)} dates between {source1} and {source2}",
                details={"symbol": symbol, "bad_dates": list(bad_dates.index[:10])},
            )

    def _canonicalize(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalize DataFrame to canonical schema regardless of source."""
        df = df.copy()
        df["source"] = source
        for col in self.CANONICAL_FIELDS:
            if col not in df.columns:
                df[col] = np.nan
        return df[self.CANONICAL_FIELDS]

    def _send_alert(self, level: str, message: str, details: dict):
        """Send alert to configured notification channel (log, email, webhook)."""
        logger.log(
            {"WARNING": logging.WARNING, "ERROR": logging.ERROR}.get(level, logging.WARNING),
            f"{message}\nDetails: {details}"
        )
# -- [AI:END]
```

### 2.3 Canonical Output Format

All adapters produce the same normalized schema:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime64[ns] | Trading date |
| `symbol` | str (6 digits) | Stock code e.g. "000001" |
| `name` | str | Stock name |
| `open` | float64 | Open price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Close price |
| `volume` | float64 | Volume (shares) |
| `amount` | float64 | Turnover amount (CNY) |
| `turnover` | float64 | Turnover rate (%) |
| `pct_change` | float64 | Daily change (%) |
| `amplitude` | float64 | Amplitude (%) |
| `pre_close` | float64 | Previous close |
| `adjust_flag` | str | "qfq" / "hfq" / "bfq" |
| `source` | str | "akshare" / "efinance" / "baostock" / "aggregated" |

---

## 3. Data Layer — DuckDB Table Design

### 3.1 Database Setup

```python
# -- [AI:BEGIN]
# src/pa_mcp/data/store.py

import duckdb
from pathlib import Path

class DuckDBStore:
    """DuckDB storage layer for PA_MCP."""

    def __init__(self, db_path: str = "data/pa_mcp.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self):
        """Create all tables if they don't exist."""
        self.conn.execute("INSTALL httpfs; LOAD httpfs;")
        for sql in ALL_CREATE_TABLE_SQL:
            self.conn.execute(sql)

    def close(self):
        self.conn.close()
# -- [AI:END]
```

### 3.2 All CREATE TABLE SQL + Indexes

```sql
-- -- [AI:BEGIN]
-- ============================================================
-- Table 1: stock_daily (Daily K-Line)
-- ============================================================

CREATE TABLE IF NOT EXISTS stock_daily (
    date        DATE NOT NULL,
    symbol      VARCHAR(6) NOT NULL,
    name        VARCHAR(20),
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,          -- shares
    amount      DOUBLE,          -- CNY
    turnover    DOUBLE,          -- turnover rate %
    pct_change  DOUBLE,          -- daily change %
    amplitude   DOUBLE,          -- amplitude %
    pre_close   DOUBLE,
    adjust_flag VARCHAR(4),      -- "qfq", "hfq", "bfq"
    source      VARCHAR(16),     -- "akshare", "efinance", "baostock"
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date, adjust_flag)
);

CREATE INDEX IF NOT EXISTS idx_daily_symbol ON stock_daily(symbol);
CREATE INDEX IF NOT EXISTS idx_daily_date ON stock_daily(date);
CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON stock_daily(symbol, date);


-- ============================================================
-- Table 2: stock_minute (Intraday K-Line)
-- ============================================================

CREATE TABLE IF NOT EXISTS stock_minute (
    date        DATE NOT NULL,
    time        TIME NOT NULL,
    symbol      VARCHAR(6) NOT NULL,
    period      VARCHAR(4) NOT NULL,  -- "1", "5", "15", "30", "60"
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    amount      DOUBLE,
    source      VARCHAR(16),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date, time, period)
);

CREATE INDEX IF NOT EXISTS idx_minute_symbol_date ON stock_minute(symbol, date, period);
CREATE INDEX IF NOT EXISTS idx_minute_date ON stock_minute(date);


-- ============================================================
-- Table 3: stock_realtime (Real-Time Spot Quote, transient)
-- ============================================================

CREATE TABLE IF NOT EXISTS stock_realtime (
    symbol          VARCHAR(6) PRIMARY KEY,
    name            VARCHAR(20),
    latest_price    DOUBLE,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    pre_close       DOUBLE,
    pct_change      DOUBLE,
    volume          DOUBLE,
    amount          DOUBLE,
    turnover        DOUBLE,
    bid1            DOUBLE, bid1_vol DOUBLE,
    ask1            DOUBLE, ask1_vol DOUBLE,
    bid2            DOUBLE, bid2_vol DOUBLE,
    ask2            DOUBLE, ask2_vol DOUBLE,
    bid3            DOUBLE, bid3_vol DOUBLE,
    ask3            DOUBLE, ask3_vol DOUBLE,
    bid4            DOUBLE, bid4_vol DOUBLE,
    ask4            DOUBLE, ask4_vol DOUBLE,
    bid5            DOUBLE, bid5_vol DOUBLE,
    ask5            DOUBLE, ask5_vol DOUBLE,
    limit_up        DOUBLE,
    limit_down      DOUBLE,
    total_market_cap DOUBLE,
    float_market_cap DOUBLE,
    pe_ttm          DOUBLE,
    pb_mrq          DOUBLE,
    update_time     TIMESTAMP,
    source          VARCHAR(16),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
);


-- ============================================================
-- Table 4: index_daily (Market Index Daily)
-- ============================================================

CREATE TABLE IF NOT EXISTS index_daily (
    date        DATE NOT NULL,
    code        VARCHAR(16) NOT NULL,
    name        VARCHAR(32),
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    amount      DOUBLE,
    source      VARCHAR(16),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (code, date)
);

CREATE INDEX IF NOT EXISTS idx_index_date ON index_daily(date);


-- ============================================================
-- Table 5: financials_balance_sheet
-- ============================================================

CREATE TABLE IF NOT EXISTS financials_balance_sheet (
    symbol          VARCHAR(6) NOT NULL,
    report_date     DATE NOT NULL,
    report_type     VARCHAR(10),
    total_assets            DOUBLE,
    total_liabilities       DOUBLE,
    total_equity            DOUBLE,
    current_assets          DOUBLE,
    current_liabilities     DOUBLE,
    cash_equivalents        DOUBLE,
    accounts_receivable     DOUBLE,
    inventory               DOUBLE,
    fixed_assets            DOUBLE,
    intangible_assets       DOUBLE,
    goodwill                DOUBLE,
    short_term_borrowing    DOUBLE,
    long_term_borrowing     DOUBLE,
    accounts_payable        DOUBLE,
    total_shares            DOUBLE,
    source      VARCHAR(16),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, report_date)
);


-- ============================================================
-- Table 6: financials_income_stmt
-- ============================================================

CREATE TABLE IF NOT EXISTS financials_income_stmt (
    symbol          VARCHAR(6) NOT NULL,
    report_date     DATE NOT NULL,
    report_type     VARCHAR(10),
    total_revenue           DOUBLE,
    operating_cost          DOUBLE,
    operating_profit        DOUBLE,
    net_profit              DOUBLE,
    net_profit_deducted     DOUBLE,
    eps                     DOUBLE,
    rd_expense              DOUBLE,
    selling_expense         DOUBLE,
    admin_expense           DOUBLE,
    finance_expense         DOUBLE,
    investment_income       DOUBLE,
    source      VARCHAR(16),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, report_date)
);


-- ============================================================
-- Table 7: financials_cash_flow
-- ============================================================

CREATE TABLE IF NOT EXISTS financials_cash_flow (
    symbol          VARCHAR(6) NOT NULL,
    report_date     DATE NOT NULL,
    report_type     VARCHAR(10),
    cf_operating    DOUBLE,
    cf_investing    DOUBLE,
    cf_financing    DOUBLE,
    free_cash_flow  DOUBLE,
    capex           DOUBLE,
    dividend_paid   DOUBLE,
    source      VARCHAR(16),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, report_date)
);


-- ============================================================
-- Table 8: capital_flow (Fund Flow)
-- ============================================================

CREATE TABLE IF NOT EXISTS capital_flow (
    date                DATE NOT NULL,
    symbol              VARCHAR(6) NOT NULL,
    main_net_inflow     DOUBLE,
    super_large_net     DOUBLE,
    large_net           DOUBLE,
    medium_net          DOUBLE,
    small_net           DOUBLE,
    main_pct            DOUBLE,
    north_bound_inflow  DOUBLE,
    margin_balance      DOUBLE,
    short_sell_volume   DOUBLE,
    source      VARCHAR(16),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_flow_date ON capital_flow(date);


-- ============================================================
-- Table 9: dragon_tiger (Dragon-Tiger Board)
-- ============================================================

CREATE TABLE IF NOT EXISTS dragon_tiger (
    date                DATE NOT NULL,
    symbol              VARCHAR(6) NOT NULL,
    name                VARCHAR(20),
    reason              VARCHAR(128),
    close               DOUBLE,
    pct_change          DOUBLE,
    lhb_buy_amount      DOUBLE,
    lhb_sell_amount     DOUBLE,
    lhb_net_amount      DOUBLE,
    buy_broker_1        VARCHAR(64),  buy_amount_1 DOUBLE,
    sell_broker_1       VARCHAR(64),  sell_amount_1 DOUBLE,
    buy_broker_2        VARCHAR(64),  buy_amount_2 DOUBLE,
    sell_broker_2       VARCHAR(64),  sell_amount_2 DOUBLE,
    buy_broker_3        VARCHAR(64),  buy_amount_3 DOUBLE,
    sell_broker_3       VARCHAR(64),  sell_amount_3 DOUBLE,
    buy_broker_4        VARCHAR(64),  buy_amount_4 DOUBLE,
    sell_broker_4       VARCHAR(64),  sell_amount_4 DOUBLE,
    buy_broker_5        VARCHAR(64),  buy_amount_5 DOUBLE,
    sell_broker_5       VARCHAR(64),  sell_amount_5 DOUBLE,
    is_institution_buy  BOOLEAN,
    is_famous_trader    BOOLEAN,
    turnover_rate       DOUBLE,
    amplitude           DOUBLE,
    source      VARCHAR(16),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_lhb_date ON dragon_tiger(date);
CREATE INDEX IF NOT EXISTS idx_lhb_symbol ON dragon_tiger(symbol);


-- ============================================================
-- Table 10: trade_calendar (Trading Calendar)
-- ============================================================

CREATE TABLE IF NOT EXISTS trade_calendar (
    date        DATE PRIMARY KEY,
    is_trade_day BOOLEAN NOT NULL,
    week_day    INTEGER,
    month       INTEGER,
    year        INTEGER,
    is_week_end BOOLEAN,
    pre_trade_date  DATE,
    next_trade_date DATE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
);

CREATE INDEX IF NOT EXISTS idx_cal_year_month ON trade_calendar(year, month);


-- ============================================================
-- Table 11: stock_basic (Stock Basic Info)
-- ============================================================

CREATE TABLE IF NOT EXISTS stock_basic (
    symbol          VARCHAR(6) PRIMARY KEY,
    name            VARCHAR(20),
    exchange        VARCHAR(4),
    stock_type      VARCHAR(8),
    ipo_date        DATE,
    delist_date     DATE,
    industry        VARCHAR(64),
    sector          VARCHAR(64),
    total_shares    DOUBLE,
    float_shares    DOUBLE,
    is_st           BOOLEAN DEFAULT FALSE,
    status          VARCHAR(8),
    source          VARCHAR(16),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
);

CREATE INDEX IF NOT EXISTS idx_stock_industry ON stock_basic(industry);


-- ============================================================
-- Table 12: stock_sector_mapping (Stock <-> Sector relationship)
-- ============================================================

CREATE TABLE IF NOT EXISTS stock_sector_mapping (
    symbol          VARCHAR(6) NOT NULL,
    sector_code     VARCHAR(16) NOT NULL,
    sector_name     VARCHAR(64),
    sector_type     VARCHAR(16),
    source          VARCHAR(16),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, sector_code)
);


-- ============================================================
-- Table 13: key_holders (Top 10 Shareholders)
-- ============================================================

CREATE TABLE IF NOT EXISTS key_holders (
    symbol          VARCHAR(6) NOT NULL,
    report_date     DATE NOT NULL,
    holder_rank     INTEGER,
    holder_name     VARCHAR(128),
    holder_type     VARCHAR(32),
    shares_held     DOUBLE,
    holding_pct     DOUBLE,
    change_amount   DOUBLE,
    change_pct      DOUBLE,
    source          VARCHAR(16),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, report_date, holder_rank)
);
-- -- [AI:END]
```

### 3.3 DuckDB Query Patterns

```python
# -- [AI:BEGIN]
# Recommended DuckDB query optimizations

# 1. Use EXCLUDE for wide tables
# SELECT * EXCLUDE (created_at, source) FROM stock_daily WHERE symbol = '000001'

# 2. Use ASOF join for nearest-date lookups
# SELECT a.*, b.close AS index_close FROM stock_daily a
# ASOF JOIN index_daily b ON a.date >= b.date AND b.code = '000300'

# 3. Window functions for rolling calculations
# SELECT symbol, date, close,
#   AVG(close) OVER (PARTITION BY symbol ORDER BY date
#     ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20
# FROM stock_daily

# 4. Efficient bulk insert via parquet intermediate
# COPY stock_daily FROM 'temp.parquet' (FORMAT PARQUET);
# -- [AI:END]
```

---

## 4. Data Layer — Redis Cache Strategy

### 4.1 Key Design Convention

```
Format: {domain}:{entity}:{subkey}
All keys have explicit TTL.
```

| Key Pattern | TTL | Value Type | Description |
|-------------|-----|------------|-------------|
| `quote:{symbol}:realtime` | 3s | JSON | Single stock 5-level quote snapshot |
| `quote:all:spot` | 5s | MsgPack | Full market real-time spot (compressed) |
| `kline:{symbol}:daily:{adjust}:{days}` | 6h | Parquet bytes | Last N days of daily kline |
| `kline:{symbol}:minute:{period}:latest` | 5m | Parquet bytes | Latest intraday minute kline |
| `info:{symbol}:basic` | 24h | JSON | Stock basic info + industry |
| `fin:{symbol}:latest` | 12h | JSON | Latest quarter financial summary |
| `cal:latest` | 24h | JSON | Latest trading calendar month |
| `index:{code}:daily:{days}` | 1h | Parquet bytes | Index daily data |
| `lhb:{date}` | 24h | JSON | Dragon-tiger board for date |
| `flow:{symbol}:daily:{days}` | 6h | Parquet bytes | Recent capital flow data |
| `hot:rank` | 5m | JSON | Current hot stock ranking |
| `market:overview` | 5s | JSON | Market overview (indices, breadth) |
| `lock:akshare:global` | 30s | String ("1") | Distributed lock for AKShare calls |
| `lock:efinance:global` | 10s | String ("1") | Distributed lock for efinance calls |
| `lock:update:daily` | 3600s | String | Update job status lock |

### 4.2 Cache Implementation

```python
# -- [AI:BEGIN]
# src/pa_mcp/data/cache.py

import redis
import json
import msgpack
import io
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis caching layer with tiered strategy."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.from_url(redis_url, decode_responses=False)

    def get_realtime_quote(self, symbol: str) -> Optional[dict]:
        """Get real-time quote from Redis (3s TTL)."""
        key = f"quote:{symbol}:realtime"
        data = self.client.get(key)
        if data:
            return msgpack.unpackb(data, raw=False)
        return None

    def set_realtime_quote(self, symbol: str, data: dict):
        """Set real-time quote with 3s TTL."""
        key = f"quote:{symbol}:realtime"
        packed = msgpack.dumps(data, use_bin_type=True)
        self.client.setex(key, 3, packed)

    def get_daily_kline(self, symbol: str, adjust: str, days: int) -> Optional[pd.DataFrame]:
        """Get cached kline as DataFrame. Uses Parquet serialization for speed."""
        key = f"kline:{symbol}:daily:{adjust}:{days}"
        data = self.client.get(key)
        if data:
            buf = io.BytesIO(data)
            return pd.read_parquet(buf)
        return None

    def set_daily_kline(self, symbol: str, adjust: str, days: int, df: pd.DataFrame):
        """Cache kline with 6h TTL."""
        key = f"kline:{symbol}:daily:{adjust}:{days}"
        buf = io.BytesIO()
        df.to_parquet(buf, compression="zstd", index=False)
        self.client.setex(key, 21600, buf.getvalue())

    def acquire_lock(self, lock_name: str, ttl: int = 30) -> bool:
        """Try to acquire a distributed lock."""
        return bool(self.client.set(f"lock:{lock_name}", "1", nx=True, ex=ttl))

    def release_lock(self, lock_name: str):
        """Release a distributed lock."""
        self.client.delete(f"lock:{lock_name}")

    def warmup_daily(self, symbols: list[str]):
        """Pre-cache daily kline for a watchlist. Called before market open."""
        for symbol in symbols:
            try:
                key = f"kline:{symbol}:daily:qfq:365"
                if not self.client.exists(key):
                    logger.info(f"[Cache] Warming up daily kline for {symbol}")
            except Exception as e:
                logger.error(f"[Cache] Warmup failed for {symbol}: {e}")
# -- [AI:END]
```

### 4.3 What to Cache vs. Direct DuckDB

| Cache | Direct to DuckDB |
|-------|-----------------|
| Real-time quotes (3s TTL) | Full historical daily kline |
| Last 365 days daily kline per symbol | Minute kline history beyond 5 days |
| Latest financial summary (12h TTL) | Complete financial statements history |
| Full market spot (5s TTL) | Full capital flow history |
| Market overview snapshot (5s TTL) | All dragon-tiger board history |
| Hot ranking (5m TTL) | Trade calendar (rarely changes) |
| Today's dragon-tiger (24h TTL) | Stock basic info (rarely changes) |
| Index latest (1h TTL) | All historical index data |

---

## 5. Data Layer — Cron Update Jobs

### 5.1 Scheduler Architecture

```python
# -- [AI:BEGIN]
# src/pa_mcp/data/scheduler.py

import asyncio
import logging
from datetime import datetime, time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class UpdateJob:
    name: str
    func: Callable
    schedule_time: time
    status: JobStatus = JobStatus.PENDING
    last_run: Optional[datetime] = None
    last_error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: int = 300       # seconds between retries
    timeout: int = 3600          # max runtime in seconds
    incremental: bool = True

class DataScheduler:
    """Manages daily data update cron jobs."""

    def __init__(self, cache: "RedisCache", aggregator: "DataAggregator", store: "DuckDBStore"):
        self.cache = cache
        self.aggregator = aggregator
        self.store = store

    def define_jobs(self) -> list[UpdateJob]:
        """Define the daily update pipeline."""
        return [
            UpdateJob(name="update_trade_calendar", func=self._update_trade_calendar,
                      schedule_time=time(17, 30), timeout=120),
            UpdateJob(name="update_stock_basic", func=self._update_stock_basic,
                      schedule_time=time(17, 32), timeout=300),
            UpdateJob(name="update_daily_kline", func=self._update_daily_kline,
                      schedule_time=time(17, 35), timeout=3600, max_retries=5, retry_delay=600),
            UpdateJob(name="update_minute_kline", func=self._update_minute_kline,
                      schedule_time=time(18, 35), timeout=1800),
            UpdateJob(name="update_financials", func=self._update_financials,
                      schedule_time=time(19, 5), timeout=1800),
            UpdateJob(name="update_capital_flow", func=self._update_capital_flow,
                      schedule_time=time(19, 35), timeout=1200),
            UpdateJob(name="update_dragon_tiger", func=self._update_dragon_tiger,
                      schedule_time=time(20, 0), timeout=600),
            UpdateJob(name="update_index_daily", func=self._update_index_daily,
                      schedule_time=time(20, 10), timeout=300),
            UpdateJob(name="compute_technical_indicators", func=self._compute_technical_indicators,
                      schedule_time=time(20, 15), timeout=1800),
        ]

    async def run_daily_pipeline(self):
        """Execute all update jobs in sequence after 17:30."""
        if not self.cache.acquire_lock("update:daily", ttl=7200):
            logger.warning("[Scheduler] Another update pipeline is already running")
            return

        try:
            jobs = self.define_jobs()
            for job in jobs:
                await self._run_job_with_retry(job)
        finally:
            self.cache.release_lock("update:daily")

    async def _run_job_with_retry(self, job: UpdateJob):
        """Run a single job with retry logic."""
        job.status = JobStatus.RUNNING
        job.last_run = datetime.now()
        job.retry_count = 0

        while job.retry_count <= job.max_retries:
            try:
                logger.info(f"[Scheduler] Starting job: {job.name} (attempt {job.retry_count+1})")
                await asyncio.wait_for(job.func(), timeout=job.timeout)
                job.status = JobStatus.SUCCESS
                logger.info(f"[Scheduler] Job completed: {job.name}")
                return
            except asyncio.TimeoutError:
                job.last_error = f"Timeout after {job.timeout}s"
                logger.error(f"[Scheduler] Job {job.name} timed out")
            except Exception as e:
                job.last_error = str(e)
                logger.error(f"[Scheduler] Job {job.name} failed: {e}")

            job.retry_count += 1
            if job.retry_count <= job.max_retries:
                job.status = JobStatus.RETRYING
                delay = job.retry_delay * job.retry_count
                logger.info(f"[Scheduler] Retrying {job.name} in {delay}s...")
                await asyncio.sleep(delay)

        job.status = JobStatus.FAILED
        self._send_alert(job)

    def _send_alert(self, job: UpdateJob):
        logger.critical(f"ALERT: Job {job.name} FAILED permanently. Last error: {job.last_error}")

    # -- Placeholder implementations for each update task --
    async def _update_trade_calendar(self): pass
    async def _update_stock_basic(self): pass
    async def _update_daily_kline(self): pass
    async def _update_minute_kline(self): pass
    async def _update_financials(self): pass
    async def _update_capital_flow(self): pass
    async def _update_dragon_tiger(self): pass
    async def _update_index_daily(self): pass
    async def _compute_technical_indicators(self): pass
# -- [AI:END]
```

### 5.2 Incremental vs Full Update Strategy

| Table | Incremental Strategy | Full Refresh Trigger |
|-------|---------------------|---------------------|
| `stock_daily` | Fetch from `MAX(date)` - 3 days buffer | On init / weekends / if gap > 30 days |
| `stock_minute` | Fetch today only | Rarely needed |
| `stock_realtime` | Always full (overwrite) | Every write |
| `financials_*` | Check latest `report_date`, fetch new quarters | On init / fiscal year rollover |
| `capital_flow` | Fetch last 3 days | On init |
| `dragon_tiger` | Fetch today's date only | On init |
| `trade_calendar` | Fetch current year, update on Dec 31 for next year | On init |
| `stock_basic` | Fetch full list, upsert by symbol | Always full (lightweight) |

---

## 6. Data Layer — Data Quality Validation

### 6.1 Validation Pipeline

```python
# -- [AI:BEGIN]
# src/pa_mcp/data/quality.py

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    PASS = "pass"
    WARN = "warn"
    ERROR = "error"

@dataclass
class ValidationResult:
    check_name: str
    level: ValidationLevel
    passed: bool
    details: str = ""
    affected_rows: int = 0
    symbol: Optional[str] = None
    date: Optional[str] = None

@dataclass
class ValidationReport:
    results: list[ValidationResult] = field(default_factory=list)
    overall_passed: bool = True

    def add(self, result: ValidationResult):
        self.results.append(result)
        if result.level == ValidationLevel.ERROR:
            self.overall_passed = False


class DataQualityValidator:
    """Data quality validation checks for incoming data."""

    def validate_daily_kline(self, df: pd.DataFrame, symbol: str) -> ValidationReport:
        report = ValidationReport()

        # 1. Row count check
        report.add(self._check_row_count(df, symbol, min_rows=1))

        # 2. Required columns check
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        report.add(self._check_required_columns(df, required_cols, symbol))

        # 3. OHLC sanity
        report.add(self._check_ohlc_sanity(df, symbol))

        # 4. Non-negative volume/amount
        report.add(self._check_non_negative(df, "volume", symbol))
        report.add(self._check_non_negative(df, "amount", symbol))

        # 5. Duplicate check
        report.add(self._check_duplicates(df, ["date"], symbol))

        # 6. Date continuity
        report.add(self._check_date_continuity(df, symbol))

        # 7. Price jumps anomaly detection
        report.add(self._check_price_jumps(df, symbol, threshold=0.30))

        # 8. Staleness check
        report.add(self._check_staleness(df, symbol, max_days_behind=3))

        # 9. Adjust flag consistency
        if "adjust_flag" in df.columns:
            report.add(self._check_adjust_consistency(df, symbol))

        return report

    def _check_row_count(self, df, symbol, min_rows=1):
        passed = len(df) >= min_rows
        return ValidationResult(
            check_name="row_count",
            level=ValidationLevel.ERROR if not passed else ValidationLevel.PASS,
            passed=passed,
            details=f"Row count: {len(df)} (min: {min_rows})",
            symbol=symbol,
        )

    def _check_required_columns(self, df, required, symbol):
        missing = [c for c in required if c not in df.columns]
        passed = len(missing) == 0
        return ValidationResult(
            check_name="required_columns",
            level=ValidationLevel.ERROR if not passed else ValidationLevel.PASS,
            passed=passed,
            details=f"Missing columns: {missing}" if missing else "All required columns present",
            symbol=symbol,
        )

    def _check_ohlc_sanity(self, df, symbol):
        bad_rows = df[
            (df["high"] < df[["open", "close"]].max(axis=1)) |
            (df["low"] > df[["open", "close"]].min(axis=1))
        ]
        passed = len(bad_rows) == 0
        return ValidationResult(
            check_name="ohlc_sanity",
            level=ValidationLevel.WARN,
            passed=passed,
            details=f"Rows with OHLC violations: {len(bad_rows)}",
            affected_rows=len(bad_rows),
            symbol=symbol,
        )

    def _check_non_negative(self, df, col, symbol):
        neg = df[df[col] < 0]
        passed = len(neg) == 0
        return ValidationResult(
            check_name=f"non_negative_{col}",
            level=ValidationLevel.ERROR if not passed else ValidationLevel.PASS,
            passed=passed,
            details=f"Negative {col} rows: {len(neg)}",
            symbol=symbol,
        )

    def _check_duplicates(self, df, keys, symbol):
        dup = df.duplicated(subset=keys).sum()
        passed = dup == 0
        return ValidationResult(
            check_name="duplicates",
            level=ValidationLevel.WARN,
            passed=passed,
            details=f"Duplicate rows: {dup}",
            affected_rows=dup,
            symbol=symbol,
        )

    def _check_date_continuity(self, df, symbol):
        if len(df) < 2:
            return ValidationResult(check_name="date_continuity", level=ValidationLevel.PASS, passed=True, symbol=symbol)
        dates = df["date"].sort_values()
        gaps = dates.diff().dt.days
        large_gaps = gaps[gaps > 5]
        passed = len(large_gaps) == 0
        return ValidationResult(
            check_name="date_continuity",
            level=ValidationLevel.WARN,
            passed=passed,
            details=f"Gaps > 5 days: {len(large_gaps)}.",
            symbol=symbol,
        )

    def _check_price_jumps(self, df, symbol, threshold=0.30):
        if len(df) < 2:
            return ValidationResult(check_name="price_jumps", level=ValidationLevel.PASS, passed=True, symbol=symbol)
        returns = df["close"].pct_change().abs()
        jumps = returns[returns > threshold]
        passed = len(jumps) == 0
        return ValidationResult(
            check_name="price_jumps",
            level=ValidationLevel.WARN,
            passed=passed,
            details=f"Price jumps > {threshold*100}%: {len(jumps)}",
            affected_rows=len(jumps),
            symbol=symbol,
        )

    def _check_staleness(self, df, symbol, max_days_behind=3):
        latest = df["date"].max()
        days_behind = (pd.Timestamp.now() - latest).days
        passed = days_behind <= max_days_behind
        return ValidationResult(
            check_name="staleness",
            level=ValidationLevel.WARN,
            passed=passed,
            details=f"Latest date: {latest.date()}, {days_behind} days behind today",
            symbol=symbol,
        )

    def _check_adjust_consistency(self, df, symbol):
        flags = df["adjust_flag"].unique()
        passed = len(flags) == 1
        return ValidationResult(
            check_name="adjust_consistency",
            level=ValidationLevel.WARN,
            passed=passed,
            details=f"Multiple adjust flags found: {list(flags)}" if not passed else "OK",
            symbol=symbol,
        )
# -- [AI:END]
```

---

## 7. Strategy Engine — Base Class & Registry

### 7.1 Base Strategy Class (Complete Interface)

```python
# -- [AI:BEGIN]
# src/pa_mcp/strategy/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import pandas as pd

# ---- Enums ----

class StrategyCategory(Enum):
    LIMIT_UP = "limit_up"       # First board / relay / leader
    TREND = "trend"             # Trend following
    SWING = "swing"             # Swing trading
    VALUE = "value"             # Value investing
    GRID = "grid"               # Grid trading
    DIP = "dip"                 # Dip buying
    HALF_WAY = "half_way"       # Mid-path entry
    MARKET_STATE = "market_state"

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ALERT = "alert"

class MarketState(Enum):
    CLIMAX = "climax"
    FERMENT = "ferment"
    STARTUP = "startup"
    DOWNTURN = "downturn"
    ICE = "ice"

class TimeFrame(Enum):
    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    M60 = "60m"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

# ---- Signal Dataclass ----

@dataclass
class Signal:
    symbol: str
    strategy_name: str
    signal_type: SignalType
    confidence: float                          # 0.0 - 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: list[float] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    indicators: dict[str, float] = field(default_factory=dict)
    timestamp: Optional[str] = None

# ---- Parameter Space ----

@dataclass
class ParamRange:
    name: str
    min_val: float
    max_val: float
    step: float
    param_type: Literal["int", "float", "choice"] = "float"
    choices: Optional[list] = None

# ---- Base Strategy ----

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    # -- Metadata (override in subclass) --
    name: str = "base"
    category: StrategyCategory = StrategyCategory.TREND
    description: str = "Base strategy"
    version: str = "1.0.0"
    timeframe: TimeFrame = TimeFrame.DAILY
    author: str = "PA_MCP"
    tags: list[str] = []

    # -- Market state suitability (0.0 = unsuitable, 1.0 = ideal) --
    market_state_suitability: dict[MarketState, float] = {
        MarketState.CLIMAX: 0.5,
        MarketState.FERMENT: 0.7,
        MarketState.STARTUP: 0.8,
        MarketState.DOWNTURN: 0.3,
        MarketState.ICE: 0.2,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = params or self.default_params()

    @abstractmethod
    def default_params(self) -> dict:
        """Return default parameters. Subclass MUST override."""
        return {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with columns:
                date, open, high, low, close, volume, amount, turnover

        Returns:
            List of Signal objects (can be empty = no signal)

        Raises:
            ValueError: If data is missing required columns
            RuntimeError: If indicator computation fails
        """
        pass

    @abstractmethod
    def get_params_space(self) -> list[ParamRange]:
        """
        Define parameter search space for optimization.
        Return empty list if strategy has no tunable params.
        """
        return []

    def validate(self) -> bool:
        """Validate strategy configuration. Raises ValueError if invalid."""
        if not self.name or self.name == "base":
            raise ValueError(f"Strategy name must be set (got: '{self.name}')")
        if not isinstance(self.category, StrategyCategory):
            raise ValueError(f"Invalid category: {self.category}")
        return True

    def get_info(self) -> dict:
        """Return strategy metadata as dict (for MCP tool list_strategies)."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "version": self.version,
            "timeframe": self.timeframe.value,
            "tags": self.tags,
            "params": self.params,
            "params_space": [
                {"name": p.name, "min": p.min_val, "max": p.max_val,
                 "step": p.step, "type": p.param_type}
                for p in self.get_params_space()
            ],
            "market_state_suitability": {
                k.value: v for k, v in self.market_state_suitability.items()
            },
        }

    def _validate_data(self, data: pd.DataFrame, required_cols: list[str]):
        """Ensure DataFrame has required columns."""
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(
                f"[{self.name}] Missing required columns: {missing}. "
                f"Available: {list(data.columns)}"
            )

    def _safe_div(self, a: float, b: float, default: float = 0.0) -> float:
        """Safe division to avoid ZeroDivisionError."""
        return a / b if b != 0 else default
# -- [AI:END]
```

### 7.2 Strategy Registry

```python
# -- [AI:BEGIN]
# src/pa_mcp/strategy/registry.py

import importlib
import pkgutil
from pathlib import Path
from typing import Type, Optional
import inspect

from .base import BaseStrategy, StrategyCategory

class StrategyRegistry:
    """Registry for discovering, registering, and listing strategies."""

    _instance: Optional["StrategyRegistry"] = None
    _strategies: dict[str, Type[BaseStrategy]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._strategies = {}
        return cls._instance

    def register(self, strategy_class: Type[BaseStrategy]):
        """Register a strategy class by its name."""
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"{strategy_class} must be a subclass of BaseStrategy")
        temp = strategy_class()
        name = temp.name
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' is already registered")
        self._strategies[name] = strategy_class

    def discover(self, package_path: str = "pa_mcp.strategy.strategies"):
        """
        Auto-discover strategies in the strategies package.
        Any class inheriting from BaseStrategy with a non-empty name will be registered.
        """
        try:
            package = importlib.import_module(package_path)
            package_dir = Path(package.__file__).parent if package.__file__ else None

            for _, module_name, _ in pkgutil.iter_modules(
                [str(package_dir)] if package_dir else []
            ):
                module = importlib.import_module(f"{package_path}.{module_name}")
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseStrategy)
                        and obj is not BaseStrategy
                        and obj.name != "base"
                    ):
                        self.register(obj)
        except ModuleNotFoundError:
            pass  # Package doesn't exist yet

    def get(self, name: str) -> Type[BaseStrategy]:
        """Get strategy class by name."""
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found. Available: {self.list_names()}")
        return self._strategies[name]

    def list_by_category(self, category: Optional[StrategyCategory] = None) -> list[dict]:
        """List all strategies, optionally filtered by category."""
        result = []
        for name, cls in self._strategies.items():
            temp = cls()
            if category is None or temp.category == category:
                result.append(temp.get_info())
        return result

    def list_names(self) -> list[str]:
        """Return all registered strategy names."""
        return list(self._strategies.keys())

    def create(self, name: str, params: Optional[dict] = None) -> BaseStrategy:
        """Create a strategy instance by name with optional param override."""
        cls = self.get(name)
        return cls(params=params)

    def __len__(self):
        return len(self._strategies)


# Global singleton
strategy_registry = StrategyRegistry()
# -- [AI:END]
```

---

## 8. Strategy Engine — 40+ Strategies (8 Categories)

### 8.1 Strategy Catalog

Based on DeepPulse's 40 strategies and QuantsPlaybook's 100+ strategy patterns.

#### Category 1: Limit-Up / First Board (Shou Ban) — 5 strategies

| # | Name | Key Logic | Trigger Conditions |
|---|------|-----------|-------------------|
| 1 | **Low-Level First Board** | Price at 60-day low quartile, first limit-up in 20 days | Price < MA60 * 0.85, volume > MA(V,20) * 2.0, is_limit_up |
| 2 | **First Board Re-Seal** | Board opens then re-seals; strength signal | limit_up, has_opened, current == limit_up, seal_time > 30min |
| 3 | **Theme First Board** | Sector leader hits first board on news catalyst | sector_rank <= 3, sector_pct_change > 3%, is_limit_up, first_in_20_days |
| 4 | **High-Turnover First Board** | High turnover first board, real buying | is_limit_up, turnover > 10%, volume_ratio > 3.0, market_cap < 5B |
| 5 | **Gap-Up First Board** | Opens at limit-up and holds; monitor for next day | open == limit_up, close == limit_up, volume > 0 |

```python
# -- [AI:BEGIN]
# Example: Low-Level First Board strategy core logic

def generate_signals_low_level_first_board(data: pd.DataFrame) -> list[Signal]:
    """
    Core logic:
    1. Price is in the lowest 25% of its 60-day range
    2. Today hits limit-up for the first time in >= 20 trading days
    3. Volume >= 2x 20-day average volume
    """
    signals = []
    if len(data) < 60:
        return signals

    latest = data.iloc[-1]
    price_low_pct = (latest["close"] - data["low"].rolling(60).min().iloc[-1]) / \
                    (data["high"].rolling(60).max().iloc[-1] - data["low"].rolling(60).min().iloc[-1])
    is_low_position = price_low_pct < 0.25
    is_limit_up = latest["pct_change"] >= 9.8

    limit_up_history = data["pct_change"] >= 9.8
    recent_limit_ups = limit_up_history.iloc[-21:-1]
    is_first_in_20 = recent_limit_ups.sum() == 0

    avg_vol_20 = data["volume"].rolling(20).mean().iloc[-2]
    volume_surge = latest["volume"] >= avg_vol_20 * 2.0 if avg_vol_20 > 0 else False

    if is_low_position and is_limit_up and is_first_in_20 and volume_surge:
        signals.append(Signal(
            symbol=data.get("symbol", [""])[-1] if "symbol" in data.columns else "",
            strategy_name="Low-Level First Board",
            signal_type=SignalType.BUY,
            confidence=0.75,
            entry_price=latest["close"],
            stop_loss=latest["close"] * 0.93,
            take_profit=[latest["close"] * 1.10, latest["close"] * 1.20],
            reasoning=[
                "Price at 60-day low quartile (deep dip)",
                "First limit-up in 20+ days (fresh catalyst)",
                f"Volume {latest['volume']:.0f} exceeds 2x MA20 ({avg_vol_20:.0f})",
            ],
        ))
    return signals
# -- [AI:END]
```

#### Category 2: Relay / Continuation (Jie Li) — 6 strategies

| # | Name | Key Logic |
|---|------|-----------|
| 6 | **2nd Board Relay** | Day-2 limit-up after Day-1 limit-up, volume expands |
| 7 | **High-Turnover 2nd Board** | 2nd board with turnover > 15%, no gap-down opening |
| 8 | **Accelerating 2nd Board** | 2nd board with shrinking volume (lock-up sign) |
| 9 | **3rd Board Relay** | 3rd consecutive board, volume controlled, sector still hot |
| 10 | **High-Level Relay** | Board > 5th day, need volume + sentiment confirmation |
| 11 | **High-Level T-Bar Board** | T-shaped candle (open=low=limit_up, close near high) |

```python
# -- [AI:BEGIN]
# 2nd Board Relay: core logic
def relay_2nd_board_signal(data):
    yest = data.iloc[-2]
    today = data.iloc[-1]
    if yest["pct_change"] >= 9.8 and today["pct_change"] >= 9.8:
        if today["volume"] >= yest["volume"] * 0.8:
            if today["open"] > yest["close"] * 0.95:
                conf = 0.70 if today["volume"] >= yest["volume"] else 0.55
                return [Signal(strategy_name="2nd Board Relay",
                               signal_type=SignalType.BUY, confidence=conf,
                               entry_price=today["close"])]
    return []
# -- [AI:END]
```

#### Category 3: Leader / Dragon Head (Long Tou) — 5 strategies

| # | Name | Key Logic |
|---|------|-----------|
| 12 | **Dragon Head Leader** | Highest consecutive boards in hottest sector, volume controlled |
| 13 | **Dragon 2nd Wave** | After pullback to MA10/MA20, rebounds with volume surge |
| 14 | **Position Stealing** | When sector leader weakens, #2 stock takes over |
| 15 | **Catch-Up Dragon** | When leader gaps too high, look for lagging sector stocks |
| 16 | **Leader Sentiment Cycle** | Track leader board height to gauge market sentiment phase |

```python
# -- [AI:BEGIN]
# Dragon Head 2nd Wave: core logic
def dragon_2nd_wave(data):
    close = data["close"]
    volume = data["volume"]
    ma20 = close.rolling(20).mean().iloc[-1]
    today = data.iloc[-1]

    pullback_to_ma20 = abs(today["close"] - ma20) / ma20 < 0.03
    vol_peak = volume.rolling(20).max().iloc[-1]
    vol_washout = volume.rolling(5).mean().iloc[-2] < vol_peak * 0.5
    vol_resurgence = today["volume"] > volume.rolling(5).mean().iloc[-2] * 1.5
    breakout = today["close"] > data["high"].iloc[-2]

    if pullback_to_ma20 and vol_washout and vol_resurgence and breakout:
        return [Signal(strategy_name="Dragon 2nd Wave", signal_type=SignalType.BUY,
                       confidence=0.65, entry_price=today["close"])]
    return []
# -- [AI:END]
```

#### Category 4: Dip Buying (Di Xi) — 6 strategies

| # | Name | Key Logic |
|---|------|-----------|
| 17 | **5-Day MA Dip** | Price pulls back to MA5 from above, volume shrinking |
| 18 | **10-Day MA Dip** | Strong trend stock dips to MA10, bounce signal |
| 19 | **Platform Support** | Price reaches prior consolidation platform bottom |
| 20 | **Dragon First Red Engulf** | Dragon stock first red day (-3% to -7%), next day green engulfing |
| 21 | **Divergence Dip** | Stock limit-up opens then drops to +3~5% |
| 22 | **Oversold Dip** | RSI(14) < 25, distance from MA60 > 25%, volume climax |

#### Category 5: Trend Following (Qu Shi) — 5 strategies

| # | Name | Key Logic |
|---|------|-----------|
| 23 | **MA Bullish Alignment** | MA5 > MA10 > MA20 > MA60, all sloping up |
| 24 | **Platform Breakout** | 15-30 day consolidation, breakout above platform high with 2x volume |
| 25 | **Cup & Handle** | 20-60 day cup + 5-10 day handle; breakout above rim |
| 26 | **Turtle Trading** | Donchian channel breakout (20-day high entry, 10-day low exit) |
| 27 | **MA Golden Cross** | MA20 crosses above MA60 |

```python
# -- [AI:BEGIN]
# Platform Breakout: core logic
def platform_breakout(data):
    consol_high = data["high"].iloc[-21:-1].max()
    consol_low = data["low"].iloc[-21:-1].min()
    is_consolidating = (consol_high / consol_low - 1) < 0.15

    today = data.iloc[-1]
    breaks_out = today["close"] > consol_high

    avg_vol = data["volume"].iloc[-21:-1].mean()
    vol_surge = today["volume"] > avg_vol * 2.0

    ema12 = data["close"].ewm(span=12).mean()
    ema26 = data["close"].ewm(span=26).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    mom_positive = macd_hist.iloc[-1] > 0

    if is_consolidating and breaks_out and vol_surge and mom_positive:
        return [Signal(strategy_name="Platform Breakout", signal_type=SignalType.BUY,
                       confidence=0.72, stop_loss=consol_low * 0.98)]
    return []
# -- [AI:END]
```

#### Category 6: Swing Trading (Bo Duan) — 5 strategies

| # | Name | Key Logic |
|---|------|-----------|
| 28 | **MACD Cross** | Golden cross buy, dead cross sell |
| 29 | **Bollinger Bands** | Buy at lower band when BB width expanding, sell at upper band |
| 30 | **RSI Overbought/Oversold** | RSI(14) < 30 buy, RSI(14) > 70 sell + trend filter |
| 31 | **KDJ Resonance** | K/D/J triple golden cross + price above MA20 |
| 32 | **Elliott Wave (simplified)** | Identify impulse wave 3 entry, wave 5 exit |

```python
# -- [AI:BEGIN]
# Bollinger Bands Mean-Reversion: core logic
def bollinger_swing(data):
    close = data["close"]
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid

    today = data.iloc[-1]

    if today["close"] < bb_lower.iloc[-1] and bb_width.iloc[-1] > bb_width.iloc[-2]:
        return [Signal(strategy_name="Bollinger Swing", signal_type=SignalType.BUY,
                       confidence=0.55, entry_price=today["close"],
                       stop_loss=today["close"] * 0.95)]
    return []
# -- [AI:END]
```

#### Category 7: Value Investing (Jia Zhi) — 5 strategies

| # | Name | Key Logic |
|---|------|-----------|
| 33 | **Low PE + High Dividend** | PE(TTM) < industry_median * 0.7, dividend yield > 3% |
| 34 | **ROE-PB Framework** | ROE > 15%, PB < industry median |
| 35 | **PEG Growth** | PEG < 0.8, revenue growth > 20%, net profit growth > 15% |
| 36 | **Graham Net-Net** | Current assets - total liabilities > market cap |
| 37 | **DuPont Screening** | ROE decomposed: high margin + high turnover |

#### Category 8: Grid / Systematic (Wang Ge) — 5 strategies

| # | Name | Key Logic |
|---|------|-----------|
| 38 | **Range Grid** | When ADX < 20, grid between support/resistance |
| 39 | **Trend Grid** | Grid skewed in trend direction |
| 40 | **Dynamic Grid** | Grid width adjusts to ATR |
| 41 | **DCA Strategy** | Fixed interval buy regardless of price |
| 42 | **Volatility Grid** | Grid spacing = ATR * multiplier |

```python
# -- [AI:BEGIN]
# Range Grid: core logic
def range_grid(data):
    adx = compute_adx(data, 14)
    if adx.iloc[-1] >= 20:
        return []

    support = data["low"].rolling(20).min().iloc[-1]
    resistance = data["high"].rolling(20).max().iloc[-1]
    grid_range = resistance - support
    if grid_range / support < 0.05:
        return []

    levels = [support + grid_range * i / 5 for i in range(6)]
    current = data["close"].iloc[-1]

    for i in range(5):
        if levels[i] <= current < levels[i+1]:
            if i <= 1:
                return [Signal(strategy_name="Range Grid", signal_type=SignalType.BUY,
                               confidence=0.50, entry_price=current,
                               stop_loss=support * 0.98)]
            elif i >= 4:
                return [Signal(strategy_name="Range Grid", signal_type=SignalType.SELL,
                               confidence=0.50)]
    return []
# -- [AI:END]
```

---

## 9. Strategy Engine — Signal Format (JSON Schema)

### 9.1 Complete JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://pa-mcp.dev/schemas/signal.json",
  "title": "Trading Signal",
  "type": "object",
  "required": ["signal_id", "symbol", "strategy_name", "signal_type", "confidence", "timestamp"],
  "properties": {
    "signal_id": {
      "type": "string",
      "description": "UUID v4 for unique signal tracking",
      "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    },
    "symbol": {
      "type": "string",
      "description": "6-digit A-share code",
      "pattern": "^[0-9]{6}$"
    },
    "name": {
      "type": "string",
      "description": "Stock name (optional)"
    },
    "strategy_name": {
      "type": "string",
      "minLength": 1
    },
    "strategy_category": {
      "type": "string",
      "enum": ["limit_up", "trend", "swing", "value", "grid", "dip", "half_way", "market_state"]
    },
    "signal_type": {
      "type": "string",
      "enum": ["buy", "sell", "hold", "alert"]
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "entry_price": {
      "type": "number",
      "exclusiveMinimum": 0
    },
    "stop_loss": {
      "type": "number",
      "exclusiveMinimum": 0
    },
    "take_profit": {
      "type": "array",
      "items": {"type": "number", "exclusiveMinimum": 0},
      "minItems": 1,
      "maxItems": 5
    },
    "position_pct": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "reasoning": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 1
    },
    "risk_factors": {
      "type": "array",
      "items": {"type": "string"}
    },
    "indicators": {
      "type": "object",
      "properties": {
        "rsi_14": {"type": "number"},
        "macd": {"type": "number"},
        "macd_signal": {"type": "number"},
        "macd_hist": {"type": "number"},
        "ma_5": {"type": "number"},
        "ma_10": {"type": "number"},
        "ma_20": {"type": "number"},
        "ma_60": {"type": "number"},
        "volume_ratio": {"type": "number"},
        "atr_14": {"type": "number"},
        "adx_14": {"type": "number"},
        "bollinger_upper": {"type": "number"},
        "bollinger_lower": {"type": "number"}
      },
      "additionalProperties": true
    },
    "market_state": {
      "type": "string",
      "enum": ["climax", "ferment", "startup", "downturn", "ice", "unknown"]
    },
    "expected_holding_days": {
      "type": "integer",
      "minimum": 1,
      "maximum": 365
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "expires_at": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

### 9.2 Aggregated Signal Output (Multi-Strategy)

```json
{
  "symbol": "000001",
  "name": "Ping An Bank",
  "timestamp": "2026-07-26T09:45:00+08:00",
  "market_state": "startup",
  "aggregated_signal": "buy",
  "aggregated_confidence": 0.72,
  "vote_summary": {
    "buy_votes": 5,
    "sell_votes": 1,
    "hold_votes": 3,
    "total_strategies_fired": 9
  },
  "weighted_score": 0.65,
  "consensus_level": "moderate",
  "individual_signals": [
    {
      "strategy_name": "Platform Breakout",
      "signal_type": "buy",
      "confidence": 0.78,
      "weight": 0.20
    }
  ],
  "conflict_analysis": {
    "has_conflict": true,
    "conflict_strategies": ["MA10 Dip Buy (buy)", "RSI Overbought (sell)"],
    "resolution": "Weighted vote favors buy (5:1)"
  }
}
```

---

## 10. Strategy Engine — Parameter Optimization

### 10.1 Architecture

```python
# -- [AI:BEGIN]
# src/pa_mcp/strategy/optimizer.py

import numpy as np
import pandas as pd
from itertools import product
from typing import Callable
from dataclasses import dataclass
import random
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    best_params: dict
    best_score: float
    all_results: pd.DataFrame
    param_sensitivity: dict[str, list[tuple[float, float]]]
    method: str
    trials: int
    elapsed_seconds: float

class StrategyOptimizer:
    """
    Three optimization methods:
    1. Grid Search -- exhaustive, for small param spaces (< 1000 combos)
    2. Monte Carlo -- random sampling, for medium spaces
    3. Genetic Algorithm -- evolutionary, for large spaces
    """

    METRICS = ["sharpe_ratio", "total_return", "calmar_ratio", "win_rate", "profit_factor"]

    def __init__(
        self,
        backtest_func: Callable[[dict], dict],
        objective: str = "sharpe_ratio",
        maximize: bool = True,
    ):
        if objective not in self.METRICS:
            raise ValueError(f"Unknown objective: {objective}")
        self.backtest_func = backtest_func
        self.objective = objective
        self.maximize = maximize

    def grid_search(self, param_space: list, max_combinations: int = 1000) -> OptimizationResult:
        """Exhaustive grid search. Falls back to Monte Carlo if too many combos."""
        grids = []
        for pr in param_space:
            if pr.param_type == "choice" and pr.choices:
                grids.append(pr.choices)
            elif pr.param_type == "int":
                vals = np.arange(pr.min_val, pr.max_val + pr.step, pr.step, dtype=int)
                grids.append(list(vals))
            else:
                vals = np.arange(pr.min_val, pr.max_val + pr.step, pr.step)
                grids.append(list(round(v, 4) for v in vals))

        all_combos = list(product(*grids))
        if len(all_combos) > max_combinations:
            logger.warning(f"Grid too large ({len(all_combos)} combos). Falling back to Monte Carlo.")
            return self.monte_carlo(param_space, max_combinations)

        return self._evaluate_combinations(all_combos, [pr.name for pr in param_space], method="grid")

    def monte_carlo(self, param_space: list, samples: int = 500) -> OptimizationResult:
        """Random sampling of parameter space."""
        all_combos = []
        for _ in range(samples):
            combo = []
            for pr in param_space:
                if pr.param_type == "choice" and pr.choices:
                    combo.append(random.choice(pr.choices))
                elif pr.param_type == "int":
                    combo.append(random.randint(int(pr.min_val), int(pr.max_val)))
                else:
                    combo.append(round(random.uniform(pr.min_val, pr.max_val), 4))
            all_combos.append(tuple(combo))
        return self._evaluate_combinations(all_combos, [pr.name for pr in param_space], method="monte_carlo")

    def genetic_algorithm(
        self, param_space: list,
        population_size: int = 50, generations: int = 20,
        mutation_rate: float = 0.1, elitism: int = 5,
    ) -> OptimizationResult:
        """Genetic algorithm for large parameter spaces."""
        param_names = [pr.name for pr in param_space]

        def _random_individual():
            ind = []
            for pr in param_space:
                if pr.param_type == "choice" and pr.choices:
                    ind.append(random.choice(pr.choices))
                elif pr.param_type == "int":
                    ind.append(random.randint(int(pr.min_val), int(pr.max_val)))
                else:
                    ind.append(round(random.uniform(pr.min_val, pr.max_val), 4))
            return tuple(ind)

        def _crossover(p1, p2):
            return tuple(p1[i] if random.random() < 0.5 else p2[i] for i in range(len(p1)))

        def _mutate(ind):
            mutated = list(ind)
            for i, pr in enumerate(param_space):
                if random.random() < mutation_rate:
                    if pr.param_type == "choice" and pr.choices:
                        mutated[i] = random.choice(pr.choices)
                    elif pr.param_type == "int":
                        mutated[i] = random.randint(int(pr.min_val), int(pr.max_val))
                    else:
                        mutated[i] = round(random.uniform(pr.min_val, pr.max_val), 4)
            return tuple(mutated)

        population = [_random_individual() for _ in range(population_size)]
        all_evaluated = {}

        for gen in range(generations):
            scores = {}
            for ind in population:
                if ind not in all_evaluated:
                    params = dict(zip(param_names, ind))
                    result = self.backtest_func(params)
                    scores[ind] = result.get(self.objective, 0)
                    all_evaluated[ind] = scores[ind]

            sorted_pop = sorted(population, key=lambda x: scores.get(x, 0), reverse=self.maximize)
            new_pop = sorted_pop[:elitism]

            while len(new_pop) < population_size:
                t1 = random.choice(sorted_pop[:population_size // 2])
                t2 = random.choice(sorted_pop[:population_size // 2])
                new_pop.append(_mutate(_crossover(t1, t2)))

            population = new_pop

        best_ind = sorted_pop[0]
        best_params = dict(zip(param_names, best_ind))
        best_score = all_evaluated.get(best_ind, 0)

        results_df = pd.DataFrame([
            {"params": dict(zip(param_names, ind)), "score": score}
            for ind, score in all_evaluated.items()
        ])

        return OptimizationResult(
            best_params=best_params, best_score=best_score,
            all_results=results_df,
            param_sensitivity=self._compute_sensitivity(all_evaluated, param_names),
            method="genetic", trials=len(all_evaluated), elapsed_seconds=0.0,
        )

    def _evaluate_combinations(self, combos, param_names, method):
        results = {}
        for combo in combos:
            params = dict(zip(param_names, combo))
            bt_result = self.backtest_func(params)
            results[combo] = bt_result.get(self.objective, 0)

        best_combo = max(results, key=results.get) if self.maximize else min(results, key=results.get)
        best_params = dict(zip(param_names, best_combo))

        results_df = pd.DataFrame([
            {"params": dict(zip(param_names, combo)), "score": score}
            for combo, score in results.items()
        ])

        return OptimizationResult(
            best_params=best_params, best_score=results[best_combo],
            all_results=results_df,
            param_sensitivity=self._compute_sensitivity(results, param_names),
            method=method, trials=len(combos), elapsed_seconds=0.0,
        )

    def _compute_sensitivity(self, results, param_names):
        sensitivity = {}
        for i, name in enumerate(param_names):
            buckets = {}
            for combo, score in results.items():
                val = combo[i]
                buckets.setdefault(val, []).append(score)
            sorted_buckets = sorted(
                [(val, np.mean(scores)) for val, scores in buckets.items()]
            )
            sensitivity[name] = sorted_buckets
        return sensitivity
# -- [AI:END]
```

### 10.2 Integration with Backtest

```python
# -- [AI:BEGIN]
def optimize_strategy_workflow(
    strategy_name: str, symbol: str, start: str, end: str,
    capital: float = 100000, method: str = "monte_carlo",
) -> OptimizationResult:
    """End-to-end optimization workflow. Called by MCP tool: optimize_strategy."""
    from ..backtest.engine import BacktestEngine
    from .registry import strategy_registry

    strat_cls = strategy_registry.get(strategy_name)
    temp = strat_cls()
    param_space = temp.get_params_space()

    if not param_space:
        raise ValueError(f"Strategy '{strategy_name}' has no tunable parameters")

    engine = BacktestEngine()

    def backtest_func(params: dict) -> dict:
        result = engine.backtest(
            symbol=symbol, strategy_name=strategy_name,
            start=start, end=end, capital=capital, params=params,
        )
        return {
            "sharpe_ratio": result["sharpe_ratio"],
            "total_return": result["total_return_pct"],
            "calmar_ratio": result["calmar_ratio"],
            "win_rate": result["win_rate"],
            "profit_factor": result["profit_factor"],
        }

    optimizer = StrategyOptimizer(backtest_func, objective="sharpe_ratio")

    if method == "grid":
        return optimizer.grid_search(param_space)
    elif method == "genetic":
        return optimizer.genetic_algorithm(param_space)
    else:
        return optimizer.monte_carlo(param_space)
# -- [AI:END]
```

---

## 11. Strategy Engine — Signal Aggregation

### 11.1 Conflict Resolution Logic

```python
# -- [AI:BEGIN]
# src/pa_mcp/strategy/signals.py

import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from collections import Counter

from .base import Signal, SignalType, MarketState

class AggregationMethod(Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    MARKET_STATE_WEIGHTED = "market_state_weighted"
    BAYESIAN_ENSEMBLE = "bayesian_ensemble"

@dataclass
class AggregatedSignal:
    symbol: str
    name: str
    aggregated_signal: SignalType
    aggregated_confidence: float
    vote_summary: dict
    weighted_score: float
    consensus_level: str
    individual_signals: list[dict]
    conflict_analysis: dict
    position_pct: float
    timestamp: str

class SignalAggregator:
    """Aggregates signals from multiple strategies into a single decision."""

    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.WEIGHTED_VOTE,
        strategy_weights: Optional[dict[str, float]] = None,
        market_state: MarketState = MarketState.DOWNTURN,
    ):
        self.method = method
        self.strategy_weights = strategy_weights or {}
        self.market_state = market_state

    def aggregate(self, signals: list[Signal]) -> Optional[AggregatedSignal]:
        if not signals:
            return None

        symbol = signals[0].symbol
        name = getattr(signals[0], "name", "")

        if self.method == AggregationMethod.MAJORITY_VOTE:
            return self._majority_vote(signals, symbol, str(name))
        elif self.method == AggregationMethod.WEIGHTED_VOTE:
            return self._weighted_vote(signals, symbol, str(name))
        elif self.method == AggregationMethod.MARKET_STATE_WEIGHTED:
            return self._market_state_weighted(signals, symbol, str(name))
        elif self.method == AggregationMethod.BAYESIAN_ENSEMBLE:
            return self._bayesian_ensemble(signals, symbol, str(name))
        return None

    def _majority_vote(self, signals, symbol, name):
        votes = Counter(s.signal_type for s in signals)
        total = len(signals)
        winner = votes.most_common(1)[0][0]
        confidence = votes[winner] / total

        return AggregatedSignal(
            symbol=symbol, name=name,
            aggregated_signal=winner, aggregated_confidence=confidence,
            vote_summary={
                "buy_votes": votes.get(SignalType.BUY, 0),
                "sell_votes": votes.get(SignalType.SELL, 0),
                "hold_votes": votes.get(SignalType.HOLD, 0),
                "total_strategies_fired": total,
            },
            weighted_score=0.5,
            consensus_level="strong" if confidence >= 0.7 else ("moderate" if confidence >= 0.5 else "weak"),
            individual_signals=[self._signal_to_dict(s) for s in signals],
            conflict_analysis=self._analyze_conflicts(signals),
            position_pct=self._suggest_position(signals, winner),
            timestamp=signals[0].timestamp or "",
        )

    def _weighted_vote(self, signals, symbol, name):
        buy_score = 0.0
        sell_score = 0.0

        for s in signals:
            strat_weight = self.strategy_weights.get(s.strategy_name, 1.0)
            if s.signal_type == SignalType.BUY:
                buy_score += s.confidence * strat_weight
            elif s.signal_type == SignalType.SELL:
                sell_score += s.confidence * strat_weight

        total_score = buy_score + sell_score
        if total_score == 0:
            winner, confidence = SignalType.HOLD, 0.0
        else:
            winner = SignalType.BUY if buy_score > sell_score else SignalType.SELL
            winner_score = max(buy_score, sell_score)
            confidence = min(winner_score / total_score, 1.0)

        return AggregatedSignal(
            symbol=symbol, name=name,
            aggregated_signal=winner, aggregated_confidence=round(confidence, 4),
            vote_summary={
                "buy_votes": sum(1 for s in signals if s.signal_type == SignalType.BUY),
                "sell_votes": sum(1 for s in signals if s.signal_type == SignalType.SELL),
                "hold_votes": sum(1 for s in signals if s.signal_type == SignalType.HOLD),
                "total_strategies_fired": len(signals),
            },
            weighted_score=round(buy_score - sell_score, 4),
            consensus_level="strong" if confidence >= 0.7 else ("moderate" if confidence >= 0.5 else "weak"),
            individual_signals=[self._signal_to_dict(s) for s in signals],
            conflict_analysis=self._analyze_conflicts(signals),
            position_pct=self._suggest_position(signals, winner),
            timestamp=signals[0].timestamp or "",
        )

    def _market_state_weighted(self, signals, symbol, name):
        return self._weighted_vote(signals, symbol, name)

    def _bayesian_ensemble(self, signals, symbol, name):
        return self._weighted_vote(signals, symbol, name)

    def _analyze_conflicts(self, signals: list[Signal]) -> dict:
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        has_conflict = len(buy_signals) > 0 and len(sell_signals) > 0

        return {
            "has_conflict": has_conflict,
            "conflict_strategies": [
                f"{s.strategy_name} ({s.signal_type.value})"
                for s in (buy_signals + sell_signals)
            ] if has_conflict else [],
            "resolution": (
                f"Weighted vote: buy={len(buy_signals)}, sell={len(sell_signals)}"
                if has_conflict else "No conflict"
            ),
        }

    def _suggest_position(self, signals, winner) -> float:
        avg_confidence = np.mean([s.confidence for s in signals])
        signal_count = len(signals)
        base = 0.10
        bonus = min(0.02 * (signal_count - 1), 0.20)
        scaled = (base + bonus) * avg_confidence
        return round(min(scaled, 0.30), 4)

    def _signal_to_dict(self, s: Signal) -> dict:
        return {
            "strategy_name": s.strategy_name,
            "signal_type": s.signal_type.value,
            "confidence": s.confidence,
            "entry_price": s.entry_price,
            "stop_loss": s.stop_loss,
            "take_profit": s.take_profit,
            "reasoning": s.reasoning,
            "weight": self.strategy_weights.get(s.strategy_name, 1.0),
        }
# -- [AI:END]
```

---

## 12. Backtest Engine — backtrader Wrapper

### 12.1 Core Wrapper

```python
# -- [AI:BEGIN]
# src/pa_mcp/backtest/engine.py

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    High-level backtest engine wrapping backtrader.
    Provides a simple API that hides backtrader complexity.
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.00025,
        stamp_duty: float = 0.0005,
        min_commission: float = 5.0,
    ):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.min_commission = min_commission

    def backtest(
        self,
        symbol: str,
        strategy_name: str,
        start: str,
        end: str,
        capital: float = 100000.0,
        params: Optional[dict] = None,
        benchmark_symbol: str = "000300",
    ) -> dict:
        """
        Run backtest for a single strategy on a single stock.

        Args:
            symbol: 6-digit stock code
            strategy_name: Registered strategy name
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"
            capital: Initial capital
            params: Strategy parameter overrides
            benchmark_symbol: Benchmark index for alpha/beta

        Returns:
            Backtest result dict with all metrics, trades, and equity curve.
        """
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(capital)

        # A-Share commission model
        from .a_share_rules import AShareCommission
        cerebro.broker.addcommissioninfo(
            AShareCommission(
                commission_rate=self.commission_rate,
                stamp_duty=self.stamp_duty,
                min_commission=self.min_commission,
            )
        )

        # Data feed from DuckDB
        data_df = self._load_data(symbol, start, end)
        data_feed = bt.feeds.PandasData(
            dataname=data_df,
            datetime="date", open="open", high="high", low="low",
            close="close", volume="volume", openinterest=-1,
        )
        cerebro.adddata(data_feed)

        # Strategy adapter
        from ..strategy.registry import strategy_registry
        from .a_share_strategy_adapter import create_strategy_class

        strat_cls = strategy_registry.get(strategy_name)
        adapted_cls = create_strategy_class(strat_cls, params or {})
        cerebro.addstrategy(adapted_cls)

        # Analyzers
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annualreturn")

        # Run
        from .a_share_rules import AShareSizer
        cerebro.addsizer(AShareSizer, stake=100)
        results = cerebro.run()
        strat_instance = results[0]

        return self._extract_results(cerebro, strat_instance, symbol, strategy_name, start, end)

    def _load_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Load kline data from DuckDB store."""
        from ..data.store import DuckDBStore
        store = DuckDBStore()
        df = store.conn.execute("""
            SELECT date, open, high, low, close, volume, amount
            FROM stock_daily
            WHERE symbol = ? AND adjust_flag = 'qfq'
              AND date BETWEEN ? AND ?
            ORDER BY date
        """, [symbol, start, end]).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _extract_results(self, cerebro, strat, symbol, strategy_name, start, end) -> dict:
        """Extract all backtest metrics from analyzers."""
        trade_analyzer = strat.analyzers.trades.get_analysis()

        total_trades = trade_analyzer.get("total", {}).get("total", 0)
        won_trades = trade_analyzer.get("won", {}).get("total", 0)
        lost_trades = trade_analyzer.get("lost", {}).get("total", 0)
        win_rate = won_trades / total_trades if total_trades > 0 else 0.0

        won_pnl = trade_analyzer.get("won", {}).get("pnl", {})
        lost_pnl = trade_analyzer.get("lost", {}).get("pnl", {})

        avg_win = won_pnl.get("average", 0) if won_pnl else 0
        avg_loss = lost_pnl.get("average", 0) if lost_pnl else 0
        max_win = won_pnl.get("max", 0) if won_pnl else 0
        max_loss = lost_pnl.get("max", 0) if lost_pnl else 0

        sharpe = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe.get("sharperatio", 0.0) or 0.0

        drawdown = strat.analyzers.drawdown.get_analysis()
        max_dd = drawdown.get("max", {}).get("drawdown", 0.0) or 0.0
        max_dd_duration = drawdown.get("max", {}).get("len", 0)

        returns_analysis = strat.analyzers.returns.get_analysis()
        total_return = returns_analysis.get("rtot", 0.0) or 0.0
        annual_return = returns_analysis.get("rnorm100", 0.0) or 0.0

        calmar_ratio = annual_return / abs(max_dd / 100) if max_dd != 0 else 0.0

        gross_win = trade_analyzer.get("won", {}).get("pnl", {}).get("total", 0) or 0
        gross_loss = abs(trade_analyzer.get("lost", {}).get("pnl", {}).get("total", 0)) or 0
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

        return {
            "symbol": symbol,
            "strategy_name": strategy_name,
            "start_date": start,
            "end_date": end,
            "initial_capital": float(cerebro.broker.startingcash),
            "final_value": float(cerebro.broker.getvalue()),
            "total_return_pct": round(total_return * 100, 2),
            "annual_return_pct": round(annual_return, 2),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "calmar_ratio": round(calmar_ratio, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "max_drawdown_duration": max_dd_duration,
            "total_trades": total_trades,
            "won_trades": won_trades,
            "lost_trades": lost_trades,
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_win": round(max_win, 2),
            "max_loss": round(max_loss, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.99,
            "expectancy": round(win_rate * avg_win - (1 - win_rate) * abs(avg_loss), 2),
            "trades": [],
            "equity_curve": [],
            "monthly_returns": [],
        }
# -- [AI:END]
```

---

## 13. Backtest Engine — A-Share Rules Implementation

### 13.1 T+1, Commission, Limit Check

```python
# -- [AI:BEGIN]
# src/pa_mcp/backtest/a_share_rules.py

import backtrader as bt
from datetime import datetime

class AShareCommission(bt.CommInfoBase):
    """
    A-Share commission model:
    - Commission: 0.025% (min 5 CNY per trade)
    - Stamp duty: 0.05% (sell only)
    - Transfer fee: 0.001%
    """
    params = (
        ("commission", 0.00025),
        ("stamp_duty", 0.0005),
        ("transfer_fee", 0.00001),
        ("min_commission", 5.0),
        ("margin", None),
        ("mult", 1.0),
        ("stocklike", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        value = abs(size) * price
        comm = max(value * self.p.commission, self.p.min_commission)
        stamp = value * self.p.stamp_duty if size < 0 else 0.0
        transfer = value * self.p.transfer_fee
        return comm + stamp + transfer


class AShareSizer(bt.Sizer):
    """
    A-Share position sizer:
    - Minimum 100 shares (1 lot)
    - Rounds down to nearest lot
    """
    params = (("stake", 100),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            return self.broker.getposition(data).size
        price = data.close[0]
        max_shares = int(cash / (price * 1.001))
        lots = max_shares // self.p.stake
        if lots < 1:
            return 0
        return lots * self.p.stake


class T1Enforcer:
    """
    T+1 enforcement: track entry dates and block same-day exits.
    """

    def __init__(self):
        self._entry_dates: dict[int, datetime] = {}

    def record_entry(self, data_idx: int, entry_date: datetime):
        self._entry_dates[data_idx] = entry_date

    def can_sell(self, data_idx: int, current_date: datetime) -> bool:
        entry_date = self._entry_dates.get(data_idx)
        if entry_date is None:
            return True
        return current_date > entry_date

    def clear_position(self, data_idx: int):
        self._entry_dates.pop(data_idx, None)
# -- [AI:END]
```

### 13.2 Strategy Adapter for Backtrader

```python
# -- [AI:BEGIN]
# src/pa_mcp/backtest/a_share_strategy_adapter.py

import backtrader as bt
from typing import Type
import pandas as pd
from ..strategy.base import BaseStrategy, SignalType
from .a_share_rules import T1Enforcer

def create_strategy_class(
    strategy_cls: Type[BaseStrategy],
    params: dict,
) -> Type[bt.Strategy]:
    """
    Dynamically create a backtrader Strategy subclass that wraps
    our BaseStrategy's generate_signals() method.
    Enforces T+1, limit-up/down, commission rules.
    """

    class AdaptedStrategy(bt.Strategy):
        # -- [AI] parameters exposed for optimization
        params = tuple(params.items()) if params else ()

        def __init__(self):
            self.strategy = strategy_cls(params=params)
            self.t1 = T1Enforcer()
            self.order = None

        def next(self):
            if self.order:
                return

            data_bars = self._collect_bars(lookback=100)
            signals = self.strategy.generate_signals(data_bars)

            for sig in signals:
                if sig.signal_type == SignalType.BUY and self.position.size == 0:
                    self._handle_buy(sig)
                elif sig.signal_type == SignalType.SELL and self.position.size > 0:
                    self._handle_sell(sig)

        def _handle_buy(self, sig):
            limit_up = self._calc_limit_up(self.data)
            if self.data.open[0] >= limit_up * 0.999:
                return
            self.order = self.buy()
            self.t1.record_entry(0, self.data.datetime.date(0))

        def _handle_sell(self, sig):
            if not self.t1.can_sell(0, self.data.datetime.date(0)):
                return
            limit_down = self._calc_limit_down(self.data)
            if self.data.open[0] <= limit_down * 1.001:
                return
            self.order = self.sell()
            self.t1.clear_position(0)

        def _collect_bars(self, lookback: int = 100):
            bars = []
            for i in range(min(lookback, len(self.data))):
                idx = -lookback + i
                bars.append({
                    "date": self.data.datetime.date(idx),
                    "open": self.data.open[idx],
                    "high": self.data.high[idx],
                    "low": self.data.low[idx],
                    "close": self.data.close[idx],
                    "volume": self.data.volume[idx],
                })
            return pd.DataFrame(bars)

        def _calc_limit_up(self, data):
            return round(data.close[-1] * 1.10, 2)

        def _calc_limit_down(self, data):
            return round(data.close[-1] * 0.90, 2)

        def notify_order(self, order):
            if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
                self.order = None

    return AdaptedStrategy
# -- [AI:END]
```

### 13.3 Trading Calendar

```python
# -- [AI:BEGIN]
# src/pa_mcp/backtest/calendar.py

import pandas as pd
from datetime import date
from functools import lru_cache

class AShareTradeCalendar:
    """Provides A-share trading calendar functions."""

    def __init__(self):
        self._calendar: pd.DataFrame = None

    def load_calendar(self, start_year: int = 1990, end_year: int = 2030):
        from ..data.store import DuckDBStore
        store = DuckDBStore()
        self._calendar = store.conn.execute("""
            SELECT date, is_trade_day, pre_trade_date, next_trade_date
            FROM trade_calendar
            WHERE year BETWEEN ? AND ?
            ORDER BY date
        """, [start_year, end_year]).df()

    @lru_cache(maxsize=128)
    def is_trading_day(self, dt: date) -> bool:
        if self._calendar is None:
            self.load_calendar()
        row = self._calendar[self._calendar["date"] == pd.Timestamp(dt)]
        return bool(row["is_trade_day"].iloc[0]) if len(row) > 0 else False

    def next_trading_day(self, dt: date) -> date:
        if self._calendar is None:
            self.load_calendar()
        future = self._calendar[
            (self._calendar["date"] > pd.Timestamp(dt)) &
            (self._calendar["is_trade_day"] == True)
        ]
        return future["date"].iloc[0].date() if len(future) > 0 else dt

    def prev_trading_day(self, dt: date) -> date:
        if self._calendar is None:
            self.load_calendar()
        past = self._calendar[
            (self._calendar["date"] < pd.Timestamp(dt)) &
            (self._calendar["is_trade_day"] == True)
        ]
        return past["date"].iloc[-1].date() if len(past) > 0 else dt


trade_calendar = AShareTradeCalendar()
# -- [AI:END]
```

---

## 14. Backtest Engine — API & Report Format

### 14.1 Clean Public API

```python
# -- [AI:BEGIN]
# Complete backtest API surface

from pa_mcp.backtest.engine import BacktestEngine

engine = BacktestEngine()

# ---- Single stock backtest ----
result = engine.backtest(
    symbol="000001",
    strategy_name="Platform Breakout",
    start="2024-01-01",
    end="2025-12-31",
    capital=100000,
    params={"ma_period": 20, "volume_ratio": 2.0},
)

# ---- Multi-stock basket backtest ----
result = engine.backtest_basket(
    symbols=["000001", "000002", "600036"],
    strategy_name="MA Golden Cross",
    start="2024-01-01",
    end="2025-12-31",
    capital=500000,
    position_sizing="equal_weight",
)

# ---- Strategy parameter optimization ----
opt_result = engine.optimize(
    symbol="000001",
    strategy_name="Platform Breakout",
    start="2023-01-01",
    end="2024-12-31",
    capital=100000,
    method="monte_carlo",
    samples=500,
    objective="sharpe_ratio",
)

# ---- Walk-forward analysis ----
wf_result = engine.walk_forward(
    symbol="000001",
    strategy_name="Platform Breakout",
    start="2022-01-01",
    end="2025-12-31",
    train_window_months=12,
    test_window_months=3,
    capital=100000,
)

# ---- Compare multiple strategies ----
comparison = engine.compare(
    symbol="000001",
    strategy_names=["Platform Breakout", "MA Golden Cross", "Dragon 2nd Wave"],
    start="2024-01-01",
    end="2025-12-31",
    capital=100000,
)
# -- [AI:END]
```

### 14.2 Complete Backtest Report Format

```json
{
  "metadata": {
    "report_id": "uuid-v4",
    "generated_at": "2026-07-26T20:00:00+08:00",
    "engine_version": "1.0.0",
    "backtrader_version": "1.9.76.123"
  },
  "config": {
    "symbol": "000001",
    "strategy_name": "Platform Breakout",
    "strategy_params": {"ma_period": 20, "volume_ratio": 2.0},
    "start_date": "2024-01-01",
    "end_date": "2025-12-31",
    "initial_capital": 100000.00,
    "benchmark": "000300",
    "commission_rate": 0.00025,
    "stamp_duty": 0.0005,
    "slippage": 0.001
  },
  "performance": {
    "total_return_pct": 34.56,
    "annual_return_pct": 18.23,
    "benchmark_return_pct": 12.45,
    "excess_return_pct": 5.78,
    "final_value": 134560.00,
    "sharpe_ratio": 1.85,
    "sortino_ratio": 2.34,
    "calmar_ratio": 1.42,
    "alpha": 0.045,
    "beta": 0.82,
    "max_drawdown_pct": -12.85,
    "max_drawdown_duration_days": 45,
    "max_drawdown_start": "2024-09-15",
    "max_drawdown_end": "2024-11-01",
    "volatility_annual_pct": 22.45
  },
  "trades": {
    "total_trades": 48,
    "won_trades": 29,
    "lost_trades": 19,
    "win_rate_pct": 60.42,
    "avg_win": 1250.50,
    "avg_loss": -850.30,
    "max_win": 5200.00,
    "max_loss": -3200.00,
    "max_consecutive_wins": 7,
    "max_consecutive_losses": 4,
    "profit_factor": 1.89,
    "expectancy": 425.80,
    "avg_holding_days": 12.5,
    "trade_list": [
      {
        "id": 1,
        "entry_date": "2024-01-15",
        "exit_date": "2024-01-28",
        "entry_price": 12.50,
        "exit_price": 13.80,
        "size": 800,
        "pnl": 1040.00,
        "pnl_pct": 10.40,
        "holding_days": 9,
        "exit_reason": "take_profit"
      }
    ]
  },
  "equity_curve": [
    {"date": "2024-01-01", "value": 100000.00},
    {"date": "2024-01-02", "value": 100050.00}
  ],
  "monthly_returns": [
    {"year": 2024, "month": 1, "return_pct": 5.2},
    {"year": 2024, "month": 2, "return_pct": -2.1}
  ],
  "annual_returns": [
    {"year": 2024, "return_pct": 22.15, "benchmark_pct": 15.30},
    {"year": 2025, "return_pct": 10.18, "benchmark_pct": 8.20}
  ],
  "risk_analysis": {
    "concentration": {
      "max_single_position_pct": 30.0,
      "avg_position_pct": 15.5
    },
    "tail_risk": {
      "worst_day_pct": -5.2,
      "best_day_pct": 8.5
    }
  },
  "comparison": {
    "vs_buy_and_hold": {
      "outperformance_pct": 12.10,
      "better_sharpe": true,
      "lower_drawdown": true
    }
  }
}
```

---

## Appendix A: Development Milestones

### Phase 1: Data Layer (Week 1)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | AKShare adapter + rate limiter + unit tests | `akshare_adapter.py` with 15 wrapped endpoints |
| 2-3 | efinance adapter + batch retry + unit tests | `efinance_adapter.py` with 12 wrapped endpoints |
| 3-4 | BaoStock adapter + field mapping + unit tests | `baostock_adapter.py` with 8 wrapped endpoints |
| 4-5 | Aggregator + consistency check + fallback logic | `aggregator.py` with 3-source fallback |
| 5 | DuckDB init scripts + all 13 tables | `store.py` + SQL migration files |
| 5-6 | Redis cache layer + key design validation | `cache.py` with tiered caching |
| 6-7 | Cron scheduler + all 8 update jobs | `scheduler.py` with retry + alert |
| 7 | Data quality validator + integration tests | `quality.py` with 9 validation checks |
| 7 | 5 MCP market data tools wired up | `market_data.py` tools in `tools/` |

### Phase 3: Strategy + Backtest (Week 2-3)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Base strategy class + registry + unit tests | `base.py`, `registry.py` |
| 2-5 | Implement 42 strategies (5-7 per category) | All files under `strategy/strategies/` |
| 5-6 | Signal format + JSON schema + aggregation logic | `signals.py` with 4 aggregation methods |
| 6-7 | Parameter optimizer (grid/MC/GA) + unit tests | `optimizer.py` |
| 7-8 | Backtrader engine wrapper + A-share rules | `engine.py`, `a_share_rules.py` |
| 8-9 | T+1 + limit-up/down + commission + calendar | Per-file implementations |
| 9-10 | Backtest API + report format + integration tests | Full backtest suite |

---

## Appendix B: Key Risk Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|------------|
| AKShare rate-limiting | High | High | 3-source fallback; rate limiter with exponential backoff |
| East Money API change | Medium | High | Semantic version pinning; CI daily smoke test on all 3 adapters |
| DuckDB concurrent write | Low | Medium | Single-writer cron job; WAL mode; write lock in scheduler |
| backtrader T+1 edge cases | Medium | High | Unit test with known examples; compare against manual PnL |
| Parameter overfitting | High | Medium | Walk-forward validation; out-of-sample test; MC sensitivity |
| Strategy signal inconsistency | Medium | Medium | Aggregator with weighted voting; market state adjustment |
| Financial data gaps (BaoStock) | High | Medium | Fill from efinance + AKShare; missing data flagged in quality report |

---

## Appendix C: Configuration Schema (`config/default.yaml`)

```yaml
# -- [AI:BEGIN]
# PA_MCP default configuration

data:
  duckdb_path: "data/pa_mcp.duckdb"
  redis_url: "redis://localhost:6379"

  sources:
    akshare:
      enabled: true
      priority: 1
      min_interval: 0.5
      burst_limit: 3
      max_retries: 3
      proxy_pool: []

    efinance:
      enabled: true
      priority: 2
      min_interval: 0.3
      max_retries: 3

    baostock:
      enabled: true
      priority: 3
      max_retries: 2

  aggregator:
    consistency_threshold: 0.02
    consistency_min_overlap: 10

  scheduler:
    daily_update_time: "17:30"
    incremental: true
    max_retries: 3
    retry_delay: 300

  quality:
    price_jump_threshold: 0.30
    staleness_max_days: 3
    min_required_columns: ["date", "open", "high", "low", "close", "volume"]

strategy:
  discovery_path: "pa_mcp.strategy.strategies"
  default_aggregation: "weighted_vote"
  strategy_weights: {}

backtest:
  initial_cash: 100000.0
  commission_rate: 0.00025
  stamp_duty: 0.0005
  min_commission: 5.0
  slippage: 0.001
  risk_free_rate: 0.02
  benchmark: "000300"

mcp:
  transport: "stdio"
  http_port: 8080
  host: "0.0.0.0"

logging:
  level: "INFO"
  file: "logs/pa_mcp.log"
  alert_webhook: ""
# -- [AI:END]
```

---

*End of Implementation Plan*

Generated by: mx + xiaolu (Claude)
Version: 2.0
Date: 2026-07-26
