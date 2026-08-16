# [AI:BEGIN]
# PA_MCP - Data: 数据前置检查与自动装载（按钮依赖自动补数据）
#
# 解决「点了按钮报无数据」：各研究工具声明数据依赖，
# ensure_readiness() 自动检查并按需装载：
#   kline_daily     → 内置池网络拉取（多源 router，30 只 ≈30s）
#   financials      → 内置池财报（AKShare，尽力而为，失败不阻塞）
#   stock_basic     → 内置池基础信息
#   sector_daily    → 东财板块装载（失败自动合成降级）
#   index_daily     → 指数拉取（router）
#   sentiment_daily → 情绪日统计（kline 现算落库）
#
# 原则：前置只在缺失时执行（有数据零开销）；全部 best-effort，
# 装载失败不影响主流程（研究工具自身还有降级）。
# [AI:END]

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 内置常用股池（与 CANSLIM 兜底池一致）
BUILTIN_POOL = [
    "000001", "600036", "601398", "601288", "600519", "000858",
    "000568", "300750", "002594", "601012", "600438", "000333",
    "000651", "600030", "601318", "600276", "300760", "002415",
    "300059", "688981", "002475", "601138", "000725", "002230",
    "300308", "603986", "600900", "601857", "600028", "601088",
]

# 各研究工具的依赖声明（表名 → 最低行数/条数）
NEED_KLINE = {"kline_daily": 60}
NEED_KLINE_FIN = {"kline_daily": 60, "financials_income": 1}
NEED_SECTOR = {"sector_daily": 21}
NEED_INDEX = {"index_daily": 100}
NEED_SENTIMENT = {"sentiment_daily": 1, "kline_daily": 60}
NEED_PORTFOLIO = {"portfolio": 1, "kline_daily": 60}


def check_readiness(need: dict[str, int]) -> dict[str, bool]:
    """检查各表是否满足最低数据量。返回 {表: 是否满足}。"""
    from pa_mcp.config import get_settings
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(get_settings().database.path)
    store.connect()
    try:
        out = {}
        for table, min_rows in need.items():
            try:
                if not store.table_exists(table):
                    out[table] = False
                    continue
                n = store.row_count(table)
                out[table] = n >= min_rows
            except Exception:
                out[table] = False
        return out
    finally:
        store.close()


async def ensure_readiness(need: dict[str, int],
                           symbols: Optional[list[str]] = None) -> dict[str, Any]:
    """检查并按需自动装载缺失数据。返回 {表: 状态} + 装载说明。

    原则：只装载缺失项；全部 best-effort，失败不抛异常。
    """
    status = check_readiness(need)
    missing = [t for t, ok in status.items() if not ok]
    actions: list[str] = []
    if not missing:
        return {"status": status, "actions": ["数据已就绪"], "loaded": False}

    pool = symbols or BUILTIN_POOL

    if "kline_daily" in missing:
        n = await _load_kline(pool)
        actions.append(f"行情：{n} 只装载")
        status["kline_daily"] = n >= 60

    if "stock_basic" in missing:
        n = _load_stock_basic(pool)
        actions.append(f"基本信息：{n} 只")
        status["stock_basic"] = n > 0

    if "financials_income" in missing:
        n = await _load_financials(pool)
        actions.append(f"财务：{n} 只（尽力而为）")
        status["financials_income"] = n > 0

    if "sector_daily" in missing:
        n = await _load_sectors()
        actions.append(f"板块：{n} 个（失败自动合成降级）")
        status["sector_daily"] = n >= 21

    if "index_daily" in missing:
        n = await _load_index()
        actions.append(f"指数：{'OK' if n else '拉取失败'}")
        status["index_daily"] = bool(n)

    if "sentiment_daily" in missing:
        n = _load_sentiment()
        actions.append(f"情绪：{'OK' if n else '计算失败'}")
        status["sentiment_daily"] = bool(n)

    if "portfolio" in missing:
        actions.append("持仓表：无持仓记录（跳过——请用 portfolio_add 添加）")
        status["portfolio"] = True  # 空持仓是合法状态

    return {"status": status, "actions": actions, "loaded": bool(missing)}


async def _load_kline(symbols: list[str]) -> int:
    """内置池行情网络拉取（多源 router），写入 kline_daily。"""
    from pa_mcp.config import get_settings
    from pa_mcp.data.store import DuckDBStore
    from pa_mcp.data.router import DataSourceRouter
    from pa_mcp.data.sources.akshare_adapter import AKShareAdapter
    from pa_mcp.data.sources.sina_adapter import SinaAdapter
    from pa_mcp.data.sources.tencent_adapter import TencentAdapter
    from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter

    s = get_settings()
    router = DataSourceRouter(
        [("akshare", AKShareAdapter()), ("sina", SinaAdapter()),
         ("tencent", TencentAdapter()), ("eastmoney", EastMoneyAdapter())],
        {n: s.router.circuit for n in
         ("akshare", "sina", "tencent", "eastmoney")})
    store = DuckDBStore(get_settings().database.path)
    store.connect()
    loaded = 0
    try:
        for sym in symbols:
            try:
                df, _ = await router.fetch_daily_kline(
                    symbol=sym, period="daily",
                    start_date=(__import__("datetime").date.today() -
                                __import__("datetime").timedelta(days=320)
                                ).strftime("%Y%m%d"),
                    end_date=__import__("datetime").date.today()
                    .strftime("%Y%m%d"),
                    adjust="qfq")
                if df is None or df.empty:
                    continue
                df["symbol"] = sym
                cols = [c for c in ("symbol", "date", "open", "high", "low",
                                    "close", "volume", "amount", "pct_change",
                                    "turnover", "change", "amplitude",
                                    "adjust_factor") if c in df.columns]
                store.insert_df("kline_daily", df[cols], mode="append")
                loaded += 1
            except Exception:
                continue
        return loaded
    finally:
        store.close()


def _load_stock_basic(symbols: list[str]) -> int:
    """内置池基础信息（名称/板块，映射缺失标未知）。"""
    from pa_mcp.config import get_settings
    from pa_mcp.data.store import DuckDBStore
    from pa_mcp.data.symbols import COMMON_NAMES

    store = DuckDBStore(get_settings().database.path)
    store.connect()
    try:
        rows = [{"symbol": sym, "name": COMMON_NAMES.get(sym, f"未知{sym}"),
                 "sector": "", "industry": "", "market_cap": None,
                 "list_date": None, "exchange": "", "board": "main",
                 "is_st": False, "is_suspended": False}
                for sym in symbols]
        store.insert_df("stock_basic", pd.DataFrame(rows), mode="append")
        return len(rows)
    except Exception as e:  # noqa: BLE001
        logger.warning("stock_basic 装载失败: %s", e)
        return 0
    finally:
        store.close()


async def _load_financials(symbols: list[str]) -> int:
    """内置池财报（复用 scheduler 财务装载；尽力而为——失败不阻塞，
    CANSLIM C/A 会自动降级）。"""
    from pa_mcp.config import get_settings
    from pa_mcp.data.store import DuckDBStore
    from pa_mcp.data.scheduler import DataUpdateScheduler

    store = DuckDBStore(get_settings().database.path)
    store.connect()
    try:
        sched = DataUpdateScheduler(store=store)
        n = await sched._update_financials(force_full=True)
        return int(n)
    except Exception as e:  # noqa: BLE001
        logger.warning("财务装载失败: %s", e)
        return 0
    finally:
        store.close()


async def _load_sectors() -> int:
    """板块装载（东财 → 失败自动合成降级）。"""
    from pa_mcp.research.sector_rotation import SectorRotationAnalyzer
    info = await SectorRotationAnalyzer().load_sector_data(top_n=40, days=120)
    return info.get("loaded", 0)


async def _load_index() -> bool:
    """指数拉取（router）写入 index_daily。"""
    from pa_mcp.config import get_settings
    from pa_mcp.data.store import DuckDBStore
    from pa_mcp.data.router import DataSourceRouter
    from pa_mcp.data.sources.akshare_adapter import AKShareAdapter
    from pa_mcp.data.sources.sina_adapter import SinaAdapter
    from pa_mcp.data.sources.tencent_adapter import TencentAdapter
    from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter

    s = get_settings()
    router = DataSourceRouter(
        [("akshare", AKShareAdapter()), ("sina", SinaAdapter()),
         ("tencent", TencentAdapter()), ("eastmoney", EastMoneyAdapter())],
        {n: s.router.circuit for n in
         ("akshare", "sina", "tencent", "eastmoney")})
    try:
        df, _ = await router.fetch_daily_kline(
            symbol="sh000001", period="daily",
            start_date=(__import__("datetime").date.today() -
                        __import__("datetime").timedelta(days=400)
                        ).strftime("%Y%m%d"),
            end_date=__import__("datetime").date.today().strftime("%Y%m%d"),
            adjust="qfq")
        if df is None or df.empty:
            return False
        df["symbol"] = "sh000001"
        store = DuckDBStore(get_settings().database.path)
        store.connect()
        try:
            cols = [c for c in ("symbol", "date", "open", "high", "low",
                                "close", "volume", "amount") if c in df.columns]
            store.insert_df("index_daily", df[cols], mode="append")
            return True
        finally:
            store.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("指数拉取失败: %s", e)
        return False


def _load_sentiment() -> bool:
    """情绪日统计（基于 kline 现算落库）。"""
    from pa_mcp.research.sentiment_cycle import SentimentCycleAnalyzer
    try:
        r = SentimentCycleAnalyzer().analyze()
        return "error" not in r
    except Exception:
        return False
