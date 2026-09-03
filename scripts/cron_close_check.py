# -*- coding: utf-8 -*-
"""收盘观察池检查:调用 PA_MCP 内部函数 get_market_sentiment / watchlist_overview / watchlist_consensus"""
import asyncio, json, sys, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, BASE)

import pa_mcp.server as S
from pa_mcp.data import DuckDBStore, CacheManager, AKShareAdapter, SinaAdapter
from pa_mcp.config import get_settings


async def main():
    # 初始化 server 全局状态(等价于 lifespan)
    S._settings = get_settings()
    S._store = DuckDBStore()
    S._store.connect()
    S._cache = CacheManager()
    S._akshare = AKShareAdapter()
    S._sina = SinaAdapter()
    try:
        from pa_mcp.data.source_factory import build_router
        S._router = build_router(
            S._settings,
            min_source_interval={"eastmoney": 1.2, "ths": 0.4},
            existing={"akshare": S._akshare, "sina": S._sina},
        )
    except Exception:
        S._router = None
    from pa_mcp.risk.guard import RiskGuard
    S._guard = RiskGuard()

    out = {}

    # 1) 市场温度
    try:
        out["sentiment"] = await S.get_market_sentiment()
    except Exception as e:\n        out["sentiment"] = {"success": False, "error": str(e)}

    # 2) 观察池实时概况
    try:
        out["overview"] = await S.watchlist_overview()
    except Exception as e:\n        out["overview"] = {"success": False, "error": str(e)}

    # 3) 多源综合信号
    try:
        out["consensus"] = await S.watchlist_consensus(
            "600096,600598,600426,601088,601225,000967,002573")
    except Exception as e:\n        out["consensus"] = {"success": False, "error": str(e)}

    if S._store:
        S._store.close()
    if S._sina:
        await S._sina.close()

    print(json.dumps(out, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
