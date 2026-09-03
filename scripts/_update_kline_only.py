"""Only update daily kline via scheduler (skip other phases)."""
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from pa_mcp.data import DuckDBStore, AKShareAdapter
from pa_mcp.config import get_settings


async def main():
    settings = get_settings()
    store = DuckDBStore()
    store.connect()
    ak = AKShareAdapter()

    from pa_mcp.data.source_factory import build_router
    router = None
    try:
        router = build_router(
            settings,
            min_source_interval={"eastmoney": 1.2, "ths": 0.4},
            existing={"akshare": ak},
        )
    except Exception as e:
        logging.warning("router init failed, single-source fallback: %s", e)

    from pa_mcp.data.scheduler import DataUpdateScheduler
    sched = DataUpdateScheduler(store=store, data_router=router, akshare_adapter=ak)

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    rows = await sched._update_daily_kline(force_full=False)
    dt = round(loop.time() - t0, 1)
    print(f"KLINE_UPDATE_RESULT rows={rows} elapsed={dt}s")

    latest = store.get_latest_date("kline_daily")
    print(f"LATEST_KLINE_DATE={latest}")


if __name__ == "__main__":
    asyncio.run(main())
