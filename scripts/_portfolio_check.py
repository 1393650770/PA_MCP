import asyncio, json, sys, traceback
sys.path.insert(0, r"d:\Project\AI\PA_MCP\PA_MCP\src")
import pa_mcp.server as S

def _short(obj, limit=4000):
    s = json.dumps(obj, ensure_ascii=False, default=str, indent=2)
    return s[:limit] + ("\n...[truncated]" if len(s) > limit else "")

async def main():
    S._settings = S.get_settings()
    S._store = S.DuckDBStore()
    S._store.connect()
    S._cache = S.CacheManager()
    S._akshare = S.AKShareAdapter()
    S._sina = S.SinaAdapter()
    try:
        from pa_mcp.data.source_factory import build_router
        S._router = build_router(S._settings, min_source_interval={"eastmoney": 1.2, "ths": 0.4}, existing={"akshare": S._akshare, "sina": S._sina})
    except Exception as e:\n        print("router init warn:", e)
        S._router = None
    S._guard = S.RiskGuard()

    print("===== 1) agent_portfolio_review =====")
    try:
        print(_short(await S.agent_portfolio_review()))
    except Exception as e:\n        print("ERROR:", e)
        traceback.print_exc()

    symbols = []
    try:
        if S._store and S._store.table_exists("portfolio"):
            df = S._store.query_df("SELECT DISTINCT symbol FROM portfolio")
            symbols = [str(x) for x in df["symbol"].tolist()]
    except Exception as e:\n        print("portfolio read err:", e)
    print("\nHOLDING SYMBOLS:", symbols)

    print("\n===== 2) get_valuation_snapshot =====")
    for sym in symbols:
        print("\n--- %s ---" % sym)
        try:
            print(_short(await S.get_valuation_snapshot(sym)))
        except Exception as e:\n            print("ERROR:", e)
            traceback.print_exc()

    print("\n===== 3) get_data_source_health =====")
    try:
        print(_short(await S.get_data_source_health()))
    except Exception as e:\n        print("ERROR:", e)
        traceback.print_exc()

    if S._store:
        S._store.close()
    if S._sina:
        try:
            await S._sina.close()
        except Exception:
            pass

asyncio.run(main())
