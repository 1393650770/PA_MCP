import asyncio, json, sys
sys.path.insert(0, r"d:\Project\AI\PA_MCP\PA_MCP\src")
from pa_mcp import server as S

async def main():
    out = {}
    out["data_source_health"] = await S.get_data_source_health()
    out["market_state"] = await S.agent_market_state()
    out["market_sentiment"] = await S.get_market_sentiment()
    out["scan"] = await S.scan_market(strategy="bollinger_mean_reversion", top_n=10)
    with open(r"d:\Project\AI\PA_MCP\PA_MCP\scripts\daily_out.json", "w", encoding="utf-8") as f:\n        json.dump(out, f, ensure_ascii=False, default=str, indent=1)\n\nasyncio.run(main())\nprint("DONE")
