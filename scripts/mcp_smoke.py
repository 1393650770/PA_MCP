# MCP 冒烟测试：模拟 OpenClaw 客户端连 stdio server，逐个调用常用工具
# 用法: python scripts/mcp_smoke.py [tool1,tool2...]
# 无参数 = 调用默认工具清单（用户反馈"经常报错"的工具）

from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_CMD = [sys.executable, "-m", "pa_mcp.server"]

DEFAULT_TOOLS = [
    ("get_market_overview", {}),
    ("get_market_sentiment", {}),
    ("scan_market", {"strategy": "bollinger_mean_reversion", "top_n": 5}),
    ("scan_canslim", {"top_n": 5}),
    ("factor_stock_selection", {"symbols": "000001,600036,300750,600519,601318", "top_n": 3}),
    ("analyze_stock", {"symbol": "000001"}),
    ("sector_hot_cold", {}),
    ("get_strategy_guide", {}),
    ("get_methodology_guide", {}),
]


async def call_tool(session: ClientSession, name: str, args: dict) -> None:
    t0 = asyncio.get_event_loop().time()
    try:
        result = await asyncio.wait_for(
            session.call_tool(name, args), timeout=90)
        text = ""
        for c in result.content:
            if getattr(c, "type", "") == "text":
                text += getattr(c, "text", "")
        elapsed = asyncio.get_event_loop().time() - t0
        is_err = result.isError
        status = "❌ ERROR" if is_err else "✅ ok"
        print(f"[{elapsed:5.1f}s] {status} {name}  ->  {text[:120]}")
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - t0
        print(f"[{elapsed:5.1f}s] 💥 EXCEPTION {name}: {type(e).__name__}: {str(e)[:200]}")


async def main() -> None:
    args = sys.argv[1:]
    tools = DEFAULT_TOOLS
    if args:
        wanted = set(args)
        tools = [(n, a) for n, a in DEFAULT_TOOLS if n in wanted]

    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:],
                                   env=None, cwd=None)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            t0 = asyncio.get_event_loop().time()
            init = await asyncio.wait_for(session.initialize(), timeout=120)
            print(f"🔌 连接成功: {init.serverInfo.name} v{init.serverInfo.version} "
                  f"({asyncio.get_event_loop().time() - t0:.1f}s 冷启动)")
            tools_resp = await session.list_tools()
            print(f"📚 工具总数: {len(tools_resp.tools)}\n")
            for name, args in tools:
                await call_tool(session, name, args)
    print("\n🏁 冒烟结束")


if __name__ == "__main__":
    asyncio.run(main())
