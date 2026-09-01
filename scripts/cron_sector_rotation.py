# 板块轮动晨报 cron 脚本：直连 stdio MCP server 调用 PA_MCP 工具
# 用法: venv\Scripts\python.exe scripts\cron_sector_rotation.py
# 输出: 分步骤 JSON 到 stdout，便于外层解析

from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_CMD = [sys.executable, "-m", "pa_mcp.server"]


def fmt(content) -> str:
    text = ""
    for c in content:
        if getattr(c, "type", "") == "text":
            text += getattr(c, "text", "")
    return text


async def call(session, name, args, timeout=180) -> dict:
    t0 = asyncio.get_event_loop().time()
    try:
        result = await asyncio.wait_for(session.call_tool(name, args), timeout=timeout)
        text = fmt(result.content)
        print(f"=== {name} args={json.dumps(args, ensure_ascii=False)} elapsed={asyncio.get_event_loop().time()-t0:.1f}s isError={result.isError} ===")
        print(text[:6000])
        print()
        return {"name": name, "isError": result.isError, "text": text}
    except Exception as e:\n        print(f"=== {name} EXCEPTION {type(e).__name__}: {str(e)[:300]} ===")
        print()
        return {"name": name, "isError": True, "text": f"EXCEPTION {type(e).__name__}: {e}"}


async def main() -> None:
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:],
                                   env=None, cwd=None)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=120)

            # 1. 板块轮动预测
            await call(session, "predict_sector_rotation", {"load_data": True})
            # 2. 轮动状态
            await call(session, "sector_rotation_status", {})

    print("🏁 DONE")


if __name__ == "__main__":
    asyncio.run(main())
