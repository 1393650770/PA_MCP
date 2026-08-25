# MCP 并发冒烟：同一 session 并行调用多工具，检测锁冲突/竞态/挂死
# 用法: python scripts/mcp_concurrency_test.py

from __future__ import annotations

import asyncio
import json
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

RESULTS: dict[str, str] = {}


async def call(session: ClientSession, name: str, args: dict, timeout: int = 90) -> str:
    t0 = time.time()
    try:
        res = await asyncio.wait_for(session.call_tool(name, args), timeout=timeout)
        text = "".join(c.text for c in res.content if getattr(c, "type", "") == "text")
        elapsed = time.time() - t0
        return f"{'❌ERR' if res.isError else '✅ ok'} {elapsed:5.1f}s {text[:40]}"
    except Exception as e:
        return f"💥EXC {time.time()-t0:5.1f}s {type(e).__name__}: {str(e)[:80]}"


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "pa_mcp.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"🔌 已连接，开始并发测试\n")

            # ── 场景 1：并发读（5 只不同股票 K 线 + 实时行情） ──
            print("═══ 场景1：并发读（5×get_kline + 3×get_realtime_quote）═══")
            t0 = time.time()
            tasks = []
            for i, sym in enumerate(["000001", "600036", "300750", "600519", "601318"]):
                tasks.append(call(session, "get_kline", {"symbol": sym,
                                                         "days": 120}))
            for sym in ["000001", "510300", "159915"]:
                tasks.append(call(session, "get_realtime_quote", {"symbol": sym}))
            results = await asyncio.gather(*tasks)
            for r in results:
                print(f"  {r}")
            print(f"  总耗时 {time.time()-t0:.1f}s\n")

            # ── 场景 2：并发混合（重工具 + 读 + 元信息） ──
            print("═══ 场景2：混合（scan_market + evaluate_predictions + 元信息×3）═══")
            t0 = time.time()
            tasks = [
                call(session, "scan_market", {"strategy": "bollinger_mean_reversion", "top_n": 5}, 180),
                call(session, "evaluate_predictions", {}, 180),
                call(session, "get_strategy_guide", {}),
                call(session, "get_methodology_guide", {}),
                call(session, "get_data_source_health", {}),
            ]
            results = await asyncio.gather(*tasks)
            for r in results:
                print(f"  {r}")
            print(f"  总耗时 {time.time()-t0:.1f}s\n")

            # ── 场景 3：并发写（portfolio_add ×3 + 读 portfolio_summary） ──
            print("═══ 场景3：并发写（3×portfolio_add + portfolio_summary）═══")
            t0 = time.time()
            tasks = [
                call(session, "portfolio_add", {"symbol": "600000", "cost": 10.0, "shares": 100}),
                call(session, "portfolio_add", {"symbol": "000002", "cost": 8.0, "shares": 100}),
                call(session, "portfolio_add", {"symbol": "510300", "cost": 4.5, "shares": 100}),
                call(session, "portfolio_summary", {}),
            ]
            results = await asyncio.gather(*tasks)
            for r in results:
                print(f"  {r}")
            print(f"  总耗时 {time.time()-t0:.1f}s\n")

            # ── 清理测试持仓 ──
            for sym in ("600000", "000002", "510300"):
                await call(session, "portfolio_remove", {"symbol": sym})

            print("🏁 并发测试结束")


if __name__ == "__main__":
    asyncio.run(main())
