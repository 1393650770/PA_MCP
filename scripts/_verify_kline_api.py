"""验证：优化后的 get_kline / get_kline_batch（payload / 窗口 / 批量）。"""
from __future__ import annotations

import asyncio
import json
import sys
import time

sys.path.insert(0, ".")

import pa_mcp.server as srv


async def main() -> None:
    import pa_mcp.server as s
    from pa_mcp.config import get_settings
    from pa_mcp.data import CacheManager, DuckDBStore

    # 复刻 lifespan 的最小初始化（不连网、不启 router）
    s._settings = get_settings()
    s._store = DuckDBStore()
    s._cache = CacheManager()

    def _kb(obj) -> float:
        return len(json.dumps(obj, ensure_ascii=False, default=str)) / 1024

    print("=== get_kline 默认（days=250, compact）===")
    t0 = time.perf_counter()
    r = await s.get_kline("000001")
    t1 = time.perf_counter()
    d = r["data"]
    print(f"rows={d['rows']} 耗时 {t1-t0:.3f}s payload {_kb(d['kline']):.1f} KB "
          f"as_of={r['data_freshness'][:10]} truncated={d['truncated']}")

    print("\n=== get_kline full=True（对照：旧行为）===")
    t0 = time.perf_counter()
    r2 = await s.get_kline("000001", full=True)
    t1 = time.perf_counter()
    d2 = r2["data"]
    print(f"rows={d2['rows']} 耗时 {t1-t0:.3f}s payload {_kb(d2['kline']):.1f} KB "
          f"truncated={d2['truncated']}")

    print("\n=== start_date/end_date 是否真正生效 ===")
    r3 = await s.get_kline("000001", start_date="20260101", end_date="20260331")
    d3 = r3["data"]
    print(f"rows={d3['rows']} 首 {d3['kline'][0]['date'] if d3['kline'] else None} "
          f"末 {d3['kline'][-1]['date'] if d3['kline'] else None}")

    print("\n=== compact=False（全列）===")
    r4 = await s.get_kline("000001", days=10, compact=False)
    print(f"rows={r4['data']['rows']} 列={list(r4['data']['kline'][0].keys())}")

    print("\n=== 缓存命中（第二次同参）===")
    t0 = time.perf_counter()
    await s.get_kline("000001")
    t1 = time.perf_counter()
    print(f"耗时 {t1-t0:.4f}s（首次应明显高于此）")

    print("\n=== get_kline_batch：20 只 × 250 根 ===")
    syms = ["000001", "600036", "601398", "601288", "600519", "000858",
            "000568", "300750", "002594", "601012", "600438", "000333",
            "000651", "600030", "601318", "600276", "300760", "002415",
            "300059", "688981"]
    t0 = time.perf_counter()
    rb = await s.get_kline_batch(",".join(syms), days=250)
    t1 = time.perf_counter()
    db = rb["data"]
    print(f"只数={len(db['symbols'])} missing={db['missing']} "
          f"rows_total={db['rows_total']} 耗时 {t1-t0:.3f}s "
          f"payload {_kb(db):.1f} KB")

    print("\n=== 逐只 get_kline 对照（N+1）===")
    t0 = time.perf_counter()
    for x in syms:
        await s.get_kline(x, days=250)
    t1 = time.perf_counter()
    print(f"20 只逐只耗时 {t1-t0:.3f}s（batch 为 {db and '见上'}）")

    print("\n=== 批量 + 日期窗口 ===")
    rb2 = await s.get_kline_batch(",".join(syms[:5]), start_date="2026-09-01",
                                    end_date="2026-09-11")
    db2 = rb2["data"]
    print(f"rows_total={db2['rows_total']} as_of 样例="
          f"{list(db2['as_of'].items())[:3]}")

    print("\n=== 参数保护 ===")
    r5 = await s.get_kline_batch(",".join(syms), max_symbols=3)
    print(f"max_symbols=3 → 取 {len(r5['data']['symbols'])} 只，"
          f"dropped={len(r5['data']['dropped_over_cap'])}")
    r6 = await s.get_kline_batch("")
    print(f"空 symbols → success={r6['success']} error={r6['error']}")

    pass


asyncio.run(main())
