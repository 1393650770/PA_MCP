"""验证：反追高 + 时效性在真实数据上的效果。只读预测，不落盘。"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from pa_mcp.agent.prediction import get_prediction_service
from pa_mcp.config import get_settings
from pa_mcp.data.store import DuckDBStore

SYMBOLS = ["603601", "600313", "000001", "601318", "601728"]


async def main() -> None:
    store = DuckDBStore(get_settings().database.path)
    store.connect()
    svc = get_prediction_service()
    print(f"{'代码':8s} {'数据截至':10s} {'末日%':>7s} {'20日%':>7s} "
          f"{'入场':6s} {'方向':6s} {'概率':>6s} {'校准后':>6s} {'期望%':>7s}")
    print("-" * 78)
    for sym in SYMBOLS:
        df = store.query_df(
            "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 160",
            [sym])
        if df.empty:
            print(f"{sym}: 无数据")
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = df["close"].astype(float)
        last = float(c.iloc[-1])
        d1 = (last / float(c.iloc[-2]) - 1) * 100
        r20 = (last / float(c.iloc[-21]) - 1) * 100
        res = await svc.predict(sym, df, horizon="5d", use_llm=False)
        note = res.entry_note[:40]
        print(f"{sym:8s} {res.as_of:10s} {d1:+7.1f} {r20:+7.1f} "
              f"{res.entry_timing:6s} {res.direction:6s} "
              f"{res.probability:6.2f} {res.calibrated_probability:6.2f} "
              f"{res.expected_return_pct:+7.2f}  {note}")
        for k in (res.key_risks or [])[:2]:
            print(f"         ↳ {k[:90]}")
    store.close()


asyncio.run(main())
