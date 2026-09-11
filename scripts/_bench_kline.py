"""基准：get_kline 全量读取的实际成本（行数 / 耗时 / payload 大小）。"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")

from pa_mcp.config import get_settings
from pa_mcp.data.store import DuckDBStore

s = DuckDBStore(get_settings().database.path)
s.connect()

print("=== 库规模 ===")
q = lambda sql: s.query_df(sql, [])
print(q("SELECT COUNT(*) n, COUNT(DISTINCT symbol) syms, "
        "CAST(MIN(date) AS VARCHAR) mn, CAST(MAX(date) AS VARCHAR) mx "
        "FROM kline_daily").to_string(index=False))

sym = "000001"
t0 = time.perf_counter()
df = s.query_df("SELECT * FROM kline_daily WHERE symbol=? ORDER BY date ASC", [sym])
t1 = time.perf_counter()
print(f"\n=== 单票全量 SELECT * ===")
print(f"行数 {len(df)} 列 {len(df.columns)} 耗时 {t1-t0:.3f}s")
print("列：", list(df.columns))

t0 = time.perf_counter()
recs = df.to_dict(orient="records")
for r in recs:
    for k, v in r.items():
        if hasattr(v, "isoformat"):
            r[k] = v.isoformat()
        elif hasattr(v, "item"):
            r[k] = float(v)
t1 = time.perf_counter()
payload = json.dumps(recs, ensure_ascii=False)
print(f"序列化耗时 {t1-t0:.3f}s  payload {len(payload)/1024:.1f} KB "
      f"（单条 {len(payload)/max(1,len(recs)):.0f} B）")

print("\n=== 只取 6 列 + 近 250 根 ===")
t0 = time.perf_counter()
df2 = s.query_df(
    "SELECT CAST(date AS VARCHAR) date, open, high, low, close, volume "
    "FROM kline_daily WHERE symbol=? ORDER BY date DESC LIMIT 250", [sym])
t1 = time.perf_counter()
p2 = json.dumps(df2.to_dict(orient="records"), ensure_ascii=False)
print(f"行数 {len(df2)} 耗时 {t1-t0:.3f}s payload {len(p2)/1024:.1f} KB "
      f"（比全量小 {len(payload)/max(1,len(p2)):.1f}×）")

print("\n=== 全市场一次性聚合读取（批量接口可行性）===")
t0 = time.perf_counter()
df3 = s.query_df(
    "SELECT symbol, CAST(date AS VARCHAR) date, close, volume FROM kline_daily "
    "WHERE date >= DATE '2026-09-07' ORDER BY symbol, date", [])
t1 = time.perf_counter()
print(f"近 5 日全市场 {len(df3)} 行 耗时 {t1-t0:.3f}s "
      f"payload {len(json.dumps(df3.to_dict(orient='records')))/1024:.0f} KB")

print("\n=== 全市场逐只查询（现状：N+1）===")
syms = [r[0] for r in s.execute(
    "SELECT DISTINCT symbol FROM kline_daily LIMIT 60").fetchall()]
t0 = time.perf_counter()
for x in syms:
    s.query_df("SELECT * FROM kline_daily WHERE symbol=? ORDER BY date ASC", [x])
t1 = time.perf_counter()
print(f"60 只逐只全量查询耗时 {t1-t0:.3f}s（外推 1096 只约 "
      f"{(t1-t0)/60*1096:.0f}s）")

print("\n=== 单条 SQL 拿 60 只全部历史 ===")
t0 = time.perf_counter()
df4 = s.query_df(
    "SELECT symbol, CAST(date AS VARCHAR) date, open, high, low, close, volume "
    "FROM kline_daily WHERE symbol IN ({}) ORDER BY symbol, date".format(
        ",".join("?" * len(syms))), syms)
t1 = time.perf_counter()
print(f"{len(df4)} 行 耗时 {t1-t0:.3f}s")

s.close()
