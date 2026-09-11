"""诊断：预测是否"追涨"——预测时股票是否已大涨，之后是否回落。

只读，不改数据。用项目 venv 运行。
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, ".")

from pa_mcp.config import get_settings
from pa_mcp.data.store import DuckDBStore

DB = get_settings().database.path
store = DuckDBStore(DB)
store.connect()

print("=== 库内数据新鲜度 ===")
for t in ("kline_daily", "prediction_log", "sentiment_daily"):
    try:
        r = store.query_df(f"SELECT MAX(date) AS d, COUNT(*) AS c FROM {t}", [])
        print(f"  {t}: last={str(r['d'].iloc[0])[:10]} rows={r['c'].iloc[0]}")
    except Exception as e:
        print(f"  {t}: ERR {e}")

print("\n=== 预测记录状态 ===")
st = store.query_df(
    "SELECT status, COUNT(*) c FROM prediction_log GROUP BY status ORDER BY c DESC", [])
print(st.to_string(index=False))

print("\n=== 最近 25 条预测（含已评估） ===")
rows = store.query_df(
    "SELECT id, symbol, predict_date, horizon, direction, probability, "
    "expected_return_pct, mode, status, actual_return_pct "
    "FROM prediction_log ORDER BY id DESC LIMIT 25", [])
print(rows.to_string(index=False))

print("\n=== 追涨诊断：预测时的动量 vs 之后的实际收益 ===")
# 取已评估的预测，回溯预测日当天的 20 日涨幅、乖离率、是否已涨停
ev = store.query_df(
    "SELECT id, symbol, predict_date, horizon, direction, probability, "
    "expected_return_pct, actual_return_pct, status "
    "FROM prediction_log WHERE status IN ('hit','miss','ambiguous') "
    "ORDER BY id DESC", [])
print(f"已评估 {len(ev)} 条")

recs = []
for _, r in ev.iterrows():
    sym = r["symbol"]
    pd_ = str(r["predict_date"])[:10]
    k = store.query_df(
        "SELECT date, close, high, low, open, volume FROM kline_daily "
        "WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 60",
        [sym, pd_])
    if len(k) < 21:
        continue
    k = k.sort_values("date").reset_index(drop=True)
    c = k["close"].astype(float)
    last = float(c.iloc[-1])
    ret5 = (last / float(c.iloc[-6]) - 1) * 100
    ret20 = (last / float(c.iloc[-21]) - 1) * 100
    ma5 = float(c.tail(5).mean())
    ma20 = float(c.tail(20).mean())
    bias5 = (last / ma5 - 1) * 100
    bias20 = (last / ma20 - 1) * 100
    # 预测日是否已涨停（涨幅 >= 9.8%）
    last_chg = (float(c.iloc[-1]) / float(c.iloc[-2]) - 1) * 100
    recs.append({
        "id": int(r["id"]), "symbol": sym, "pd": pd_, "dir": r["direction"],
        "prob": round(float(r["probability"]), 2),
        "exp": round(float(r["expected_return_pct"]), 2),
        "ret5": round(ret5, 1), "ret20": round(ret20, 1),
        "bias5": round(bias5, 1), "bias20": round(bias20, 1),
        "last_chg": round(last_chg, 1),
        "actual": round(float(r["actual_return_pct"]), 2),
        "status": r["status"],
    })

d = pd.DataFrame(recs)
if d.empty:
    print("无可用样本")
else:
    print(f"\n样本 {len(d)} 条")
    print("\n-- A. 预测时已大涨（20日涨幅>15%）的预测，之后实际收益 --")
    hot = d[d["ret20"] > 15]
    print(f"   共 {len(hot)} 条，平均实际收益 {hot['actual'].mean():+.2f}%"
          f"，命中率 {(hot['status']=='hit').mean():.1%}")
    cool = d[d["ret20"] <= 15]
    print(f"   对照 冷静组 {len(cool)} 条，平均实际收益 {cool['actual'].mean():+.2f}%"
          f"，命中率 {(cool['status']=='hit').mean():.1%}")

    print("\n-- B. 预测日当天已大涨（单日>=9.8%，多为涨停） --")
    lu = d[d["last_chg"] >= 9.8]
    print(f"   共 {len(lu)} 条，平均实际收益 {lu['actual'].mean():+.2f}%"
          f"，命中率 {(lu['status']=='hit').mean():.1%}  (n={len(lu)})")

    print("\n-- C. 看涨预测按预测时乖离率(bias20)分组 --")
    up = d[d["dir"] == "up"].copy()
    up["bias_bin"] = pd.cut(up["bias20"], [-100, 5, 10, 20, 100],
                            labels=["<5%", "5-10%", "10-20%", ">20%"])
    g = up.groupby("bias_bin", observed=True).agg(
        n=("actual", "size"), avg_actual=("actual", "mean"),
        hit=("status", lambda s: (s == "hit").mean()))
    print(g.round(3).to_string())

    print("\n-- D. 看涨预测中预测时 20 日涨幅分档 --")
    up["ret_bin"] = pd.cut(up["ret20"], [-100, 0, 10, 20, 40, 100],
                           labels=["<0", "0-10", "10-20", "20-40", ">40"])
    g2 = up.groupby("ret_bin", observed=True).agg(
        n=("actual", "size"), avg_actual=("actual", "mean"),
        hit=("status", lambda s: (s == "hit").mean()))
    print(g2.round(3).to_string())

    print("\n-- E. 期望收益与实际收益相关性（追涨是否被放大） --")
    print(f"   corr(exp, actual) = {d['exp'].corr(d['actual']):.3f}")
    print(f"   corr(ret20, actual) = {d['ret20'].corr(d['actual']):.3f}")

store.close()
