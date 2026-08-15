# [AI:BEGIN]
# PA_MCP - 板块基准事件研究测试（风格匹配）
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd

from pa_mcp.research.event_study import signal_forward_returns


def _make_df(n=120, seed=1, trend=0.001):
    rng = np.random.default_rng(seed)
    close = 10.0
    rows = []
    for i in range(n):
        close *= 1 + trend + rng.normal(0, 0.008)
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6,
                     "symbol": "000001"})
    return pd.DataFrame(rows)


def _sig12(df):
    """12 个信号（跨日期，满足 ≥10 事件门槛）。"""
    return pd.DataFrame([
        {"symbol": "000001", "date": str(df["date"].iloc[i])[:10],
         "direction": "bullish", "strategy_name": "t"}
        for i in range(5, 65, 5)])


def test_benchmark_series_style():
    """Series 形态基准：所有 horizon 共用。"""
    df = _make_df()
    sig = _sig12(df)
    bench = pd.Series({str(d)[:10]: 0.5 for d in df["date"]})
    results = signal_forward_returns(df, sig, [5], benchmark_returns=bench)
    assert results
    assert results[0].benchmark_avg_return_pct == 0.5
    assert results[0].excess_return_pct == round(
        results[0].avg_return_pct - 0.5, 2)


def test_benchmark_dict_style():
    """dict[horizon] 形态：各 horizon 独立基准。"""
    df = _make_df()
    sig = _sig12(df)
    bench_maps = {
        5: pd.Series({str(d)[:10]: 1.0 for d in df["date"]}),
        10: pd.Series({str(d)[:10]: 2.0 for d in df["date"]}),
    }
    results = signal_forward_returns(df, sig, [5, 10],
                                     benchmark_returns=bench_maps)
    r5 = next(r for r in results if r.horizon == 5)
    r10 = next(r for r in results if r.horizon == 10)
    assert r5.benchmark_avg_return_pct == 1.0
    assert r10.benchmark_avg_return_pct == 2.0


def test_no_benchmark_backward_compat():
    """无基准参数：行为与旧版一致（无条件基准）。"""
    df = _make_df()
    sig = _sig12(df)
    r1 = signal_forward_returns(df, sig, [5])
    r2 = signal_forward_returns(df, sig, [5], benchmark_returns=None)
    assert r1[0].benchmark_avg_return_pct == r2[0].benchmark_avg_return_pct


def test_sector_alpha_detection():
    """板块基准检验：超额 = 个股收益 - 板块基准。"""
    df = _make_df(seed=3)
    sig = _sig12(df)
    bench = pd.Series({str(d)[:10]: 1.0 for d in df["date"]})
    results = signal_forward_returns(df, sig, [5], benchmark_returns=bench)
    r = results[0]
    assert r.benchmark_avg_return_pct == 1.0
    assert r.excess_return_pct == round(r.avg_return_pct - 1.0, 2)
