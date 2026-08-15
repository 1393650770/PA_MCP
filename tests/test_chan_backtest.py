# [AI:BEGIN]
# PA_MCP - 缠论背驰组合回测测试
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd

from pa_mcp.research.chan_backtest import (
    backtest_beichi_signals,
    format_beichi_backtest,
)


def _zigzag(n=260, seed=1, amp=0.006):
    """锯齿衰减行情（产生背驰信号）。"""
    rng = np.random.default_rng(seed)
    close = 10.0
    rows = []
    for i in range(n):
        phase = (i // 40) % 4
        if phase in (0, 2):
            close *= 1 + amp + rng.normal(0, 0.003)
        else:
            close *= 1 - amp * 0.8 + rng.normal(0, 0.003)
        rows.append({"date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6,
                     "symbol": "000001"})
    return pd.DataFrame(rows)


def test_backtest_beichi_signals(tmp_path):
    """背驰信号组合回测：返回组合/基准/超额/信号数。"""
    klines = {f"6000{i:02d}": _zigzag(seed=i, amp=0.006 if i % 2 else 0.008)
              for i in range(1, 5)}
    r = backtest_beichi_signals(klines, window=60, step=5)
    assert "error" not in r
    assert r["n_stock"] == 4
    assert r["total_signals"] >= 1
    assert r["portfolio"]["total_return_pct"] is not None
    assert r["benchmark"]["total_return_pct"] is not None
    assert r["excess_return_pct"] is not None
    text = format_beichi_backtest(r)
    assert "背驰信号组合回测" in text


def test_backtest_no_signals():
    """无背驰信号 → 明确 error。"""
    # 单调上涨（无动能衰竭）→ 无背驰
    rows = []
    close = 10.0
    for i in range(200):
        close *= 1.004
        rows.append({"date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6,
                     "symbol": "000001"})
    df = pd.DataFrame(rows)
    r = backtest_beichi_signals({"600001": df, "600002": df})
    assert "error" in r


def test_backtest_short_data():
    r = backtest_beichi_signals({"600001": _zigzag(n=80)})
    assert "error" in r
