# [AI:BEGIN]
# PA_MCP - 全策略事件研究对比测试
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd

from pa_mcp.research.strategy_compare import (
    compare_all_strategies,
    format_compare,
)


def _regime_df(n=300, seed=1, up_bias=0.0):
    """分段趋势行情（含少量上涨偏移）。"""
    rng = np.random.default_rng(seed)
    close = 10.0
    rows = []
    regime = 1.0
    for i in range(n):
        if i % 90 == 0:
            regime = rng.choice([-1.0, 1.0])
        close *= 1 + regime * 0.003 + up_bias + rng.normal(0, 0.006)
        rows.append({"date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6})
    return pd.DataFrame(rows)


def test_compare_all_strategies():
    """多股票多策略：返回全部注册策略的聚合对比。"""
    from pa_mcp.engine.strategies.base import StrategyRegistry
    registry = StrategyRegistry()
    registry.auto_discover()
    n_registered = len(registry.list_all())

    klines = {f"6000{i:02d}": _regime_df(seed=i) for i in range(1, 5)}
    r = compare_all_strategies(klines, horizons=(5, 10))
    assert "error" not in r
    assert r["n_strategies"] >= 3  # 部分策略有足够信号
    assert len(r["strategies"]) == r["n_strategies"]
    # 按 5 日超额降序
    excess = [s["excess_5d_pct"] for s in r["strategies"]]
    assert excess == sorted(excess, reverse=True)
    # 每策略字段齐全
    s0 = r["strategies"][0]
    assert "total_events" in s0 and "horizons" in s0
    assert "5" in s0["horizons"]
    assert isinstance(s0["useful"], bool)
    text = format_compare(r)
    assert "全策略" in text and "有效策略" in text or "无策略" in text


def test_compare_no_signal():
    """无信号策略被过滤（不产生 0 事件行）。"""
    klines = {"600001": _regime_df(n=100, seed=1)}  # 太短 → 多数策略无信号
    r = compare_all_strategies(klines, min_events=5)
    assert "error" not in r or "无" in str(r.get("error", ""))
    if "error" not in r:
        for s in r["strategies"]:
            assert s["total_events"] >= 5


def test_compare_empty():
    r = compare_all_strategies({})
    assert "error" in r
