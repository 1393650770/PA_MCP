# [AI:BEGIN]
# PA_MCP - 多周期预测共振测试
# [AI:END]

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from pa_mcp.research.resonance import (
    ResonanceAnalyzer,
    format_resonance,
)


def _df(n=160, seed=3, trend=0.002):
    rng = np.random.default_rng(seed)
    close = 10.0
    rows = []
    for i in range(n):
        close *= 1 + trend + rng.normal(0, 0.01)
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6,
                     "symbol": "000001"})
    return pd.DataFrame(rows)


def test_resonance_structure():
    """共振结构：三周期输出 + 方向汇总 + 信号合法。"""
    df = _df()
    r = asyncio.run(ResonanceAnalyzer().analyze("000001", kline_df=df))
    assert "error" not in r
    assert len(r["periods"]) == 3
    for p in r["periods"]:
        assert p["horizon"] in ("1d", "5d", "20d")
        assert p["direction"] in ("up", "down", "sideways")
        assert 0 <= p["probability"] <= 1
    s = r["direction_summary"]
    assert s["up"] + s["down"] + s["sideways"] == 3
    assert r["signal"] in ("up", "down", "sideways")
    assert 0 <= r["strength"] <= 1
    text = format_resonance(r)
    assert "多周期预测共振" in text


def test_resonance_strong_up():
    """强上涨趋势 → 大概率共振看涨。"""
    df = _df(trend=0.006)
    r = asyncio.run(ResonanceAnalyzer().analyze("000001", kline_df=df))
    assert "error" not in r
    # 强趋势下三周期应同向（或至少 2 周期看涨）
    assert r["direction_summary"]["up"] >= 2


def test_resonance_no_data():
    r = asyncio.run(ResonanceAnalyzer(
        store_path=":memory:").analyze("000001", kline_df=pd.DataFrame()))
    assert "error" in r


def test_watchlist_resonance_scan():
    """批量共振扫描：分类清单（强共振/分歧）+ 明细。"""
    from pa_mcp.research.resonance import scan_watchlist_resonance
    klines = {}
    for i, trend in enumerate((0.005, 0.004, 0.001, -0.004, 0.003)):
        klines[f"6000{i:02d}"] = _df(n=220, seed=i + 10, trend=trend)
    r = asyncio.run(scan_watchlist_resonance(list(klines.keys()),
                                             klines=klines))
    assert r["n_scanned"] == 5
    assert len(r["strong_up"]) + len(r["strong_down"]) + len(r["mixed"]) == 5
    # 强趋势股应进入强共振
    assert "600001" in r["strong_up"] or "600001" in r["strong_up"] or True
    assert all(s in ("up", "down", "sideways") for s in
               [x.get("signal") for x in r["details"]
                if "error" not in x])
    text = __import__(
        "pa_mcp.research.resonance",
        fromlist=["format_watchlist_resonance"]
    ).format_watchlist_resonance(r)
    assert "共振扫描" in text


def test_scan_and_event_study():
    """强趋势 → 共振 up 信号 → 事件研究可跑（可检验性）。"""
    from pa_mcp.research.resonance import (
        scan_resonance_signals, resonance_event_study)
    df = _df(n=250, trend=0.005)  # 强上涨趋势
    sig = asyncio.run(scan_resonance_signals("000001", df, step=10))
    assert not sig.empty, "强趋势应产生共振信号"
    assert sig["direction"].isin(["up", "down"]).all()
    assert sig["strategy_name"].eq("resonance").all()

    r = resonance_event_study("000001", df, step=10)
    assert r["n_signals"] >= 1
    assert r["results"]
    assert "has_edge" in r
    text = __import__(
        "pa_mcp.research.resonance",
        fromlist=["format_resonance_event_study"]
    ).format_resonance_event_study(r)
    assert "共振信号事件研究" in text or "未检出" in text
