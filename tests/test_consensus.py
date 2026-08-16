# [AI:BEGIN]
# PA_MCP - 综合决策信号测试
# [AI:END]

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from pa_mcp.research.consensus import (
    ConsensusAnalyzer,
    format_consensus,
)


def _df(n=200, seed=3, trend=0.004):
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


def test_consensus_structure():
    """综合信号：多源聚合 + 投票 + 强度/一致度。"""
    df = _df()
    r = asyncio.run(ConsensusAnalyzer().analyze("000001", kline_df=df))
    assert "error" not in r
    assert r["sources"], "至少一个信号源"
    assert r["signal"] in ("up", "down", "sideways")
    assert 0 <= r["strength"] <= 1
    assert 0 <= r["agreement"] <= 1
    assert r["level"] in ("强", "中", "弱")
    # 强趋势 → 共振+预测+策略大概率看涨
    if r["sources"].get("resonance"):
        assert r["sources"]["resonance"]["signal"] in ("up", "down", "sideways")
    text = format_consensus(r)
    assert "综合决策信号" in text and "投票" in text


def test_consensus_up_bias():
    """强上涨趋势 → 综合信号偏涨。"""
    df = _df(trend=0.007)
    r = asyncio.run(ConsensusAnalyzer().analyze("000001", kline_df=df))
    assert "error" not in r
    votes = r["votes"]
    assert votes["up"] >= votes["down"]


def test_consensus_no_data():
    r = asyncio.run(ConsensusAnalyzer(
        store_path=":memory:").analyze("000001", kline_df=pd.DataFrame()))
    assert "error" in r


def test_watchlist_consensus_scan():
    """批量综合信号：分类清单 + 明细（信号/强度/源数）。"""
    from pa_mcp.research.consensus import (
        scan_watchlist_consensus, format_watchlist_consensus)
    klines = {}
    for i, trend in enumerate((0.005, 0.004, -0.003, 0.001, 0.003)):
        klines[f"6000{i:02d}"] = _df(n=220, seed=i + 20, trend=trend)
    r = asyncio.run(scan_watchlist_consensus(list(klines.keys()),
                                             klines=klines))
    assert r["n_scanned"] == 5
    assert len(r["strong_up"]) + len(r["strong_down"]) + len(r["mixed"]) == 5
    for x in r["details"]:
        if "error" not in x:
            assert x["signal"] in ("up", "down", "sideways")
            assert 0 <= x["strength"] <= 1
            assert x["n_sources"] >= 1
    text = format_watchlist_consensus(r)
    assert "综合信号扫描" in text


def test_consensus_figure_build():
    """投票分布图构建：柱状 + 源标注 + 三方向。"""
    from pa_mcp.ui.gradio_app import _build_consensus_figure
    votes = {"up": 5.4, "down": 2.1, "sideways": 0.5}
    sources = {
        "resonance": {"signal": "up", "strength": 1.0},
        "prediction": {"signal": "up", "strength": 0.62},
        "strategy": {"signal": "down", "strength": 0.67},
        "market": {"signal": "up", "strength": 0.7},
    }
    fig = _build_consensus_figure(votes, sources)
    assert len(fig.data) == 2  # 柱状 + 源散点
    assert len(fig.data[0].x) == 3  # 三方向
    assert len(fig.data[1].x) == 4  # 四源


def test_consensus_event_study():
    """强趋势 → 综合信号 → 事件研究可跑（可检验性）。"""
    from pa_mcp.research.consensus import (
        scan_consensus_signals, consensus_event_study,
        format_consensus_event_study)
    df = _df(n=300, trend=0.005)
    sig = asyncio.run(scan_consensus_signals("000001", df, step=10))
    assert not sig.empty, "强趋势应产生综合信号"
    assert sig["direction"].isin(["up", "down"]).all()
    assert sig["strategy_name"].eq("consensus").all()

    r = consensus_event_study("000001", df, step=10)
    assert r["n_signals"] >= 1
    assert r["results"]
    assert "has_edge" in r
    text = format_consensus_event_study(r)
    assert "综合信号事件研究" in text or "未检出" in text
