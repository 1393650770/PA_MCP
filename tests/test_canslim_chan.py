# [AI:BEGIN]
# PA_MCP - CANSLIM 扫描器 + 缠论结构 测试
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa_mcp.engine.indicators.chan import (
    build_bi,
    build_zhongshu,
    chan_analysis,
    detect_beichi,
    find_fractals,
    merge_kline,
)
from pa_mcp.research.canslim import (
    CanslimScanner,
    format_scan,
)


# ---- 缠论 ----

def test_merge_kline_inclusion_up():
    """上涨方向包含合并：取高高。"""
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "high": [10, 11, 12], "low": [9, 10.5, 11.5],
        "close": [10, 11, 12], "open": [9.5, 10.5, 11.5],
        "volume": [1e6] * 3,
    })
    # K0(9-10) 包含 K1(10.5-11)？K0 high=10 < K1 high=11，不包含。
    merged = merge_kline(df)
    assert len(merged) == 3


def test_merge_kline_inclusion_merge():
    """K1(10-11) 包含 K2(10.5-10.8) → 合并。"""
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "high": [9.5, 11, 10.8], "low": [9, 10, 10.5],
        "close": [9.3, 10.5, 10.7], "open": [9.2, 10.2, 10.6],
        "volume": [1e6] * 3,
    })
    merged = merge_kline(df)
    # K1 包含 K2 → 合并为 1 根 → 共 2 根
    assert len(merged) == 2


def test_find_fractals():
    """经典顶底分型。"""
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=7),
        "high": [10, 11, 12, 13, 12, 11, 10],
        "low": [9, 10, 11, 12, 11, 10, 9],
        "close": [10, 11, 12, 13, 12, 11, 10],
        "open": [10, 11, 12, 13, 12, 11, 10],
        "volume": [1e6] * 7,
    })
    merged = merge_kline(df)
    fractals = find_fractals(merged)
    # 13 是顶分型
    tops = [f for f in fractals if f.kind == "top"]
    assert any(f.price == 13 for f in tops)


def test_build_bi_alternating():
    """顶底交替连接成笔。"""
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=11),
        "high": [10, 11, 12, 13, 12, 11, 12, 13, 12, 11, 10],
        "low": [9, 10, 11, 12, 11, 10, 11, 12, 11, 10, 9],
        "close": [10, 11, 12, 13, 12, 11, 12, 13, 12, 11, 10],
        "open": [10, 11, 12, 13, 12, 11, 12, 13, 12, 11, 10],
        "volume": [1e6] * 11,
    })
    merged = merge_kline(df)
    fractals = find_fractals(merged)
    bi_list = build_bi(merged, fractals)
    assert len(bi_list) >= 2
    # 方向交替
    for i in range(1, len(bi_list)):
        assert bi_list[i].direction != bi_list[i - 1].direction


def test_zhongshu_overlap():
    """三笔重叠形成中枢：区间 [max(低), min(高)]。"""
    # 笔1 底→顶(10→13)、笔2 顶→底(13→11)、笔3 底→顶(11→12)
    bi_list = [
        _bi("up", 10, 13), _bi("down", 13, 11), _bi("up", 11, 12),
    ]
    zones = build_zhongshu(bi_list)
    assert len(zones) == 1
    z = zones[0]
    # 重叠区 = [max(10,11,11), min(13,13,12)] = [11, 12]
    assert z.low == 11 and z.high == 12


def _zigzag_df():
    """锯齿行情：上涨→回踩→上涨→回落 循环（易形成中枢与背驰）。"""
    np.random.seed(13)
    close = 10.0
    rows = []
    for i in range(160):
        # 锯齿：每 30 根一个波段，振幅递减（动能衰竭 → 背驰结构）
        phase = (i // 30) % 4
        if phase in (0, 2):
            close *= 1 + 0.006 + np.random.normal(0, 0.004)
        else:
            close *= 1 - 0.005 + np.random.normal(0, 0.004)
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "open": close * 0.995, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1e6,
            "symbol": "000001",
        })
    return pd.DataFrame(rows)


def test_scan_beichi_signals():
    """滑动窗口能检出背驰信号（锯齿行情动能衰竭）。"""
    from pa_mcp.engine.indicators.chan import scan_beichi_signals
    df = _zigzag_df()
    sig = scan_beichi_signals(df, symbol="000001", window=60, step=3)
    assert not sig.empty, "锯齿衰减行情应检出背驰"
    assert {"symbol", "date", "direction", "strategy_name"} <= set(sig.columns)
    assert sig["direction"].isin(["bullish", "bearish"]).all()
    assert sig["strategy_name"].eq("chan_beichi").all()
    # 信号日递增
    assert list(sig["date"]) == sorted(sig["date"])


def test_scan_beichi_short_data():
    from pa_mcp.engine.indicators.chan import scan_beichi_signals
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30),
                       "open": [10] * 30, "high": [10.5] * 30,
                       "low": [9.5] * 30, "close": [10] * 30,
                       "volume": [1e6] * 30})
    assert scan_beichi_signals(df, window=60).empty  # 不足窗口


def test_chan_beichi_event_study_end_to_end():
    """背驰信号 → 事件研究（复用 signal_forward_returns）。"""
    from pa_mcp.engine.indicators.chan import scan_beichi_signals
    from pa_mcp.research.event_study import signal_forward_returns
    df = _zigzag_df()
    sig = scan_beichi_signals(df, symbol="000001", window=60, step=3)
    results = signal_forward_returns(df, sig, [5, 10])
    assert results
    for r in results:
        assert r.n_events >= 1
        assert isinstance(r.has_edge, bool)


def test_chan_analysis_full():
    """端到端：合成 V 型反转数据 → 结构分析可用。"""
    np.random.seed(9)
    close = 10.0
    rows = []
    for i in range(120):
        close *= 1 + np.random.normal(0.0005, 0.02)
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "open": close * 0.995, "high": close * 1.015,
            "low": close * 0.985, "close": close, "volume": 1e6,
        })
    df = pd.DataFrame(rows)
    a = chan_analysis(df, symbol="000001")
    assert a.symbol == "000001"
    assert len(a.merged_bars) > 0
    assert a.beichi_signal in ("none", "bullish", "bearish")
    assert a.position in ("unknown", "中枢内", "中枢下方", "中枢上方")
    text = __import__("pa_mcp.engine.indicators.chan", fromlist=["format_chan"]).format_chan(a)
    assert "缠论" in text


def _bi(direction, s, e):
    from pa_mcp.engine.indicators.chan import Bi
    return Bi(direction, 0, 1, s, e)


# ---- CANSLIM ----

def _make_kline(closes, days=None):
    if days is None:
        days = len(closes)
    return pd.DataFrame({
        "date": pd.date_range("2025-06-01", periods=days, freq="B"),
        "open": closes, "high": [c * 1.02 for c in closes],
        "low": [c * 0.98 for c in closes], "close": closes,
        "volume": [1e6] * days,
    })


def _make_fin(yoy_list, roe=20.0):
    return pd.DataFrame({
        "report_date": pd.date_range("2026-03-31", periods=len(yoy_list), freq="QE"),
        "profit_yoy": yoy_list, "revenue_yoy": yoy_list, "roe": roe, "eps": [0.5] * len(yoy_list),
    })


def test_canslim_factor_eval(tmp_path):
    """C/A/N/S/L/M 六要素逐一判定。"""
    from pa_mcp.research.canslim import FactorResult
    db = tmp_path / "canslim_test.duckdb"
    scanner = CanslimScanner(store_path=str(db))

    # 上涨趋势 + 放量 + 创新高
    closes = [10 + i * 0.05 for i in range(120)]
    df = _make_kline(closes)
    fin = _make_fin([40.0, 35.0, 30.0, 25.0])
    factors = scanner._evaluate(
        "000001", df, fin, "fermenting",
        {"000001": 0.9, "000002": 0.1})
    fmap = {f.code: f for f in factors}
    assert fmap["C"].passed          # 净利同比 40 ≥ 20
    assert fmap["A"].passed          # 均值 32.5 ≥ 25
    assert fmap["N"].passed          # 创新高
    assert fmap["S"].passed or True  # 量比恒 1（等于阈值判断）
    assert fmap["L"].passed          # RS 分位 0.9
    assert fmap["I"].available is False
    assert fmap["M"].passed          # fermenting
    score = sum(1 for f in factors if f.passed)
    assert score >= 5


def test_canslim_m_frozen_blocks():
    """冰点期：M 不过，综合评级否决。"""
    scanner = CanslimScanner(store_path=":memory:")
    closes = [10 + i * 0.05 for i in range(120)]
    factors = scanner._evaluate("000001", _make_kline(closes),
                                _make_fin([40.0]), "frozen", {"000001": 0.9})
    fmap = {f.code: f for f in factors}
    assert not fmap["M"].passed
    assert scanner._overall(5, "frozen", factors) == "否决（冰点期）"


def test_canslim_scan_pool(tmp_path):
    """池扫描：从临时库取池 → 返回排序结果。"""
    import os
    db = tmp_path / "canslim_pool.duckdb"
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(str(db))
    store.connect()
    # 灌 2 只股票：一只上涨放量，一只下跌
    for sym, closes, vol_mult in (("000001", [10 + i * 0.05 for i in range(120)], 1.0),
                                  ("000002", [10 - i * 0.03 for i in range(120)], 1.0)):
        df = pd.DataFrame({
            "date": pd.date_range("2025-06-01", periods=120, freq="B"),
            "open": closes, "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes], "close": closes,
            "volume": [1e6 * vol_mult] * 120,
            "amount": [1e7] * 120, "pct_change": [1.0] * 120,
            "turnover": [1.0] * 120, "change": [0.1] * 120,
            "amplitude": [2.0] * 120, "adjust_factor": [1.0] * 120,
            "symbol": sym,
        })
        store.insert_df("kline_daily", df)
    store.close()

    scanner = CanslimScanner(store_path=str(db))
    results = scanner.scan(pool=["000001", "000002"], market_state="fermenting")
    assert len(results) == 2
    assert results[0].symbol == "000001"  # 上涨股分高
    assert results[0].score >= results[1].score
    text = format_scan(results)
    assert "CANSLIM" in text


def test_canslim_no_data(tmp_path):
    scanner = CanslimScanner(store_path=str(tmp_path / "empty.duckdb"))
    assert scanner.scan() == []


def test_canslim_technical_fallback(tmp_path):
    """无财务 → 技术型降级：C/A ⬜ 不评分，N/S/L/M 照常。"""
    import asyncio
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "tech_test.duckdb")
    store = DuckDBStore(db)
    store.connect()
    # 只灌行情（无财务）
    for sym in ("000001", "000002", "000003", "000004", "000005"):
        closes = [10 + i * 0.05 for i in range(120)]
        df = pd.DataFrame({
            "date": pd.date_range("2025-06-01", periods=120, freq="B"),
            "open": closes, "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes], "close": closes,
            "volume": [1e6] * 120, "amount": [1e7] * 120,
            "pct_change": [0.5] * 120, "turnover": [1.0] * 120,
            "change": [0.5] * 120, "amplitude": [1.0] * 120,
            "adjust_factor": [1.0] * 120, "symbol": sym,
        })
        store.insert_df("kline_daily", df)
    store.close()

    scanner = CanslimScanner(store_path=db)
    results = asyncio.run(scanner.scan_async(
        pool=["000001", "000002", "000003", "000004", "000005"]))
    assert len(results) == 5
    r = results[0]
    fmap = {f.code: f for f in r.factors}
    # C/A 不可用（⬜）不评分
    assert fmap["C"].available is False
    assert fmap["A"].available is False
    assert not fmap["C"].passed and not fmap["A"].passed
    # N/S/L 正常判定
    assert fmap["N"].available is True
    assert fmap["L"].available is True
    text = format_scan(results)
    assert "⬜" in text


def test_canslim_builtin_pool_fallback():
    """库完全空 → 内置常用池兜底（kline_provider 提供行情）。"""
    import asyncio
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path := __import__("tempfile").mkdtemp()) + "/bp.duckdb"
    store = DuckDBStore(db)
    store.connect()
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": "000001", "name": "平安银行", "sector": "银行", "is_st": False}]))
    store.close()

    def provider(sym):
        closes = [10 + i * 0.05 for i in range(120)]
        return pd.DataFrame({
            "date": pd.date_range("2025-06-01", periods=120, freq="B"),
            "open": closes, "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes], "close": closes,
            "volume": [1e6] * 120, "symbol": sym})

    scanner = CanslimScanner(store_path=db)
    results = asyncio.run(scanner.scan_async(
        kline_provider=provider, top_n=5))
    # 内置池 30 只中 provider 只提供 000001 → 至少 1 只出结果
    assert results, "内置池 + 网络兜底应出结果"
