# 全市场动态扫描池（market_pool）测试：mock 快照，零网络

from __future__ import annotations

import pytest

from pa_mcp.research import market_pool as mp

SNAPSHOT = [
    {"symbol": "sh600001", "name": "涨停票", "changepercent": 10.05,
     "amount": 5e9, "turnoverratio": 12.3, "trade": 10.5},
    {"symbol": "sz000002", "name": "强势票", "changepercent": 6.5,
     "amount": 8e9, "turnoverratio": 8.2, "trade": 12.1},
    {"symbol": "sz300003", "name": "弱票", "changepercent": -2.1,
     "amount": 9e9, "turnoverratio": 5.0, "trade": 30.0},
    {"symbol": "sh600004", "name": "僵尸票", "changepercent": 3.0,
     "amount": 1e7, "turnoverratio": 0.5, "trade": 5.0},
    {"symbol": "bj999999", "name": "北交所票", "changepercent": 4.5,
     "amount": 6e9, "turnoverratio": 10.0, "trade": 8.0},
    {"symbol": "bad", "name": "坏数据", "changepercent": "x",
     "amount": 6e9, "turnoverratio": 10.0, "trade": 8.0},
]


@pytest.fixture(autouse=True)
def _mock_snapshot(monkeypatch):
    """mock 新浪全市场快照：返回固定数据。"""
    async def _fake(*a, **k):
        return SNAPSHOT
    monkeypatch.setattr(
        "pa_mcp.research.sentiment_cycle.SentimentCycleAnalyzer"
        "._fetch_market_snapshot", _fake)
    mp.clear_pool_cache()


def test_basic_filter():
    """涨幅区间 + 成交额门槛过滤。"""
    pool = asyncio_run(mp.build_market_scan_pool(limit=10))
    syms = [p["symbol"] for p in pool]
    assert "000002" in syms          # 强势票通过
    assert "600001" not in syms      # 涨停票剔除（≥9%）
    assert "000003" not in syms      # 弱票剔除
    assert "600004" not in syms      # 成交额不足 5 亿剔除


def test_sort_and_limit():
    """涨幅降序 + limit 截断。"""
    pool = asyncio_run(mp.build_market_scan_pool(limit=1))
    assert len(pool) == 1
    assert pool[0]["change_pct"] > 0


def test_norm_symbol_and_metadata():
    """代码标准化 + 元数据齐全。"""
    pool = asyncio_run(mp.build_market_scan_pool(limit=10))
    hit = next(p for p in pool if p["symbol"] == "000002")
    assert hit["name"] == "强势票"
    assert hit["change_pct"] == 6.5
    assert hit["amount_billion"] == 80.0
    assert hit["turnover"] == 8.2
    assert hit["price"] == 12.1
    assert hit["source"] == "sina_snapshot"


def test_bad_rows_skipped():
    """坏数据行跳过不崩溃。"""
    pool = asyncio_run(mp.build_market_scan_pool(limit=10))
    assert all(p["symbol"] != "bad" for p in pool)


def test_cache_hit(monkeypatch):
    """TTL 缓存：第二次调用不重新抓快照。"""
    calls = {"n": 0}
    real = mp._snapshot_cached

    async def counting():
        async def _fake(*a, **k):
            calls["n"] += 1
            return SNAPSHOT
        monkeypatch.setattr(
            "pa_mcp.research.sentiment_cycle.SentimentCycleAnalyzer"
            "._fetch_market_snapshot", _fake)
        await mp.build_market_scan_pool(limit=10)
        await mp.build_market_scan_pool(limit=5)

    asyncio_run(counting())
    assert calls["n"] == 1  # 第二次走缓存


def test_no_network_fallback(monkeypatch):
    """快照全失败 → 空池（调用方降级），不抛异常。"""
    async def _empty(*a, **k):
        return []
    monkeypatch.setattr(
        "pa_mcp.research.sentiment_cycle.SentimentCycleAnalyzer"
        "._fetch_market_snapshot", _empty)
    mp.clear_pool_cache()
    pool = asyncio_run(mp.build_market_scan_pool(limit=10))
    assert pool == []


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)
