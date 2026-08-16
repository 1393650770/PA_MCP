# [AI:BEGIN]
# PA_MCP - 数据前置检查与自动装载 测试
# [AI:END]

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest


def _seed_db():
    """临时库：kline 有 100 根，financials 空。"""
    from pa_mcp.data.store import DuckDBStore
    db = os.path.join(tempfile.mkdtemp(), "readiness.duckdb")
    store = DuckDBStore(db)
    store.connect()
    dates = pd.date_range("2026-05-01", periods=100, freq="B")
    close = 10.0
    rows = []
    for i in range(100):
        close *= 1.001
        rows.append({"symbol": "000001", "date": dates[i],
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close,
                     "volume": 1e6, "amount": 1e7, "pct_change": 0.1,
                     "turnover": 1.0, "change": 0.1, "amplitude": 1.0,
                     "adjust_factor": 1.0})
    store.insert_df("kline_daily", pd.DataFrame(rows))
    store.close()
    return db


def test_check_readiness(tmp_path, monkeypatch):
    """检查各表状态：有行情/无财务。"""
    db = _seed_db()
    # 指向临时库
    import pa_mcp.config as cfg
    real = cfg.get_settings()
    d = real.model_dump()
    d["database"] = {"path": db, "read_only": False}
    monkeypatch.setattr(cfg, "_settings", type(real)(**d))

    from pa_mcp.data.readiness import check_readiness
    status = check_readiness({"kline_daily": 60, "financials_income": 1})
    assert status["kline_daily"] is True
    assert status["financials_income"] is False


def test_ensure_no_missing_is_noop(tmp_path, monkeypatch):
    """数据齐备 → 零装载动作。"""
    db = _seed_db()
    import pa_mcp.config as cfg
    real = cfg.get_settings()
    d = real.model_dump()
    d["database"] = {"path": db, "read_only": False}
    monkeypatch.setattr(cfg, "_settings", type(real)(**d))

    from pa_mcp.data.readiness import ensure_readiness
    import asyncio
    r = asyncio.run(ensure_readiness({"kline_daily": 60}))
    assert r["loaded"] is False
    assert "已就绪" in r["actions"][0]


def test_ensure_missing_kline_network_fail(tmp_path, monkeypatch):
    """行情缺失 + 网络不可达 → 不崩溃，标记未满足。"""
    db = os.path.join(str(tmp_path), "empty.duckdb")  # 空库
    import pa_mcp.config as cfg
    real = cfg.get_settings()
    d = real.model_dump()
    d["database"] = {"path": db, "read_only": False}
    monkeypatch.setattr(cfg, "_settings", type(real)(**d))

    from pa_mcp.data.readiness import ensure_readiness
    import asyncio
    # 网络会被 router 重试拖慢——用极短超时的 monkeypatch
    async def fast_load(*a, **k):
        return 0
    import pa_mcp.data.readiness as rd
    monkeypatch.setattr(rd, "_load_kline", fast_load)
    r = asyncio.run(ensure_readiness({"kline_daily": 60}))
    assert r["loaded"] is True
    assert r["status"]["kline_daily"] is False  # 装载失败标记，不抛异常
