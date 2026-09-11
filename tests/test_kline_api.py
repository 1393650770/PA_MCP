# [AI:BEGIN]
# PA_MCP - K 线读取接口测试（临时数据库，不接触真实 data/pa_mcp.duckdb）
#
# 覆盖的优化点：
#   1. start_date/end_date 真正生效（旧版日线路径完全忽略这两个参数）
#   2. 默认窗口收敛 + 列裁剪（单票全量 2400 行 ≈ 700KB，会撑爆客户端上下文）
#   3. get_kline_batch：单条 SQL 批量取，替代逐只调用的 N+1
# [AI:END]

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pandas as pd
import pytest

import pa_mcp.server as srv
from pa_mcp.data import DuckDBStore


def _kline_df(symbol: str, n: int = 400, start: str = "2024-01-01") -> pd.DataFrame:
    d0 = date.fromisoformat(start)
    rows = []
    for i in range(n):
        d = d0 + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        rows.append({
            "symbol": symbol,
            "date": d,
            "open": 10.0 + i * 0.01,
            "high": 10.5 + i * 0.01,
            "low": 9.5 + i * 0.01,
            "close": 10.2 + i * 0.01,
            "volume": 1e6 + i,
            "amount": 1e7,
            "amplitude": 2.0,
            "pct_change": 0.5,
            "change": 0.05,
            "turnover": 1.2,
            "adjust_factor": 1.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def store(tmp_path):
    """临时 DuckDB，写入两只票的合成日线，并挂到 server 全局。"""
    db = DuckDBStore(str(tmp_path / "t.duckdb"))
    db.connect()
    db.insert_df("kline_daily", _kline_df("000001"))
    db.insert_df("kline_daily", _kline_df("600036", n=120))
    old = srv._store
    srv._store = db
    srv._cache = None       # 关缓存，保证每次都真实查询
    yield db
    srv._store = old
    db.close()


# ---- 参数归一化 ----

def test_norm_date_yyyymmdd():
    assert srv._norm_date("20260101") == "2026-01-01"


def test_norm_date_iso_passthrough():
    assert srv._norm_date("2026-01-01") == "2026-01-01"


def test_norm_date_empty():
    assert srv._norm_date("") == ""


def test_norm_date_today_only_for_end():
    assert srv._norm_date("today", end=True) == date.today().isoformat()


# ---- SQL 构造 ----

def test_kline_query_uses_window_when_days_set():
    sql, params = srv._kline_query(srv._KLINE_CORE_COLS, ["000001"], "", "", 250)
    assert "ROW_NUMBER()" in sql, "days>0 必须用窗口函数按 symbol 取最近 N 根"
    assert params[-1] == 250


def test_kline_query_date_range():
    sql, params = srv._kline_query(srv._KLINE_CORE_COLS, ["000001"],
                                   "2026-01-01", "2026-03-31", 0)
    assert "date >= CAST(? AS DATE)" in sql
    assert "date <= CAST(? AS DATE)" in sql
    assert "2026-01-01" in params and "2026-03-31" in params


def test_kline_query_multi_symbol_single_sql():
    sql, params = srv._kline_query(srv._KLINE_CORE_COLS,
                                   ["000001", "600036"], "", "", 100)
    assert sql.count("?") >= 3, "多 symbol 必须合成一条 SQL（IN 子句）"


# ---- get_kline ----

@pytest.mark.asyncio
async def test_get_kline_default_window_is_bounded(store):
    """默认不得返回全量历史（旧版单票 2400 行 ≈ 700KB）。"""
    r = await srv.get_kline("000001")
    assert r["success"]
    assert r["data"]["rows"] <= 250
    assert r["data"]["window"]["days"] == 250


@pytest.mark.asyncio
async def test_get_kline_respects_date_range(store):
    """防回归：start_date/end_date 必须在库路径生效（旧版直接忽略）。"""
    r = await srv.get_kline("000001", start_date="20240101", end_date="20240331",
                            days=0)
    recs = r["data"]["kline"]
    assert recs, "日期窗口内应有数据"
    assert recs[0]["date"] >= "2024-01-01"
    assert recs[-1]["date"] <= "2024-03-31"
    assert r["data"]["rows"] == len(recs)


@pytest.mark.asyncio
async def test_get_kline_compact_columns(store):
    r = await srv.get_kline("000001", days=5, compact=True)
    assert set(r["data"]["kline"][0].keys()) == {
        "date", "open", "high", "low", "close", "volume"}


@pytest.mark.asyncio
async def test_get_kline_full_columns(store):
    r = await srv.get_kline("000001", days=5, compact=False)
    assert "turnover" in r["data"]["kline"][0]
    assert "adjust_factor" in r["data"]["kline"][0]


@pytest.mark.asyncio
async def test_get_kline_full_flag_returns_all(store):
    r = await srv.get_kline("000001", full=True)
    assert r["data"]["rows"] > 250
    assert r["data"]["window"]["days"] is None


@pytest.mark.asyncio
async def test_get_kline_reports_as_of(store):
    r = await srv.get_kline("000001", days=30)
    assert r["data"]["kline"], "应有数据"
    assert r["data_freshness"].startswith(r["data"]["kline"][-1]["date"])


@pytest.mark.asyncio
async def test_get_kline_unknown_symbol_fails(store):
    r = await srv.get_kline("999999")
    # 库里没有该票：要么明确 NOT_FOUND，要么走网络源失败——都不该抛异常
    assert isinstance(r, dict)
    assert "success" in r


# ---- get_kline_batch ----

@pytest.mark.asyncio
async def test_batch_returns_all_symbols(store):
    r = await srv.get_kline_batch("000001,600036", days=100)
    assert r["success"]
    assert set(r["data"]["symbols"]) == {"000001", "600036"}
    assert r["data"]["missing"] == []
    assert r["data"]["rows_total"] == sum(
        len(v) for v in r["data"]["kline"].values())


@pytest.mark.asyncio
async def test_batch_per_symbol_window(store):
    """每只票各自取最近 N 根（窗口函数按 symbol 分区）。"""
    r = await srv.get_kline_batch("000001,600036", days=50)
    for sym, recs in r["data"]["kline"].items():
        assert len(recs) <= 50
    # 600036 只有约 86 根，应少于 000001 的 50 根窗口上限
    assert len(r["data"]["kline"]["600036"]) <= len(r["data"]["kline"]["000001"])


@pytest.mark.asyncio
async def test_batch_reports_missing(store):
    r = await srv.get_kline_batch("000001,999999", days=50)
    assert "999999" in r["data"]["missing"]


@pytest.mark.asyncio
async def test_batch_caps_symbol_count(store):
    r = await srv.get_kline_batch("000001,600036,999998,999997", max_symbols=2)
    assert len(r["data"]["symbols"]) <= 2
    assert len(r["data"]["dropped_over_cap"]) == 2


@pytest.mark.asyncio
async def test_batch_rejects_empty(store):
    r = await srv.get_kline_batch("")
    assert r["success"] is False
    assert r["error_type"] == "INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_batch_supports_chinese_separators(store):
    r = await srv.get_kline_batch("000001、600036", days=10)
    assert set(r["data"]["symbols"]) == {"000001", "600036"}


@pytest.mark.asyncio
async def test_batch_as_of_per_symbol(store):
    r = await srv.get_kline_batch("000001,600036", days=10)
    assert set(r["data"]["as_of"].keys()) == {"000001", "600036"}
    assert all(len(v) == 10 for v in r["data"]["as_of"].values())


def test_max_rows_guard_is_reasonable():
    """单票全量约 2400 行；guard 必须大于它但远小于'全市场全量'。"""
    assert 2400 < srv.MAX_KLINE_ROWS <= 10000
    assert srv.MAX_BATCH_SYMBOLS >= 20
