# [AI:BEGIN]
# PA_MCP - 数据质量体检测试
# [AI:END]

from __future__ import annotations

import pandas as pd

from pa_mcp.data.quality_report import (
    DataQualityReport,
    format_report,
)


def _seed(tmp_path, inject_bad=False):
    """灌数据：正常行情 + 可选注入异常（OHLC 违反/缺口/负价）。"""
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "quality_test.duckdb")
    store = DuckDBStore(db)
    store.connect()
    # 正常股票 200 根（工作日无缺口）
    dates = pd.date_range("2026-01-01", periods=200, freq="B")
    rows = []
    for sym in ("000001", "000002"):
        close = 10.0
        for i in range(200):
            close *= 1.001
            rows.append({"symbol": sym, "date": dates[i],
                         "open": close * 0.995, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1e6, "amount": 1e7,
                         "pct_change": 0.1, "turnover": 1.0,
                         "change": 0.1, "amplitude": 2.0,
                         "adjust_factor": 1.0})
    if inject_bad:
        # 异常股票：OHLC 违反（high < close）+ 负价 + 缺口（跳过 10 天）
        bad_dates = pd.date_range("2026-01-01", periods=100, freq="B")
        bad_dates = bad_dates[:50].append(
            bad_dates[60:])  # 50-59 缺失 → 缺口
        for i, d in enumerate(bad_dates):
            rows.append({"symbol": "999999", "date": d,
                         "open": 5.0, "high": 4.0,   # high < close 违反
                         "low": 1.0, "close": 6.0,
                         "volume": -100,              # 负量
                         "amount": 1e7, "pct_change": 1.0,
                         "turnover": 1.0, "change": 1.0,
                         "amplitude": 2.0, "adjust_factor": 1.0})
    store.insert_df("kline_daily", pd.DataFrame(rows))
    # 其他表少量数据
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": "000001", "name": "a", "sector": "s", "is_st": False},
        {"symbol": "000002", "name": "b", "sector": "s", "is_st": False},
    ]))
    store.close()
    return db


def test_quality_report_healthy(tmp_path):
    db = _seed(tmp_path, inject_bad=False)
    r = DataQualityReport(store_path=db).generate()
    assert r["score"] >= 90
    assert r["kline_checks"]["sampled"] == 2
    assert r["kline_checks"]["ohlc_bad_rows"] == 0
    assert r["kline_checks"]["date_gaps"] == 0
    assert "kline_daily" in r["tables"]
    assert r["tables"]["kline_daily"]["rows"] == 400
    text = format_report(r)
    assert "数据质量体检" in text and "表覆盖" in text


def test_quality_report_detects_issues(tmp_path):
    db = _seed(tmp_path, inject_bad=True)
    r = DataQualityReport(store_path=db).generate()
    assert r["kline_checks"]["ohlc_bad_rows"] >= 50   # 999999 全部违反
    assert r["kline_checks"]["non_positive_rows"] >= 50  # 负量
    assert r["kline_checks"]["date_gaps"] >= 1        # 缺口
    assert r["score"] < 90
    assert any("999999" in i for i in r["issues"])


def test_quality_report_empty(tmp_path):
    r = DataQualityReport(store_path=str(tmp_path / "empty.duckdb")).generate()
    assert r["kline_checks"]["issues"]
    assert r["score"] <= 70
