# [AI:BEGIN]
# PA_MCP - 价值×动量复合选股测试
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd

from pa_mcp.research.value_momentum import (
    ValueMomentumScreen,
    format_value_momentum,
)


def _seed(tmp_path):
    """灌 4 只股票：A 便宜且走强 / B 便宜但走弱 / C 贵但走强 / D 双弱。"""
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "vm_test.duckdb")
    store = DuckDBStore(db)
    store.connect()
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": "000001", "name": "便宜走强", "sector": "银行",
         "market_cap": 800e8, "is_st": False},
        {"symbol": "000002", "name": "便宜走弱", "sector": "银行",
         "market_cap": 700e8, "is_st": False},
        {"symbol": "000003", "name": "昂贵走强", "sector": "科技",
         "market_cap": 300e8, "is_st": False},
        {"symbol": "000004", "name": "双弱", "sector": "制造",
         "market_cap": 60e8, "is_st": False},
    ]))
    # 财务：A/B 优质（ROE 15/12，净利正，增长正），C 亏损高增长，D 亏损
    rows = []
    for sym, roe, yoy, profit in (
            ("000001", 15.0, 15.0, 10e8),
            ("000002", 12.0, 10.0, 8e8),
            ("000003", -3.0, 50.0, -1e8),
            ("000004", -8.0, -20.0, -5e8)):
        for i in range(4):
            rows.append({"symbol": sym,
                         "report_date": pd.Timestamp(f"2025-{12-i*3:02d}-01"),
                         "profit_yoy": yoy, "revenue_yoy": yoy * 0.8,
                         "roe": roe, "net_profit_parent": profit,
                         "net_margin": 10.0, "eps": 1.0 if profit > 0 else -0.2})
    store.insert_df("financials_income", pd.DataFrame(rows))
    # 行情：A/C 上涨（动量强），B/D 下跌
    dates = pd.date_range("2025-06-01", periods=100, freq="B")
    for sym, trend in (("000001", 0.004), ("000002", -0.003),
                       ("000003", 0.005), ("000004", -0.005)):
        close = 10.0
        krows = []
        for i in range(100):
            close *= 1 + trend
            krows.append({"symbol": sym, "date": dates[i],
                          "open": close * 0.995, "high": close * 1.01,
                          "low": close * 0.99, "close": close,
                          "volume": 1e6, "amount": 1e7,
                          "pct_change": trend * 100, "turnover": 1.0,
                          "change": trend * 100, "amplitude": 2.0,
                          "adjust_factor": 1.0})
        store.insert_df("kline_daily", pd.DataFrame(krows))
    store.close()
    return db


def test_value_momentum_screen(tmp_path):
    db = _seed(tmp_path)
    quotes = {
        "000001": {"pe": 8.0, "pb": 0.9, "price": 10.0},
        "000002": {"pe": 9.0, "pb": 1.0, "price": 10.0},
        "000003": {"pe": 60.0, "pb": 8.0, "price": 50.0},
        "000004": {"pe": -5.0, "pb": 1.5, "price": 3.0},
    }
    r = ValueMomentumScreen(store_path=db).screen(
        ["000001", "000002", "000003", "000004"], quotes=quotes)
    assert "error" not in r
    assert r["n_scored"] == 4
    # 000001（便宜+走强）应居首
    assert r["top_symbols"][0] == "000001"
    top = {x["symbol"]: x for x in r["selection"]}
    assert top["000001"]["quadrant"] == "价值+动量双高"
    assert "000001" in r["best_candidates"]
    # 000002 便宜但走弱 → 价值高/动量弱
    assert top["000002"]["quadrant"] == "价值高/动量弱"
    # 000003 昂贵但走强 → 动量强/价值弱
    assert top["000003"]["quadrant"] == "动量强/价值弱"
    text = format_value_momentum(r)
    assert "价值 × 动量" in text and "双高" in text


def test_value_momentum_no_data(tmp_path):
    r = ValueMomentumScreen(store_path=str(tmp_path / "none.db")).screen(
        ["000001", "000002", "000003"])
    assert "error" in r
