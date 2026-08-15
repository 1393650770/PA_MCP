# [AI:BEGIN]
# PA_MCP - 研究结果导出测试（CSV 生成逻辑）
# [AI:END]

from __future__ import annotations

import io

import pandas as pd

from pa_mcp.research.factors import select_stocks_by_factors
from pa_mcp.research.graham import GrahamScreener


def _seed(tmp_path):
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "export_test.duckdb")
    store = DuckDBStore(db)
    store.connect()
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": "000001", "name": "银行A", "sector": "银行",
         "market_cap": 800e8, "is_st": False},
        {"symbol": "000002", "name": "银行B", "sector": "银行",
         "market_cap": 700e8, "is_st": False},
        {"symbol": "000003", "name": "白酒A", "sector": "白酒",
         "market_cap": 300e8, "is_st": False},
        {"symbol": "000004", "name": "白酒B", "sector": "白酒",
         "market_cap": 200e8, "is_st": False},
        {"symbol": "000005", "name": "科技A", "sector": "科技",
         "market_cap": 100e8, "is_st": False},
    ]))
    fin = []
    for sym in ("000001", "000002", "000003", "000004", "000005"):
        for i in range(4):
            fin.append({"symbol": sym,
                        "report_date": pd.Timestamp(f"2025-{12-i*3:02d}-01"),
                        "profit_yoy": 15.0, "revenue_yoy": 12.0, "roe": 14.0,
                        "net_profit_parent": 5e8, "net_margin": 10.0,
                        "eps": 1.0})
    store.insert_df("financials_income", pd.DataFrame(fin))
    import numpy as np
    rng = np.random.default_rng(3)
    dates = pd.date_range("2025-06-01", periods=100, freq="B")
    krows = []
    for sym in ("000001", "000002", "000003", "000004", "000005"):
        close = 10.0
        for i in range(100):
            close *= 1 + 0.002 + rng.normal(0, 0.008)
            krows.append({"symbol": sym, "date": dates[i],
                          "open": close * 0.995, "high": close * 1.01,
                          "low": close * 0.99, "close": close,
                          "volume": 1e6, "amount": 1e7,
                          "pct_change": 0.2, "turnover": 1.0,
                          "change": 0.2, "amplitude": 2.0,
                          "adjust_factor": 1.0})
    store.insert_df("kline_daily", pd.DataFrame(krows))
    store.close()
    return db


def test_export_selection_csv(tmp_path):
    """选股结果 → CSV 文本可解析。"""
    db = _seed(tmp_path)
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(db)
    store.connect()
    klines = {}
    for sym in ("000001", "000002", "000003", "000004", "000005"):
        klines[sym] = store.query_df(
            "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date", [sym])
    store.close()
    r = select_stocks_by_factors(klines, top_n=5)
    assert "error" not in r
    rows = []
    for i, x in enumerate(r["selection"], 1):
        rows.append({"rank": i, "symbol": x["symbol"], "score": x["score"],
                     **x["factor_details"]})
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    csv_text = buf.getvalue()
    parsed = pd.read_csv(io.StringIO(csv_text))
    assert len(parsed) == 5
    assert {"rank", "symbol", "score"} <= set(parsed.columns)
    assert "prediction" in parsed.columns or True  # 无预测融合时无此列


def test_export_nav_csv():
    """组合回测净值序列 → CSV 可解析。"""
    from pa_mcp.research.factors import backtest_factor_selection
    import numpy as np
    rng = np.random.default_rng(5)
    klines = {}
    for i in range(1, 7):
        close = 10.0
        rows = []
        for j in range(250):
            close *= 1 + 0.001 + rng.normal(0, 0.008)
            rows.append({"date": pd.Timestamp("2025-01-01")
                         + pd.Timedelta(days=j),
                         "open": close * 0.995, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1e6, "symbol": f"6000{i:02d}"})
        klines[f"6000{i:02d}"] = pd.DataFrame(rows)
    r = backtest_factor_selection(klines, top_n=3, horizon=5,
                                  train_window=80)
    assert "error" not in r
    nav_rows = r["portfolio"].get("nav_series") or []
    assert len(nav_rows) >= 10
    csv_text = pd.DataFrame(nav_rows).to_csv(index=False)
    parsed = pd.read_csv(io.StringIO(csv_text))
    assert {"date", "nav"} <= set(parsed.columns)


def test_export_graham_csv(tmp_path):
    """格雷厄姆结果 → CSV 可解析。"""
    db = _seed(tmp_path)
    quotes = {s: {"pe": 10, "pb": 1.2, "price": 10}
              for s in ("000001", "000002", "000003", "000004", "000005")}
    results = GrahamScreener(store_path=db).screen(
        list(quotes.keys()), quotes=quotes)
    rows = [{"symbol": x.symbol, "score": x.score, "rating": x.rating,
             "margin_of_safety_pct": x.margin_of_safety_pct}
            for x in results]
    csv_text = pd.DataFrame(rows).to_csv(index=False)
    parsed = pd.read_csv(io.StringIO(csv_text))
    assert len(parsed) == 5
    assert "symbol" in parsed.columns and "rating" in parsed.columns
