# [AI:BEGIN]
# PA_MCP - 持仓风险面板测试
# [AI:END]

from __future__ import annotations

import asyncio

import pandas as pd

from pa_mcp.research.portfolio_risk import (
    PortfolioRiskDashboard,
    format_risk_dashboard,
)


def _seed(tmp_path):
    """灌持仓 + 行情 + 板块（3 只持仓，2 只集中）。"""
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "risk_test.duckdb")
    store = DuckDBStore(db)
    store.connect()
    # portfolio 表（直接建：无 DDL 定义 → 手动 CREATE）
    store.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            cost DOUBLE NOT NULL,
            shares DOUBLE NOT NULL,
            added_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    store.insert_df("portfolio", pd.DataFrame([
        {"id": 1, "symbol": "000001", "cost": 10.0, "shares": 10000,
         "added_date": "2026-07-01"},
        {"id": 2, "symbol": "000002", "cost": 10.0, "shares": 5000,
         "added_date": "2026-07-01"},
        {"id": 3, "symbol": "000003", "cost": 10.0, "shares": 1000,
         "added_date": "2026-07-01"},
    ]))
    # 行情：000001 涨（11 元）、000002 涨（10.5）、000003 跌（9）
    dates = pd.date_range("2026-06-01", periods=40, freq="B")
    for sym, end_close in (("000001", 11.0), ("000002", 10.5), ("000003", 9.0)):
        closes = [10.0 + (end_close - 10.0) * i / 39 for i in range(40)]
        krows = []
        for i in range(40):
            krows.append({"symbol": sym, "date": dates[i],
                          "open": closes[i] * 0.995,
                          "high": closes[i] * 1.01,
                          "low": closes[i] * 0.99, "close": closes[i],
                          "volume": 1e6, "amount": 1e7,
                          "pct_change": 0.1, "turnover": 1.0,
                          "change": 0.1, "amplitude": 2.0,
                          "adjust_factor": 1.0})
        store.insert_df("kline_daily", pd.DataFrame(krows))
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": "000001", "name": "银行A", "sector": "银行", "is_st": False},
        {"symbol": "000002", "name": "银行B", "sector": "银行", "is_st": False},
        {"symbol": "000003", "name": "科技A", "sector": "科技", "is_st": False},
    ]))
    store.close()
    return db


def test_risk_dashboard(tmp_path):
    db = _seed(tmp_path)
    r = asyncio.run(PortfolioRiskDashboard(store_path=db).analyze())
    assert "error" not in r
    assert r["n_holdings"] == 3
    assert r["total_value"] > 0
    assert r["total_cost"] == 160000  # 16000 股 × 10 元
    # 000001 权重最高（110000/总值）
    top = max(r["holdings"], key=lambda h: h["value"])
    assert top["symbol"] == "000001"
    assert top["pnl_pct"] > 0
    # 集中度：HHI > 0.3（000001 占比 ~53%）
    assert r["concentration"]["hhi"] > 0.3
    assert r["concentration"]["top_weight_pct"] > 40
    # 行业：银行 ~77%
    assert r["concentration"]["sector_weights"]["银行"] > 60
    # 每只都有预测（确定性模式）
    assert all(h["prediction"] for h in r["holdings"])
    # 风险评分：集中度高 → 中/高风险
    assert r["risk_score"] >= 40
    assert r["risk_level"] in ("中", "高")
    assert any("集中" in n for n in r["risk_notes"])
    text = format_risk_dashboard(r)
    assert "持仓风险面板" in text and "风险评分" in text


def test_risk_dashboard_empty(tmp_path):
    r = asyncio.run(PortfolioRiskDashboard(
        store_path=str(tmp_path / "empty.duckdb")).analyze())
    assert "error" in r
