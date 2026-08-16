# [AI:BEGIN]
# PA_MCP - 一站式分析（流水线整合报告）测试
# [AI:END]

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from pa_mcp.research.one_click import one_click_report


def _seed(tmp_path):
    """临时库：6 股票 × 150 根 + 财务 + 板块 + 指数 + 情绪 + 持仓。"""
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "oneclick.duckdb")
    store = DuckDBStore(db)
    store.connect()
    rng = np.random.default_rng(9)
    dates = pd.date_range("2026-01-01", periods=150, freq="B")
    symbols = [f"00000{i}" for i in range(1, 7)]
    krows = []
    for sym in symbols:
        close = 10.0
        for i in range(150):
            close *= 1 + 0.001 + rng.normal(0, 0.01)
            krows.append({"symbol": sym, "date": dates[i],
                          "open": close * 0.995, "high": close * 1.01,
                          "low": close * 0.99, "close": close,
                          "volume": 1e6, "amount": 1e7, "pct_change": 0.2,
                          "turnover": 1.0, "change": 0.2, "amplitude": 1.0,
                          "adjust_factor": 1.0})
    store.insert_df("kline_daily", pd.DataFrame(krows))
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": s, "name": f"股{s}", "sector": "银行" if i < 3 else "科技",
         "market_cap": 500e8, "is_st": False}
        for i, s in enumerate(symbols)]))
    fin = []
    for sym in symbols:
        for i in range(4):
            fin.append({"symbol": sym,
                        "report_date": pd.Timestamp(f"2025-{12-i*3:02d}-01"),
                        "profit_yoy": 15.0, "revenue_yoy": 12.0, "roe": 14.0,
                        "net_profit_parent": 5e8, "net_margin": 10.0,
                        "eps": 1.0})
    store.insert_df("financials_income", pd.DataFrame(fin))
    idx = []
    close = 3000.0
    for i in range(150):
        close *= 1 + 0.0005 + rng.normal(0, 0.004)
        idx.append({"symbol": "sh000001", "date": dates[i],
                    "open": close * 0.995, "high": close * 1.01,
                    "low": close * 0.99, "close": close, "volume": 1e8,
                    "amount": 1e9})
    store.insert_df("index_daily", pd.DataFrame(idx))
    srows = []
    for code, name, ret in (("BK0001", "银行", 0.003), ("BK0002", "科技", 0.002)):
        close = 100.0
        for i in range(120):
            close *= 1 + ret
            srows.append({"sector_code": code, "name": name,
                          "date": dates[i + 30], "open": close * 0.99,
                          "close": close, "high": close * 1.01,
                          "low": close * 0.99, "volume": 1e7,
                          "amount": 1e9, "pct_change": ret * 100,
                          "turnover": 2.0})
    store.insert_df("sector_daily", pd.DataFrame(srows))
    store.insert_df("sentiment_daily", pd.DataFrame([{
        "date": str(dates[-1])[:10], "limit_up_count": 30,
        "limit_down_count": 3, "max_board_height": 3, "board2_count": 5,
        "board3_count": 2, "board4p_count": 1, "first_board_count": 20,
        "promotion_rate": 0.5, "sentiment_score": 60, "stage": ""}]))
    store.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY,
            symbol VARCHAR(10), cost DOUBLE, shares INTEGER,
            added_date DATE, created_at TIMESTAMP
        )
    """)
    store.insert_df("portfolio", pd.DataFrame([
        {"id": 1, "symbol": "000001", "cost": 10.0, "shares": 10000,
         "added_date": "2026-07-01", "created_at": "2026-07-01"}]))
    store.close()
    return db


def test_one_click_report_structure(tmp_path, monkeypatch):
    """流水线整合：各阶段段齐全 + 报告可读。"""
    db = _seed(tmp_path)
    import pa_mcp.config as cfg
    real = cfg.get_settings()
    d = real.model_dump()
    d["database"] = {"path": db, "read_only": False}
    monkeypatch.setattr(cfg, "_settings", type(real)(**d))

    r = asyncio.run(one_click_report(
        ["000001", "000002", "000003", "000004", "000005", "000006"]))
    titles = [s["title"] for s in r["sections"]]
    assert "🩺 数据体检" in titles
    assert "🏛️ 市场结构" in titles
    assert "🧭 情绪矩阵" in titles
    assert "🔄 板块轮动" in titles
    assert "🎯 因子选股" in titles
    assert "⚖️ 价值动量" in titles
    assert "📗 格雷厄姆" in titles
    assert "🧮 综合信号" in titles
    assert "🛡️ 持仓风险" in titles
    assert "一站式分析报告" in r["report"]
    # 每段有内容
    assert all(s["body"] for s in r["sections"])


def test_one_click_no_data(tmp_path, monkeypatch):
    """空库 → 各段降级（不可用标注），不崩溃。"""
    import pa_mcp.config as cfg
    real = cfg.get_settings()
    d = real.model_dump()
    d["database"] = {"path": str(tmp_path / "none.duckdb"),
                     "read_only": False}
    monkeypatch.setattr(cfg, "_settings", type(real)(**d))

    r = asyncio.run(one_click_report(["000001"]))
    assert "一站式分析报告" in r["report"]
    # 报告仍生成（各段降级但整体不崩；选股段无数据可跳过）
    assert len(r["sections"]) >= 6
