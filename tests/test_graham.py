# [AI:BEGIN]
# PA_MCP - 格雷厄姆价值筛选测试
# [AI:END]

from __future__ import annotations

import pandas as pd

from pa_mcp.research.graham import (
    GrahamScreener,
    format_graham,
)


def _seed(tmp_path):
    """灌 4 只股票：一只低估值优质（银行），一只高估值（科技），
    一只亏损（亏损股），一只无财务（新股）。"""
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "graham_test.duckdb")
    store = DuckDBStore(db)
    store.connect()
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": "000001", "name": "优质银行", "sector": "银行",
         "market_cap": 800e8, "is_st": False},
        {"symbol": "000002", "name": "高估值科技", "sector": "科技",
         "market_cap": 300e8, "is_st": False},
        {"symbol": "000003", "name": "亏损股", "sector": "制造",
         "market_cap": 60e8, "is_st": False},
        {"symbol": "000004", "name": "无财务", "sector": "新股",
         "market_cap": 10e8, "is_st": False},
    ]))
    # 财务：000001 优质（ROE 15%，净利全正，同比 +20%），000002 高 ROE 但
    # 高增速，000003 亏损
    rows = []
    for sym, roe, yoy, profit in (
            ("000001", 15.0, 20.0, 10e8),
            ("000002", 18.0, 40.0, 5e8),
            ("000003", -5.0, -30.0, -2e8)):
        for i in range(4):
            rows.append({"symbol": sym,
                         "report_date": pd.Timestamp(f"2025-{12-i*3:02d}-01"),
                         "profit_yoy": yoy, "revenue_yoy": yoy * 0.8,
                         "roe": roe, "net_profit_parent": profit,
                         "net_margin": 10.0, "eps": 1.0 if sym != "000003" else -0.2})
    store.insert_df("financials_income", pd.DataFrame(rows))
    store.close()
    return db


def test_graham_screen(tmp_path):
    db = _seed(tmp_path)
    # 传入 quote 模拟估值（避免网络依赖）
    quotes = {
        "000001": {"pe": 8.0, "pb": 0.9, "price": 10.0},
        "000002": {"pe": 40.0, "pb": 5.0, "price": 50.0},
        "000003": {"pe": -10.0, "pb": 1.0, "price": 3.0},
        "000004": {"pe": 25.0, "pb": 2.0, "price": 8.0},
    }
    results = GrahamScreener(store_path=db).screen(
        ["000001", "000002", "000003", "000004"], quotes=quotes)
    assert len(results) == 4
    by_sym = {r.symbol: r for r in results}
    r1 = by_sym["000001"]
    # 优质银行：规模 ✓ ROE ✓ 盈利稳 ✓ 增长 ✓ PE ✓ PB ✓ → ≥5 条
    assert r1.score >= 5
    assert r1.rating in ("价值", "深度价值", "符合")
    # 内在价值 = 1.0 × (8.5 + 2×20) = 48.5 → 安全边际 vs 10 元 = +385%
    assert r1.intrinsic_value is not None
    assert r1.margin_of_safety_pct > 50
    # 高估值科技：PE/PB 不过
    assert by_sym["000002"].score <= 4
    # 亏损股：盈利稳定 ✗ 增长 ✗ → 低分
    assert by_sym["000003"].score <= 3
    # 无财务：只能评规模/估值 → 低分
    assert by_sym["000004"].total_scored <= 3
    # 排序：优质银行第一
    assert results[0].symbol == "000001"
    text = format_graham(results)
    assert "格雷厄姆" in text and "安全边际" in text


def test_graham_no_data(tmp_path):
    r = GrahamScreener(store_path=str(tmp_path / "empty.duckdb")).screen(
        ["000001", "000002", "000003"], quotes={
            "000001": {"pe": 10, "pb": 1, "price": 5},
            "000002": {"pe": 10, "pb": 1, "price": 5},
            "000003": {"pe": 10, "pb": 1, "price": 5}})
    # 无财务 → 各标准 unavailable，但可评分标准仍能跑
    assert r
    assert all(x.total_scored >= 2 for x in r)


def test_graham_intrinsic_value_formula():
    """成长公式：V = EPS × (8.5 + 2g)。"""
    import tempfile, os
    from pa_mcp.data.store import DuckDBStore
    db = os.path.join(tempfile.mkdtemp(), "g2.duckdb")
    store = DuckDBStore(db)
    store.connect()
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": "000001", "name": "x", "sector": "s",
         "market_cap": 100e8, "is_st": False}]))
    store.insert_df("financials_income", pd.DataFrame([
        {"symbol": "000001", "report_date": pd.Timestamp("2025-03-31"),
         "profit_yoy": 15.0, "revenue_yoy": 10.0, "roe": 12.0,
         "net_profit_parent": 1e8, "net_margin": 10.0, "eps": 2.0}]))
    store.close()
    r = GrahamScreener(store_path=db).screen(
        ["000001"], quotes={"000001": {"pe": 12, "pb": 1.2, "price": 20}})[0]
    # V = 2.0 × (8.5 + 2×15) = 2.0 × 38.5 = 77
    assert r.intrinsic_value == 77.0
    # 安全边际 = (77-20)/77 = 74%
    assert abs(r.margin_of_safety_pct - 74.0) < 1.5
