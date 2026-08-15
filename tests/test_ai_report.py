# [AI:BEGIN]
# PA_MCP - AI 市场研究报告测试
# [AI:END]

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from pa_mcp.research.ai_report import AiMarketReport


def _seed(tmp_path):
    """灌最小数据集：行情 + 财务 + 板块（供各聚合段取数）。"""
    import numpy as np
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "report_test.duckdb")
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
        {"symbol": "000006", "name": "科技B", "sector": "科技",
         "market_cap": 80e8, "is_st": False},
    ]))
    fin = []
    for sym in ("000001", "000002", "000003", "000004", "000005", "000006"):
        for i in range(4):
            fin.append({"symbol": sym,
                        "report_date": pd.Timestamp(f"2025-{12-i*3:02d}-01"),
                        "profit_yoy": 15.0, "revenue_yoy": 12.0, "roe": 14.0,
                        "net_profit_parent": 5e8, "net_margin": 10.0,
                        "eps": 1.0})
    store.insert_df("financials_income", pd.DataFrame(fin))
    # 行情：6 只上涨趋势
    rng = np.random.default_rng(3)
    dates = pd.date_range("2025-06-01", periods=150, freq="B")
    krows = []
    for sym in ("000001", "000002", "000003", "000004", "000005", "000006"):
        close = 10.0
        for i in range(150):
            close *= 1 + 0.002 + rng.normal(0, 0.008)
            krows.append({"symbol": sym, "date": dates[i],
                          "open": close * 0.995, "high": close * 1.01,
                          "low": close * 0.99, "close": close,
                          "volume": 1e6, "amount": 1e7,
                          "pct_change": 0.2, "turnover": 1.0,
                          "change": 0.2, "amplitude": 2.0,
                          "adjust_factor": 1.0})
    store.insert_df("kline_daily", pd.DataFrame(krows))
    # 板块日线（轮动段）
    srows = []
    for code, name, ret in (("BK0001", "银行", 0.004), ("BK0002", "白酒", 0.003),
                            ("BK0003", "科技", 0.002)):
        close = 100.0
        for i in range(40):
            close *= 1 + ret
            srows.append({"sector_code": code, "name": name,
                          "date": dates[i + 110], "open": close * 0.99,
                          "close": close, "high": close * 1.01,
                          "low": close * 0.99, "volume": 1e7,
                          "amount": 1e9, "pct_change": ret * 100,
                          "turnover": 2.0})
    store.insert_df("sector_daily", pd.DataFrame(srows))
    store.close()
    return db


def test_report_template_fallback(tmp_path):
    """无 LLM：确定性模板降级，各段齐全。"""
    db = _seed(tmp_path)
    gen = AiMarketReport(store_path=db)
    result = asyncio.run(gen.generate(
        ["000001", "000002", "000003", "000004", "000005", "000006"]))
    assert result["llm_analysis"] is None
    s = result["sections"]
    assert "market_state" in s and "matrix" in s
    assert "factor_selection" in s and "value_momentum" in s
    assert "prediction_eval" in s
    assert "AI 市场研究报告" in result["report"]


def test_report_with_llm_mock(tmp_path):
    """有 LLM（mock）：LLM 综述并入报告。"""
    db = _seed(tmp_path)

    class MockAdapter:
        provider_name = "mock"

        async def chat_json(self, params):
            return {
                "summary": "市场处于发酵期，银行板块强势，建议关注低估值银行。",
                "focus_areas": [{"area": "银行", "reason": "RS 居前且估值低"}],
                "risks": [{"risk": "板块集中", "reason": "强势板块过于集中"}],
                "suggestions": ["关注因子选股 top 标的"],
                "confidence": 0.7,
            }

    from pa_mcp.agent import llm_port
    orig = llm_port._adapter
    llm_port.register_adapter(MockAdapter())
    try:
        gen = AiMarketReport(store_path=db)
        result = asyncio.run(gen.generate(
            ["000001", "000002", "000003", "000004", "000005", "000006"]))
        assert result["llm_analysis"] is not None
        assert "发酵期" in result["llm_analysis"]["summary"]
        assert "🤖 AI 综述" in result["report"]
        assert "银行" in result["report"]
        assert "置信度" in result["report"]
    finally:
        llm_port.register_adapter(orig)


def test_report_llm_fails_fallback(tmp_path):
    """LLM 返回错误 → 降级模板（不崩）。"""
    db = _seed(tmp_path)

    class BadAdapter:
        provider_name = "bad"

        async def chat_json(self, params):
            raise RuntimeError("rate limited")

    from pa_mcp.agent import llm_port
    orig = llm_port._adapter
    llm_port.register_adapter(BadAdapter())
    try:
        gen = AiMarketReport(store_path=db)
        result = asyncio.run(gen.generate(["000001", "000002", "000003",
                                           "000004", "000005", "000006"]))
        assert result["llm_analysis"] is None
        assert "AI 市场研究报告" in result["report"]
    finally:
        llm_port.register_adapter(orig)
