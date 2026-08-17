# [AI:BEGIN]
# PA_MCP - 板块轮动分析/预测/验证 测试（临时数据库）
# [AI:END]

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pa_mcp.research.sector_rotation import (
    SectorRotationAnalyzer,
    format_rotation,
)


def _seed_sector_daily(tmp_path):
    """灌 6 个板块 40 根日线：BK1/BK2 上涨+加速，BK3 上涨减速，BK4-6 下跌。"""
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(str(tmp_path / "sector_test.duckdb"))
    store.connect()
    dates = pd.date_range("2026-05-01", periods=40, freq="B")
    specs = {
        "BK0001": ("银行", 0.015, 0.01),    # 上涨且加速
        "BK0002": ("白酒", 0.012, 0.008),   # 上涨且加速
        "BK0003": ("光伏", 0.010, -0.005),  # 上涨但减速
        "BK0004": ("地产", -0.008, 0.0),    # 下跌
        "BK0005": ("钢铁", -0.005, 0.0),    # 下跌
        "BK0006": ("煤炭", -0.003, 0.0),    # 下跌
    }
    rows = []
    for code, (name, base_ret, accel) in specs.items():
        close = 100.0
        for i in range(40):
            ret = base_ret + accel * (i / 40)
            close *= 1 + ret
            rows.append({
                "sector_code": code, "name": name, "date": dates[i],
                "open": close * 0.99, "close": close, "high": close * 1.01,
                "low": close * 0.99, "volume": 1e7, "amount": 1e9,
                "pct_change": ret * 100, "turnover": 2.0,
            })
    store.insert_df("sector_daily", pd.DataFrame(rows))
    store.close()
    return store.db_path


def test_analyze_ranking(tmp_path):
    db = _seed_sector_daily(tmp_path)
    a = SectorRotationAnalyzer(store_path=db).analyze()
    assert "error" not in a
    assert a["board_count"] == 6
    top = a["ranked_sectors"][0]
    assert top["sector_code"] == "BK0001"  # 涨最快
    assert top["rs_pct"] > 0
    assert a["rotated_in"], "加速板块应标记轮入"
    assert a["rotated_out"], "减速板块应标记轮出"
    assert a["rotation_speed"] in ("高", "中", "低")


def test_predict_deterministic(tmp_path, monkeypatch):
    db = _seed_sector_daily(tmp_path)
    analyzer = SectorRotationAnalyzer(store_path=db)
    # 明确模拟未配置 LLM（ensure_llm_adapter 会主动读真实配置）
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)

    import asyncio
    pred = asyncio.run(analyzer.predict())
    assert pred["mode"] == "deterministic"
    assert len(pred["top_sectors_next_week"]) >= 2
    # 首推应为 BK0001 银行（RS 最高且加速）
    assert pred["top_sectors_next_week"][0]["sector"] == "银行"
    # 每个候选都带 sector_code（验证用）
    assert pred["top_sectors_next_week"][0]["sector_code"] == "BK0001"
    assert pred["sectors_to_avoid"]
    assert 0 <= pred["confidence"] <= 1


def test_save_and_evaluate(tmp_path):
    db = _seed_sector_daily(tmp_path)
    analyzer = SectorRotationAnalyzer(store_path=db)
    pred = {
        "mode": "deterministic",
        "predict_date": "2026-05-01",
        "top_sectors_next_week": [
            {"sector": "银行", "sector_code": "BK0001", "probability": 0.7},
            {"sector": "白酒", "sector_code": "BK0002", "probability": 0.6},
            {"sector": "光伏", "sector_code": "BK0003", "probability": 0.5},
        ],
        "rotation_logic": "动量延续",
        "sectors_to_avoid": [], "confidence": 0.6,
        "key_drivers": [], "risks": [],
    }
    pid = analyzer.save_prediction(pred)
    assert pid > 0

    # 预测日 05-01 之后 5 个交易日的板块收益：
    # BK0001 继续涨 → top3 平均 > 全板块平均（BK4-6 下跌）
    summary = analyzer.evaluate_predictions(days_forward=5)
    assert summary["evaluated"] == 1
    assert summary["avg_excess_pct"] > 0
    assert summary["hit_rate"] == 1.0


def _seed_stock_basic(tmp_path, db):
    """灌 stock_basic 板块映射 + 每板块 2 只股票行情。"""
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(db)
    store.connect()
    dates = pd.date_range("2026-05-01", periods=80, freq="B")
    specs = {
        ("000001", "平安银行", "银行"): 0.01,
        ("000002", "招商银行", "银行"): 0.005,
        ("600519", "贵州茅台", "白酒"): 0.008,
        ("000858", "五粮液", "白酒"): -0.002,
        ("601012", "隆基绿能", "光伏"): -0.008,
    }
    sb_rows = [{"symbol": s, "name": n, "sector": sec, "industry": sec,
                "is_st": False} for s, n, sec in specs]
    store.insert_df("stock_basic", pd.DataFrame(sb_rows))
    k_rows = []
    for (sym, _, _), daily in specs.items():
        close = 10.0
        for i in range(80):
            close *= 1 + daily
            k_rows.append({
                "symbol": sym, "date": dates[i], "open": close * 0.99,
                "close": close, "high": close * 1.01, "low": close * 0.99,
                "volume": 1e6, "amount": 1e7, "pct_change": daily * 100,
                "turnover": 1.0, "change": 0.1, "amplitude": 2.0,
                "adjust_factor": 1.0,
            })
    store.insert_df("kline_daily", pd.DataFrame(k_rows))
    store.close()


def test_leaders_in_sector(tmp_path):
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "leaders_test.duckdb")
    _seed_stock_basic(tmp_path, db)
    analyzer = SectorRotationAnalyzer(store_path=db)
    r = analyzer.leaders_in_sector("银行", top_n=5)
    assert "error" not in r
    assert r["leader_count"] == 2
    assert r["leaders"][0]["symbol"] == "000001"  # 平安银行涨更快
    assert r["leaders"][0]["rs60_pct"] > r["leaders"][1]["rs60_pct"]
    # 上涨 60 日 → 接近新高
    assert r["leaders"][0]["near_60d_high"] is True


def test_leaders_in_sector_no_mapping(tmp_path):
    db = str(tmp_path / "leaders_empty.duckdb")
    analyzer = SectorRotationAnalyzer(store_path=db)
    r = analyzer.leaders_in_sector("不存在板块")
    assert "error" in r


def test_sector_context_injection(tmp_path):
    """预测 prompt 板块上下文：_sector_context 返回板块 RS 文本。"""
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "ctx_test.duckdb")
    _seed_stock_basic(tmp_path, db)
    # 板块日线：银行上涨（模拟）
    store = DuckDBStore(db)
    store.connect()
    dates = pd.date_range("2026-05-01", periods=40, freq="B")
    close = 100.0
    rows = []
    for i in range(40):
        close *= 1.008
        rows.append({"sector_code": "BK0001", "name": "银行",
                     "date": dates[i], "open": close * 0.99, "close": close,
                     "high": close * 1.01, "low": close * 0.99,
                     "volume": 1e7, "amount": 1e9, "pct_change": 0.8,
                     "turnover": 2.0})
    store.insert_df("sector_daily", pd.DataFrame(rows))
    store.close()

    from pa_mcp.agent.prediction import PredictionService
    svc = PredictionService(store_path=db)
    ctx = svc._sector_context("000001")
    assert "银行" in ctx
    assert "强势" in ctx  # 板块上涨且加速→强势
    # 无映射股票 → 空串
    assert svc._sector_context("999999") == ""


def test_synthetic_sector_fallback(tmp_path):
    """东财不可用 → 合成板块降级（stock_basic.sector + kline 聚合）。"""
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "syn_test.duckdb")
    store = DuckDBStore(db)
    store.connect()
    # 2 板块 × 2 股票（kline 100 根 + sector 映射）
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": "000001", "name": "银行A", "sector": "银行", "is_st": False},
        {"symbol": "000002", "name": "银行B", "sector": "银行", "is_st": False},
        {"symbol": "000003", "name": "白酒A", "sector": "白酒", "is_st": False},
        {"symbol": "000004", "name": "白酒B", "sector": "白酒", "is_st": False},
    ]))
    dates = pd.date_range("2026-05-01", periods=100, freq="B")
    krows = []
    for sym, base in (("000001", 0.004), ("000002", 0.003),
                      ("000003", 0.002), ("000004", -0.001)):
        close = 10.0
        for i in range(100):
            close *= 1 + base
            krows.append({"symbol": sym, "date": dates[i],
                          "open": close * 0.995, "high": close * 1.01,
                          "low": close * 0.99, "close": close,
                          "volume": 1e6, "amount": 1e7,
                          "pct_change": base * 100, "turnover": 1.0,
                          "change": base * 100, "amplitude": 2.0,
                          "adjust_factor": 1.0})
    store.insert_df("kline_daily", pd.DataFrame(krows))
    store.close()

    a = SectorRotationAnalyzer(store_path=db).analyze()
    assert "error" not in a
    assert a["synthetic"] is True
    assert "合成板块" in a["data_source"]
    assert a["board_count"] == 2
    assert a["ranked_sectors"][0]["sector_code"] == "SYN_银行"


def test_format_rotation():
    pred = {
        "mode": "deterministic", "predict_date": "2026-08-15",
        "top_sectors_next_week": [
            {"sector": "银行", "sector_code": "BK0001", "probability": 0.7,
             "reason": "RS 居前"}] ,
        "rotation_logic": "强者恒强",
        "sectors_to_avoid": [{"sector": "地产", "reason": "减速"}],
        "confidence": 0.6, "key_drivers": ["资金流"], "risks": ["回调"],
        "analysis": {"rotation_speed": "高", "rotated_in": [], "rotated_out": []},
    }
    text = format_rotation(pred)
    assert "板块轮动" in text and "银行" in text and "强者恒强" in text


# ---- 热门/冷门板块（新浪实时榜） ----

async def test_hot_cold_sectors(monkeypatch):
    """热门/冷门排序与渲染。"""
    from pa_mcp.research.sector_rotation import SectorRotationAnalyzer

    async def _fake_boards(self):
        return [
            {"code": "new_a", "name": "板块A", "change_pct": 3.5,
             "amount": 1e9, "leader_code": "sh600001", "leader_name": "龙头A"},
            {"code": "new_b", "name": "板块B", "change_pct": 1.2,
             "amount": 2e9, "leader_code": "sh600002", "leader_name": "龙头B"},
            {"code": "new_c", "name": "板块C", "change_pct": -2.1,
             "amount": 3e9, "leader_code": "sh600003", "leader_name": "龙头C"},
        ]

    monkeypatch.setattr(SectorRotationAnalyzer, "fetch_sina_boards",
                        _fake_boards)
    r = await SectorRotationAnalyzer().hot_cold_sectors(top_n=2)
    assert [b["name"] for b in r["hot"]] == ["板块A", "板块B"]
    assert [b["name"] for b in r["cold"]] == ["板块C", "板块B"]  # 涨幅最低 2 个
    assert "热门板块" in r["report"] and "冷门板块" in r["report"]
    assert "+3.50%" in r["report"] and "-2.10%" in r["report"]
    assert "龙头A" in r["report"]


async def test_hot_cold_fallback_failure(monkeypatch):
    """新浪接口全失败 → error 降级（不抛异常）。"""
    from pa_mcp.research.sector_rotation import SectorRotationAnalyzer

    async def _no_boards(self):
        return []

    monkeypatch.setattr(SectorRotationAnalyzer, "fetch_sina_boards",
                        _no_boards)
    r = await SectorRotationAnalyzer().hot_cold_sectors()
    assert "error" in r
    assert "新浪板块接口不可用" in r["error"]
