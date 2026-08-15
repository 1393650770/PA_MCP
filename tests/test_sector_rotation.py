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


def test_predict_deterministic(tmp_path):
    db = _seed_sector_daily(tmp_path)
    analyzer = SectorRotationAnalyzer(store_path=db)

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
