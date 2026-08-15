# [AI:BEGIN]
# PA_MCP - 游资情绪周期测试（临时数据库）
# [AI:END]

from __future__ import annotations

import pandas as pd

from pa_mcp.research.sentiment_cycle import (
    SentimentCycleAnalyzer,
    format_sentiment,
)


def _seed_market(tmp_path, day_specs):
    """灌多日行情。

    day_specs: {date: {symbol: pct_change}} 每日每只股票涨跌幅。
    """
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(str(tmp_path / "sentiment_test.duckdb"))
    store.connect()
    rows = []
    for day, specs in day_specs.items():
        for sym, pct in specs.items():
            base = 10.0
            rows.append({
                "symbol": sym, "date": pd.Timestamp(day),
                "open": base, "close": base * (1 + pct / 100),
                "high": base * 1.05, "low": base * 0.95,
                "volume": 1e6, "amount": 1e7, "pct_change": pct,
                "turnover": 1.0, "change": pct, "amplitude": 5.0,
                "adjust_factor": 1.0,
            })
    store.insert_df("kline_daily", pd.DataFrame(rows))
    store.close()
    return store.db_path


def test_fermenting_stage(tmp_path):
    """发酵期：连板高度 3+、晋级率高、涨停 ≥30。"""
    db = _seed_market(tmp_path, {
        "2026-08-03": {"600001": 10.0, "600002": 10.0, "600003": 10.0},
        # 8-04：3 只二连板 + 30 只首板 → 晋级率 100%
        "2026-08-04": {**{f"6000{i:02d}": 10.0 for i in range(4, 34)},
                       "600001": 10.0, "600002": 10.0, "600003": 10.0},
        # 8-05：龙头三连板 + 晋级
        "2026-08-05": {**{f"6000{i:02d}": 10.0 for i in range(4, 34)},
                       "600001": 10.0, "600002": 10.0, "600003": 10.0},
    })
    a = SentimentCycleAnalyzer(store_path=db).analyze("2026-08-05")
    assert "error" not in a
    assert a["max_board_height"] == 3
    assert a["limit_up_count"] >= 30
    assert a["promotion_rate"] is not None
    assert a["stage"] in ("fermenting", "climax", "starting")
    assert a["sentiment_score"] >= 40
    text = format_sentiment(a)
    assert "情绪周期" in text and "梯队" in text


def test_recess_stage(tmp_path):
    """退潮期：晋级率骤降（昨日涨停今日大多断板）。"""
    db = _seed_market(tmp_path, {
        "2026-08-03": {**{f"6000{i:02d}": 10.0 for i in range(1, 31)}},
        # 8-04：30 只首板晋级到 2 板（全晋级）→ 高度 2
        "2026-08-04": {**{f"6000{i:02d}": 10.0 for i in range(1, 31)}},
        # 8-05：高度 3（3 只晋级）+ 大量断板（其余跌）
        "2026-08-05": {"600001": 10.0, "600002": 10.0, "600003": 10.0,
                       **{f"6000{i:02d}": -8.0 for i in range(4, 31)}},
    })
    a = SentimentCycleAnalyzer(store_path=db).analyze("2026-08-05")
    # 昨日涨停 30 只 → 今日仅 3 只晋级 → 晋级率 10% < 25% → 退潮
    assert a["promotion_rate"] is not None and a["promotion_rate"] < 0.25
    assert a["stage"] == "recess"
    assert a["warnings"], "退潮应产生预警"


def test_ice_stage(tmp_path):
    """冰点期：涨停不足 + 跌停潮。"""
    db = _seed_market(tmp_path, {
        "2026-08-03": {**{f"6000{i:02d}": -10.0 for i in range(1, 31)},
                       "600100": 5.0},
    })
    a = SentimentCycleAnalyzer(store_path=db).analyze("2026-08-03")
    assert a["stage"] == "ice"
    assert a["limit_up_count"] < 10


def test_climax_stage(tmp_path):
    """高潮期：连板高度 ≥5 且涨停 ≥50。"""
    db = _seed_market(tmp_path, {
        "2026-08-03": {**{f"6000{i:02d}": 10.0 for i in range(1, 51)}},
        "2026-08-04": {**{f"6000{i:02d}": 10.0 for i in range(1, 51)}},
        "2026-08-05": {**{f"6000{i:02d}": 10.0 for i in range(1, 51)}},
        "2026-08-06": {**{f"6000{i:02d}": 10.0 for i in range(1, 51)}},
        "2026-08-07": {**{f"6000{i:02d}": 10.0 for i in range(1, 51)}},
    })
    a = SentimentCycleAnalyzer(store_path=db).analyze("2026-08-07")
    assert a["max_board_height"] == 5
    assert a["limit_up_count"] >= 50
    assert a["stage"] == "climax"
