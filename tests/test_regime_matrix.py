# [AI:BEGIN]
# PA_MCP - 情绪×轮动矩阵测试
# [AI:END]

from __future__ import annotations

import pandas as pd

from pa_mcp.research.regime_matrix import (
    REGIME_MATRIX,
    RegimeMatrixAnalyzer,
    format_matrix,
)


def _seed_sentiment(tmp_path, stage_spec):
    """灌情绪数据（直接写 sentiment_daily，绕开连板计算）。"""
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(str(tmp_path / "matrix_test.duckdb"))
    store.connect()
    store.insert_df("sentiment_daily", pd.DataFrame([{
        "date": stage_spec["date"], "limit_up_count": stage_spec.get(
            "limit_up", 30), "limit_down_count": stage_spec.get("limit_down", 3),
        "max_board_height": stage_spec.get("max_h", 3),
        "board2_count": 5, "board3_count": 2, "board4p_count": 1,
        "first_board_count": 20,
        "promotion_rate": stage_spec.get("promotion_rate", 0.5),
        "sentiment_score": stage_spec.get("score", 60), "stage": "",
    }]))
    store.close()


def _seed_rotation(tmp_path, speed):
    """灌板块数据（6 板块 40 根日线，按 speed 构造加速/减速/恒定）。"""
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(str(tmp_path / "matrix_test.duckdb"))
    store.connect()
    dates = pd.date_range("2026-05-01", periods=40, freq="B")
    # 收益曲线：high=前慢后快（加速），low=前快后慢（减速），medium=恒定
    if speed == "high":
        rets = {f"BK{i:04d}": (0.002, 0.012) for i in range(1, 7)}
    elif speed == "low":
        rets = {f"BK{i:04d}": (0.012, 0.002) for i in range(1, 7)}
    else:
        rets = {f"BK{i:04d}": (0.006, 0.006) for i in range(1, 7)}
    names = {f"BK{i:04d}": f"板块{i}" for i in range(1, 7)}
    rows = []
    for code, (slow, fast) in rets.items():
        close = 100.0
        for i in range(40):
            # 加速须在窗口尾部（最近 5 日）才产生正 accel：前 35 日 slow
            ret = slow if i < 35 else fast
            close *= 1 + ret
            rows.append({"sector_code": code, "name": names[code],
                         "date": dates[i], "open": close * 0.99,
                         "close": close, "high": close * 1.01,
                         "low": close * 0.99, "volume": 1e7,
                         "amount": 1e9, "pct_change": ret * 100,
                         "turnover": 2.0})
    store.insert_df("sector_daily", pd.DataFrame(rows))
    store.close()


def test_matrix_fermenting_high_speed(tmp_path):
    """发酵期 + 高轮动 → 亢奋期。"""
    _seed_sentiment(tmp_path, {"date": "2026-08-05", "limit_up": 45, "max_h": 3})
    _seed_rotation(tmp_path, "high")
    r = RegimeMatrixAnalyzer(store_path=str(tmp_path / "matrix_test.duckdb")).analyze()
    assert r["sentiment_stage"] == "fermenting"
    assert r["rotation_speed"] == "高"
    assert r["regime_label"] == "亢奋期"
    assert "控制追高" in r["advice"]
    text = format_matrix(r)
    assert "情绪 × 轮动" in text and "亢奋期" in text


def test_matrix_recess_low_speed(tmp_path):
    """退潮期 + 低轮动 → 收缩期（空仓）。"""
    # 退潮判定：晋级率 10% < 25% 且连板高度 ≥3
    _seed_sentiment(tmp_path, {"date": "2026-08-05", "limit_up": 12,
                               "limit_down": 18, "max_h": 3, "score": 25,
                               "promotion_rate": 0.1})
    _seed_rotation(tmp_path, "low")
    r = RegimeMatrixAnalyzer(store_path=str(tmp_path / "matrix_test.duckdb")).analyze()
    assert r["sentiment_stage"] == "recess"
    assert r["regime_label"] == "收缩期"
    assert "空仓" in r["advice"]


def test_matrix_ice_any_speed(tmp_path):
    """冰点期 + 中轮动 → 冰点（空仓等待）。"""
    _seed_sentiment(tmp_path, {"date": "2026-08-05", "limit_up": 5,
                               "limit_down": 30, "max_h": 1, "score": 10})
    _seed_rotation(tmp_path, "medium")
    r = RegimeMatrixAnalyzer(store_path=str(tmp_path / "matrix_test.duckdb")).analyze()
    assert r["regime_label"] == "冰点"


def test_matrix_missing_data(tmp_path, monkeypatch):
    """无数据 + 实时也失败 → 降级结论（数据不足）。"""
    from pa_mcp.research import sentiment_cycle as sc
    async def _no_realtime():
        return None
    monkeypatch.setattr(sc.SentimentCycleAnalyzer,
                        "_fetch_realtime_stats", _no_realtime)
    r = RegimeMatrixAnalyzer(store_path=str(tmp_path / "none.db")).analyze()
    assert r["sentiment_stage"] == "unknown"
    assert r["rotation_speed"] == "unknown"
    assert "数据不足" in r["advice"]


def test_matrix_full_coverage():
    """矩阵 9 格齐全（5 阶段 × 3 速度）。"""
    for stage, speeds in REGIME_MATRIX.items():
        for sp in ("low", "medium", "high"):
            label, advice, risk = speeds[sp]
            assert label and advice and risk
