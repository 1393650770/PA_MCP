# [AI:BEGIN]
# PA_MCP - 经验库（RAG）测试（临时数据库）
# [AI:END]

from __future__ import annotations

import pandas as pd

from pa_mcp.agent.experience import ExperienceEntry, ExperienceService


def test_save_and_search(tmp_path):
    db = tmp_path / "exp_test.duckdb"
    svc = ExperienceService(store_path=str(db))

    id1 = svc.save_analysis("000001", {
        "direction": "bullish", "overall_strength_score": 72,
        "dimension_scores": {"technical": 80, "capital": 60},
        "key_risks": ["大盘调整"],
        "key_evidence": [{"dimension": "technical", "finding": "均线多头", "impact": "positive"}],
        "disclaimer": "研究参考",
    })
    id2 = svc.save_analysis("000001", {
        "direction": "bearish", "overall_strength_score": 30,
        "dimension_scores": {"technical": 20},
        "key_risks": ["破位"],
    })
    assert id2 > id1

    entries = svc.search_experience(symbol="000001", limit=5)
    assert len(entries) == 2
    # 时间倒序：最新在前
    assert entries[0].direction == "bearish"
    assert entries[0].strength_score == 30.0

    # 方向过滤
    bullish = svc.search_experience(symbol="000001", direction="bullish", limit=5)
    assert len(bullish) == 1 and bullish[0].direction == "bullish"

    # 无匹配
    assert svc.search_experience(symbol="600519", limit=5) == []


def test_format_experience_empty():
    svc = ExperienceService()
    assert svc.format_experience([]) == ""


def test_format_experience_text():
    svc = ExperienceService()
    entry = ExperienceEntry(
        symbol="000001", analyzed_date="2026-08-10",
        cycle_position="normal_channel", direction="bullish",
        strength_score=72.0, conclusion="多头格局，回踩可关注",
        dimension_scores={"technical": 80}, key_risks=["大盘调整"],
        outcome="hit", actual_return_5d=3.2)
    text = svc.format_experience([entry])
    assert "000001" in text and "正常通道" not in text or "normal_channel" in text
    assert "hit" in text and "3.2" in text
    assert "历史参考案例" in text


def test_evaluate_history_roundtrip(tmp_path):
    db = tmp_path / "exp_test2.duckdb"
    svc = ExperienceService(store_path=str(db))
    svc.save_analysis("000001", {"direction": "bullish", "overall_strength_score": 70})
    svc.save_analysis("000001", {"direction": "bearish", "overall_strength_score": 30})

    # 模拟行情：08-01 后 5 日上涨 4%
    closes = [10.0 + 0.08 * i for i in range(10)]
    dates = pd.date_range("2026-07-28", periods=10, freq="B")

    def fake_kline(symbol):
        return pd.DataFrame({
            "date": dates, "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [1e6] * 10,
        })

    # analyzed_date 默认今天 → 无法回填；手动构造已过期的历史记录
    svc2 = ExperienceService(store_path=str(db))
    _direct_insert(svc2, "000001", "2026-08-01", "bullish")
    _direct_insert(svc2, "000001", "2026-08-01", "bearish")

    summary = svc2.evaluate_history(kline_provider=fake_kline)
    assert summary["evaluated"] == 2
    assert summary["hit_rate"] == 0.5

    entries = svc2.search_experience(symbol="000001", limit=5)
    statuses = {e.summary: e.outcome for e in entries if e.summary == "outcome-check"}
    assert entries[0].outcome in ("hit", "miss")


def _direct_insert(svc, symbol, date_str, direction):
    """直接写入一条带指定日期的历史记录（绕过默认今天）。"""
    from pa_mcp.agent.experience import ExperienceService
    import json
    store = svc._store()
    try:
        max_id = store.query_df("SELECT COALESCE(MAX(id),0) AS m FROM analysis_history", [])
        new_id = int(max_id.iloc[0]["m"]) + 1
        store.insert_df("analysis_history", pd.DataFrame([{
            "id": new_id, "symbol": symbol, "analyzed_date": date_str,
            "cycle_position": "trading_range", "direction": direction,
            "strength_score": 60.0, "summary": "outcome-check",
            "conclusion": "", "dimension_scores": "{}", "key_risks": "[]",
            "outcome": "pending",
        }]))
    finally:
        store.close()


def test_stats(tmp_path):
    db = tmp_path / "exp_test3.duckdb"
    svc = ExperienceService(store_path=str(db))
    svc.save_analysis("000001", {"direction": "bullish", "overall_strength_score": 70})
    stats = svc.stats()
    assert stats["total"] == 1
    assert stats["by_direction"].get("bullish") == 1
