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


def test_sentiment_summary(tmp_path):
    """摘要方法：关键字段齐全；失败返回空 dict（best-effort）。"""
    db = _seed_market(tmp_path, {
        "2026-08-03": {"600001": 10.0, "600002": 10.0},
        "2026-08-04": {"600001": 10.0, "600002": 10.0, "600003": 10.0},
    })
    a = SentimentCycleAnalyzer(store_path=db)
    s = a.sentiment_summary("2026-08-04")
    assert s.get("date") == "2026-08-04"
    assert "stage" in s and "sentiment_score" in s
    assert "max_board_height" in s and "promotion_rate" in s
    # 空库 + 实时也失败 → 空 dict（实时模式成功时返回真实情绪）
    import pa_mcp.research.sentiment_cycle as sc
    empty = SentimentCycleAnalyzer(store_path=str(tmp_path / "none.db"))

    async def _no_realtime():
        return None
    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(sc.SentimentCycleAnalyzer,
                        "_fetch_realtime_stats", _no_realtime)
    assert empty.sentiment_summary() == {}
    monkeypatch.undo()


def test_diagnosis_with_sentiment_context(tmp_path):
    """市场诊断集成：情绪上下文注入 LLM prompt + 确定性冰点降级。"""
    import asyncio
    from pa_mcp.agent.orchestrator import AgentOrchestrator

    # 确定性：情绪冰点 → frozen
    orch = AgentOrchestrator()
    d = orch._diagnosis_deterministic({
        "limit_up_count": 5, "turnover_billion": 300,
        "sentiment": {"stage": "ice", "stage_zh": "冰点期",
                      "sentiment_score": 12, "max_board_height": 1},
    })
    assert d["market_state"] == "frozen"
    assert any("冰点" in o for o in d["key_observations"])

    # LLM 路径：情绪格式化注入 user_prompt
    class P:
        system_prompt = ""
        user_prompt = ""
        mode = "fast"
        max_tokens = 100

    class MockAdapter:
        provider_name = "mock"

        async def chat_json(self, params):
            self.last = params.user_prompt
            return {"market_state": "fermenting", "confidence": 70,
                    "suggested_max_position_pct": 60, "risk_level": "medium",
                    "key_observations": []}

    from pa_mcp.agent import llm_port
    orig = llm_port._adapter
    adapter = MockAdapter()
    llm_port.register_adapter(adapter)
    try:
        d2 = asyncio.run(orch.market_diagnosis({
            "limit_up_count": 40,
            "sentiment": {"stage_zh": "发酵期", "sentiment_score": 65,
                          "max_board_height": 4, "promotion_rate": 0.5,
                          "board2_count": 10, "board3_count": 5,
                          "board4p_count": 2},
        }))
        assert d2["market_state"] == "fermenting"
        assert "游资情绪" in adapter.last
        assert "发酵期" in adapter.last
    finally:
        llm_port.register_adapter(orig)


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
