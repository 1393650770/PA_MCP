# 递归自我改进（self_improve + memory.lessons）测试

from __future__ import annotations

import pytest

import pandas as pd

from pa_mcp.agent.memory import LongTermMemory
from pa_mcp.research import self_improve as si


def test_lesson_recorded_on_miss(tmp_path):
    """决策 miss（看涨负收益）→ 自动沉淀教训。"""
    mem = LongTermMemory(db_path=str(tmp_path / "m.db"))
    mem.record_decision("000001", 80, "up", "climax", "llm",
                        [{"k": "v"}], ["风险"])
    # 回填负收益 → record_outcome 自动沉淀教训
    mem.record_outcome(1, -3.5, days=5)
    lessons = mem.get_lessons(symbol="000001")
    assert len(lessons) >= 1
    assert "追高" in lessons[0]["lesson"] or "看涨" in lessons[0]["lesson"]
    assert lessons[0]["forward_return"] == -3.5


def test_no_lesson_on_win(tmp_path):
    """盈利决策不沉淀教训。"""
    mem = LongTermMemory(db_path=str(tmp_path / "m.db"))
    mem.record_decision("000001", 60, "up", "fermenting", "llm",
                        [{"k": "v"}], [])
    mem.record_outcome(1, 2.0, days=5)
    assert mem.get_lessons() == []


def test_lessons_stats_and_filter(tmp_path):
    mem = LongTermMemory(db_path=str(tmp_path / "m.db"))
    for i in range(3):
        mem.record_decision("000001", 80, "up", "climax", "llm", [{"k": i}], [])
        mem.record_outcome(i + 1, -4.0, days=5)
    stats = mem.lessons_stats()
    assert stats["total"] >= 3
    assert stats["by_direction"].get("up", 0) >= 3
    # 按方向过滤
    assert len(mem.get_lessons(direction="up", limit=10)) >= 3
    assert mem.get_lessons(direction="down") == []


def test_deterministic_report(tmp_path, monkeypatch):
    """无 LLM → 统计摘要报告。"""
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)
    monkeypatch.setattr(si, "_gather_evidence",
                        lambda: {"prediction": {"evaluated": 10,
                                                "hit_rate": 0.6},
                                 "lessons": {"total": 2}})
    r = si._deterministic_report(si._gather_evidence())
    assert r["mode"] == "rule"
    assert "预测验证" in r["report"]
    assert "教训库" in r["report"]


async def test_run_self_improve_llm(monkeypatch):
    """LLM 报告：结构完整。"""

    class _FakeAdapter:
        async def chat_json(self, params):
            return {
                "summary": "系统运行正常",
                "lessons_learned": ["追高失败重复出现"],
                "method_adjustments": [{"method": "bollinger",
                                        "action": "降低追高权重"}],
                "data_gaps": ["分钟线"],
                "next_actions": ["补分钟数据"],
            }

    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter",
        lambda: _FakeAdapter())
    monkeypatch.setattr(si, "_gather_evidence", lambda: {"lessons": {}})
    r = await si.run_self_improve()
    assert r["mode"] == "llm"
    assert r["summary"] == "系统运行正常"
    assert "方法调整" in r["report"]
    assert "下一步" in r["report"]


# ---- 上下文压缩 ----

from pa_mcp.research import context_compress as cc


def test_compress_kline_two_level():
    """K 线两级压缩：整体统计 + 近期明细 + 早期周聚合。"""
    dates = pd.date_range("2025-01-01", periods=150, freq="B")
    df = pd.DataFrame({
        "date": dates, "open": 10.0, "high": 10.5, "low": 9.5,
        "close": [10.0 + i * 0.01 for i in range(150)],
        "volume": 1_000_000, "pct_change": 0.1,
    })
    text = cc.compress_kline(df, detail_bars=10, summary_weeks=4)
    assert "K线150根" in text
    assert "近10日" in text
    assert "早期周聚合" in text
    assert "区间" in text and "收" in text


def test_compress_kline_short():
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10,
                                             freq="B"),
                       "close": [1.0] * 10, "high": [1.1] * 10,
                       "low": [0.9] * 10, "volume": [100] * 10})
    text = cc.compress_kline(df, detail_bars=30)
    assert "近10日" in text  # 全量明细
    assert "早期周聚合" not in text  # 无早期部分


async def test_compress_text_short_passthrough():
    r = await cc.compress_text("短文本", max_chars=1000)
    assert r["mode"] == "none"
    assert r["compressed"] == "短文本"


async def test_compress_text_truncate(monkeypatch):
    """无 LLM：保头尾截断。"""
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter", lambda: None)
    long = "关键数字123 " * 200
    r = await cc.compress_text(long, max_chars=500)
    assert r["mode"] == "truncate"
    assert len(r["compressed"]) <= 500
    assert "关键数字123" in r["compressed"]
    assert "已压缩" in r["compressed"]


async def test_compress_text_llm(monkeypatch):
    """LLM 精炼压缩。"""

    class _FakeAdapter:
        async def chat(self, params):
            from pa_mcp.agent.llm_port import LLMResponse
            return LLMResponse(content="压缩后：净利+23%，PE 19，风险：追高",
                               model="fake", provider="fake")

    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.ensure_llm_adapter",
        lambda: _FakeAdapter())
    r = await cc.compress_text("很长内容 " * 300, max_chars=500)
    assert r["mode"] == "llm"
    assert "净利+23%" in r["compressed"]
    assert r["compressed_len"] <= 500
