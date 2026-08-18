# 递归自我改进（self_improve + memory.lessons）测试

from __future__ import annotations

import pytest

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
