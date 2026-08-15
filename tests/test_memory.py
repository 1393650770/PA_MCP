# [AI:BEGIN]
# PA_MCP - 长期记忆（SQLite）测试
# [AI:END]

from __future__ import annotations

import os

from pa_mcp.agent.memory import LongTermMemory


def test_record_and_performance(tmp_path):
    db = os.path.join(tmp_path, "mem_test.db")
    mem = LongTermMemory(db_path=db)

    # 3 胜 2 负
    for i in range(5):
        did = mem.record_decision(
            symbol="000001", strength_score=60 + i, direction="bullish",
            market_state="fermenting", mode="deep",
            evidence=[{"dimension": "technical", "finding": "x"}],
            risks=["risk"],
        )
        mem.record_outcome(did, forward_return=0.03 if i < 3 else -0.02, days=5)

    perf = mem.get_performance_summary(days=60)
    assert perf["total_decisions"] == 5
    assert perf["win_rate"] == 60.0
    assert perf["avg_return_pct"] == 1.0
    assert perf["avg_win_pct"] == 3.0
    assert perf["avg_loss_pct"] == -2.0


def test_strategy_weight_bayesian(tmp_path):
    db = os.path.join(tmp_path, "mem_test2.db")
    mem = LongTermMemory(db_path=db)
    for i in range(10):
        mem.update_strategy_weight("bollinger_mean_reversion", win=(i < 7))
    w = mem.get_strategy_weight("bollinger_mean_reversion")
    assert 0.5 < w < 1.0  # 7/10 胜 → 权重升但向 0.5 衰减
    assert mem.get_strategy_weight("unknown_strategy") == 1.0


def test_bias_detection_overconfidence(tmp_path):
    db = os.path.join(tmp_path, "mem_test3.db")
    mem = LongTermMemory(db_path=db)
    # 5 条高分决策 + 负收益 → 触发过度自信
    for i in range(5):
        did = mem.record_decision(
            symbol="600000", strength_score=80 + i, direction="bullish",
            market_state="climax", mode="deep",
            evidence=[], risks=[],
        )
        mem.record_outcome(did, forward_return=-0.05, days=5)
    biases = mem.detect_bias()
    assert any(b["type"] == "overconfidence" for b in biases)

    # 低分决策 + 负收益 → 不触发
    mem2 = LongTermMemory(db_path=os.path.join(tmp_path, "mem_test4.db"))
    for i in range(5):
        did = mem2.record_decision(
            symbol="600000", strength_score=40, direction="neutral",
            market_state="dull", mode="fast", evidence=[], risks=[])
        mem2.record_outcome(did, forward_return=-0.05, days=5)
    assert mem2.detect_bias() == []


def test_default_path_is_absolute():
    """默认路径必须绝对（避免 cwd 依赖）。"""
    mem = LongTermMemory(db_path=None)
    assert os.path.isabs(mem.db_path)
    assert mem.db_path.endswith("memory.db")
