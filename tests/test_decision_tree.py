# [AI:BEGIN]
# PA_MCP - 决策树构建测试
# [AI:END]

from __future__ import annotations

from pa_mcp.agent.decision_tree import (
    build_decision_tree,
    tree_summary,
)


def test_tree_with_prediction_only():
    """仅有预测输入 → 出树，方向/仓位/结论逐层正确。"""
    pred = {
        "direction": "up", "probability": 0.62,
        "probability_distribution": {"up": 0.62, "down": 0.2, "sideways": 0.18},
        "expected_return_pct": 3.5, "expected_range_pct": [-1.5, 6.0],
        "cycle_position": "normal_channel", "cycle_forecast": "broad_channel",
        "key_levels": {"support": [10.2], "resistance": [11.5]},
        "scenarios": [{"name": "放量突破", "probability": 0.4}],
        "key_risks": ["大盘调整"], "horizon": "5d", "confidence": 0.6,
    }
    tree = build_decision_tree("000001", prediction=pred, stock_name="平安银行")
    assert tree["symbol"] == "000001"
    assert tree["stock_name"] == "平安银行"
    assert tree["tree"]["type"] == "root"
    market = tree["tree"]["children"][0]
    assert market["type"] == "decision"
    # 无诊断 → 保守假设
    assert "保守" in market["reason"]
    route = market["children"][0]
    assert route["type"] == "branch"
    assert "bollinger_mean_reversion" in route["detail"]
    direction_node = route["children"][0]
    assert "看涨" in direction_node["label"]
    position_node = direction_node["children"][0]
    assert "仓位" in position_node["label"]
    leaf = position_node["children"][0]
    assert leaf["type"] == "leaf"
    assert "支撑" in leaf["detail"] and "10.2" in leaf["detail"]
    # 看涨 + 无分析建议 → 回退 10%（medium 假设）
    assert "10" in position_node["detail"]


def test_tree_with_diagnosis():
    """诊断注入：冰点 → 直接观望叶；发酵 → 正常路径 + 分析建议生效。"""
    diag = {
        "market_state": "fermenting", "market_state_zh": "发酵期",
        "suggested_max_position_pct": 60, "risk_level": "medium",
        "key_observations": ["成交额放量", "涨停40家"],
        "strategy_routing": {"strategies": ["ma_golden_cross"], "label": "x"},
    }
    class FakeAnalysis:
        suggested_max_position_pct = 12.0
        direction = "bullish"

    tree = build_decision_tree("000001", diagnosis=diag,
                               analysis=FakeAnalysis(), prediction={
                                   "direction": "up", "probability": 0.6,
                                   "horizon": "5d",
                                   "expected_return_pct": 2.0,
                                   "expected_range_pct": [0, 4],
                                   "cycle_position": "normal_channel",
                                   "cycle_forecast": "trading_range",
                                   "key_levels": {}, "scenarios": [],
                                   "key_risks": []})
    assert tree["market_state"] == "fermenting"
    assert tree["position_cap_pct"] == 60
    # 建议 12% 受 60% 上限约束 → 12%
    leaf = tree["tree"]["children"][0]["children"][0]["children"][0]["children"][0]
    assert "12%" in leaf["detail"] or "≤12%" in tree["tree"]["children"][0][
        "children"][0]["children"][0]["detail"]


def test_tree_frozen_shortcut():
    """冰点期 → 跳过中间层直接观望叶。"""
    diag = {"market_state": "frozen", "market_state_zh": "冰点期",
            "suggested_max_position_pct": 5, "risk_level": "extreme",
            "key_observations": ["跌停40家"]}
    tree = build_decision_tree("000001", diagnosis=diag)
    leaf = tree["tree"]["children"][0]
    assert leaf["type"] == "leaf"
    assert "观望" in leaf["label"]


def test_tree_down_direction():
    """看跌 → 回避叶，仓位 0。"""
    pred = {
        "direction": "down", "probability": 0.7, "horizon": "5d",
        "expected_return_pct": -4.0, "expected_range_pct": [-8, 0],
        "cycle_position": "trading_range", "cycle_forecast": "trading_range",
        "key_levels": {}, "scenarios": [], "key_risks": [],
    }
    tree = build_decision_tree("000001", prediction=pred)
    # root → market → route → direction → position → leaf
    pos_node = tree["tree"]["children"][0]["children"][0]["children"][0]["children"][0]
    leaf = pos_node["children"][0]
    assert "回避" in leaf["label"]
    assert "0%" in pos_node["detail"]


def test_tree_with_market_bias():
    """指数偏空：看涨预测降级中性 + 仓位上限收紧。"""
    pred = {
        "direction": "up", "probability": 0.7, "horizon": "5d",
        "expected_return_pct": 3.0, "expected_range_pct": [-1, 5],
        "cycle_position": "normal_channel", "cycle_forecast": "broad_channel",
        "key_levels": {}, "scenarios": [], "key_risks": [],
    }
    tree = build_decision_tree("000001", prediction=pred,
                               market_bias="偏空")
    # 方向节点：看涨被降级为中性
    dir_node = tree["tree"]["children"][0]["children"][0]["children"][0]
    assert "指数偏空修正" in dir_node["label"]
    # 观察含指数结构说明
    assert any("指数空头" in o for o in tree["observations"])
    # 仓位上限收紧到 30
    assert tree["position_cap_pct"] <= 30

    # 偏多：看跌降级 + 上限不低于 40
    pred2 = dict(pred, direction="down")
    tree2 = build_decision_tree("000001", prediction=pred2,
                                market_bias="偏多")
    assert "指数偏多修正" in tree2["tree"]["children"][0]["children"][0]["children"][0]["label"]
    assert tree2["position_cap_pct"] >= 40


def test_tree_summary_text():
    """文本摘要含各层标签。"""
    pred = {"direction": "sideways", "probability": 0.5, "horizon": "5d",
            "expected_return_pct": 0.5, "expected_range_pct": [-2, 2],
            "cycle_position": "trading_range", "cycle_forecast": "trading_range",
            "key_levels": {}, "scenarios": [], "key_risks": []}
    tree = build_decision_tree("000001", prediction=pred)
    s = tree_summary(tree)
    assert "决策树" in s
    assert "市场状态" in s
    assert "策略路由" in s
    assert "预测方向" in s
    assert "仓位" in s
