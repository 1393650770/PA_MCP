# [AI:BEGIN]
# PA_MCP - Agent Layer: 决策树（市场诊断 → 策略路由 → 方向 → 仓位 → 建议）
#
# 借鉴 PA_Agent「决策树可视化 + 闸门→策略路径」机制：
#   1. 确定性树构建：根(标的) → 市场状态 → 策略路由 → 预测方向 →
#      仓位分级 → 叶结论（每层都是「闸门」，条件不满足即换路径）
#   2. 纯数据输出（dict/JSON），UI 负责画图、MCP 直接返回结构
#   3. 输入可选：诊断(diagnosis) / 分析(AnalysisResult) / 预测(prediction dict)，
#      缺省层自动用确定性规则生成——任何输入组合都能出树
#   4. 每个节点带 reason（判定依据）与 detail（证据），可追溯、不编造
# [AI:END]

from __future__ import annotations

import json
from typing import Any, Optional

# ---- 节点类型 ----
ROOT = "root"
DECISION = "decision"   # 闸门节点：条件判定
BRANCH = "branch"       # 路径分叉：策略候选
LEAF = "leaf"           # 结论/建议


def _node(nid: str, label: str, ntype: str,
          reason: str = "", detail: str = "",
          children: Optional[list[dict]] = None) -> dict:
    """构造一个树节点（统一结构，UI 与 MCP 共用）。"""
    return {
        "id": nid,
        "label": label,
        "type": ntype,
        "reason": reason,
        "detail": detail,
        "children": children or [],
    }


def _position_band(pct: float) -> str:
    """仓位分级（确定性）：0=观望 1=试探 2=正常 3=积极。"""
    if pct is None or pct <= 0:
        return "观望（不建仓）"
    if pct <= 3:
        return "试探（≤3%）"
    if pct <= 8:
        return "轻仓（≤8%）"
    if pct <= 15:
        return "正常（≤15%）"
    return "积极（≤20%，上限内）"


def build_decision_tree(
    symbol: str,
    diagnosis: Optional[dict] = None,
    analysis: Optional[Any] = None,
    prediction: Optional[dict] = None,
    stock_name: str = "",
    market_bias: Optional[str] = None,
) -> dict:
    """构建决策树：根 → 市场闸门 → 策略路由 → 方向闸门 → 仓位闸门 → 叶结论。

    任何一层输入缺失时用确定性规则补全，保证始终可出树（可追溯）。

    Args:
        symbol: 股票代码
        diagnosis: 市场诊断（含策略路由）
        analysis: 分析结果（含建议仓位）
        prediction: 预测 dict（方向/概率/区间）
        stock_name: 股票名称
        market_bias: 指数结构方向（"偏多"/"偏空"/"中性"，
            来自 market_structure 联合分析）——市场闸门增强维度
    """
    # ---- 层 1：市场状态（闸门） ----
    market_state = "unknown"
    market_zh = "未知"
    pos_cap = 50
    risk_level = "medium"
    observations: list[str] = []
    if diagnosis:
        market_state = str(diagnosis.get("market_state", "unknown")).lower()
        market_zh = str(diagnosis.get("market_state_zh") or market_state)
        pos_cap = float(diagnosis.get("suggested_max_position_pct", 50) or 50)
        risk_level = str(diagnosis.get("risk_level", "medium"))
        observations = [str(o) for o in (diagnosis.get("key_observations") or [])]
    if market_state == "unknown":
        # 无诊断输入 → 中性保守
        market_state, market_zh = "dull", "未知（保守假设低迷）"
        pos_cap, risk_level = 30, "medium"
        observations = ["无市场诊断数据，保守判定"]

    # 市场闸门增强：指数结构方向（缠论，来自 market_structure）
    if market_bias:
        bias_note = {
            "偏多": "指数多头结构（缠论），顺大势",
            "偏空": "指数空头结构（缠论），逆风环境",
            "中性": "指数结构中性",
        }.get(market_bias, f"指数结构：{market_bias}")
        observations.append(bias_note)
        # 偏空环境收紧仓位上限与风险等级
        if market_bias == "偏空":
            pos_cap = min(pos_cap, 30)
            risk_level = "high" if risk_level in ("medium", "low") else risk_level
            observations.append("指数偏空 → 总仓位上限收紧至 30%")
        elif market_bias == "偏多":
            pos_cap = max(pos_cap, 40)

    # 市场闸门：极端状态（冰点）→ 直接到观望叶
    if market_state in ("frozen",):
        frozen = _node(
            "leaf_frozen", "❄️ 冰点期：空仓观望",
            LEAF, reason="市场冰点（跌停潮/缩量），风险收益比极差",
            detail="等待情绪修复信号（涨停家数回升/成交额回暖）再入场",
        )
        return _tree(symbol, stock_name, market_state, market_zh,
                     pos_cap, risk_level, observations, frozen)

    # ---- 层 2：策略路由（分支） ----
    routing = {}
    if diagnosis:
        routing = diagnosis.get("strategy_routing") or {}
    strategies = [str(s) for s in (routing.get("strategies") or [])]
    if not strategies:
        strategies = ["bollinger_mean_reversion", "ma_golden_cross",
                      "volume_price_momentum"]
    route_label = str(routing.get("label") or f"适配 {market_zh} 的策略")
    risk_notes = str(routing.get("risk_notes") or "")

    route_node = _node(
        "route", f"🧭 策略路由：{route_label}", BRANCH,
        reason=f"市场状态 {market_zh} 对应的策略清单",
        detail="；".join(strategies),
    )

    # ---- 层 3：预测方向（闸门） ----
    direction = "neutral"
    prob = 0.5
    exp_ret = 0.0
    horizon = "5d"
    cycle_pos = cycle_forecast = "unknown"
    if prediction:
        direction = str(prediction.get("direction", "sideways")).lower()
        if direction not in ("up", "down", "sideways"):
            direction = "sideways"
        prob = float(prediction.get("probability", 0.5) or 0.5)
        exp_ret = float(prediction.get("expected_return_pct", 0.0) or 0.0)
        horizon = str(prediction.get("horizon", "5d"))
        cycle_pos = str(prediction.get("cycle_position", "unknown"))
        cycle_forecast = str(prediction.get("cycle_forecast", "unknown"))

    # 指数结构修正预测方向（偏空环境看涨降级为中性提醒）
    if market_bias == "偏空" and direction == "up":
        direction = "sideways"
        dir_zh = f"➡️ 震荡（{prob:.0%}，指数偏空修正）"
        dir_reason = (f"预测看涨 {prob:.0%}，但指数空头结构——逆风环境"
                      f"降级为中性（期望 {exp_ret:+.1f}%）")
    elif market_bias == "偏多" and direction == "down":
        direction = "sideways"
        dir_zh = f"➡️ 震荡（{prob:.0%}，指数偏多修正）"
        dir_reason = (f"预测看跌 {prob:.0%}，但指数多头结构——"
                      f"顺势环境降级为中性（期望 {exp_ret:+.1f}%）")
    elif direction == "up":
        dir_zh = f"📈 看涨（{prob:.0%}）"
        dir_reason = f"预测期望收益 {exp_ret:+.1f}%，多头信号占优"
    elif direction == "down":
        dir_zh = f"📉 看跌（{prob:.0%}）"
        dir_reason = f"预测期望收益 {exp_ret:+.1f}%，空头信号占优"
    else:
        dir_zh = f"➡️ 震荡（{prob:.0%}）"
        dir_reason = f"期望收益 {exp_ret:+.1f}%，多空平衡"

    dir_node = _node(
        "direction", f"🎯 预测方向：{dir_zh}", DECISION,
        reason=dir_reason,
        detail=f"周期位置 {cycle_pos} → {cycle_forecast}（{horizon}）",
    )

    # ---- 层 4：仓位分级（闸门） ----
    # 综合建议仓位 = min(诊断仓位上限, 分析建议, 方向加权)
    suggested = 0.0
    pos_reason = ""
    if analysis is not None:
        try:
            suggested = float(getattr(analysis, "suggested_max_position_pct", 0.0) or 0.0)
            pos_reason = f"分析师团建议 ≤{suggested:.0f}%"
        except Exception:
            suggested = 0.0
    if direction == "up":
        fallback = 10.0 if risk_level in ("low", "medium") else 5.0
    elif direction == "down":
        fallback = 0.0
    else:
        fallback = 3.0
    if suggested <= 0:
        suggested = fallback
        pos_reason = f"无分析建议，按方向确定性回退（{_position_band(suggested)}）"
    else:
        pos_reason = (pos_reason + f"，且受市场仓位上限 {pos_cap:.0f}% 约束")
    suggested = max(0.0, min(suggested, pos_cap))

    pos_node = _node(
        "position", f"💼 仓位闸门：{_position_band(suggested)}", DECISION,
        reason=pos_reason, detail=f"最终建议仓位 ≤{suggested:.0f}%",
    )

    # ---- 层 5：叶结论（建议，不越权下单） ----
    if direction == "up":
        leaf_label = f"✅ 研究倾向参与：轻仓跟随 {horizon} 方向，跌破支撑止损"
    elif direction == "down":
        leaf_label = "⛔ 研究倾向回避：等待企稳或做空条件（当前不做空）"
    else:
        leaf_label = "⏸️ 研究倾向观望：区间操作或等待方向明确"
    leaf_detail_parts = []
    if prediction and prediction.get("key_levels"):
        lv = prediction["key_levels"]
        if lv.get("support"):
            leaf_detail_parts.append(f"支撑 {lv['support']} / 压力 {lv['resistance']}")
    if prediction and prediction.get("scenarios"):
        sc = prediction["scenarios"][0]
        leaf_detail_parts.append(f"主情景：{sc.get('name', '')}（{sc.get('probability', 0):.0%}）")
    if prediction and prediction.get("key_risks"):
        leaf_detail_parts.append("风险：" + "；".join(
            str(r) for r in prediction["key_risks"][:2]))
    if risk_notes:
        leaf_detail_parts.append(f"路由风险提示：{risk_notes}")
    leaf = _node(
        "leaf", leaf_label, LEAF,
        reason="由市场状态/策略路由/方向/仓位逐层闸门推导",
        detail="；".join(leaf_detail_parts) or "研究参考，非投资建议",
    )

    pos_node["children"] = [leaf]
    dir_node["children"] = [pos_node]
    route_node["children"] = [dir_node]
    market_node = _node(
        "market", f"🌡️ 市场状态：{market_zh}", DECISION,
        reason="；".join(observations) or "无观测数据",
        detail=f"风险等级 {risk_level}，建议总仓位上限 {pos_cap:.0f}%",
        children=[route_node],
    )

    return _tree(symbol, stock_name, market_state, market_zh,
                 pos_cap, risk_level, observations, market_node)


def _tree(symbol: str, stock_name: str, market_state: str, market_zh: str,
          pos_cap: float, risk_level: str, observations: list[str],
          child: dict) -> dict:
    return {
        "symbol": symbol,
        "stock_name": stock_name or symbol,
        "market_state": market_state,
        "market_state_zh": market_zh,
        "position_cap_pct": pos_cap,
        "risk_level": risk_level,
        "observations": observations,
        "tree": {
            "id": "root",
            "label": f"决策树：{stock_name or symbol}",
            "type": "root",
            "reason": "逐层闸门：市场 → 策略 → 方向 → 仓位 → 结论",
            "detail": "研究参考，非投资建议",
            "children": [child],
        },
    }


def tree_to_json(tree: dict) -> str:
    """决策树 → JSON 字符串（MCP 返回用）。"""
    return json.dumps(tree, ensure_ascii=False, indent=2)


def tree_summary(tree: dict) -> str:
    """决策树 → 文本路径摘要（沿「主路径」打印各层结论）。"""
    lines = [f"### 🌳 {tree.get('stock_name', tree['symbol'])} 决策树"]
    node = tree["tree"]

    def walk(n: dict, depth: int) -> None:
        indent = "  " * depth
        lines.append(f"{indent}- {n['label']}")
        if n.get("reason"):
            lines.append(f"{indent}  ⚙️ {n['reason']}")
        if n.get("detail") and n["type"] in (LEAF, DECISION):
            lines.append(f"{indent}  📋 {n['detail']}")
        for c in n.get("children", []):
            walk(c, depth + 1)

    walk(node, 0)
    return "\n".join(lines)
