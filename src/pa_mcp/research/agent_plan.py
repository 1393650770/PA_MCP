# [AI:BEGIN]
# PA_MCP - Research: Agent 自主规划（plan → execute → 反馈重规划）
#
# 借鉴开源 Agent 规划的通用模式（Anthropic《Building Effective Agents》
# 的 workflow/agent 分类、Claude Code 的 plan-then-execute、LangGraph
# 状态机、ai-hedge-fund 的分阶段 pipeline）：
#   1. create_plan：给定目标 → LLM 生成结构化计划
#      {steps: [{id, action(工具名), args, purpose, depends_on}]}
#      工具目录（97 个 MCP 工具名+一句话）注入 prompt，让 LLM 知道能调什么
#   2. update_plan：执行完一步后反馈结果摘要 → LLM 调整剩余步骤
#      （evaluator-optimizer 风格的动态重规划：数据缺口自动补采）
#   3. 确定性降级：无 LLM 时按目标关键词路由到预设流程（研究/选股/持仓/
#      板块/预测五类）
#
# 与 methodology_guide（静态四步地图）互补：地图是"方法体系"，本模块是
# "本次任务的可执行计划"。
# [AI:END]

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_STEPS = 8

# 工具目录（name: 一句话）——注入 prompt 供 LLM 规划
TOOLS_INDEX: dict[str, str] = {
    "agent_market_diagnosis": "市场状态诊断（5 态+仓位建议）",
    "get_market_overview": "大盘总览（指数/涨跌家数/涨停数）",
    "sentiment_cycle": "游资情绪周期（涨停梯队/连板/阶段）",
    "sector_hot_cold": "今日热门/冷门板块",
    "predict_sector_rotation": "板块轮动预测（未来一周强势板块）",
    "sector_rotation_status": "板块轮动当前状态（RS 排名/轮入轮出）",
    "market_structure": "市场结构联合分析（指数缠论×情绪矩阵）",
    "scan_market": "按策略扫描股票池（当前买入信号）",
    "scan_canslim": "CANSLIM 成长股扫描",
    "factor_stock_selection": "多因子选股（IC 加权）",
    "graham_screen": "格雷厄姆价值筛选",
    "value_momentum_screen": "价值×动量复合选股",
    "strategy_compare": "全策略事件研究对比（有效性检验）",
    "get_strategy_guide": "策略速查（按市场状态推荐）",
    "get_methodology_guide": "新手决策地图（四步体系）",
    "research_event_study": "信号事件研究（预测力检验）",
    "research_walk_forward": "Walk-Forward 样本外检验",
    "backtest_overfit_diagnosis": "回测过拟合诊断（DSR/PBO）",
    "factor_neutralize": "因子中性化（风格正交）",
    "portfolio_risk_dashboard": "持仓风险面板（盈亏/集中度/评分）",
    "portfolio_summary": "持仓清单",
    "signal_consensus": "个股综合决策信号（5 源投票）",
    "predict_resonance": "多周期预测共振",
    "predict_market": "个股未来走势预测（1d/5d/20d）",
    "predict_future_chart": "未来 K 线路径预测（LLM 三情景）",
    "predict_position_size": "预测→仓位建议",
    "evaluate_predictions": "预测验证成绩单（命中率/Brier/IC）",
    "agent_analyze_stock": "AI 多维分析（fast/deep/debate）",
    "agent_debate_picks": "选股多 Agent 辩论（Bull/Bear+大师团）",
    "ai_market_report": "AI 市场研究报告（聚合+LLM 综述）",
    "trading_actions": "今日操作面板（止盈止损/买入）",
    "one_click_analysis": "一站式分析（全流水线）",
    "get_decision_tree": "决策树（市场→策略→仓位）",
    "evaluate_methods": "开源方法评价（可信度+持仓×方法）",
    "get_valuation_snapshot": "实时估值快照（PE/PB/市值）",
    "get_kline": "历史 K 线",
    "get_realtime_quote": "实时行情（含指数）",
    "get_stock_capital_flow": "个股资金流",
    "review_dragon_tiger": "龙虎榜席位",
    "watchlist_consensus": "自选股综合信号扫描",
    "agent_experience_search": "经验库检索（历史案例）",
    "agent_memory_status": "长期记忆（决策胜率/偏差）",
    "get_data_source_health": "数据源健康",
    "data_quality_report": "数据质量体检",
}

# 确定性降级：目标关键词 → 预设流程
_RULES: list[tuple[list[str], list[str]]] = [
    (["研究", "深度", "调研"], ["get_kline", "get_realtime_quote",
     "agent_analyze_stock", "signal_consensus", "predict_market",
     "predict_future_chart", "ai_market_report"]),
    (["选股", "股票", "候选"], ["get_market_overview", "sentiment_cycle",
     "scan_canslim", "factor_stock_selection", "strategy_compare",
     "agent_debate_picks"]),
    (["持仓", "组合", "仓位"], ["portfolio_summary",
     "portfolio_risk_dashboard", "signal_consensus",
     "predict_position_size", "trading_actions"]),
    (["板块", "热门", "冷门", "轮动"], ["get_market_overview",
     "sector_hot_cold", "sector_rotation_status",
     "predict_sector_rotation"]),
    (["预测", "走势", "未来"], ["get_kline", "predict_market",
     "predict_future_chart", "predict_resonance",
     "evaluate_predictions"]),
]

_PLAN_PROMPT = """你是 A 股研究规划师。把用户目标拆解为可执行的工具调用计划。

【可用工具】{tools}

【用户目标】{goal}

输出计划 JSON（只输出 JSON）：
{{"goal": "目标重述一句话",
  "steps": [{{"id": 1, "action": "工具名", "args": "工具参数（如 symbol=601728）",
              "purpose": "这一步要解决什么问题（≤30字）",
              "depends_on": []}}],
  "risks": ["数据缺口/注意点"],
  "data_needs": ["执行前需确认的数据"]}}

要求：
1. 3-{max_steps} 步，顺序合理（环境→选股→验证→结论），deps 标依赖
2. 只用上面列出的工具；args 给出合理默认（研究标的从目标提取）
3. 计划是研究参考，非投资建议"""


def _tools_text() -> str:
    return "；".join(f"{k}({v})" for k, v in TOOLS_INDEX.items())


def _parse_plan(raw: dict) -> Optional[dict]:
    """校验 LLM 输出结构。"""
    if not isinstance(raw, dict) or "steps" not in raw:
        return None
    steps = []
    for s in raw.get("steps", [])[:MAX_STEPS]:
        action = str(s.get("action", "")).strip()
        if action not in TOOLS_INDEX:
            continue  # 只保留已知工具
        steps.append({
            "id": int(s.get("id", len(steps) + 1)),
            "action": action,
            "args": str(s.get("args", "")),
            "purpose": str(s.get("purpose", ""))[:40],
            "depends_on": [int(d) for d in s.get("depends_on", [])
                           if str(d).isdigit()],
        })
    if not steps:
        return None
    return {"goal": raw.get("goal", ""), "steps": steps,
            "risks": raw.get("risks", []),
            "data_needs": raw.get("data_needs", [])}


def _rule_plan(goal: str) -> dict:
    """确定性降级：目标关键词 → 预设流程。"""
    steps = []
    seq = None
    for keys, actions in _RULES:
        if any(k in goal for k in keys):
            seq = actions
            break
    seq = seq or ["get_market_overview", "agent_market_diagnosis",
                  "ai_market_report"]
    for i, a in enumerate(seq, 1):
        steps.append({"id": i, "action": a, "args": "",
                      "purpose": TOOLS_INDEX.get(a, "")[:30],
                      "depends_on": [i - 1] if i > 1 else []})
    return {"goal": goal, "steps": steps, "risks": ["无 LLM，规则路由计划"],
            "data_needs": [], "mode": "rule"}


async def create_plan(goal: str, max_steps: int = 6) -> dict[str, Any]:
    """给定目标 → 可执行计划（LLM 规划，无 LLM 规则降级）。"""
    goal = goal.strip()
    if not goal:
        return {"error": "目标不能为空（如：研究 601728 并给出操作建议）"}
    max_steps = max(3, min(max_steps, MAX_STEPS))
    plan: Optional[dict] = None
    try:
        from pa_mcp.agent.llm_factory import ensure_llm_adapter
        from pa_mcp.agent.llm_port import LLMCallParams
        adapter = ensure_llm_adapter()
        if adapter is not None:
            params = LLMCallParams(
                system_prompt="你是 A 股研究规划师。只输出合法 JSON。"
                              "计划是研究参考，非投资建议。",
                user_prompt=_PLAN_PROMPT.format(
                    tools=_tools_text(), goal=goal, max_steps=max_steps),
                mode="fast", max_tokens=2000,
            )
            raw = await adapter.chat_json(params)
            plan = _parse_plan(raw)
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM 规划失败，规则降级: %s", str(e)[:80])

    if plan is None:
        plan = _rule_plan(goal)
        plan["mode"] = "rule"
    else:
        plan["mode"] = "llm"
    plan["report"] = format_plan(plan)
    return plan


async def update_plan(plan: dict, completed_id: int,
                      result_summary: str) -> dict[str, Any]:
    """执行完一步后反馈结果 → 动态调整剩余步骤（评估-优化）。

    result_summary 含已执行步骤的输出摘要；LLM 判断后续步骤是否
    需要增删（数据缺口补采/方向修正）。
    """
    steps = plan.get("steps", [])
    if not steps:
        return {"error": "计划为空"}
    remaining = [s for s in steps if s["id"] > completed_id]
    done = [s for s in steps if s["id"] <= completed_id]
    if not remaining:
        plan["report"] = format_plan(plan)
        return plan  # 已全部完成

    updated: Optional[dict] = None
    try:
        from pa_mcp.agent.llm_factory import ensure_llm_adapter
        from pa_mcp.agent.llm_port import LLMCallParams
        adapter = ensure_llm_adapter()
        if adapter is not None:
            prompt = (
                "已执行计划步骤并得到结果，评估是否需要调整剩余步骤。\n"
                f"【原计划】{steps}\n"
                f"【已完成】步骤 {completed_id} 结果：{result_summary[:800]}\n"
                f"【剩余步骤】{remaining}\n"
                "若剩余步骤仍合适输出 keep=true；若有缺口输出调整后的"
                "剩余步骤（只输出 JSON）："
                '{"keep": true/false, "remaining": [{"id","action","args",'
                '"purpose","depends_on"}], "reason": "一句话"}')
            params = LLMCallParams(
                system_prompt="只输出合法 JSON。研究参考，非投资建议。",
                user_prompt=prompt, mode="fast", max_tokens=1500)
            raw = await adapter.chat_json(params)
            if isinstance(raw, dict) and raw.get("keep") is False:
                rem = _parse_plan({"steps": raw.get("remaining", [])})
                if rem:
                    updated = rem["steps"]
    except Exception as e:  # noqa: BLE001
        logger.warning("规划调整失败，保持原计划: %s", str(e)[:80])

    if updated:
        # 保留已完成的，替换剩余
        plan["steps"] = done + updated
        plan["report"] = format_plan(plan)
    return plan


def format_plan(plan: dict) -> str:
    """计划 → markdown。"""
    mode_txt = "🤖 LLM 规划" if plan.get("mode") == "llm" else "📐 规则路由"
    lines = [f"## 🗺️ AI 研究计划（{mode_txt}）",
             f"**目标**：{plan.get('goal', '')}", ""]
    for s in plan.get("steps", []):
        dep = f"（依赖 {s['depends_on']}）" if s.get("depends_on") else ""
        args = f" `{s['args']}`" if s.get("args") else ""
        lines.append(f"{s['id']}. **{s['action']}**{args} — "
                     f"{s.get('purpose', '')}{dep}")
    if plan.get("data_needs"):
        lines.append("\n**数据前置**：" + "；".join(plan["data_needs"]))
    if plan.get("risks"):
        lines.append("**注意**：" + "；".join(plan["risks"][:3]))
    lines.append("\n*按步骤执行，每步完成后可调 agent_plan_update 反馈"
                 "结果实现动态重规划。研究参考，非投资建议。*")
    return "\n".join(lines)
