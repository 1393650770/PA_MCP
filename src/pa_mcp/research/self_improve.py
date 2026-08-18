# [AI:BEGIN]
# PA_MCP - Research: 递归自我改进（RSI 落地：评估 → 教训 → 下次更好）
#
# 借鉴开源做法（受限于安全约束，落地为"受控改进闭环"而非自治进化）：
#   - Reflexion：决策失败 → 反思 → 教训沉淀 → 后续行动参考
#     （memory.record_lesson 已在决策回填 miss 时自动沉淀）
#   - Voyager 技能库：成功经验沉淀复用（经验库 hit/miss 已做）
#   - Evaluator-Optimizer：评估产出质量 → 生成改进建议（本模块）
#   - MemGPT/LangMem 记忆分层：情景（决策记录）+ 语义（经验/教训）
#
# run_self_improve() 聚合四路证据：
#   ① 预测验证成绩单（命中率/Brier/IC）
#   ② 全策略事件研究（有效/失效策略）
#   ③ 经验库统计（hit/miss 率）
#   ④ 教训库（最近 miss 决策的教训）
# → LLM 生成改进报告（哪些方法失效/教训反复出现/调优方向/下一步）；
#   无 LLM 时确定性统计摘要。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _gather_evidence() -> dict[str, Any]:
    """聚合自我评估的四路证据（best-effort，失败给空段）。"""
    ev: dict[str, Any] = {}

    # ① 预测成绩单
    try:
        import asyncio
        from pa_mcp.agent.prediction import get_prediction_service
        r = asyncio.run(get_prediction_service().evaluate_predictions())
        if "error" not in r:
            ev["prediction"] = {
                "evaluated": r.get("evaluated", 0),
                "hit_rate": r.get("hit_rate"),
                "brier": r.get("brier_score"),
                "ic": r.get("ic"),
                "by_direction": r.get("by_direction", {}),
            }
    except Exception:  # noqa: BLE001
        pass

    # ② 策略评估（事件研究对比）
    try:
        from pa_mcp.research.strategy_compare import compare_all_strategies
        from pa_mcp.data.store import DuckDBStore
        from pa_mcp.config import get_settings
        store = DuckDBStore(get_settings().database.path)
        store.connect()
        try:
            klines = {}
            for sym in ("000001", "600036", "300750", "600519", "601318"):
                df = store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 250", [sym])
                if df is not None and not df.empty:
                    klines[sym] = df.sort_values("date").reset_index(drop=True)
        finally:
            store.close()
        if len(klines) >= 3:
            r = compare_all_strategies(klines)
            if "error" not in r:
                ev["strategies"] = {
                    "useful": r.get("useful_strategies", []),
                    "ranking": r.get("ranking", [])[:8],
                }
    except Exception:  # noqa: BLE001
        pass

    # ③ 经验库统计
    try:
        from pa_mcp.agent.experience import get_experience_service
        ev["experience"] = get_experience_service().stats()
    except Exception:  # noqa: BLE001
        pass

    # ④ 教训库
    try:
        from pa_mcp.agent.memory import LongTermMemory
        mem = LongTermMemory()
        ev["lessons"] = mem.lessons_stats()
        ev["recent_lessons"] = mem.get_lessons(limit=8)
        ev["bias"] = mem.detect_bias()
    except Exception:  # noqa: BLE001
        pass

    return ev


_IMPROVE_PROMPT = """你是 A 股研究系统自我改进评审。基于以下系统运行证据，
给出下一轮改进建议。

【证据】{evidence}

输出 JSON（只输出 JSON）：
{{"summary": "系统当前状态一句话",
  "lessons_learned": ["教训1（哪些判断模式反复失败）"],
  "method_adjustments": [{{"method": "方法名", "action": "调整建议"}}],
  "data_gaps": ["数据缺口/需补采"],
  "next_actions": ["下一步改进动作（≤3条）"]}}
研究参考，非投资建议。"""


def _deterministic_report(ev: dict[str, Any]) -> dict[str, Any]:
    """无 LLM：统计摘要（确定性）。"""
    lines = ["## 🔄 自我改进评估（📐 统计摘要）", ""]
    p = ev.get("prediction")
    if p:
        lines.append(f"**预测验证**：样本 {p.get('evaluated', 0)}，"
                     f"命中率 {p.get('hit_rate', '—')}，"
                     f"IC {p.get('ic', '—')}")
    s = ev.get("strategies")
    if s:
        lines.append(f"**策略评估**：有效 {'、'.join(s.get('useful', []) or ['无'])}；"
                     f"排名 {', '.join(s.get('ranking', [])[:5])}")
    e = ev.get("experience")
    if e:
        lines.append(f"**经验库**：{e}")
    l = ev.get("lessons")
    if l:
        lines.append(f"**教训库**：{l.get('total', 0)} 条"
                     f"（{l.get('by_direction', {})}）")
    b = ev.get("bias")
    if b:
        lines.append(f"**认知偏差**：{b}")
    lines.append("\n*配置 LLM 后生成深度改进建议（方法调整/下一步动作）。"
                 "研究参考，非投资建议。*")
    return {"mode": "rule", "summary": "统计摘要（无 LLM）",
            "lessons_learned": [], "method_adjustments": [],
            "data_gaps": [], "next_actions": [],
            "report": "\n".join(lines)}


async def run_self_improve() -> dict[str, Any]:
    """递归自我改进评估：聚合证据 → LLM 改进建议（确定性降级）。"""
    ev = _gather_evidence()
    result: Optional[dict] = None
    try:
        import json
        from pa_mcp.agent.llm_factory import ensure_llm_adapter
        from pa_mcp.agent.llm_port import LLMCallParams
        adapter = ensure_llm_adapter()
        if adapter is not None:
            params = LLMCallParams(
                system_prompt="你是 A 股研究系统自我改进评审。只输出合法 JSON。"
                              "研究参考，非投资建议。",
                user_prompt=_IMPROVE_PROMPT.format(
                    evidence=json.dumps(ev, ensure_ascii=False)[:3000]),
                mode="deep", max_tokens=2000,
            )
            raw = await adapter.chat_json(params)
            if isinstance(raw, dict) and "summary" in raw:
                raw["mode"] = "llm"
                result = raw
    except Exception as e:  # noqa: BLE001
        logger.warning("自我改进 LLM 失败，统计降级: %s", str(e)[:80])

    if result is None:
        result = _deterministic_report(ev)
    else:
        lines = [f"## 🔄 自我改进评估（🤖 LLM）",
                 f"**总结**：{result.get('summary', '')}", ""]
        if result.get("lessons_learned"):
            lines.append("**教训**：" + "；".join(result["lessons_learned"]))
        if result.get("method_adjustments"):
            lines.append("**方法调整**：")
            for m in result["method_adjustments"][:5]:
                lines.append(f"- {m.get('method', '')}：{m.get('action', '')}")
        if result.get("data_gaps"):
            lines.append("**数据缺口**：" + "；".join(result["data_gaps"]))
        if result.get("next_actions"):
            lines.append("**下一步**：" + "；".join(result["next_actions"]))
        lines.append("\n*评估基于系统真实运行证据（预测/策略/经验/教训）。"
                     "研究参考，非投资建议。*")
        result["report"] = "\n".join(lines)
    result["evidence"] = {k: v for k, v in ev.items()
                          if k not in ("recent_lessons",)}
    return result
