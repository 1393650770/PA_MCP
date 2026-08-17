# [AI:BEGIN]
# PA_MCP - Research: 一站式分析（流水线整合报告）
#
# 参考开源 agent 做法（ai-hedge-fund pipeline / TradingAgents 分阶段），
# 把全部确定性研究按流水线顺序执行并整合为一份报告：
#   阶段1 市场体检：数据质量评分
#   阶段2 市场结构：指数缠论 × 情绪矩阵
#   阶段3 板块轮动：RS top5 + 轮动速度
#   阶段4 选股：因子选股（AI 预测融合）→ 价值动量 → 格雷厄姆
#   阶段5 个股信号：池内每只综合信号（前 6 只）
#   阶段6 持仓风险：组合评分 + 强信号看跌告警
#   阶段7 汇总：整合报告 + 风险提示
#
# 全部确定性（use_llm=False）快速执行；LLM 可选（--llm 时附综述）。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_POOL = ["000001", "600036", "300750", "000858", "600519", "601318"]


async def one_click_report(symbols: Optional[list[str]] = None,
                           include_llm: bool = False) -> dict[str, Any]:
    """一站式分析：流水线整合报告。

    Args:
        symbols: 股票池（缺省 DEFAULT_POOL）
        include_llm: 是否附加 AI 综述（LLM 可用时）
    """
    pool = [s.strip() for s in (symbols or DEFAULT_POOL) if s.strip()][:6]
    sections: list[tuple[str, str]] = []

    # 前置：行情/板块/指数/情绪 缺失自动补（网络可用时拉真实，失败降级）
    try:
        from pa_mcp.data.readiness import ensure_readiness
        from pa_mcp.data.readiness import (
            NEED_SECTOR, NEED_INDEX, NEED_SENTIMENT)
        await ensure_readiness({**NEED_SENTIMENT, **NEED_SECTOR,
                                **NEED_INDEX})
    except Exception:
        pass

    # 阶段1：市场体检
    try:
        from pa_mcp.data.quality_report import get_quality_report
        q = get_quality_report().generate()
        sections.append(("🩺 数据体检", q["summary"]))
    except Exception as e:  # noqa: BLE001
        sections.append(("🩺 数据体检", f"不可用：{e}"))

    # 阶段2：市场结构（允许网络拉指数）
    try:
        from pa_mcp.research.market_structure import MarketStructureAnalyzer
        ms = await MarketStructureAnalyzer().analyze(use_network=True)
        if ms["index"]["rows"] > 0:
            j = ms["joint"]
            sections.append(("🏛️ 市场结构",
                             f"指数 {ms['index']['last_close']}（{ms['index']['last_date']}）"
                             f"：{j['bias']}——{j['structure']}；{j['environment']}"))
        else:
            sections.append(("🏛️ 市场结构", "指数数据不可用（网络拉取也失败）"))
    except Exception as e:  # noqa: BLE001
        sections.append(("🏛️ 市场结构", f"不可用：{e}"))

    # 阶段3：情绪×轮动矩阵
    try:
        from pa_mcp.research.regime_matrix import RegimeMatrixAnalyzer
        m = RegimeMatrixAnalyzer().analyze()
        sections.append(("🧭 情绪矩阵", f"{m['matrix_cell']} → {m['regime_label']}"
                                        f"（建议：{m['advice'][:40]}）"))
    except Exception as e:  # noqa: BLE001
        sections.append(("🧭 情绪矩阵", f"不可用：{e}"))

    # 阶段4：板块轮动（自动合成降级）
    try:
        from pa_mcp.research.sector_rotation import SectorRotationAnalyzer
        a = SectorRotationAnalyzer().analyze()
        if "error" not in a:
            top = "、".join(s["name"] for s in a["ranked_sectors"][:5])
            src = a.get("data_source", "")
            sections.append(("🔄 板块轮动", f"速度{a.get('rotation_speed')}；"
                                            f"强势：{top}（{src}）"))
        else:
            sections.append(("🔄 板块轮动", a["error"]))
    except Exception as e:  # noqa: BLE001
        sections.append(("🔄 板块轮动", f"不可用：{e}"))

    # 阶段5：选股（因子 + 价值动量 + 格雷厄姆）
    try:
        from pa_mcp.research.factors import select_stocks_by_factors
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        store = DuckDBStore(get_settings().database.path)
        store.connect()
        try:
            klines = {}
            for sym in pool:
                df = store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 150", [sym])
                if not df.empty:
                    klines[sym] = df
        finally:
            store.close()
        if len(klines) >= 5:
            sel = select_stocks_by_factors(klines, top_n=5,
                                           prediction_weight=0.5)
            if "error" not in sel:
                sections.append(("🎯 因子选股",
                                 "top5：" + "、".join(sel["top_symbols"])))
    except Exception as e:  # noqa: BLE001
        sections.append(("🎯 因子选股", f"不可用：{e}"))

    try:
        from pa_mcp.research.value_momentum import ValueMomentumScreen
        vm = ValueMomentumScreen().screen(pool, top_n=5)
        if "error" not in vm:
            best = "、".join(vm.get("best_candidates", [])) or "无"
            sections.append(("⚖️ 价值动量",
                             f"top5：{'、'.join(vm['top_symbols'])}；双高：{best}"))
    except Exception as e:  # noqa: BLE001
        sections.append(("⚖️ 价值动量", f"不可用：{e}"))

    try:
        from pa_mcp.research.graham import GrahamScreener
        g = GrahamScreener().screen(pool)
        if g:
            top = g[0]
            sections.append(("📗 格雷厄姆",
                             f"{top.name} 通过 {top.score}/{top.total_scored}"
                             f"（安全边际 "
                             f"{top.margin_of_safety_pct:+.0f}%）"))
    except Exception as e:  # noqa: BLE001
        sections.append(("📗 格雷厄姆", f"不可用：{e}"))

    # 阶段6：个股综合信号（前 6 只）
    sig_parts = []
    try:
        from pa_mcp.research.consensus import ConsensusAnalyzer
        for sym in pool[:6]:
            c = await ConsensusAnalyzer().analyze(sym)
            if "error" not in c:
                sig_parts.append(
                    f"{sym}: {c['signal']}（{c['level']} {c['strength']:.0%}）")
    except Exception as e:  # noqa: BLE001
        sig_parts.append(f"不可用：{e}")
    sections.append(("🧮 综合信号", "；".join(sig_parts) or "无"))

    # 阶段7：持仓风险
    try:
        from pa_mcp.research.portfolio_risk import PortfolioRiskDashboard
        pr = await PortfolioRiskDashboard().analyze(use_llm=False)
        if "error" not in pr:
            down = [h["symbol"] for h in pr["holdings"]
                    if h.get("consensus")
                    and h["consensus"]["signal"] == "down"
                    and h["consensus"]["strength"] >= 0.6]
            sections.append(("🛡️ 持仓风险",
                             f"评分 {pr['risk_score']}（{pr['risk_level']}），"
                             f"盈亏 {pr['total_pnl_pct']:+.1f}%"
                             + (f"，综合看跌：{'、'.join(down)}" if down else "")))
        else:
            sections.append(("🛡️ 持仓风险", "无持仓"))
    except Exception as e:  # noqa: BLE001
        sections.append(("🛡️ 持仓风险", f"不可用：{e}"))

    # 整合报告
    lines = [f"## 🚀 一站式分析报告（{pool[0]} 等 {len(pool)} 只）", ""]
    for title, body in sections:
        lines.append(f"### {title}")
        lines.append(body)
        lines.append("")
    lines.append("---")
    lines.append("*全部为确定性研究结果（可追溯）；研究参考，非投资建议。*")

    # 可选 LLM 综述
    llm_summary = None
    if include_llm:
        try:
            from pa_mcp.agent.llm_port import get_llm_adapter, LLMCallParams
            from pa_mcp.agent.llm_factory import ensure_llm_adapter
            adapter = ensure_llm_adapter()
            if adapter is not None:
                data = "\n".join(f"[{t}] {b}" for t, b in sections)
                params = LLMCallParams(
                    system_prompt=(
                        "你是有经验的 A 股研究综述编辑。只输出合法 JSON。"
                        "只解释给定数据，不编造。输出是研究参考，非投资建议。"),
                    user_prompt=(
                        f"基于以下一站式分析结果，给出 3 句总结、2 个关注方向、"
                        f"2 个风险点（JSON：summary/focus_areas/risks）。\n\n{data}"),
                    mode="fast", max_tokens=1500,
                )
                raw = await adapter.chat_json(params)
                if isinstance(raw, dict) and "error" not in raw and raw.get("summary"):
                    llm_summary = raw
        except Exception:
            pass

    return {
        "pool": pool,
        "sections": [{"title": t, "body": b} for t, b in sections],
        "llm_summary": llm_summary,
        "report": "\n".join(lines),
    }
