# [AI:BEGIN]
# PA_MCP - Research: AI 市场研究报告（确定性研究 → LLM 综述）
#
# LLM 融合的收尾形态：把全部确定性研究结果（市场诊断/情绪矩阵/板块
# 轮动/因子选股/价值动量/预测）聚合 → LLM 综述为自然语言投资环境
# 报告（结构化 JSON：总结/关注/风险/建议）；无 LLM 时确定性模板降级。
#
# 原则：LLM 只解释确定性结果，不产生新数据（不编造）。
# [AI:END]

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPORT_PROMPT = """你是 A 股市场研究综述编辑。基于以下确定性研究数据，撰写一份简明的市场环境报告。

【市场状态】
{market_state}

【情绪×轮动矩阵】
{matrix}

【板块轮动】
{sector_rotation}

【因子选股（含 AI 预测融合）】
{factor_selection}

【价值×动量复合】
{value_momentum}

【持仓风险】
{portfolio_risk}

【预测验证成绩单】
{prediction_eval}

【报告要求】
1. 只输出一个 JSON 对象（不要 markdown）
2. summary：3-5 句市场环境总结（中文，基于数据不编造）
3. focus_areas：2-3 个值得关注的板块/方向（含依据）
4. risks：2-3 个风险点（含依据）
5. suggestions：2-3 条研究性操作思路（注明是研究参考）
6. confidence：0-1（数据完整性置信度）

【JSON 格式】
{{
  "summary": "...",
  "focus_areas": [{{"area": "...", "reason": "..."}}],
  "risks": [{{"risk": "...", "reason": "..."}}],
  "suggestions": ["..."],
  "confidence": 0.7
}}"""


class AiMarketReport:
    """AI 市场研究报告生成器（聚合 + LLM 综述）。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    # ---- 数据聚合 ----
    def _collect(self, symbols: list[str]) -> dict[str, str]:
        """收集各研究模块的摘要文本（best-effort，失败给空段）。"""
        out: dict[str, str] = {}

        # 市场状态（确定性诊断）
        try:
            from pa_mcp.agent.orchestrator import AgentOrchestrator
            d = AgentOrchestrator()._diagnosis_deterministic(None)
            out["market_state"] = (
                f"状态 {d['market_state_zh']}，风险 {d['risk_level']}，"
                f"总仓位上限 {d['suggested_max_position_pct']}%；"
                + "；".join(d["key_observations"]))
        except Exception:
            out["market_state"] = "无数据"

        # 情绪×轮动矩阵
        try:
            from pa_mcp.research.regime_matrix import RegimeMatrixAnalyzer
            m = RegimeMatrixAnalyzer(self._store_path).analyze()
            out["matrix"] = (f"{m['matrix_cell']} → {m['regime_label']}；"
                             f"建议：{m['advice']}；风险：{m['risk']}")
        except Exception:
            out["matrix"] = "无数据"

        # 板块轮动
        try:
            from pa_mcp.research.sector_rotation import SectorRotationAnalyzer
            a = SectorRotationAnalyzer(self._store_path).analyze()
            if "error" not in a:
                top = "、".join(s["name"] for s in a["ranked_sectors"][:5])
                out["sector_rotation"] = (
                    f"轮动速度 {a.get('rotation_speed')}，"
                    f"强势板块：{top}，新进："
                    + "、".join(s["name"] for s in a.get("rotated_in", [])[:3])
                    or "无")
            else:
                out["sector_rotation"] = "无板块数据"
        except Exception:
            out["sector_rotation"] = "无数据"

        # 因子选股（预测融合 50%）
        try:
            from pa_mcp.research.factors import select_stocks_by_factors
            klines = self._load_klines(symbols)
            if len(klines) >= 5:
                sel = select_stocks_by_factors(klines, top_n=5,
                                               prediction_weight=0.5)
                if "error" not in sel:
                    out["factor_selection"] = (
                        f"top5：{'、'.join(sel['top_symbols'])}（{sel['method']}）")
                else:
                    out["factor_selection"] = f"因子选股不可用：{sel['error']}"
            else:
                out["factor_selection"] = f"股票池行情不足（{len(klines)} < 5）"
        except Exception as e:
            out["factor_selection"] = f"因子选股失败：{str(e)[:80]}"

        # 价值×动量
        try:
            from pa_mcp.research.value_momentum import ValueMomentumScreen
            vm = ValueMomentumScreen(self._store_path).screen(symbols, top_n=5)
            if "error" not in vm:
                out["value_momentum"] = (
                    f"top5：{'、'.join(vm['top_symbols'])}；双高候选："
                    + "、".join(vm.get("best_candidates", []) or ["无"]))
            else:
                out["value_momentum"] = f"价值动量不可用：{vm['error']}"
        except Exception as e:
            out["value_momentum"] = f"价值动量失败：{str(e)[:80]}"

        # 预测验证成绩单
        try:
            from pa_mcp.agent.prediction import PredictionService
            import asyncio
            s = asyncio.run(PredictionService(
                self._store_path).evaluate_predictions())
            out["prediction_eval"] = (
                f"已评估 {s.get('evaluated', 0)} 条，命中率 "
                f"{s.get('hit_rate', '—')}"
                + (f"，IC {s['ic']}" if s.get("ic") is not None else ""))
        except Exception:
            out["prediction_eval"] = "无预测验证数据"

        # 持仓风险（best-effort，无持仓时跳过）
        try:
            from pa_mcp.research.portfolio_risk import PortfolioRiskDashboard
            import asyncio
            pr = asyncio.run(PortfolioRiskDashboard(
                self._store_path).analyze(use_llm=False))
            if "error" not in pr:
                down = [h["symbol"] for h in pr["holdings"]
                        if h.get("prediction")
                        and h["prediction"]["direction"] == "down"]
                out["portfolio_risk"] = (
                    f"组合市值 {pr['total_value']:,.0f} 元，盈亏 "
                    f"{pr['total_pnl_pct']:+.1f}%，风险评分 "
                    f"{pr['risk_score']}（{pr['risk_level']}），"
                    f"单票最大 {pr['concentration']['top_weight_pct']:.0f}%"
                    + (f"，预测看跌：{'、'.join(down)}" if down else ""))
            else:
                out["portfolio_risk"] = "无持仓数据"
        except Exception:
            out["portfolio_risk"] = "无持仓数据"

        return out

    def _load_klines(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        path = self._store_path or get_settings().database.path
        store = DuckDBStore(path)
        store.connect()
        try:
            out = {}
            for sym in symbols:
                df = store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 150", [sym])
                if not df.empty:
                    out[sym] = df
            return out
        finally:
            store.close()

    # ---- 报告生成 ----
    async def generate(self, symbols: list[str]) -> dict[str, Any]:
        """生成 AI 市场研究报告。

        Returns:
            sections: 各确定性数据段
            llm_analysis: LLM 综述（无 LLM 时为 None）
            report: 完整 markdown
        """
        sections = self._collect(symbols)
        llm_analysis = None
        try:
            from pa_mcp.agent.llm_port import get_llm_adapter, LLMCallParams
            adapter = get_llm_adapter()
            if adapter is not None:
                user_prompt = REPORT_PROMPT.format(
                    market_state=sections.get("market_state", "无"),
                    matrix=sections.get("matrix", "无"),
                    sector_rotation=sections.get("sector_rotation", "无"),
                    factor_selection=sections.get("factor_selection", "无"),
                    value_momentum=sections.get("value_momentum", "无"),
                    portfolio_risk=sections.get("portfolio_risk", "无"),
                    prediction_eval=sections.get("prediction_eval", "无"),
                )
                params = LLMCallParams(
                    system_prompt=(
                        "你是有经验的 A 股研究综述编辑。只输出合法 JSON。"
                        "只解释给定数据，不编造新事实。输出是研究参考，非投资建议。"
                    ),
                    user_prompt=user_prompt, mode="fast", max_tokens=1200,
                )
                raw = await adapter.chat_json(params)
                if isinstance(raw, dict) and "error" not in raw \
                        and raw.get("summary"):
                    llm_analysis = {
                        "summary": raw["summary"],
                        "focus_areas": raw.get("focus_areas", []),
                        "risks": raw.get("risks", []),
                        "suggestions": raw.get("suggestions", []),
                        "confidence": raw.get("confidence", 0.5),
                    }
        except Exception as e:  # noqa: BLE001
            logger.warning("AI report LLM failed, template fallback: %s", e)

        return {
            "date": date.today().isoformat(),
            "sections": sections,
            "llm_analysis": llm_analysis,
            "report": self._render(sections, llm_analysis),
        }

    @staticmethod
    def _render(sections: dict[str, str],
                llm: Optional[dict]) -> str:
        lines = [
            f"## 📋 AI 市场研究报告（{date.today().isoformat()}）",
            f"- **市场状态**：{sections.get('market_state', '—')}",
            f"- **情绪×轮动**：{sections.get('matrix', '—')}",
            f"- **板块轮动**：{sections.get('sector_rotation', '—')}",
            f"- **因子选股**：{sections.get('factor_selection', '—')}",
            f"- **价值×动量**：{sections.get('value_momentum', '—')}",
            f"- **持仓风险**：{sections.get('portfolio_risk', '—')}",
            f"- **预测验证**：{sections.get('prediction_eval', '—')}",
        ]
        if llm:
            lines += [
                "",
                "## 🤖 AI 综述",
                llm["summary"],
            ]
            if llm.get("focus_areas"):
                lines.append("\n**关注方向**：")
                for f in llm["focus_areas"]:
                    lines.append(f"- **{f.get('area', '')}**：{f.get('reason', '')}")
            if llm.get("risks"):
                lines.append("\n**风险点**：")
                for r in llm["risks"]:
                    lines.append(f"- {r.get('risk', '')}：{r.get('reason', '')}")
            if llm.get("suggestions"):
                lines.append("\n**研究思路**：")
                for s in llm["suggestions"]:
                    lines.append(f"- {s}")
            lines.append(f"\n*综述置信度 {llm.get('confidence', 0.5):.0%}*")
        lines.append("\n*研究参考，非投资建议。*")
        return "\n".join(lines)


_report_gen: Optional[AiMarketReport] = None


def get_ai_report_generator() -> AiMarketReport:
    """单例获取研报生成器。"""
    global _report_gen
    if _report_gen is None:
        _report_gen = AiMarketReport()
    return _report_gen
