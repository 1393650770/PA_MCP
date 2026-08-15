# [AI:BEGIN]
# PA_MCP - Research: 情绪 × 轮动 联合矩阵（Regime Matrix）
#
# 两个已有分析的合成视图（非必要不造轮子，纯复用）：
#   - 游资情绪周期（sentiment_cycle）：启动/发酵/高潮/退潮/冰点
#   - 板块轮动速度（sector_rotation）：高/中/低
#   合成 9 格矩阵 → 市场状态标签 + 操作建议 + 风险提示
#
# 方法论：游资情绪周期 × 动量轮动（强者恒强/高低切换）的经典组合——
# 「情绪决定做不做，轮动决定做什么」。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 矩阵定义：{情绪阶段: {轮动速度: (标签, 操作建议, 风险)}}
REGIME_MATRIX: dict[str, dict[str, tuple[str, str, str]]] = {
    "starting": {
        "low": ("布局期", "左侧低吸超跌龙头，试探性建仓，仓位 ≤30%",
                "情绪刚回暖，主线未明，容易反复"),
        "medium": ("回暖期", "小仓试探参与首板梯队，跟随题材启动",
                   "确认信号不足，谨防一日游"),
        "high": ("躁动期", "题材快速启动，快进快出（首板/2板），仓位 ≤40%",
                 "轮动过快，追高易吃面"),
    },
    "fermenting": {
        "low": ("主线聚焦", "主线明确：重仓强势板块龙头（仓位 ≤50%）",
                "主线高度集中，警惕单一板块回撤"),
        "medium": ("扩散期", "主线 + 轮动双线操作，跟随板块 RS 排名",
                   "扩散期尾声可能出现分化"),
        "high": ("亢奋期", "热点高速扩散，控制追高，仓位 ≤40%",
                 "赚钱效应强但分歧加大，注意情绪退潮前兆"),
    },
    "climax": {
        "low": ("高位抱团", "只持有不追高，逐步兑现盈利",
                "抱团松动时踩踏风险大"),
        "medium": ("分歧顶", "减仓为主，仅保留最强龙头",
                   "连板高度见顶，分歧转一致后再转分歧"),
        "high": ("亢奋顶", "坚决兑现，仓位降至 ≤20%",
                 "情绪高潮 + 高速轮动 = 最后一波博弈，风险极高"),
    },
    "recess": {
        "low": ("收缩期", "空仓等待，不参与反抽",
                "亏钱效应扩散，反抽是逃命机会"),
        "medium": ("退潮中", "仅做低位补涨/超跌反弹，仓位 ≤20%",
                   "退潮期反弹持续性差"),
        "high": ("高低切换", "只做低位轮动补涨，规避高位股",
                 "切换失败则普跌，控制仓位"),
    },
    "ice": {
        "low": ("冰点", "空仓等待情绪修复（涨停回暖/连板高度回升）",
                "跌停潮未止，抄底危险"),
        "medium": ("冰点", "空仓等待，观察晋级率是否回升",
                   "情绪冰点期无主线"),
        "high": ("冰点异动", "极轻仓观察异动板块（≤10%），等待确认",
                 "冰点+高速轮动 = 资金试错，方向未定"),
    },
}


class RegimeMatrixAnalyzer:
    """情绪 × 轮动联合矩阵分析（复用 sentiment_cycle + sector_rotation）。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    def _read_sentiment_cached(self) -> dict[str, Any]:
        """从 sentiment_daily 缓存读最新情绪（绕开 kline 依赖，纯缓存）。

        阶段判定复用 SentimentCycleAnalyzer._stage（与主分析一致）。
        """
        from pa_mcp.research.sentiment_cycle import SentimentCycleAnalyzer
        analyzer = SentimentCycleAnalyzer(self._store_path)
        store = analyzer._store()
        try:
            rows = store.query_df(
                "SELECT * FROM sentiment_daily ORDER BY date DESC LIMIT 2", [])
            if rows.empty:
                return {}
            latest = rows.iloc[0]
            stats = {k: (str(latest[k])[:10] if k == "date"
                         else float(latest[k]) if k in (
                             "promotion_rate", "sentiment_score")
                         else int(latest[k]))
                     for k in ("date", "limit_up_count", "limit_down_count",
                               "max_board_height", "board2_count",
                               "board3_count", "board4p_count",
                               "first_board_count", "promotion_rate",
                               "sentiment_score")
                     if k in latest.index and latest[k] is not None}
            prev_stats = None
            if len(rows) > 1:
                p = rows.iloc[1]
                prev_stats = {k: (float(p[k]) if k in (
                    "promotion_rate", "sentiment_score") else int(p[k]))
                              for k in ("limit_up_count", "limit_down_count",
                                        "max_board_height", "promotion_rate",
                                        "sentiment_score")
                              if k in p.index and p[k] is not None}
            stage, stage_zh = SentimentCycleAnalyzer._stage(stats, prev_stats)
            return {**stats, "stage": stage, "stage_zh": stage_zh}
        finally:
            store.close()

    def analyze(self) -> dict[str, Any]:
        """获取当前矩阵位置 + 操作建议。

        任一模块数据缺失时：该维度标 unknown，矩阵给出降级结论。
        """
        from pa_mcp.research.sentiment_cycle import SentimentCycleAnalyzer
        from pa_mcp.research.sector_rotation import SectorRotationAnalyzer

        sentiment: dict = {}
        rotation: dict = {}
        try:
            sentiment = self._read_sentiment_cached()
        except Exception as e:  # noqa: BLE001
            logger.warning("sentiment unavailable in matrix", error=str(e))
        try:
            rotation = SectorRotationAnalyzer(
                self._store_path).analyze()
        except Exception as e:  # noqa: BLE001
            logger.warning("rotation unavailable in matrix", error=str(e))

        # 情绪缓存缺失时回退实时分析（需 kline 数据）
        if not sentiment:
            try:
                sentiment = SentimentCycleAnalyzer(
                    self._store_path).sentiment_summary()
            except Exception:  # noqa: BLE001
                sentiment = {}

        stage = sentiment.get("stage", "unknown")
        speed_zh = rotation.get("rotation_speed", "unknown")
        # 中文轮动速度 → 矩阵 key（英文）
        speed = {"高": "high", "中": "medium", "低": "low"}.get(
            speed_zh, "unknown")

        label, advice, risk = "—", "数据不足，先装载行情/板块数据", "—"
        if stage in REGIME_MATRIX and speed in REGIME_MATRIX[stage]:
            label, advice, risk = REGIME_MATRIX[stage][speed]
        elif stage == "unknown" and speed != "unknown":
            label = "板块主导"
            advice = "无情绪数据：以板块轮动为准，跟随 RS 排名"
            risk = "情绪维度缺失，判断不完整"
        elif stage != "unknown" and speed == "unknown":
            label = "情绪主导"
            advice = "无板块数据：以情绪阶段为准"
            risk = "轮动维度缺失，判断不完整"

        return {
            "sentiment_stage": stage,
            "sentiment_stage_zh": sentiment.get("stage_zh", stage),
            "sentiment_score": sentiment.get("sentiment_score"),
            "rotation_speed": speed_zh,
            "rotation_speed_note": rotation.get("note", ""),
            "matrix_cell": f"{sentiment.get('stage_zh', stage)} × {speed_zh}轮动",
            "regime_label": label,
            "advice": advice,
            "risk": risk,
            "matrix": REGIME_MATRIX,
        }


_analyzer: Optional[RegimeMatrixAnalyzer] = None


def get_regime_analyzer() -> RegimeMatrixAnalyzer:
    """单例获取矩阵分析器。"""
    global _analyzer
    if _analyzer is None:
        _analyzer = RegimeMatrixAnalyzer()
    return _analyzer


def format_matrix(result: dict[str, Any]) -> str:
    """矩阵结果 → markdown（UI/MCP 共用）。"""
    lines = [
        f"## 🧭 情绪 × 轮动矩阵（{result['matrix_cell']}）",
        f"**当前位置**：**{result['regime_label']}**",
        f"- **情绪**：{result['sentiment_stage_zh']}"
        f"（分 {result['sentiment_score']}）"
        if result["sentiment_score"] is not None else
        f"- **情绪**：{result['sentiment_stage_zh']}",
        f"- **轮动速度**：{result['rotation_speed']}",
        f"- **操作建议**：{result['advice']}",
        f"- **风险提示**：{result['risk']}",
        "",
        "### 全矩阵参考",
        "| 情绪 \\ 轮动 | 低 | 中 | 高 |",
        "|---|---|---|---|",
    ]
    stage_zh = {"starting": "启动期", "fermenting": "发酵期",
                "climax": "高潮期", "recess": "退潮期", "ice": "冰点期"}
    for st, speeds in REGIME_MATRIX.items():
        cells = [speeds[sp][0] for sp in ("low", "medium", "high")]
        lines.append(f"| {stage_zh[st]} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("\n*「情绪决定做不做，轮动决定做什么」。研究参考，非投资建议。*")
    return "\n".join(lines)
