# [AI:BEGIN]
# PA_MCP - Research: 市场结构联合分析（指数缠论 × 情绪矩阵）
#
# 大牛方法应用到市场层面：
#   - 指数缠论结构：上证指数分型/笔/中枢/背驰（复用 chan_analysis）
#   - 情绪×轮动矩阵：市场状态标签（复用 RegimeMatrixAnalyzer）
#   联合解读：指数结构方向 × 情绪阶段 → 市场环境综合判断
#
# 数据：上证指数（sh000001，多源 router 拉取，index_daily 兜底）。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

INDEX_SYMBOL = "sh000001"


class MarketStructureAnalyzer:
    """指数缠论结构 × 情绪矩阵联合分析。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    async def analyze(self, use_network: bool = True) -> dict[str, Any]:
        """市场结构联合分析。

        Args:
            use_network: 库中无指数数据时是否走网络拉取（测试传 False
                避免慢速重试）
        """
        # 1) 上证指数行情
        index_df = await self._load_index(use_network=use_network)
        # 2) 缠论结构
        chan = None
        chan_err = None
        if index_df is not None and not index_df.empty:
            try:
                from pa_mcp.engine.indicators.chan import chan_analysis
                chan = chan_analysis(index_df, symbol="上证指数")
            except Exception as e:  # noqa: BLE001
                chan_err = str(e)[:80]
        # 3) 情绪矩阵
        matrix = None
        try:
            from pa_mcp.research.regime_matrix import RegimeMatrixAnalyzer
            matrix = RegimeMatrixAnalyzer(self._store_path).analyze()
        except Exception as e:  # noqa: BLE001
            logger.warning("matrix unavailable: %s", e)

        # 4) 联合判断
        joint = self._joint(chan, matrix)

        return {
            "index": {
                "symbol": INDEX_SYMBOL,
                "rows": len(index_df) if index_df is not None else 0,
                "last_close": round(float(index_df["close"].iloc[-1]), 2)
                if index_df is not None and not index_df.empty else None,
                "last_date": str(index_df["date"].iloc[-1])[:10]
                if index_df is not None and not index_df.empty else None,
            },
            "chan": chan.to_dict() if chan is not None else None,
            "chan_error": chan_err,
            "matrix": matrix if matrix is not None else None,
            "joint": joint,
            "report": self._render(chan, matrix, joint, index_df),
        }

    async def _load_index(self, use_network: bool = True) -> Optional[pd.DataFrame]:
        """上证指数行情：index_daily 兜底 → 多源 router。"""
        try:
            from pa_mcp.config import get_settings
            from pa_mcp.data.store import DuckDBStore
            path = self._store_path or get_settings().database.path
            store = DuckDBStore(path)
            store.connect()
            try:
                df = store.query_df(
                    "SELECT date, open, high, low, close, volume "
                    "FROM index_daily WHERE symbol = ? ORDER BY date",
                    [INDEX_SYMBOL])
                if not df.empty and len(df) >= 100:
                    return df
            finally:
                store.close()
        except Exception:
            pass
        if not use_network:
            return None
        try:
            from datetime import date, timedelta
            from pa_mcp.config import get_settings
            from pa_mcp.data.router import DataSourceRouter
            from pa_mcp.data.sources.akshare_adapter import AKShareAdapter
            from pa_mcp.data.sources.sina_adapter import SinaAdapter
            from pa_mcp.data.sources.tencent_adapter import TencentAdapter
            from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter
            s = get_settings()
            router = DataSourceRouter(
                [("akshare", AKShareAdapter()), ("sina", SinaAdapter()),
                 ("tencent", TencentAdapter()),
                 ("eastmoney", EastMoneyAdapter())],
                {n: s.router.circuit for n in
                 ("akshare", "sina", "tencent", "eastmoney")})
            end = date.today().isoformat()
            start = (date.today() - timedelta(days=400)).isoformat()
            df, _ = await router.fetch_daily_kline(
                INDEX_SYMBOL, start_date=start, end_date=end)
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            logger.warning("index fetch failed: %s", e)
        return None

    @staticmethod
    def _joint(chan, matrix: Optional[dict]) -> dict[str, str]:
        """联合判断：指数方向 × 情绪阶段 → 环境结论。"""
        if chan is None:
            return {"verdict": "数据不足", "detail": "指数行情不可用"}
        pos = chan.position
        sig = chan.beichi_signal
        stage = matrix.get("sentiment_stage", "unknown") if matrix else "unknown"
        regime = matrix.get("regime_label", "") if matrix else ""

        # 指数结构方向
        if pos == "中枢上方" and sig == "none":
            struct = "多头结构（指数站上中枢，无背驰）"
        elif pos == "中枢上方" and sig == "bearish":
            struct = "多头结构但上涨背驰——涨势衰竭预警"
        elif pos == "中枢下方" and sig == "bullish":
            struct = "空头结构但下跌背驰——跌势衰竭，关注反转"
        elif pos == "中枢下方":
            struct = "空头结构（指数跌破中枢）"
        elif sig == "bearish":
            struct = "中枢内但上涨背驰"
        elif sig == "bullish":
            struct = "中枢内但下跌背驰"
        else:
            struct = "中枢内震荡"

        # 情绪配合
        if stage == "frozen":
            env = "情绪冰点——即使结构转好也需等待确认"
        elif stage in ("climax",) and sig == "bearish":
            env = "情绪高潮 + 指数背驰——风险信号叠加，警惕兑现"
        elif stage == "recess":
            env = "情绪退潮——指数结构以防守为主"
        elif stage in ("fermenting", "starting"):
            env = "情绪回暖——结构转好时可积极跟踪"
        else:
            env = "情绪中性——跟随指数结构"

        verdict = f"{struct}；{env}"
        bias = ("偏多" if ("多头" in struct and "衰竭" not in struct
                           and "冰点" not in env)
                else "偏空" if ("空头" in struct or "高潮" in env
                               and "背驰" in struct)
                else "中性")
        return {"verdict": verdict, "bias": bias,
                "structure": struct, "environment": env,
                "regime": regime}

    @staticmethod
    def _render(chan, matrix: Optional[dict], joint: dict,
                index_df: Optional[pd.DataFrame]) -> str:
        lines = [
            "## 🏛️ 市场结构联合分析（上证指数 × 情绪矩阵）",
            f"- **指数**：{joint.get('detail', '—')}"
            if index_df is None or index_df.empty else
            f"- **指数**：{INDEX_SYMBOL} 收 "
            f"{round(float(index_df['close'].iloc[-1]), 2)}"
            f"（{str(index_df['date'].iloc[-1])[:10]}）",
            f"- **联合判断**：**{joint.get('bias', '—')}** —— {joint.get('verdict', '—')}",
        ]
        if chan is not None:
            lines.extend([
                f"- **缠论结构**：{chan.position}；"
                f"笔 {len(chan.bi_list)} 段 / 中枢 {len(chan.zhongshu_list)} 个；"
                f"背驰：{chan.beichi_detail[:60]}",
            ])
        if matrix is not None:
            lines.append(
                f"- **情绪矩阵**：{matrix.get('matrix_cell', '—')} → "
                f"{matrix.get('regime_label', '—')}（{matrix.get('advice', '')[:40]}）")
        if joint.get("regime"):
            lines.append(f"- **当前 regime**：{joint['regime']}")
        lines.append("\n*指数结构（缠论）× 情绪阶段（游资）双重验证。"
                     "研究参考，非投资建议。*")
        return "\n".join(lines)


_analyzer: Optional[MarketStructureAnalyzer] = None


def get_market_structure() -> MarketStructureAnalyzer:
    """单例获取市场结构分析器。"""
    global _analyzer
    if _analyzer is None:
        _analyzer = MarketStructureAnalyzer()
    return _analyzer
