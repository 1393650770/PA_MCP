# [AI:BEGIN]
# PA_MCP - Research: 价值 × 动量 复合选股
#
# 学术依据：Asness, Moskowitz & Pedersen (2013)「Value and Momentum
# Everywhere」——价值与动量在全球市场均为稳健因子且低相关，复合
# 组合的风险调整收益优于任一单因子。
#
# 本实现（非必要不造轮子，纯复用）：
#   - 价值分：格雷厄姆评分（graham.GrahamScreener）
#   - 动量分：60 日收益截面 z-score（factors 的 ret60）
#   - 复合分 = 0.5 × 价值z + 0.5 × 动量z
#   - 四象限：价值×动量 双高 = 最佳候选（「便宜且正在走强」）
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValueMomentumScreen:
    """价值 × 动量复合选股（格雷厄姆评分 × 60 日动量）。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    def screen(self, symbols: list[str],
               quotes: Optional[dict[str, dict]] = None,
               top_n: int = 10,
               value_weight: float = 0.5) -> dict[str, Any]:
        """复合选股。

        Args:
            symbols: 股票池
            quotes: 估值快照（透传格雷厄姆筛选）
            top_n: 返回数量
            value_weight: 价值权重（动量 = 1 - value_weight）
        """
        from pa_mcp.research.graham import GrahamScreener

        symbols = [s.strip() for s in symbols if s.strip()]
        if not symbols:
            return {"error": "请输入股票代码"}

        # 1) 价值分（格雷厄姆）
        graham_results = GrahamScreener(
            self._store_path).screen(symbols, quotes=quotes)
        gmap = {r.symbol: r for r in graham_results}

        # 2) 动量分（60 日收益，截面 z-score）
        mom: dict[str, float] = {}
        for sym in symbols:
            df = self._load_kline(sym)
            if df is None or len(df) < 61:
                continue
            close = df["close"].astype(float)
            mom[sym] = float(close.iloc[-1] / close.iloc[-61] - 1) * 100
        if mom:
            vals = list(mom.values())
            mean, std = float(np.mean(vals)), float(np.std(vals))
            if std > 1e-12:
                mom = {s: (v - mean) / std for s, v in mom.items()}

        # 3) 复合
        rows = []
        for sym in symbols:
            g = gmap.get(sym)
            if g is None:
                continue
            mz = mom.get(sym)
            # 价值分标准化（0-1 → z 近似：score/total）
            gz = (g.score / g.total_scored - 0.5) * 2 if g.total_scored else 0.0
            if mz is None:
                continue
            score = value_weight * gz + (1 - value_weight) * mz
            # 象限
            v_hi = gz >= 0
            m_hi = mz >= 0
            if v_hi and m_hi:
                quadrant = "价值+动量双高"
            elif v_hi:
                quadrant = "价值高/动量弱"
            elif m_hi:
                quadrant = "动量强/价值弱"
            else:
                quadrant = "双弱"
            rows.append({
                "symbol": sym, "name": g.name,
                "value_score": g.score, "value_total": g.total_scored,
                "value_z": round(gz, 3),
                "momentum_z": round(mz, 3),
                "composite": round(score, 4),
                "quadrant": quadrant,
                "rating": g.rating,
                "margin_of_safety_pct": g.margin_of_safety_pct,
            })

        if not rows:
            return {"error": "无足够数据（需财务/估值 + ≥61 根行情）"}
        rows.sort(key=lambda r: r["composite"], reverse=True)
        top = rows[:top_n]
        best = [r for r in top if r["quadrant"] == "价值+动量双高"]
        return {
            "method": (f"价值×动量复合：{value_weight:.0%}×格雷厄姆评分 + "
                       f"{1 - value_weight:.0%}×60日动量 z-score"
                       "（Asness et al. 2013）"),
            "n_scored": len(rows),
            "selection": top,
            "top_symbols": [r["symbol"] for r in top],
            "best_candidates": [r["symbol"] for r in best],
            "note": "「价值+动量双高」= 便宜且正在走强（复合最优象限）",
        }

    def _load_kline(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            from pa_mcp.config import get_settings
            from pa_mcp.data.store import DuckDBStore
            path = self._store_path or get_settings().database.path
            store = DuckDBStore(path)
            store.connect()
            try:
                df = store.query_df(
                    "SELECT date, close FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 80", [symbol])
                return df.sort_values("date").reset_index(drop=True) \
                    if not df.empty else None
            finally:
                store.close()
        except Exception:
            return None


_screener: Optional[ValueMomentumScreen] = None


def get_value_momentum_screen() -> ValueMomentumScreen:
    """单例获取复合筛选器。"""
    global _screener
    if _screener is None:
        _screener = ValueMomentumScreen()
    return _screener


def format_value_momentum(result: dict[str, Any]) -> str:
    """复合结果 → markdown。"""
    if "error" in result:
        return f"价值×动量复合不可用：{result['error']}"
    lines = [
        f"## ⚖️ 价值 × 动量 复合选股",
        f"**方法**：{result['method']}",
        f"**样本**：{result['n_scored']} 只完成评分",
        "",
        "| 排名 | 代码 | 名称 | 复合分 | 价值z | 动量z | 象限 | 评级 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(result["selection"], 1):
        lines.append(
            f"| {i} | {r['symbol']} | {r['name']} | **{r['composite']:+.3f}** | "
            f"{r['value_z']:+.2f} | {r['momentum_z']:+.2f} | "
            f"{r['quadrant']} | {r['rating']} |")
    if result.get("best_candidates"):
        lines.append("\n**最佳候选**（价值+动量双高）："
                     + "、".join(result["best_candidates"]))
    lines.append(f"\n*{result['note']}。研究参考，非投资建议。*")
    return "\n".join(lines)
