# [AI:BEGIN]
# PA_MCP - Research: 综合决策信号（多信号源加权投票融合）
#
# 信号融合（借鉴 ensemble voting 思想，非必要不造轮子——全部复用已有模块）：
#   1. 多周期共振（强度 ×3）          — 趋势确认
#   2. 预测 5d（方向概率 ×2）         — 量化预测
#   3. 策略信号（近 10 日买入/卖出 ×2）— 规则策略
#   4. 缠论背驰（×1）                — 大牛方法
#   5. 市场结构（大盘 bias ×2）       — 环境
#   加权投票 → 综合方向 / 强度 / 一致度
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

WEIGHTS = {
    "resonance": 3.0,
    "prediction": 2.0,
    "strategy": 2.0,
    "chan": 1.0,
    "market": 2.0,
}


class ConsensusAnalyzer:
    """综合决策信号（多信号源加权投票）。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    async def analyze(self, symbol: str,
                      kline_df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
        """综合信号。

        Args:
            symbol: 股票代码
            kline_df: 可选行情（缺省从库拉）
        """
        from pa_mcp.agent.prediction import PredictionService

        if kline_df is None:
            kline_df = self._load_kline(symbol)
        if kline_df is None or kline_df.empty:
            return {"error": f"{symbol} 无行情数据"}

        sources: dict[str, dict] = {}

        # 1) 多周期共振
        try:
            from pa_mcp.research.resonance import ResonanceAnalyzer
            res = await ResonanceAnalyzer(self._store_path).analyze(
                symbol, kline_df=kline_df)
            if "error" not in res:
                sources["resonance"] = {
                    "signal": res["signal"],
                    "strength": res["strength"],
                    "detail": res["resonance"],
                }
        except Exception:
            pass

        # 2) 预测 5d
        try:
            p = (await PredictionService(self._store_path).predict(
                symbol, kline_df, horizon="5d", use_llm=False)).to_dict()
            sources["prediction"] = {
                "signal": p["direction"],
                "strength": p["probability"],
                "detail": f"期望 {p['expected_return_pct']:+.1f}%",
            }
        except Exception:
            pass

        # 3) 策略信号（近 10 日注册策略信号计数）
        try:
            from pa_mcp.engine.strategies.base import StrategyRegistry
            from pa_mcp.engine.strategies.base import SignalDirection
            registry = StrategyRegistry()
            registry.auto_discover()
            last_date = str(kline_df["date"].iloc[-1])[:10]
            cutoff = (pd.to_datetime(last_date) - pd.Timedelta(days=15))\
                .strftime("%Y-%m-%d")
            n_buy = n_sell = 0
            for name in registry.list_all():
                try:
                    sigs = registry.get(name).generate_signals(kline_df.copy())
                    for s in sigs:
                        st = getattr(s, "signal_time", None) \
                            or str(getattr(s, "timestamp", ""))[:10]
                        if st and cutoff <= str(st)[:10] <= last_date:
                            d = getattr(s, "direction", None)
                            dv = d.value if hasattr(d, "value") else str(d)
                            if dv == "bullish":
                                n_buy += 1
                            elif dv == "bearish":
                                n_sell += 1
                except Exception:
                    continue
            if n_buy or n_sell:
                signal = "up" if n_buy > n_sell else "down"
                strength = min(1.0, max(n_buy, n_sell) / 3)
                sources["strategy"] = {
                    "signal": signal, "strength": strength,
                    "detail": f"近15日 买入{n_buy}/卖出{n_sell} 信号",
                }
        except Exception:
            pass

        # 4) 缠论背驰
        try:
            from pa_mcp.engine.indicators.chan import chan_analysis
            c = chan_analysis(kline_df)
            if c.beichi_signal in ("bullish", "bearish"):
                sources["chan"] = {
                    "signal": "up" if c.beichi_signal == "bullish" else "down",
                    "strength": 0.8,
                    "detail": c.beichi_detail[:50],
                }
        except Exception:
            pass

        # 5) 市场结构（大盘）
        try:
            from pa_mcp.research.market_structure import (
                MarketStructureAnalyzer)
            ms = await MarketStructureAnalyzer(
                self._store_path).analyze(use_network=False)
            if ms["index"]["rows"] > 0:
                bias = ms["joint"]["bias"]
                if bias == "偏多":
                    sources["market"] = {"signal": "up", "strength": 0.7,
                                         "detail": "指数多头结构"}
                elif bias == "偏空":
                    sources["market"] = {"signal": "down", "strength": 0.7,
                                         "detail": "指数空头结构"}
        except Exception:
            pass

        if not sources:
            return {"error": "无可用信号源"}

        # 加权投票
        votes = {"up": 0.0, "down": 0.0, "sideways": 0.0}
        total_w = 0.0
        for name, src in sources.items():
            w = WEIGHTS.get(name, 1.0)
            sig = src["signal"]
            strength = max(0.1, min(1.0, float(src["strength"])))
            votes[sig] = votes.get(sig, 0.0) + w * strength
            total_w += w
        final = max(votes, key=votes.get)
        final_score = votes[final] / total_w if total_w > 0 else 0.0
        # 一致度：最大票占比
        agreement = votes[final] / max(sum(votes.values()), 1e-9)

        level = ("强" if final_score >= 0.6 else "中" if final_score >= 0.4
                 else "弱")
        return {
            "symbol": symbol,
            "sources": sources,
            "votes": {k: round(v, 2) for k, v in votes.items()},
            "signal": final,
            "strength": round(final_score, 2),
            "level": level,
            "agreement": round(agreement, 2),
            "note": ("加权投票（共振×3/预测×2/策略×2/背驰×1/大盘×2）。"
                     "研究参考，非投资建议。"),
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
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 200", [symbol])
                return df if not df.empty else None
            finally:
                store.close()
        except Exception:
            return None


_analyzer: Optional[ConsensusAnalyzer] = None


def get_consensus_analyzer() -> ConsensusAnalyzer:
    """单例获取综合信号分析器。"""
    global _analyzer
    if _analyzer is None:
        _analyzer = ConsensusAnalyzer()
    return _analyzer


def format_consensus(result: dict[str, Any]) -> str:
    """综合信号 → markdown。"""
    if "error" in result:
        return f"综合信号不可用：{result['error']}"
    dir_zh = {"up": "📈 看涨", "down": "📉 看跌", "sideways": "➡️ 震荡"}
    lines = [
        f"## 🧮 综合决策信号：{result['symbol']}",
        f"**{dir_zh.get(result['signal'], result['signal'])}"
        f"（{result['level']}强度，{result['strength']:.0%}，"
        f"一致度 {result['agreement']:.0%}）**",
        "",
        "### 信号源",
        "| 来源 | 方向 | 强度 | 明细 |",
        "|---|---|---|---|",
    ]
    for name, src in result["sources"].items():
        label = {"resonance": "多周期共振", "prediction": "预测5d",
                 "strategy": "策略信号", "chan": "缠论背驰",
                 "market": "市场结构"}.get(name, name)
        lines.append(
            f"| {label} | {dir_zh.get(src['signal'], src['signal'])} | "
            f"{float(src['strength']):.0%} | {src.get('detail', '')} |")
    lines.append(
        f"\n**投票**：看涨 {result['votes']['up']:.2f} / "
        f"看跌 {result['votes']['down']:.2f} / "
        f"震荡 {result['votes']['sideways']:.2f}")
    lines.append(f"\n*{result['note']}*")
    return "\n".join(lines)
