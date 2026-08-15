# [AI:BEGIN]
# PA_MCP - Research: 多周期预测共振（1d/5d/20d 方向一致性）
#
# 多周期确认（源自技术分析共振理念 + 预测验证闭环）：
#   1d / 5d / 20d 三周期预测同向 → 强共振（趋势确认）
#   方向分歧 → 观望（周期打架 = 趋势不明）
#
# 复用 PredictionService.predict（确定性模式控成本）。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

HORIZONS = ("1d", "5d", "20d")


class ResonanceAnalyzer:
    """多周期预测共振分析。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    async def analyze(self, symbol: str,
                      kline_df: Optional[pd.DataFrame] = None,
                      use_llm: bool = False) -> dict[str, Any]:
        """三周期预测共振。

        Args:
            symbol: 股票代码
            kline_df: 可选行情（缺省从 store 拉）
            use_llm: 预测是否用 LLM（默认确定性控成本）
        """
        from pa_mcp.agent.prediction import PredictionService
        svc = PredictionService(self._store_path)
        if kline_df is None:
            kline_df = self._load_kline(symbol)
        if kline_df is None or kline_df.empty:
            return {"error": f"{symbol} 无行情数据"}

        periods = []
        for h in HORIZONS:
            try:
                r = await svc.predict(symbol, kline_df, horizon=h,
                                      use_llm=use_llm)
                p = r.to_dict()
                periods.append({
                    "horizon": h,
                    "direction": p["direction"],
                    "probability": p["probability"],
                    "expected_return_pct": p["expected_return_pct"],
                    "cycle": p["cycle_position"],
                })
            except Exception as e:  # noqa: BLE001
                periods.append({"horizon": h, "error": str(e)[:60]})

        valid = [p for p in periods if "error" not in p]
        if not valid:
            return {"error": "全部周期预测失败"}

        # 方向分布
        dirs = [p["direction"] for p in valid]
        n_up = dirs.count("up")
        n_down = dirs.count("down")
        n_side = dirs.count("sideways")

        # 共振判定
        if n_up == len(valid):
            resonance = "强共振看涨"
            signal = "up"
            strength = 1.0
        elif n_down == len(valid):
            resonance = "强共振看跌"
            signal = "down"
            strength = 1.0
        elif n_up >= 2 and n_down == 0:
            resonance = "共振看涨（有周期震荡）"
            signal = "up"
            strength = 0.7
        elif n_down >= 2 and n_up == 0:
            resonance = "共振看跌（有周期震荡）"
            signal = "down"
            strength = 0.7
        elif n_up == n_down:
            resonance = "方向分歧（多空对半）"
            signal = "sideways"
            strength = 0.2
        else:
            resonance = "轻度分歧"
            signal = "up" if n_up > n_down else "down" if n_down > n_up \
                else "sideways"
            strength = 0.4

        return {
            "symbol": symbol,
            "periods": periods,
            "direction_summary": {"up": n_up, "down": n_down,
                                  "sideways": n_side},
            "resonance": resonance,
            "signal": signal,
            "strength": strength,
            "note": ("三周期同向 = 强共振（趋势确认）；分歧 = 观望。"
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
                    "ORDER BY date DESC LIMIT 160", [symbol])
                return df if not df.empty else None
            finally:
                store.close()
        except Exception:
            return None


# ---- 共振信号事件研究（可检验性） ----

async def scan_resonance_signals(symbol: str, kline_df: pd.DataFrame,
                                 step: int = 5,
                                 min_strength: float = 0.7) -> pd.DataFrame:
    """滚动窗口扫描共振信号（供事件研究验证预测力）。

    每 step 日：窗口尾部三周期预测 → 共振 → strength ≥ min_strength
    时记录信号（up/down）。返回 sig_df [symbol, date, direction, strategy_name]。
    """
    from pa_mcp.agent.prediction import PredictionService
    svc = PredictionService()
    data = kline_df.sort_values("date").reset_index(drop=True)
    n = len(data)
    rows = []
    for i in range(60, n, step):
        win = data.iloc[max(0, i - 159):i + 1]
        if len(win) < 120:
            continue
        try:
            r = await ResonanceAnalyzer().analyze(symbol, kline_df=win)
            if "error" in r:
                continue
            if r["strength"] >= min_strength and r["signal"] in ("up", "down"):
                rows.append({
                    "symbol": symbol,
                    "date": str(data["date"].iloc[i])[:10],
                    "direction": r["signal"],
                    "strategy_name": "resonance",
                    "strength_score": 70.0 if r["signal"] == "up" else 45.0,
                    "resonance": r["resonance"],
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


def resonance_event_study(symbol: str, kline_df: pd.DataFrame,
                          step: int = 5) -> dict[str, Any]:
    """共振信号事件研究：信号后 5/10/20 日收益 vs 无条件基准。

    Args:
        symbol: 股票代码
        kline_df: 日线（≥180 根）
        step: 扫描步长
    """
    import asyncio as _asyncio
    from pa_mcp.research.event_study import signal_forward_returns

    sig_df = _asyncio.run(scan_resonance_signals(symbol, kline_df, step=step))
    if sig_df.empty:
        return {"symbol": symbol, "n_signals": 0,
                "message": "未检出强共振信号（strength ≥70% 需三周期同向）"}

    results = signal_forward_returns(kline_df, sig_df, [5, 10, 20])
    return {
        "symbol": symbol,
        "n_signals": len(sig_df),
        "n_up": int((sig_df["direction"] == "up").sum()),
        "n_down": int((sig_df["direction"] == "down").sum()),
        "signal_range": [sig_df["date"].iloc[0], sig_df["date"].iloc[-1]],
        "results": [
            {"horizon": r.horizon, "n_events": r.n_events,
             "win_rate_pct": r.win_rate_pct,
             "avg_return_pct": r.avg_return_pct,
             "benchmark_avg_return_pct": r.benchmark_avg_return_pct,
             "excess_return_pct": r.excess_return_pct,
             "has_edge": r.has_edge}
            for r in results],
        "has_edge": any(r.has_edge for r in results),
    }


def format_resonance_event_study(result: dict[str, Any]) -> str:
    """事件研究 → markdown。"""
    if result.get("n_signals", 0) == 0:
        return (f"🎯 共振事件研究：{result.get('symbol', '')} 未检出强共振信号"
                f"（{result.get('message', '')}）")
    lines = [
        f"## 🎯 共振信号事件研究：{result['symbol']}",
        f"扫描到 **{result['n_signals']} 个强共振信号**"
        f"（看多 {result['n_up']} / 看空 {result['n_down']}，"
        f"{result['signal_range'][0]} ~ {result['signal_range'][1]}）",
        "",
        "| 前瞻日 | 信号数 | 胜率% | 平均收益% | 基准% | 超额% |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["results"]:
        lines.append(
            f"| {r['horizon']} | {r['n_events']} | {r['win_rate_pct']:.1f} | "
            f"{r['avg_return_pct']:+.2f} | {r['benchmark_avg_return_pct']:+.2f} | "
            f"{r['excess_return_pct']:+.2f} |")
    verdict = "✅ 共振有预测力" if result["has_edge"] else "❌ 无显著预测力"
    lines.append(f"\n**结论：{verdict}**")
    lines.append("\n*共振 = 1d/5d/20d 三周期同向（强度 ≥70%）。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)


_analyzer: Optional[ResonanceAnalyzer] = None


def get_resonance_analyzer() -> ResonanceAnalyzer:
    """单例获取共振分析器。"""
    global _analyzer
    if _analyzer is None:
        _analyzer = ResonanceAnalyzer()
    return _analyzer


def format_resonance(result: dict[str, Any]) -> str:
    """共振结果 → markdown。"""
    if "error" in result:
        return f"多周期共振不可用：{result['error']}"
    dir_zh = {"up": "📈 看涨", "down": "📉 看跌", "sideways": "➡️ 震荡"}
    lines = [
        f"## 🎯 多周期预测共振：{result['symbol']}",
        f"**{result['resonance']}**（信号：{dir_zh.get(result['signal'], result['signal'])}"
        f"，强度 {result['strength']:.0%}）",
        "",
        "| 周期 | 方向 | 概率 | 期望% | 周期位置 |",
        "|---|---|---|---|---|",
    ]
    for p in result["periods"]:
        if "error" in p:
            lines.append(f"| {p['horizon']} | ❌ {p['error']} |")
            continue
        lines.append(
            f"| {p['horizon']} | {dir_zh.get(p['direction'], p['direction'])} | "
            f"{p['probability']:.0%} | {p['expected_return_pct']:+.1f} | "
            f"{p['cycle']} |")
    lines.append(f"\n*{result['note']}*")
    return "\n".join(lines)
