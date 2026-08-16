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


# ---- 综合信号事件研究（可检验性） ----

async def scan_consensus_signals(symbol: str, kline_df: pd.DataFrame,
                                 step: int = 5,
                                 min_strength: float = 0.6) -> pd.DataFrame:
    """滚动窗口扫描综合信号（供事件研究验证预测力）。

    每 step 日：窗口尾部综合投票 → strength ≥ min_strength 时记录
    信号（up/down）。返回 sig_df [symbol, date, direction, strategy_name]。
    """
    analyzer = ConsensusAnalyzer()
    data = kline_df.sort_values("date").reset_index(drop=True)
    n = len(data)
    rows = []
    for i in range(120, n, step):
        win = data.iloc[max(0, i - 199):i + 1]
        if len(win) < 150:
            continue
        try:
            r = await analyzer.analyze(symbol, kline_df=win)
            if "error" in r:
                continue
            if r["strength"] >= min_strength and r["signal"] in ("up", "down"):
                rows.append({
                    "symbol": symbol,
                    "date": str(data["date"].iloc[i])[:10],
                    "direction": r["signal"],
                    "strategy_name": "consensus",
                    "strength_score": 70.0 if r["signal"] == "up" else 45.0,
                    "detail": f"一致度 {r['agreement']:.0%}",
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


def consensus_event_study(symbol: str, kline_df: pd.DataFrame,
                          step: int = 5) -> dict[str, Any]:
    """综合信号事件研究（同步包装，供 UI 同步函数使用）。"""
    import asyncio as _asyncio
    return _asyncio.run(consensus_event_study_async(
        symbol, kline_df, step=step))


async def consensus_event_study_async(symbol: str, kline_df: pd.DataFrame,
                                      step: int = 5) -> dict[str, Any]:
    """综合信号事件研究：信号后 5/10/20 日收益 vs 无条件基准。

    Args:
        symbol: 股票代码
        kline_df: 日线（≥250 根）
        step: 扫描步长
    """
    from pa_mcp.research.event_study import signal_forward_returns

    sig_df = await scan_consensus_signals(symbol, kline_df, step=step)
    if sig_df.empty:
        return {"symbol": symbol, "n_signals": 0,
                "message": "未检出强综合信号（strength ≥60% 需多数源一致）"}

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


def format_consensus_event_study(result: dict[str, Any]) -> str:
    """事件研究 → markdown。"""
    if result.get("n_signals", 0) == 0:
        return (f"🧮 综合信号事件研究：{result.get('symbol', '')} 未检出强信号"
                f"（{result.get('message', '')}）")
    lines = [
        f"## 🧮 综合信号事件研究：{result['symbol']}",
        f"扫描到 **{result['n_signals']} 个强综合信号**"
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
    verdict = "✅ 综合信号有预测力" if result["has_edge"] else "❌ 无显著预测力"
    lines.append(f"\n**结论：{verdict}**")
    lines.append("\n*综合信号 = 共振/预测/策略/背驰/大盘 加权投票（强度 ≥60%）。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)


async def scan_watchlist_consensus(
    symbols: list[str],
    klines: Optional[dict[str, pd.DataFrame]] = None,
) -> dict[str, Any]:
    """自选股批量综合信号扫描：强看涨/看跌/分歧分类清单。

    Args:
        symbols: 股票代码列表（2-20 只）
        klines: 可选行情覆盖（缺省每只从库拉取）
    """
    analyzer = ConsensusAnalyzer()
    rows = []
    for sym in symbols[:20]:
        try:
            df = klines.get(sym) if klines else None
            if df is None:
                df = analyzer._load_kline(sym)
            if df is None or df.empty:
                rows.append({"symbol": sym, "error": "无数据"})
                continue
            r = await analyzer.analyze(sym, kline_df=df)
            if "error" in r:
                rows.append({"symbol": sym, "error": r["error"]})
                continue
            rows.append({
                "symbol": sym,
                "signal": r["signal"],
                "strength": r["strength"],
                "level": r["level"],
                "agreement": r["agreement"],
                "n_sources": len(r["sources"]),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"symbol": sym, "error": str(e)[:60]})

    strong_up = [r for r in rows if r.get("signal") == "up"
                 and r.get("strength", 0) >= 0.6]
    strong_down = [r for r in rows if r.get("signal") == "down"
                   and r.get("strength", 0) >= 0.6]
    mixed = [r for r in rows if "error" not in r
             and r.get("strength", 1) < 0.6]
    return {
        "n_scanned": len(rows),
        "strong_up": [r["symbol"] for r in strong_up],
        "strong_down": [r["symbol"] for r in strong_down],
        "mixed": [r["symbol"] for r in mixed],
        "details": rows,
        "note": ("强信号 = 5 源加权投票强度 ≥60%（多数一致）；"
                 "mixed = 分歧/未确认。研究参考，非投资建议。"),
    }


def format_watchlist_consensus(result: dict[str, Any]) -> str:
    """批量综合信号 → markdown。"""
    lines = [
        f"## 🧮 自选股综合信号扫描（{result['n_scanned']} 只）",
        f"- **📈 强看涨**（{len(result['strong_up'])}）："
        + ("、".join(result["strong_up"]) or "无"),
        f"- **📉 强看跌**（{len(result['strong_down'])}）："
        + ("、".join(result["strong_down"]) or "无"),
        f"- **➡️ 分歧**（{len(result['mixed'])}）："
        + ("、".join(result["mixed"]) or "无"),
        "",
        "| 代码 | 信号 | 强度 | 等级 | 一致度 | 信号源数 |",
        "|---|---|---|---|---|---|",
    ]
    dir_zh = {"up": "📈", "down": "📉", "sideways": "➡️"}
    for r in result["details"]:
        if "error" in r:
            lines.append(f"| {r['symbol']} | ❌ {r['error']} |")
            continue
        lines.append(
            f"| {r['symbol']} | {dir_zh.get(r['signal'], r['signal'])} | "
            f"{r['strength']:.0%} | {r['level']} | "
            f"{r['agreement']:.0%} | {r['n_sources']} |")
    lines.append(f"\n*{result['note']}*")
    return "\n".join(lines)


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
