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
               value_weight: float = 0.5,
               klines: Optional[dict[str, pd.DataFrame]] = None) -> dict[str, Any]:
        """复合选股。

        Args:
            symbols: 股票池
            quotes: 估值快照（透传格雷厄姆筛选）
            top_n: 返回数量
            value_weight: 价值权重（动量 = 1 - value_weight）
            klines: 可选 {symbol: DataFrame} 行情覆盖（滚动回测用——
                窗口内切片注入，避免从 store 读全历史造成前视）
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
            df = klines.get(sym) if klines else self._load_kline(sym)
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


# ---- 价值×动量组合回测（滚动调仓，复用 PortfolioBacktestEngine） ----

def backtest_value_momentum(
    symbols: list[str],
    klines: dict[str, pd.DataFrame],
    quotes: Optional[dict[str, dict]] = None,
    top_n: int = 3,
    horizon: int = 5,
    train_window: int = 120,
    value_weight: float = 0.5,
    initial_cash: float = 100_000.0,
) -> dict[str, Any]:
    """滚动价值×动量组合回测。

    每 horizon 日用窗口内行情切片评分（动量无前视；财务/估值用最新
    快照——标注近似）→ top N 持仓 → 共享账本组合回测 vs 全池等权。

    Args:
        symbols: 股票池
        klines: {symbol: 日线（升序）}（各股 ≥ train_window + 61 根）
        quotes: 估值快照（缺省 None → 格雷厄姆估值标准 unavailable）
        top_n: 每期持仓数量
        horizon: 调仓周期
        train_window: 动量窗口（滚动切片 = train_window 根）
        value_weight: 价值权重
        initial_cash: 初始资金
    """
    from pa_mcp.portfolio.backtest import PortfolioBacktestEngine

    aligned: dict[str, pd.DataFrame] = {}
    for sym, df in klines.items():
        d = df.sort_values("date").reset_index(drop=True)
        if len(d) >= train_window + 61:
            aligned[sym] = d
    if len(aligned) < 3:
        return {"error": f"满足窗口的股票不足（{len(aligned)} < 3）"}

    calendar = sorted(set().union(
        *[set(df["date"].astype(str).str[:10]) for df in aligned.values()]))
    n = len(calendar)
    if n <= train_window + horizon:
        return {"error": f"日历 {n} 天不足（需 > {train_window + horizon}）"}

    screen = ValueMomentumScreen()
    signals_by_symbol: dict[str, list[dict]] = {s: [] for s in aligned}
    for t in range(train_window, n - 1, horizon):
        end_date = calendar[t]
        window_klines = {}
        for sym, d in aligned.items():
            sub = d[d["date"].astype(str).str[:10] <= end_date]
            window_klines[sym] = sub.tail(train_window + 61)
        sel = screen.screen(symbols, quotes=quotes, top_n=top_n,
                            value_weight=value_weight,
                            klines=window_klines)
        if "error" in sel:
            continue
        top = sel["top_symbols"]
        for sym in aligned:
            signals_by_symbol[sym].append({
                "date": end_date, "symbol": sym,
                "direction": "bullish" if sym in top else "bearish",
                "strength_score": 60.0 if sym in top else 40.0,
            })

    sig_dfs = {s: pd.DataFrame(lst) if lst else pd.DataFrame(
        columns=["date", "symbol", "direction", "strength_score"])
        for s, lst in signals_by_symbol.items()}

    engine = PortfolioBacktestEngine(initial_cash=initial_cash)
    result = engine.run(aligned, sig_dfs)

    # 基准：全池等权
    bench_signals = {}
    for sym in aligned:
        bench_signals[sym] = pd.DataFrame([
            {"date": calendar[t], "symbol": sym, "direction": "bullish",
             "strength_score": 50.0}
            for t in range(train_window, n - 1, horizon)])
    bench = PortfolioBacktestEngine(initial_cash=initial_cash).run(
        aligned, bench_signals)

    return {
        "method": (f"滚动 {train_window} 日价值×动量评分（权重 {value_weight:.0%}）"
                   f"→ 每 {horizon} 日调仓 top {top_n}，等权组合回测"),
        "n_stock": len(aligned),
        "n_rebalances": len(next(iter(sig_dfs.values()))),
        "portfolio": {
            "total_return_pct": getattr(result, "total_return_pct", None),
            "annual_return_pct": getattr(result, "annual_return_pct", None),
            "max_drawdown_pct": getattr(result, "max_drawdown_pct", None),
            "sharpe_ratio": getattr(result, "sharpe_ratio", None),
            "total_trades": getattr(result, "total_trades", None),
        },
        "benchmark": {
            "total_return_pct": getattr(bench, "total_return_pct", None),
            "max_drawdown_pct": getattr(bench, "max_drawdown_pct", None),
        },
        "excess_return_pct": round(
            float(getattr(result, "total_return_pct", 0) or 0)
            - float(getattr(bench, "total_return_pct", 0) or 0), 2),
        "note": ("动量用窗口内切片（无前视）；财务/估值用最新快照（近似）。"
                 "基准 = 全池等权。研究参考，非投资建议。"),
    }


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


def format_vm_backtest(result: dict[str, Any]) -> str:
    """组合回测 → markdown。"""
    if "error" in result:
        return f"价值×动量回测不可用：{result['error']}"
    p, b = result["portfolio"], result["benchmark"]
    return (
        f"## 🏆 价值×动量组合回测\n"
        f"**方法**：{result['method']}\n"
        f"**样本**：{result['n_stock']} 只股票 × {result['n_rebalances']} 次调仓\n"
        f"- **组合**：总收益 {p['total_return_pct']}% | 年化 {p['annual_return_pct']}% | "
        f"回撤 {p['max_drawdown_pct']}% | Sharpe {p['sharpe_ratio']}\n"
        f"- **基准**（全池等权）：{b['total_return_pct']}% | 回撤 {b['max_drawdown_pct']}%\n"
        f"- **超额收益**：**{result['excess_return_pct']:+.2f}%**\n"
        f"- 交易 {p['total_trades']} 笔\n"
        f"*{result['note']}*"
    )
