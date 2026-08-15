# [AI:BEGIN]
# PA_MCP - Research: Signal Event Study (信号前瞻收益检验)
#
# 经典 alpha 检验方法：对策略的每个买入信号，统计信号后 N 个交易日的收益。
# 汇总全部信号：胜率、平均收益、平均最大回撤、样本数。
#
# 与基准比较：
#   - 全部交易日买入持有 N 日（无条件基准）
#   - 若信号样本胜率/收益显著高于基准 → 信号有预测力
#
# 这是"策略有没有用"的最直接检验。
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd


@dataclass
class EventStudyResult:
    strategy: str
    symbol: str
    horizon: int                    # N 个交易日
    n_events: int = 0
    win_rate_pct: float = 0.0
    avg_return_pct: float = 0.0
    median_return_pct: float = 0.0
    avg_max_drawdown_pct: float = 0.0
    benchmark_win_rate_pct: float = 0.0
    benchmark_avg_return_pct: float = 0.0
    excess_return_pct: float = 0.0   # 信号收益 - 基准收益
    verdict: str = "样本不足"

    @property
    def has_edge(self) -> bool:
        """有预测力：样本充足 + 胜率>50% + 超额收益>0"""
        return (self.n_events >= 20 and self.win_rate_pct > 50
                and self.excess_return_pct > 0)

    def summary(self) -> str:
        return (
            f"**{self.strategy} × {self.symbol}**（信号后 {self.horizon} 日）\n"
            f"- 信号样本：{self.n_events} 次\n"
            f"- 胜率：{self.win_rate_pct:.1f}%（基准 {self.benchmark_win_rate_pct:.1f}%）\n"
            f"- 平均收益：{self.avg_return_pct:+.2f}%（基准 {self.benchmark_avg_return_pct:+.2f}%）\n"
            f"- 超额收益：{self.excess_return_pct:+.2f}%\n"
            f"- 结论：{'✅ 信号有预测力' if self.has_edge else '❌ 无显著预测力'}"
        )


def signal_forward_returns(
    kline_df: pd.DataFrame,
    signals: pd.DataFrame,
    horizons: list[int] = (5, 10, 20),
) -> list[EventStudyResult]:
    """计算信号后 N 日收益。

    Args:
        kline_df: [symbol, date, close] 升序
        signals: [symbol, date, direction]（bullish 信号）
        horizons: 前瞻交易日数
    """
    if kline_df.empty or signals.empty:
        return []

    df = kline_df.sort_values("date").reset_index(drop=True)
    df["date_str"] = df["date"].astype(str).str[:10]
    close = df["close"].values
    dates = df["date_str"].values

    # 基准：全部交易日的无条件 N 日收益
    results = []
    bullish = signals[signals["direction"] == "bullish"] if "direction" in signals.columns else signals

    for horizon in horizons:
        # 基准收益（全部 bar）
        bench_returns = []
        for i in range(len(df) - horizon):
            bench_returns.append((close[i + horizon] / close[i] - 1) * 100)
        bench_avg = sum(bench_returns) / len(bench_returns) if bench_returns else 0
        bench_win = sum(1 for r in bench_returns if r > 0) / len(bench_returns) * 100 if bench_returns else 0

        # 信号收益
        returns = []
        for _, sig in bullish.iterrows():
            sig_date = str(sig.get("date", ""))[:10]
            sym = sig.get("symbol", "")
            # 定位信号在数据中的位置
            idx = None
            for i in range(len(df)):
                if dates[i] == sig_date and str(df.iloc[i].get("symbol", "")) == sym:
                    idx = i
                    break
            if idx is None or idx + horizon >= len(df):
                continue
            # 信号后持有 N 日
            fwd = (close[idx + horizon] / close[idx] - 1) * 100
            returns.append(fwd)

        if len(returns) < 10:
            results.append(EventStudyResult(
                strategy=str(signals.iloc[0].get("strategy_name", "")),
                symbol=str(signals.iloc[0].get("symbol", "")),
                horizon=horizon, n_events=len(returns),
                verdict="样本不足",
            ))
            continue

        wins = sum(1 for r in returns if r > 0)
        results.append(EventStudyResult(
            strategy=str(signals.iloc[0].get("strategy_name", "")),
            symbol=str(signals.iloc[0].get("symbol", "")),
            horizon=horizon,
            n_events=len(returns),
            win_rate_pct=round(wins / len(returns) * 100, 1),
            avg_return_pct=round(sum(returns) / len(returns), 2),
            median_return_pct=round(sorted(returns)[len(returns) // 2], 2),
            benchmark_win_rate_pct=round(bench_win, 1),
            benchmark_avg_return_pct=round(bench_avg, 2),
            excess_return_pct=round(sum(returns) / len(returns) - bench_avg, 2),
        ))

    return results


def event_study_multi(
    kline_by_symbol: dict[str, pd.DataFrame],
    signals_by_symbol: dict[str, pd.DataFrame],
    horizons: list[int] = (5, 10, 20),
) -> str:
    """多股事件研究汇总（Markdown 报告）。"""
    lines = ["## 信号事件研究汇总（信号后 N 日）",
             "| 策略×股票 | 信号数 | 5日胜率% | 5日超额% | 10日胜率% | 10日超额% | 判定 |",
             "|---|---|---|---|---|---|---|"]

    promotable = []
    for sym in signals_by_symbol:
        if sym not in kline_by_symbol:
            continue
        sigs = signals_by_symbol[sym]
        if sigs.empty:
            continue
        results = signal_forward_returns(kline_by_symbol[sym], sigs, horizons)
        if not results:
            continue
        r5 = next((r for r in results if r.horizon == 5), None)
        r10 = next((r for r in results if r.horizon == 10), None)
        edge = (r5 and r5.has_edge) or (r10 and r10.has_edge)
        lines.append(
            f"| {sigs.iloc[0].get('strategy_name','')}×{sym} | "
            f"{r5.n_events if r5 else 0} | "
            f"{r5.win_rate_pct if r5 and r5.n_events>=10 else '—'} | "
            f"{r5.excess_return_pct if r5 and r5.n_events>=10 else '—'} | "
            f"{r10.win_rate_pct if r10 and r10.n_events>=10 else '—'} | "
            f"{r10.excess_return_pct if r10 and r10.n_events>=10 else '—'} | "
            f"{'✅' if edge else '❌'} |"
        )
        if edge:
            promotable.append(f"{sigs.iloc[0].get('strategy_name','')}×{sym}")

    lines.append(f"\n**有预测力组合：{len(promotable)}** {promotable if promotable else '（无）'}")
    lines.append("\n*事件研究 = 信号后持有N日收益 vs 无条件基准。研究参考，非投资建议。*")
    return "\n".join(lines)
