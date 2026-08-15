# [AI:BEGIN]
# PA_MCP - Portfolio: 信号 → 组合管线
#
# 把策略信号转化为可交易的组合目标：
#   1. 多票信号聚合：每只股票的策略信号 → alpha_score
#   2. PortfolioConstructor.build() → 约束目标权重
#   3. 组合净值合成（按权重加权各票独立回测净值）
#
# 设计：研究候选（信号级）→ 组合目标（权重级）→ 组合回测（净值级）
# [AI:END]

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from pa_mcp.backtest.engine import BacktestEngine, BacktestReport
from pa_mcp.backtest.broker import FeeSchedule
from pa_mcp.portfolio.construction import (
    PortfolioConstructor, PortfolioTarget, TargetWeight, ConstraintSet,
)


@dataclass
class SignalCandidate:
    """一只股票的策略信号聚合（作为组合候选）。"""
    symbol: str
    alpha_score: float          # 信号强度聚合（0-100）
    price: float                # 最新收盘
    adv: float                  # 日均成交额（元）
    sector: str = ""


@dataclass
class PortfolioRunResult:
    """组合回测结果。"""
    target: PortfolioTarget
    reports: dict[str, BacktestReport]      # 各票独立回测
    nav_series: list[dict]                  # 组合加权净值
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0

    def summary(self) -> str:
        lines = [
            "## 组合回测结果",
            f"- 持仓 {self.target.total_stocks} 只 | 仓位 {self.target.total_exposure:.0%} | "
            f"现金 {self.target.cash_weight:.0%}",
            f"- 组合收益 {self.total_return_pct:+.2f}% | 年化 {self.annual_return_pct:+.2f}% | "
            f"夏普 {self.sharpe:.2f} | 回撤 {self.max_drawdown_pct:.2f}%",
            "",
            "| 股票 | 权重 | 收益% | 交易 |",
            "|---|---|---|---|",
        ]
        for sym, wt in sorted(self.target.holdings, key=lambda h: h.weight, reverse=True):
            rep = self.reports.get(sym.symbol)
            if rep:
                lines.append(f"| {sym.symbol} | {wt.weight:.0%} | {rep.total_return_pct:+.2f} | {rep.total_trades} |")
        lines.append("\n*等权信号组合，各票独立资金回测后按权重合成。研究参考，非投资建议。*")
        return "\n".join(lines)


def build_portfolio_from_signals(
    kline_by_symbol: dict[str, pd.DataFrame],
    signal_fn: Callable[[pd.DataFrame, str], pd.DataFrame],
    initial_cash: float = 100_000.0,
    constraints: Optional[ConstraintSet] = None,
) -> PortfolioRunResult:
    """从多票信号构建组合并回测。

    Args:
        kline_by_symbol: {symbol: 日线DataFrame}
        signal_fn: callable(kline_df, symbol) -> signals DataFrame
        initial_cash: 组合初始资金
    """
    # 1. 聚合信号 → 候选
    candidates: list[SignalCandidate] = []
    reports: dict[str, BacktestReport] = {}

    for sym, df in kline_by_symbol.items():
        if df.empty:
            continue
        sigs = signal_fn(df, sym)
        if sigs is None or sigs.empty:
            continue

        price = float(df["close"].iloc[-1])
        adv = float(df["amount"].tail(20).mean()) if "amount" in df.columns else price * 1e6

        # alpha_score：信号强度均值（0-100）
        if "strength_score" in sigs.columns:
            alpha = float(sigs["strength_score"].mean())
        else:
            alpha = 60.0
        candidates.append(SignalCandidate(symbol=sym, alpha_score=alpha,
                                          price=price, adv=adv))

        # 各票独立回测（用于组合净值合成）
        engine = BacktestEngine(initial_cash=initial_cash / max(len(kline_by_symbol), 1),
                                fee_schedule=FeeSchedule())
        reports[sym] = engine.run(df, sigs)

    if not candidates:
        return PortfolioRunResult(target=PortfolioTarget(), reports=reports,
                                  nav_series=[])

    # 2. 组合构建（约束目标权重）
    pc = PortfolioConstructor(constraints or ConstraintSet())
    target = pc.build([
        TargetWeight(c.symbol, c.alpha_score, c.alpha_score, c.sector,
                     c.price, c.adv)
        for c in candidates
    ])

    # 3. 组合净值合成（按权重加权各票净值）
    weight_map = {h.symbol: h.weight for h in target.holdings}
    nav_map: dict[str, pd.DataFrame] = {}
    for sym, rep in reports.items():
        nav = pd.DataFrame(rep.nav_series)
        if not nav.empty:
            nav["trade_date"] = pd.to_datetime(nav["trade_date"])
            nav_map[sym] = nav

    if nav_map:
        # 对齐日期，加权合成
        all_dates = sorted(set().union(*[set(n["trade_date"]) for n in nav_map.values()]))
        combo: list[dict] = []
        prev = {}
        for d in all_dates:
            nav_val = 0.0
            for sym, n in nav_map.items():
                wt = weight_map.get(sym, 0)
                if wt <= 0:
                    continue
                sub = n[n["trade_date"] <= d]
                if sub.empty:
                    continue
                prev[sym] = float(sub["nav"].iloc[-1])
                nav_val += wt * prev[sym]
            # 现金部分
            cash_wt = target.cash_weight
            nav_val += cash_wt * initial_cash
            combo.append({"trade_date": str(d)[:10], "nav": nav_val})
        combo_df = pd.DataFrame(combo)
    else:
        combo_df = pd.DataFrame()

    result = PortfolioRunResult(target=target, reports=reports,
                                nav_series=combo_df.to_dict(orient="records"))

    # 组合指标
    if not combo_df.empty and len(combo_df) > 1:
        navs = combo_df["nav"].values
        result.total_return_pct = round((navs[-1] / navs[0] - 1) * 100, 2)
        daily = pd.Series(navs).pct_change().dropna()
        if len(daily) > 1 and daily.std() > 0:
            result.annual_return_pct = round(float((1 + daily.mean()) ** 252 - 1) * 100, 2)
            result.sharpe = round(float(daily.mean() / daily.std() * (252 ** 0.5)), 2)
            cum = (1 + daily).cumprod()
            result.max_drawdown_pct = round(float(((cum - cum.cummax()) / cum.cummax()).min()) * 100, 2)

    return result
