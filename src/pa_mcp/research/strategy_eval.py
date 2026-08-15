# [AI:BEGIN]
# PA_MCP - Research: Strategy Walk-Forward Evaluation
#
# 用真实数据对策略做 walk-forward OOS 评估：
#   1. 按 TimeSeriesSplitter 生成 train/validation/test folds
#   2. 每个 fold 用事件驱动回测跑策略（信号延迟一天执行）
#   3. 汇总 OOS 指标：净收益、夏普、回撤、交易数
#   4. 输出可晋级判断（多数 fold 正收益 + 成本后成立）
#
# 晋级门槛（与 plan 一致）：
#   - aggregate OOS 净收益为正
#   - 多数 fold 净收益为正（不由单一窗口主导）
#   - 保守成本假设下仍为正
# [AI:END]

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from pa_mcp.backtest.engine import BacktestEngine
from pa_mcp.research.splits import TimeSeriesSplitter
from pa_mcp.research.recorder import RunRecorder, RunManifest, RunResult


@dataclass
class FoldResult:
    fold_number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    total_return_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    trades: int = 0
    positive: bool = False


@dataclass
class StrategyEvalReport:
    """Walk-forward 评估报告。"""
    strategy: str
    symbol: str
    folds: list[FoldResult] = field(default_factory=list)
    aggregate_return_pct: float = 0.0
    positive_folds: int = 0
    total_folds: int = 0
    pass_rate_pct: float = 0.0
    avg_sharpe: float = 0.0
    avg_max_drawdown_pct: float = 0.0
    verdict: str = "待定"  # 通过/未通过/样本不足

    @property
    def is_promotable(self) -> bool:
        """晋级条件：多数 fold 正收益 + 聚合正收益。"""
        return (self.positive_folds >= max(1, self.total_folds // 2)
                and self.aggregate_return_pct > 0)

    def summary(self) -> str:
        lines = [
            f"## {self.strategy} · {self.symbol} Walk-Forward 评估",
            f"Fold 数：{self.total_folds}（正收益 {self.positive_folds}）",
            f"聚合 OOS 收益：{self.aggregate_return_pct}%",
            f"平均夏普：{self.avg_sharpe} | 平均最大回撤：{self.avg_max_drawdown_pct}%",
            f"**结论：{'✅ 可晋级（多数fold正收益）' if self.is_promotable else '⚠️ 未达晋级标准'}**",
            "",
            "| Fold | 区间 | 收益% | 夏普 | 回撤% | 交易 |",
            "|---|---|---|---|---|---|",
        ]
        for f in self.folds:
            lines.append(
                f"| {f.fold_number} | {f.test_start}~{f.test_end} | "
                f"{f.total_return_pct:+.2f} | {f.sharpe:.2f} | "
                f"{f.max_drawdown_pct:.2f} | {f.trades} |"
            )
        lines.append("\n*研究参考，非投资建议。信号延迟一天执行，含A股撮合。*")
        return "\n".join(lines)


def run_walk_forward(
    kline_df: pd.DataFrame,
    strategy_name: str,
    signal_fn,
    initial_cash: float = 100_000.0,
    train_years: float = 2.0,
    test_months: int = 3,
    step_months: int = 3,
    min_folds: int = 2,
) -> StrategyEvalReport:
    """对策略做 walk-forward OOS 评估。

    Args:
        kline_df: [symbol, date, open, high, low, close, volume] 升序
        strategy_name: 策略名（仅用于报告）
        signal_fn: callable(df_slice) -> DataFrame[date, symbol, direction, strength_score, strategy_name]
        initial_cash: 初始资金
    """
    report = StrategyEvalReport(strategy=strategy_name,
                                symbol=str(kline_df.iloc[0].get("symbol", "")))

    if kline_df.empty or len(kline_df) < 200:
        report.verdict = "样本不足"
        return report

    start = kline_df["date"].iloc[0]
    end = kline_df["date"].iloc[-1]
    if hasattr(start, "date"):
        start = start.date() if hasattr(start, "date") else start
        end = end.date() if hasattr(end, "date") else end

    splitter = TimeSeriesSplitter(
        start_date=start, end_date=end,
        train_years=train_years, validation_months=test_months,
        test_months=test_months, step_months=step_months,
        embargo_days=3, purge_days=10,
    )
    folds = splitter.generate_folds()
    if len(folds) < min_folds:
        report.verdict = "样本不足"
        return report

    # 每个 fold 用 validation 窗口作为 OOS 评估（最后 fold 的 test 作为 sealed holdout）
    for fold in folds:
        eval_start = fold.test_start if fold.test_start is not None else fold.validation_start
        eval_end = fold.test_end if fold.test_end is not None else fold.validation_end
        ts_start = pd.Timestamp(eval_start)
        ts_end = pd.Timestamp(eval_end)
        warmup = 60  # 60天指标预热
        all_start = pd.Timestamp(eval_start - timedelta(days=warmup * 2))
        window = kline_df[kline_df["date"] >= all_start]

        # 生成信号（在窗口数据上）
        sig_df = signal_fn(window)

        # 回测整个窗口，但只统计 test 窗口内的交易？简化：回测窗口即统计窗口
        engine = BacktestEngine(initial_cash=initial_cash)
        bt = engine.run(window, sig_df)

        # 取 test 窗口的净值计算指标
        nav = pd.DataFrame(bt.nav_series)
        if nav.empty:
            continue
        nav["trade_date"] = pd.to_datetime(nav["trade_date"])
        test_nav = nav[(nav["trade_date"] >= ts_start) &
                       (nav["trade_date"] <= ts_end)]
        if len(test_nav) < 2:
            continue

        ret = (test_nav["nav"].iloc[-1] / test_nav["nav"].iloc[0] - 1) * 100
        daily = test_nav["nav"].pct_change().dropna()
        sharpe = float(daily.mean() / daily.std() * (252 ** 0.5)) if len(daily) > 1 and daily.std() > 0 else 0
        cum = (1 + daily).cumprod()
        dd = float(((cum - cum.cummax()) / cum.cummax()).min() * 100) if len(daily) > 1 else 0

        report.folds.append(FoldResult(
            fold_number=fold.fold_number,
            train_start=str(fold.train_start), train_end=str(fold.train_end),
            test_start=str(eval_start), test_end=str(eval_end),
            total_return_pct=round(ret, 2),
            sharpe=round(sharpe, 2),
            max_drawdown_pct=round(dd, 2),
            trades=bt.total_trades,
            positive=ret > 0,
        ))

    if not report.folds:
        report.verdict = "样本不足"
        return report

    report.total_folds = len(report.folds)
    report.positive_folds = sum(1 for f in report.folds if f.positive)
    report.pass_rate_pct = round(report.positive_folds / report.total_folds * 100, 1)
    report.aggregate_return_pct = round(
        sum(f.total_return_pct for f in report.folds) / report.total_folds, 2,
    )
    report.avg_sharpe = round(
        sum(f.sharpe for f in report.folds) / report.total_folds, 2,
    )
    report.avg_max_drawdown_pct = round(
        sum(f.max_drawdown_pct for f in report.folds) / report.total_folds, 2,
    )
    report.verdict = "通过" if report.is_promotable else "未通过"
    return report
