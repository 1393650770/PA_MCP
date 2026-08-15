# [AI:BEGIN]
# PA_MCP - Research: 缠论背驰信号组合回测验证
#
# 大牛方法可检验性：背驰信号（下跌背驰=看多 / 上涨背驰=看空）作为
# 调仓信号进入共享账本组合回测，直接回答「缠论背驰能否构建组合」。
#
# 复用（非必要不造轮子）：
#   - scan_beichi_signals：滑动窗口背驰信号扫描
#   - PortfolioBacktestEngine：共享账本/T+1/费用/单票10%
#   - 基准：全池等权持有
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def backtest_beichi_signals(
    klines: dict[str, pd.DataFrame],
    window: int = 60,
    step: int = 5,
    initial_cash: float = 100_000.0,
) -> dict[str, Any]:
    """缠论背驰信号组合回测。

    流程：
      1. 每股票滑动窗口扫描背驰信号（bullish=买 / bearish=卖）
      2. 信号按日对齐 → 延迟一天执行（组合引擎语义）
      3. 共享账本组合回测 vs 全池等权基准
      4. 输出：组合/基准/超额/回撤/信号统计

    Args:
        klines: {symbol: 日线（升序）}（各股 ≥ 150 根）
        window: 背驰扫描窗口
        step: 扫描步长
        initial_cash: 初始资金
    """
    from pa_mcp.engine.indicators.chan import scan_beichi_signals
    from pa_mcp.portfolio.backtest import PortfolioBacktestEngine

    aligned: dict[str, pd.DataFrame] = {}
    for sym, df in klines.items():
        d = df.sort_values("date").reset_index(drop=True)
        if len(d) >= window + 20:
            aligned[sym] = d
    if len(aligned) < 2:
        return {"error": f"满足窗口的股票不足（{len(aligned)} < 2）"}

    calendar = sorted(set().union(
        *[set(df["date"].astype(str).str[:10]) for df in aligned.values()]))
    if len(calendar) < 20:
        return {"error": "日历过短"}

    # 信号装配：{symbol: DataFrame[date, symbol, direction, strength]}
    signals_by_symbol: dict[str, pd.DataFrame] = {}
    total_signals = 0
    for sym, d in aligned.items():
        sig = scan_beichi_signals(d, symbol=sym, window=window, step=step)
        if sig.empty:
            signals_by_symbol[sym] = pd.DataFrame(
                columns=["date", "symbol", "direction", "strength_score"])
            continue
        sig = sig[["date", "symbol", "direction", "strength_score"]]
        signals_by_symbol[sym] = sig
        total_signals += len(sig)

    if total_signals == 0:
        return {"error": "无背驰信号（窗口内无动能衰竭结构），无法回测"}

    engine = PortfolioBacktestEngine(initial_cash=initial_cash)
    result = engine.run(aligned, signals_by_symbol)

    # 基准：全池等权
    bench_signals = {}
    first_day = calendar[min(window, len(calendar) - 1)]
    for sym in aligned:
        bench_signals[sym] = pd.DataFrame([{
            "date": first_day, "symbol": sym, "direction": "bullish",
            "strength_score": 50.0}])
    bench = PortfolioBacktestEngine(initial_cash=initial_cash).run(
        aligned, bench_signals)

    return {
        "method": (f"缠论背驰信号调仓（窗口 {window}/步长 {step}，"
                   f"下跌背驰买/上涨背驰卖，延迟一天执行）"),
        "n_stock": len(aligned),
        "total_signals": total_signals,
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
        "note": "背驰信号稀疏（动能衰竭结构不常见）→ 组合可能长期空仓。"
                "研究参考，非投资建议。",
    }


def format_beichi_backtest(result: dict[str, Any]) -> str:
    """组合回测 → markdown。"""
    if "error" in result:
        return f"背驰组合回测不可用：{result['error']}"
    p, b = result["portfolio"], result["benchmark"]
    return (
        f"## 🌀 缠论背驰信号组合回测\n"
        f"**方法**：{result['method']}\n"
        f"**样本**：{result['n_stock']} 只股票，共 {result['total_signals']} 个信号\n"
        f"- **组合**：总收益 {p['total_return_pct']}% | 年化 {p['annual_return_pct']}% | "
        f"回撤 {p['max_drawdown_pct']}% | Sharpe {p['sharpe_ratio']}\n"
        f"- **基准**（全池等权）：{b['total_return_pct']}% | 回撤 {b['max_drawdown_pct']}%\n"
        f"- **超额收益**：**{result['excess_return_pct']:+.2f}%**\n"
        f"- 交易 {p['total_trades']} 笔\n"
        f"*{result['note']}*"
    )
