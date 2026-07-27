# [AI:BEGIN]
# PA_MCP - Command-Line Backtest Runner
# Usage: python scripts/run_backtest.py --symbol 000001 --strategy platform_breakout
# [AI:END]

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="PA_MCP Backtest Runner")
    parser.add_argument("--symbol", required=True, help="Stock code (e.g., 000001)")
    parser.add_argument("--strategy", required=True, help="Strategy name (e.g., platform_breakout)")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--output", default="", help="Output file (JSON), empty = print")
    args = parser.parse_args()

    print(f"Running backtest: {args.symbol} x {args.strategy}")
    print(f"Period: {args.start} → {args.end}, Capital: {args.capital:,.0f} CNY")

    # Initialize store
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore()
    store.connect()

    # Get strategy info
    from pa_mcp.engine.strategies.base import StrategyRegistry
    registry = StrategyRegistry()
    registry.auto_discover("pa_mcp.engine.strategies")

    try:
        strategy_info = registry.get(args.strategy).get_info()
        print(f"\nStrategy: {strategy_info['name']}")
        print(f"Category: {strategy_info['category']}")
        print(f"Description: {strategy_info['description']}")
    except KeyError:
        available = registry.list_all()
        print(f"\nStrategy '{args.strategy}' not found. Available: {available}")
        sys.exit(1)

    # Fetch data
    print("\nFetching data...")
    df = store.query_df(
        "SELECT * FROM kline_daily WHERE symbol = ? AND date BETWEEN ? AND ? ORDER BY date",
        [args.symbol, args.start, args.end],
    )

    if df.empty:
        # Generate synthetic data for demo
        print("  No real data found, using synthetic data for demo...")
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        close = 10.0
        rows = []
        dates = pd.date_range(args.start, args.end, freq="B")
        for d in dates:
            ret = np.random.normal(0.0008, 0.018)
            open_p = close
            close = close * (1 + ret)
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "symbol": args.symbol,
                "open": round(open_p, 2),
                "high": round(max(open_p, close) * 1.005, 2),
                "low": round(min(open_p, close) * 0.995, 2),
                "close": round(close, 2),
                "volume": np.random.uniform(5e6, 2e7),
                "amount": np.random.uniform(5e7, 2e8),
            })
        df = pd.DataFrame(rows)
        print(f"  Generated {len(df)} synthetic trading days")

    print(f"  Data: {len(df)} trading days, {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")

    # Run strategy
    print("\nRunning strategy...")
    strategy = registry.get(args.strategy)
    signals = strategy.generate_signals(df)

    print(f"  Generated {len(signals)} signals")
    if signals:
        print(f"  Top signal: strength={signals[0].strength_score}, direction={signals[0].direction.value}")

    # Run backtest
    print("\nRunning backtest...")
    from pa_mcp.backtest.engine import DuckDBBacktester

    backtester = DuckDBBacktester(store)
    report = backtester.backtest(
        args.symbol, entry_sql="close > ma20 AND volume > avg_vol_20 * 1.5",
        exit_sql="close < ma10",
        start=args.start, end=args.end, capital=args.capital,
    )

    result = report.to_dict()

    # Print summary
    perf = result["performance"]
    print(f"\n{'='*50}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"  Total Return:     {perf['total_return_pct']:>8.1f}%")
    print(f"  Annual Return:    {perf['annual_return_pct']:>8.1f}%")
    print(f"  Sharpe Ratio:     {perf['sharpe_ratio']:>8.2f}")
    print(f"  Max Drawdown:     {perf['max_drawdown_pct']:>8.1f}%")
    print(f"  Win Rate:         {perf['win_rate_pct']:>8.1f}%")
    print(f"  Trades:           {result['trades']['total']:>8d}")
    print(f"  Total Fees:       {result['trades']['total_fees']:>8.0f} CNY")
    print(f"{'='*50}")
    print(f"\n  DISCLAIMER: Simulated past performance.")
    print(f"  Real A-share returns are typically 30-50% lower after")
    print(f"  liquidity, slippage, and limit-up/down constraints.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nFull report saved to: {args.output}")
    else:
        print(f"\nUse --output report.json to save full report.")


if __name__ == "__main__":
    main()
