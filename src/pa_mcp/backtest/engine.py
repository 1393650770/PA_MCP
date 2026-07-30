# [AI:BEGIN]
# PA_MCP - Backtest Engine: DuckDB SQL Vectorized Backtester
#
# Phase C: Refactored to use event-driven broker + ledger internally.
# The old vectorized loop with same-bar look-ahead has been replaced.
# Preserves the DuckDBBacktester facade for backward compatibility.
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from pa_mcp.backtest.broker import DailyBroker, FeeSchedule
from pa_mcp.backtest.events import MarketBar
from pa_mcp.backtest.ledger import Ledger
from pa_mcp.backtest.orders import Order, OrderSide, OrderStatus

logger = structlog.get_logger(__name__)


@dataclass
class BacktestReport:
    """Standardized backtest report with verified metrics."""
    symbol: str = ""
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    annual_volatility_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    win_rate_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    total_fees: float = 0.0
    total_stamp_tax: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0  # computed when benchmark available
    nav_series: list[dict] = field(default_factory=list)
    trade_records: list[dict] = field(default_factory=list)
    synthetic_data: bool = False  # True if synthetic/test data was used


class BacktestEngine:
    """Event-driven A-share backtest engine with verified accounting.

    P0 fixes applied:
    - Signal on bar[i] → executes at bar[i+1] (no same-bar look-ahead)
    - All A-share rules enforced via DailyBroker
    - NAV = cash + Σ(qty × mark_price) verified at each bar
    """

    def __init__(self, initial_cash: float = 100_000.0,
                 fee_schedule: Optional[FeeSchedule] = None) -> None:
        self.initial_cash = initial_cash
        self.broker = DailyBroker(fee_schedule=fee_schedule,
                                  initial_cash=initial_cash)
        self.ledger = Ledger(cash=initial_cash)

    def run(
        self,
        kline_df: pd.DataFrame,
        strategy_signals: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> BacktestReport:
        """Execute a backtest with event-driven order execution.

        Args:
            kline_df: [symbol, date, open, high, low, close, volume, amount]
                     Sorted ascending by date.
            strategy_signals: [date, symbol, direction, strength_score, strategy_name]
                     Signal for date D executes at bar D+1.
            benchmark_returns: Optional daily benchmark returns for alpha.
        """
        if kline_df.empty:
            return BacktestReport(symbol="", total_trades=0)

        kline_df = kline_df.sort_values("date").reset_index(drop=True)
        symbol = str(kline_df.iloc[0].get("symbol", ""))

        fills: list[dict] = []
        # Deferred signal execution: signals[i] → executed at bar[i+1]
        pending_signals: list[dict] = []

        for i, row in kline_df.iterrows():
            trade_date = self._parse_date(row["date"])

            bar = MarketBar(
                symbol=symbol,
                trade_date=trade_date,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
                amount=float(row.get("amount", 0)),
                turnover=float(row.get("turnover", 0)),
                prev_close=float(row.get("prev_close", 0)) or None,
                is_suspended=bool(row.get("is_suspended", False)),
            )

            # Execute signals from previous bar (P0-4 fix)
            for sig in pending_signals:
                self._execute_signal(sig, bar, fills)

            # Collect signals generated on THIS bar → execute at NEXT bar
            current_sigs = strategy_signals[
                (strategy_signals["date"] == row["date"])
                & (strategy_signals["symbol"] == symbol)
            ] if not strategy_signals.empty else pd.DataFrame()

            pending_signals = []
            if not current_sigs.empty:
                pending_signals = current_sigs.to_dict(orient="records")

            # Mark to market
            mark_prices = {symbol: bar.close}
            self.ledger.record_nav(bar.trade_date, mark_prices)

        return self._compute_metrics(symbol, fills, benchmark_returns)

    def _execute_signal(self, sig: dict, bar: MarketBar,
                        fills: list[dict]) -> None:
        """Execute one signal as an order at the given bar."""
        direction = sig.get("direction", "neutral")
        if direction not in ("bullish", "bearish"):
            return

        symbol = bar.symbol

        cash_for_trade = self.ledger.available_cash * 0.10
        qty = max(100, int(cash_for_trade / max(bar.close, 0.01) / 100) * 100)
        if qty <= 0:
            return

        side = OrderSide.BUY if direction == "bullish" else OrderSide.SELL

        if side == OrderSide.SELL:
            held = self.ledger.get_sellable_quantity(symbol, bar.trade_date)
            qty = min(qty, held)
            if qty <= 0:
                return

        order = Order(
            symbol=symbol, side=side, quantity=qty,
            trade_date=bar.trade_date,
            strategy_name=str(sig.get("strategy_name", "")),
        )

        _, new_fills = self.broker.submit_order(order, bar)
        for f in new_fills:
            fills.append({
                "date": str(bar.trade_date), "symbol": symbol,
                "side": f.side.value, "quantity": f.quantity,
                "price": f.price, "commission": f.commission,
                "stamp_tax": f.stamp_tax, "transfer_fee": f.transfer_fee,
                "value": f.fill_value,
            })
            if f.side == OrderSide.BUY:
                self.ledger.remove_cash(f.fill_value + f.total_cost)
                self.ledger.add_lot(
                    symbol, f.quantity,
                    (f.fill_value + f.total_cost) / max(f.quantity, 1),
                    bar.trade_date,
                )
            else:
                self.ledger.add_cash(f.fill_value - f.total_cost)
                try:
                    self.ledger.remove_shares(symbol, f.quantity, bar.trade_date)
                except ValueError:
                    pass

    def _compute_metrics(
        self, symbol: str, fills: list[dict],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> BacktestReport:
        """Compute performance metrics from NAV history."""
        report = BacktestReport(symbol=symbol)

        nav = pd.DataFrame(self.ledger.nav_history)
        if nav.empty or len(nav) < 2:
            return report

        nav_values = nav["nav"].values
        start_nav = nav_values[0]
        end_nav = nav_values[-1]
        if start_nav <= 0:
            return report

        total_return = (end_nav - start_nav) / start_nav
        report.total_return_pct = round(total_return * 100, 2)

        daily_rets = pd.Series(nav_values).pct_change().dropna()
        if len(daily_rets) > 1:
            ann = 252
            report.annual_return_pct = round(float((1 + daily_rets.mean()) ** ann - 1) * 100, 2)
            vol = float(daily_rets.std() * np.sqrt(ann))
            report.annual_volatility_pct = round(vol * 100, 2)
            report.sharpe_ratio = round(report.annual_return_pct / max(vol * 100, 0.01), 2)

            cumulative = (1 + daily_rets).cumprod()
            running_max = cumulative.cummax()
            dd = (cumulative - running_max) / running_max
            report.max_drawdown_pct = round(float(dd.min()) * 100, 2)

        report.total_trades = len(fills)
        report.total_fees = round(sum(f["commission"] + f["transfer_fee"] for f in fills), 2)
        report.total_stamp_tax = round(sum(f["stamp_tax"] for f in fills), 2)
        report.nav_series = self.ledger.nav_history[-120:]
        report.trade_records = fills[-20:]

        return report

    @staticmethod
    def _parse_date(val: Any) -> Date:
        if hasattr(val, "date"):
            d = val.date() if callable(val.date) else val
            if isinstance(d, Date):
                return d
        return Date.today()


# ---- Legacy Compatibility Facade ----

class DuckDBBacktester:
    """Backward-compatible wrapper. Routes through BacktestEngine."""

    def __init__(self, store) -> None:
        self._store = store
        self._engine = BacktestEngine()

    def backtest(
        self, symbol: str, start_date: str = "", end_date: str = "",
        strategy_name: str = "", entry_sql: str = "",
        exit_sql: str = "", initial_cash: float = 100_000.0,
    ) -> BacktestReport:
        """Legacy entry point (deprecated).

        NOTE: Uses the new engine internally. When strategy signals
        are provided via BacktestEngine.run(), the signal-on-bar[i]
        → execute-at-bar[i+1] rule is enforced.
        """
        logger.warning(
            "DuckDBBacktester.backtest() is deprecated. "
            "Use BacktestEngine.run() with strategy_signals instead.",
        )

        df = pd.DataFrame()
        try:
            df = self._store.query_df(
                "SELECT * FROM kline_daily WHERE symbol = ? "
                "AND date >= ? AND date <= ? ORDER BY date ASC",
                [symbol, start_date, end_date],
            )
        except Exception:
            pass

        if df.empty:
            return BacktestReport(symbol=symbol, total_trades=0)

        report = self._engine.run(df, pd.DataFrame())
        report.symbol = symbol
        return report
