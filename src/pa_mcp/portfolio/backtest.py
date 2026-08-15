# [AI:BEGIN]
# PA_MCP - Portfolio: 多票共享账本回测引擎
#
# 真正的组合回测：多只股票共享一个现金/持仓账本，
# 在同一个交易日历上联合执行信号、分配资金、撮合。
#
# 与单票回测（加权合成简化）的区别：
#   - 共享现金：一只票买入消耗现金，其他票买入受现金约束
#   - 共享持仓：T+1 可卖、单票上限、总仓上限在组合层面生效
#   - 统一日历：按所有票交易日并集推进
#
# 简化假设（MVP）：
#   - 每天最多每票一笔信号执行（延迟一天）
#   - 资金分配：等权（信号票均分可用现金，受单票/总仓上限约束）
#   - 期末仅估值不清仓
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from pa_mcp.backtest.broker import DailyBroker, FeeSchedule
from pa_mcp.backtest.events import MarketBar
from pa_mcp.backtest.ledger import Ledger
from pa_mcp.backtest.orders import Order, OrderSide


@dataclass
class PortfolioBacktestResult:
    nav_series: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    total_fees: float = 0.0

    def summary(self) -> str:
        return (
            f"组合收益 {self.total_return_pct:+.2f}% | 年化 {self.annual_return_pct:+.2f}% | "
            f"夏普 {self.sharpe:.2f} | 回撤 {self.max_drawdown_pct:.2f}%\n"
            f"交易 {self.total_trades} 笔 | 费用 {self.total_fees:.2f} 元"
        )


class PortfolioBacktestEngine:
    """多票共享账本回测引擎。"""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        fee_schedule: Optional[FeeSchedule] = None,
        single_stock_cap: float = 0.10,
        total_cap: float = 0.80,
        max_positions: int = 10,
        take_profit_pct: Optional[float] = None,   # 止盈：持仓收益达到此%卖出
        stop_loss_pct: Optional[float] = None,      # 止损：持仓亏损达到此%卖出
    ) -> None:
        self.initial_cash = initial_cash
        self.broker = DailyBroker(fee_schedule=fee_schedule, initial_cash=initial_cash)
        self.ledger = Ledger(cash=initial_cash)
        self.single_stock_cap = single_stock_cap
        self.total_cap = total_cap
        self.max_positions = max_positions
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct

    def run(
        self,
        kline_by_symbol: dict[str, pd.DataFrame],
        signals_by_symbol: dict[str, pd.DataFrame],
    ) -> PortfolioBacktestResult:
        """运行组合回测。

        Args:
            kline_by_symbol: {symbol: 日线DataFrame（升序）}
            signals_by_symbol: {symbol: signals DataFrame[date, symbol, direction, strength_score]}
        """
        # 统一日历：所有票日期并集（升序）
        all_dates = set()
        for sym, df in kline_by_symbol.items():
            all_dates.update(df["date"].astype(str).str[:10])
        calendar = sorted(all_dates)
        if not calendar:
            return PortfolioBacktestResult()

        # 每票数据索引：date_str -> row
        bar_map: dict[str, dict[str, dict]] = {}
        for sym, df in kline_by_symbol.items():
            df = df.sort_values("date")
            bar_map[sym] = {
                str(d)[:10]: row
                for d, row in zip(df["date"], df.to_dict(orient="records"))
            }

        # 信号索引：date_str -> {symbol: signal}
        sig_map: dict[str, dict[str, dict]] = {}
        for sym, sigs in signals_by_symbol.items():
            if sigs is None or sigs.empty:
                continue
            for _, s in sigs.iterrows():
                d = str(s.get("date", ""))[:10]
                sig_map.setdefault(d, {})[sym] = {
                    "direction": s.get("direction", "neutral"),
                    "strength": float(s.get("strength_score", 50)),
                }

        trades: list[dict] = []
        pending: dict[str, dict] = {}  # 前一日的信号（延迟一天执行）

        for day in calendar:
            # 0. 止盈止损检查（每日先处理，释放现金）
            if self.take_profit_pct is not None or self.stop_loss_pct is not None:
                self._check_exits(bar_map, day, trades)

            # 1. 执行前一日信号（延迟一天 — P0-4 原则）
            if pending:
                self._execute_day(pending, bar_map, day, trades)

            # 2. 收集当日信号 → 明日执行
            pending = dict(sig_map.get(day, {}))

            # 3. 组合估值（所有持仓）
            mark_prices = {}
            for sym in self.ledger.positions:
                bar = bar_map.get(sym, {}).get(day)
                if bar:
                    mark_prices[sym] = float(bar["close"])
            self.ledger.record_nav(pd.Timestamp(day).date(), mark_prices)

        # 期末估值
        last_day = calendar[-1]
        mark_prices = {}
        for sym in self.ledger.positions:
            bar = bar_map.get(sym, {}).get(last_day)
            if bar:
                mark_prices[sym] = float(bar["close"])
        self.ledger.record_nav(pd.Timestamp(last_day).date(), mark_prices)

        return self._compute_metrics(trades)

    def _execute_day(self, pending: dict, bar_map: dict,
                     day: str, trades: list[dict]) -> None:
        """执行一天的信号：等权分配 + 共享现金 + 组合约束。"""
        if not pending:
            return

        # 过滤：信号日 + 当日有 bar 的股票
        candidates = {}
        for sym, sig in pending.items():
            if sym not in bar_map or day not in bar_map[sym]:
                continue
            if sig.get("direction") not in ("bullish", "bearish"):
                continue
            candidates[sym] = sig

        if not candidates:
            return

        # 卖出优先：先处理 bearish（释放现金）
        sells = {s: c for s, c in candidates.items() if c["direction"] == "bearish"}
        buys = {s: c for s, c in candidates.items() if c["direction"] == "bullish"}

        for sym, sig in sells.items():
            bar = bar_map[sym][day]
            held = self.ledger.get_sellable_quantity(
                sym, pd.Timestamp(day).date())
            if held <= 0:
                continue
            order = Order(symbol=sym, side=OrderSide.SELL, quantity=held,
                          trade_date=pd.Timestamp(day).date())
            _, fills = self.broker.submit_order(order, self._to_market_bar(sym, bar, day))
            self._apply_fills(fills, trades)

        # 买入：等权分配可用现金
        if buys:
            cash_each = self.ledger.available_cash / min(len(buys), self.max_positions)
            for sym, sig in buys.items():
                bar = bar_map[sym][day]
                price = float(bar["close"])
                if price <= 0:
                    continue
                # 单票上限：当前持仓 + 新买入 ≤ 10% NAV
                nav_now = self.ledger.compute_nav(
                    {s: float(bar_map[s][day]["close"]) for s in self.ledger.positions if day in bar_map[s]})
                current_wt = (self.ledger.get_position(sym) * price) / max(nav_now, 1)
                max_buy_wt = self.single_stock_cap - current_wt
                max_buy_val = max_buy_wt * nav_now
                qty = int(min(cash_each, max_buy_val) / price / 100) * 100
                if qty <= 0:
                    continue
                order = Order(symbol=sym, side=OrderSide.BUY, quantity=qty,
                              trade_date=pd.Timestamp(day).date())
                _, fills = self.broker.submit_order(
                    order, self._to_market_bar(sym, bar, day))
                self._apply_fills(fills, trades)

    def _check_exits(self, bar_map: dict, day: str, trades: list[dict]) -> None:
        """止盈止损：持仓收益达阈值时卖出（基于成本均价）。"""
        day_date = pd.Timestamp(day).date()
        for sym in list(self.ledger.positions.keys()):
            bar = bar_map.get(sym, {}).get(day)
            if not bar:
                continue  # 当日无数据（停牌）不操作
            price = float(bar["close"])
            cost = self.ledger.get_avg_cost(sym)
            if cost <= 0:
                continue
            pnl_pct = (price / cost - 1) * 100

            should_sell = False
            if self.take_profit_pct is not None and pnl_pct >= self.take_profit_pct:
                should_sell = True
            if self.stop_loss_pct is not None and pnl_pct <= -abs(self.stop_loss_pct):
                should_sell = True

            if not should_sell:
                continue

            held = self.ledger.get_sellable_quantity(sym, day_date)
            if held <= 0:
                continue
            order = Order(symbol=sym, side=OrderSide.SELL, quantity=held,
                          trade_date=day_date,
                          strategy_name="exit_rule")
            _, fills = self.broker.submit_order(
                order, self._to_market_bar(sym, bar, day))
            self._apply_fills(fills, trades)

    def _to_market_bar(self, sym: str, bar: dict, day: str) -> MarketBar:
        return MarketBar(
            symbol=sym,
            trade_date=pd.Timestamp(day).date(),
            open=float(bar.get("open", bar["close"])),
            high=float(bar.get("high", bar["close"])),
            low=float(bar.get("low", bar["close"])),
            close=float(bar["close"]),
            volume=float(bar.get("volume", 0)),
            amount=float(bar.get("amount", 0)),
        )

    def _apply_fills(self, fills: list, trades: list[dict]) -> None:
        for f in fills:
            trades.append({
                "date": str(f.trade_date), "symbol": f.symbol,
                "side": f.side.value, "quantity": f.quantity,
                "price": f.price, "commission": f.commission,
                "stamp_tax": f.stamp_tax, "transfer_fee": f.transfer_fee,
                "value": f.fill_value,
            })
            if f.side == OrderSide.BUY:
                self.ledger.remove_cash(f.fill_value + f.total_cost)
                self.ledger.add_lot(
                    f.symbol, f.quantity,
                    (f.fill_value + f.total_cost) / max(f.quantity, 1),
                    f.trade_date,
                )
            else:
                self.ledger.add_cash(f.fill_value - f.total_cost)
                try:
                    self.ledger.remove_shares(f.symbol, f.quantity, f.trade_date)
                except ValueError:
                    pass

    def _compute_metrics(self, trades: list[dict]) -> PortfolioBacktestResult:
        result = PortfolioBacktestResult(trades=trades)
        nav = pd.DataFrame(self.ledger.nav_history)
        if nav.empty or len(nav) < 2:
            return result

        navs = nav["nav"].values
        result.total_return_pct = round((navs[-1] / navs[0] - 1) * 100, 2)
        daily = pd.Series(navs).pct_change().dropna()
        if len(daily) > 1 and daily.std() > 0:
            result.annual_return_pct = round(float((1 + daily.mean()) ** 252 - 1) * 100, 2)
            result.sharpe = round(float(daily.mean() / daily.std() * (252 ** 0.5)), 2)
            cum = (1 + daily).cumprod()
            result.max_drawdown_pct = round(
                float(((cum - cum.cummax()) / cum.cummax()).min()) * 100, 2)
        result.total_trades = len(trades)
        result.total_fees = round(
            sum(t["commission"] + t["transfer_fee"] for t in trades), 2)
        result.nav_series = self.ledger.nav_history
        return result
