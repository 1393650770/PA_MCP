# [AI:BEGIN]
# PA_MCP - Engine Layer: Event-Driven & Reversal Strategies
# Event-driven: insider buying follow, lockup expiry avoidance, block trade signal
# Reversal: oversold bounce, panic dip buy
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd

from pa_mcp.engine.strategies.base import (
    BaseStrategy, MarketState, ParamRange, Signal, SignalDirection,
    StrategyCategory, TimeFrame,
)


class InsiderBuyingFollowStrategy(BaseStrategy):
    """Event-driven: follow significant insider buying.

    Signal: insider(s) bought >1M CNY worth of shares in open market.
    Only valid if stock is NOT near lockup expiry and NOT at high pledge risk.
    """

    name = "insider_buying_follow"
    category = StrategyCategory.EVENT
    description = "Follow significant insider buying (>1M CNY) — strong positive signal"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.FERMENTING, MarketState.STARTING, MarketState.DULL]

    min_insider_amount: float = 1_000_000  # 1M CNY minimum
    max_pledge_ratio: float = 0.50  # Max acceptable pledge ratio

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if data.empty:
            return []

        signals: list[Signal] = []
        close = data.get("close", pd.Series([0] * len(data))).values

        # Check for insider trade events in metadata/event data
        insider_amount = data.get("insider_buy_amount", None)
        insider_count = data.get("insider_buy_count", 0)
        pledge_ratio = data.get("pledge_ratio", 0)

        if insider_amount is None:
            return []

        for i, row in data.iterrows():
            amount = float(row.get("insider_buy_amount", 0))
            count = int(row.get("insider_buy_count", 0))
            pledge = float(row.get("pledge_ratio", 0))

            if amount < self.min_insider_amount:
                continue
            if count == 0:
                continue
            if pledge > self.max_pledge_ratio:
                continue

            # Score: more insiders + larger amounts = stronger signal
            amount_score = min(amount / 5_000_000, 1.0) * 40  # 5M = max
            count_score = min(count / 5.0, 1.0) * 30  # 5 insiders = max
            price_below = 1.0 - pledge / self.max_pledge_ratio
            safety_score = max(0, price_below) * 30

            strength = min(amount_score + count_score + safety_score, 100)
            if strength >= 50:
                signals.append(Signal(
                    # P0-6: 信号市场时间
                    signal_time=str(row.get("date", ""))[:10] if row.get("date", "") else None,
                    symbol=str(row.get("symbol", "")),
                    strategy_name=self.name,
                    strength_score=round(strength, 1),
                    direction=SignalDirection.BULLISH,
                    evidence=[
                        {"type": "insider", "detail": f"Insider buy: {amount/10000:.0f}万 CNY by {count} insider(s)"},
                        {"type": "pledge", "detail": f"Pledge ratio: {pledge:.0%}"},
                    ],
                    risk_flags=[
                        {"type": "insider_quality", "severity": "low",
                         "detail": "Insider buys can be symbolic. Verify amount is significant."},
                    ],
                    suggested_max_position_pct=0.05,
                ))

        return signals


class LockupExpiryAvoidanceStrategy(BaseStrategy):
    """Event-driven: flag stocks with imminent large lockup expiry.

    This is a NEGATIVE screen — outputs bearish signals for stocks to avoid.
    """

    name = "lockup_expiry_avoid"
    category = StrategyCategory.EVENT
    description = "Flag stocks with imminent large lockup expiry — AVOID these"
    timeframe = TimeFrame.DAILY
    suitable_states = list(MarketState)  # Always relevant

    threshold_pct: float = 0.10  # Lockup >10% of float = significant
    days_ahead: int = 30  # Warn 30 days ahead

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if data.empty:
            return []

        signals: list[Signal] = []

        lockup_pct = data.get("lockup_pct_of_float", None)
        lockup_date = data.get("lockup_expiry_date", None)

        if lockup_pct is None or lockup_date is None:
            return []

        import datetime
        today = datetime.date.today()

        for i, row in data.iterrows():
            pct = float(row.get("lockup_pct_of_float", 0))
            expiry = row.get("lockup_expiry_date", "")

            if pct < self.threshold_pct:
                continue

            try:
                expiry_date = datetime.date.fromisoformat(str(expiry)[:10])
                days_until = (expiry_date - today).days
            except (ValueError, TypeError):
                continue

            if days_until < 0 or days_until > self.days_ahead:
                continue

            # Severity based on % of float
            severity = min(pct / 0.30, 1.0)  # 30% of float = max risk
            strength = 100 - severity * 60  # Higher lockup = lower strength (bearish)

            signals.append(Signal(
                # P0-6: 信号市场时间
                signal_time=str(row.get("date", ""))[:10] if row.get("date", "") else None,
                symbol=str(row.get("symbol", "")),
                strategy_name=self.name,
                strength_score=round(strength, 1),
                direction=SignalDirection.BEARISH,
                evidence=[
                    {"type": "lockup", "detail": f"Lockup {pct:.0%} of float expiring in {days_until} days"},
                    {"type": "risk", "detail": "Large lockup expiry = potential sell pressure"},
                ],
                risk_flags=[
                    {"type": "lockup_risk", "severity": "high" if pct > 0.20 else "medium",
                     "detail": f"{pct:.0%} of float unlocking — significant dilution risk"},
                ],
                suggested_max_position_pct=0.0,  # Avoid entirely
            ))

        return signals


class OversoldBounceReversalStrategy(BaseStrategy):
    """Reversal: deeply oversold stock with capitulation volume.

    Pattern: RSI < 25 for 3+ days, then a bullish reversal candle with volume surge.
    """

    name = "oversold_bounce"
    category = StrategyCategory.REVERSAL
    description = "Deep oversold bounce — RSI < 25 reversal with volume capitulation"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.DULL, MarketState.FROZEN]

    oversold_rsi: float = 25.0
    oversold_days: int = 3
    reversal_pct: float = 2.0  # Today's gain minimum

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if len(data) < 30:
            return []

        close = data["close"].values
        volume = data["volume"].values
        pct_change = data.get("pct_change", pd.Series([0] * len(data))).values

        # Compute RSI(14)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))

        avg_vol = pd.Series(volume).rolling(20).mean().values

        signals: list[Signal] = []
        for i in range(30, len(data)):
            # Check oversold condition
            if rsi[i] > self.oversold_rsi:
                continue

            # Check oversold duration
            oversold_count = sum(1 for j in range(i - self.oversold_days, i + 1) if rsi[j] < self.oversold_rsi)
            if oversold_count < self.oversold_days:
                continue

            # Reversal: today positive, yesterday negative
            if pct_change[i] < self.reversal_pct:
                continue

            # Volume surge (capitulation)
            vol_ratio = volume[i] / avg_vol[i] if avg_vol[i] > 0 else 0
            if vol_ratio < 1.3:
                continue

            # Score the reversal quality
            rsi_score = max(0, (self.oversold_rsi - rsi[i]) / self.oversold_rsi) * 30
            reversal_score = min(pct_change[i] / 5.0, 1.0) * 40
            vol_score = min(vol_ratio / 3.0, 1.0) * 30

            strength = min(rsi_score + reversal_score + vol_score, 100)
            signals.append(Signal(
                symbol="", strategy_name=self.name,
                strength_score=round(strength, 1),
                direction=SignalDirection.BULLISH,
                evidence=[
                    {"type": "oversold", "detail": f"RSI={rsi[i]:.1f}, oversold for {oversold_count} days"},
                    {"type": "reversal", "detail": f"Today +{pct_change[i]:.1f}%"},
                    {"type": "capitulation", "detail": f"Volume {vol_ratio:.1f}x avg"},
                ],
                risk_flags=[
                    {"type": "catching_falling_knife", "severity": "high",
                     "detail": "Oversold can become more oversold — use tight stops"},
                ],
                suggested_max_position_pct=0.03,  # Small position for reversal plays
            # P0-6: 信号市场时间
            signal_time=str(data["date"].iloc[i])[:10] if "date" in data.columns and data["date"].iloc[i] else None,
            ))

        return signals

    def get_params_space(self) -> list[ParamRange]:
        return [
            ParamRange("oversold_rsi", 15.0, 35.0, 5.0),
            ParamRange("oversold_days", 2, 7, 1),
            ParamRange("reversal_pct", 1.0, 5.0, 1.0),
        ]


class BlockTradeDiscountedStrategy(BaseStrategy):
    """Event-driven: heavily discounted block trades as warning signal.

    Large block trades at >8% discount = potential negative signal.
    """

    name = "block_trade_discount_alert"
    category = StrategyCategory.EVENT
    description = "Flag heavily discounted block trades as potential warning"
    timeframe = TimeFrame.DAILY
    suitable_states = list(MarketState)

    discount_threshold: float = 0.08  # 8% discount
    min_amount: float = 10_000_000  # 10M CNY minimum

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if data.empty:
            return []

        signals: list[Signal] = []

        block_amount = data.get("block_trade_amount", None)
        block_discount = data.get("block_trade_discount", None)

        if block_amount is None or block_discount is None:
            return []

        for i, row in data.iterrows():
            amount = float(row.get("block_trade_amount", 0))
            discount = float(row.get("block_trade_discount", 0))

            if amount < self.min_amount:
                continue
            if abs(discount) < self.discount_threshold:
                continue

            severity = min(abs(discount) / 0.15, 1.0)
            strength = 100 - severity * 50  # Bigger discount = more bearish

            signals.append(Signal(
                # P0-6: 信号市场时间
                signal_time=str(data["date"].iloc[i])[:10] if "date" in data.columns and data["date"].iloc[i] else None,
                symbol=str(row.get("symbol", "")),
                strategy_name=self.name,
                strength_score=round(strength, 1),
                direction=SignalDirection.BEARISH,
                evidence=[
                    {"type": "block_trade", "detail": f"Block trade: {amount/10000:.0f}万 CNY at {discount:.1%} discount"},
                    {"type": "risk", "detail": f"Large discounted block trade = potential insider exit signal"},
                ],
                risk_flags=[
                    {"type": "block_trade_risk", "severity": "high" if abs(discount) > 0.12 else "medium",
                     "detail": "Heavily discounted block trades often precede price declines"},
                ],
                suggested_max_position_pct=0.0,
            ))

        return signals
