# [AI:BEGIN]
# PA_MCP - Engine Layer: 海龟交易法则（唐奇安通道突破 + ATR 波动率仓位）
#
# 源自 Richard Dennis 海龟实验（经典趋势跟踪体系），本实现为 A 股适配简化版：
#   1. 入场（系统1）：close 突破前 entry_period 日最高价 → 做多信号
#   2. 离场参考：close 跌破前 exit_period 日最低价 → 趋势结束
#      （不反向做空，metadata 标记 trend_end 供回测/分析参考）
#   3. 仓位：波动率目标——1 单位 = 账户资金 × risk_percent% ÷ (ATR × 100)，
#      折算为建议仓位百分比（受 10% 上限约束）
#   4. strength_score：突破强度（突破幅度/ATR）+ 量能确认 + 距 52 周新高
#
# 自动注册：StrategyRegistry.auto_discover() 发现本类；
# 自动接入：事件驱动回测 / 事件研究 / 市场扫描（scan_market 的候选策略）。
# [AI:END]

from __future__ import annotations

import pandas as pd

from pa_mcp.engine.strategies.base import (
    BaseStrategy, MarketState, ParamRange, Signal, SignalDirection,
    StrategyCategory, TimeFrame,
)


class TurtleBreakoutStrategy(BaseStrategy):
    """海龟交易法则：唐奇安 20 日新高突破入场，ATR 波动率目标仓位。

    Params:
        entry_period: 唐奇安入场通道天数（经典 20）
        exit_period: 唐奇安离场通道天数（经典 10）
        atr_period: ATR 计算周期（经典 20）
        risk_percent: 每单风险预算（账户 %，经典 1%）
        volume_confirm_ratio: 突破放量确认阈值（0 = 不要求放量）
    """

    name = "turtle"
    category = StrategyCategory.TREND
    description = "海龟交易：唐奇安20日突破 + ATR波动率仓位（经典趋势跟踪）"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.FERMENTING, MarketState.CLIMAX,
                       MarketState.STARTING]

    entry_period: int = 20
    exit_period: int = 10
    atr_period: int = 20
    risk_percent: float = 1.0
    volume_confirm_ratio: float = 0.0

    def get_params_space(self) -> list[ParamRange]:
        return [
            ParamRange("entry_period", 10, 55, 5, "int"),
            ParamRange("exit_period", 5, 20, 5, "int"),
            ParamRange("atr_period", 10, 30, 5, "int"),
            ParamRange("risk_percent", 0.5, 2.0, 0.25, "float"),
        ]

    def _atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """ATR(period)：TR = max(高-低, |高-前收|, |低-前收|) 的滚动均值。"""
        high, low, close = data["high"], data["low"], data["close"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low,
                        (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if data is None or data.empty or len(data) < self.entry_period + 2:
            return []
        df = data.sort_values("date").reset_index(drop=True)
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else ""

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else None
        atr = self._atr(df, self.atr_period)
        # 唐奇安通道（用前一日之前的数据，无未来函数）
        entry_channel = high.rolling(self.entry_period).max().shift(1)
        exit_channel = low.rolling(self.exit_period).min().shift(1)
        high52 = high.rolling(250).max()

        if volume is not None:
            avg_vol20 = volume.rolling(20).mean()
        else:
            avg_vol20 = None

        signals: list[Signal] = []
        for i in range(self.entry_period + 1, len(df)):
            a = atr.iloc[i]
            if pd.isna(a) or a <= 0:
                continue
            c = close.iloc[i]
            entry_level = entry_channel.iloc[i]
            if pd.isna(entry_level) or c <= entry_level:
                continue

            # 突破强度：突破幅度 / ATR
            gap = (c - entry_level) / a
            vol_ok = True
            vol_detail = ""
            if volume is not None and self.volume_confirm_ratio > 0:
                v20 = avg_vol20.iloc[i]
                ratio = volume.iloc[i] / v20 if v20 and v20 > 0 else 0.0
                vol_ok = ratio >= self.volume_confirm_ratio
                vol_detail = f"量比 {ratio:.1f}"

            # 距 52 周新高（越接近越强）
            h52 = high52.iloc[i]
            near_high = 0.0
            if pd.notna(h52) and h52 > 0:
                near_high = min(1.0, c / h52 - 0.9)  # 0.9 起线性 0-1

            strength = min(100, round(
                50 + gap * 8 + (20 if vol_ok else 0) + near_high * 30, 1))

            # ATR 波动率目标仓位：1 单位风险 = risk_percent% 账户 / ATR%
            atr_pct = a / c * 100
            pos_pct = min(10.0, round(self.risk_percent / max(atr_pct, 0.1) * 100, 1))
            exit_level = exit_channel.iloc[i]

            evidence = [{
                "finding": (f"突破 {self.entry_period} 日高点 {entry_level:.2f}"
                            f"（现价 {c:.2f}，超出 {gap:.2f} ATR）"),
                "impact": "positive",
            }]
            if vol_detail:
                evidence.append({"finding": vol_detail, "impact": "positive"})
            evidence.append({
                "finding": f"ATR {a:.3f}（{atr_pct:.2f}%），止损参考 "
                           f"{exit_level:.2f}（{self.exit_period} 日通道）",
                "impact": "neutral",
            })

            signals.append(Signal(
                symbol=symbol,
                strategy_name=self.name,
                strength_score=strength,
                direction=SignalDirection.BULLISH,
                evidence=evidence,
                suggested_max_position_pct=pos_pct,
                signal_time=str(df["date"].iloc[i])[:10],
                metadata={
                    "entry_level": round(float(entry_level), 3),
                    "exit_level": round(float(exit_level), 3),
                    "atr": round(float(a), 4),
                    "atr_pct": round(atr_pct, 2),
                    "position_pct": pos_pct,
                    "trend_end": float(exit_level) if pd.notna(exit_level) else None,
                },
            ))
        return signals
