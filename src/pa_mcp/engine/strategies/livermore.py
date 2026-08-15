# [AI:BEGIN]
# PA_MCP - Engine Layer: 利弗莫尔关键点策略（Pivot Point / 最小阻力线）
#
# 源自 Jesse Livermore《股票大作手回忆录》交易哲学，A 股做多适配：
#   1. 关键点（Pivot）：前 pivot_period 日的最高价 = 阻力关键点；
#      前低点 = 支撑关键点（趋势确认基准）
#   2. 进场：close 放量突破关键点（最小阻力线被击穿）→ 做多信号
#   3. 趋势确认：关键点突破必须发生在「顺势」环境——close > MA60
#      （利弗莫尔：只在上升趋势中做多，不抄底）
#   4. 离场参考：跌破前 pivot 低点（趋势破坏）→ metadata 标记
#   5. 强度：突破幅度/ATR + 量比 + 距 52 周新高
#
# 自动注册：StrategyRegistry.auto_discover()；自动接入回测/事件研究/扫描。
# [AI:END]

from __future__ import annotations

import pandas as pd

from pa_mcp.engine.strategies.base import (
    BaseStrategy, MarketState, ParamRange, Signal, SignalDirection,
    StrategyCategory, TimeFrame,
)


class LivermorePivotStrategy(BaseStrategy):
    """利弗莫尔关键点：枢轴突破 + 趋势确认 + 放量。

    Params:
        pivot_period: 枢轴窗口（前 N 日高低点，默认 20）
        trend_ma: 趋势确认均线（默认 60）
        volume_confirm_ratio: 突破放量阈值（0 = 不要求）
    """

    name = "livermore_pivot"
    category = StrategyCategory.TREND
    description = "利弗莫尔关键点：枢轴突破+趋势确认+放量（最小阻力线）"
    timeframe = TimeFrame.DAILY
    suitable_states = [MarketState.FERMENTING, MarketState.CLIMAX,
                       MarketState.STARTING]

    pivot_period: int = 20
    trend_ma: int = 60
    volume_confirm_ratio: float = 0.0

    def get_params_space(self) -> list[ParamRange]:
        return [
            ParamRange("pivot_period", 10, 40, 5, "int"),
            ParamRange("trend_ma", 30, 120, 30, "int"),
            ParamRange("volume_confirm_ratio", 0.0, 2.0, 0.5, "float"),
        ]

    def generate_signals(
        self, data: pd.DataFrame, market_state: MarketState | None = None,
    ) -> list[Signal]:
        if data is None or data.empty or len(data) < self.trend_ma + 5:
            return []
        df = data.sort_values("date").reset_index(drop=True)
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else ""

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else None

        # 枢轴关键点（shift(1)：不含当日，无未来函数）
        pivot_high = high.rolling(self.pivot_period).max().shift(1)
        pivot_low = low.rolling(self.pivot_period).min().shift(1)
        trend_line = close.rolling(self.trend_ma).mean()
        high52 = high.rolling(250).max()

        if volume is not None:
            avg_vol20 = volume.rolling(20).mean()
        else:
            avg_vol20 = None

        # ATR（突破强度）
        prev_close = close.shift(1)
        tr = pd.concat([high - low,
                        (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        signals: list[Signal] = []
        for i in range(self.trend_ma + 1, len(df)):
            c = close.iloc[i]
            ph = pivot_high.iloc[i]
            pl = pivot_low.iloc[i]
            ma = trend_line.iloc[i]
            a = atr.iloc[i]
            if pd.isna(ph) or pd.isna(pl) or pd.isna(ma) or pd.isna(a) or a <= 0:
                continue
            # 关键点突破 + 顺势确认
            if c <= ph or c <= ma:
                continue
            gap = (c - ph) / a

            # 放量确认（利弗莫尔：关键点须被大成交量击穿，无量突破视为假突破）
            vol_ok = True
            vol_detail = ""
            if volume is not None and self.volume_confirm_ratio > 0:
                v20 = avg_vol20.iloc[i]
                ratio = volume.iloc[i] / v20 if v20 and v20 > 0 else 0.0
                if ratio < self.volume_confirm_ratio:
                    continue  # 量能不足 → 假突破风险，不进场
                vol_ok = True
                vol_detail = f"量比 {ratio:.1f}"

            h52 = high52.iloc[i]
            near_high = 0.0
            if pd.notna(h52) and h52 > 0:
                near_high = min(1.0, c / h52 - 0.9)

            strength = min(100, round(
                45 + gap * 8 + (20 if vol_ok else 0) + near_high * 30
                + (10 if c > ma * 1.05 else 0), 1))

            evidence = [{
                "finding": (f"突破关键点 {ph:.2f}（前 {self.pivot_period} 日高点，"
                            f"超出 {gap:.2f} ATR），且站稳 MA{self.trend_ma} "
                            f"{ma:.2f}——最小阻力线被击穿"),
                "impact": "positive",
            }]
            if vol_detail:
                evidence.append({"finding": vol_detail, "impact": "positive"})
            evidence.append({
                "finding": f"离场参考：跌破枢轴低点 {pl:.2f}（趋势破坏）",
                "impact": "neutral",
            })

            signals.append(Signal(
                symbol=symbol,
                strategy_name=self.name,
                strength_score=strength,
                direction=SignalDirection.BULLISH,
                evidence=evidence,
                signal_time=str(df["date"].iloc[i])[:10],
                metadata={
                    "pivot_high": round(float(ph), 3),
                    "pivot_low": round(float(pl), 3),
                    "trend_ma": round(float(ma), 3),
                    "atr": round(float(a), 4),
                    "trend_break": float(pl) if pd.notna(pl) else None,
                },
            ))
        return signals
