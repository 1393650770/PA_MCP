# [AI:BEGIN]
# PA_MCP - 策略中文说明（tips）与最优策略检测
# [AI:END]

from __future__ import annotations

from typing import Optional

STRATEGY_TIPS: dict[str, str] = {
    "ma_golden_cross": (
        "**均线金叉**：短期均线上穿长期均线（默认 MA5×MA20）+ 放量确认。\n"
        "适用：趋势行情起步（发酵/启动期）。\n"
        "风险：震荡市频繁假金叉；信号滞后。"
    ),
    "platform_breakout": (
        "**平台突破**：长时间横盘整理后放量突破箱体高点。\n"
        "适用：横盘后启动的突破行情。\n"
        "风险：假突破（诱多）常见，需量能确认。"
    ),
    "first_board_breakout": (
        "**首板启动**：首次涨停（连板梯队起点）+ 放量。\n"
        "适用：短线打板情绪周期启动。\n"
        "风险：涨停可能买不进；炸板率高；高波动。"
    ),
    "dragon_second_wave": (
        "**龙回头/二波**：龙头股回调后二次启动。\n"
        "适用：强势题材的二波机会。\n"
        "风险：龙头切换失败，回调深。"
    ),
    "bollinger_mean_reversion": (
        "**布林带均值回归**：价格跌破下轨后反转确认买入。\n"
        "适用：震荡市超卖反弹（低迷期最佳）。\n"
        "风险：单边下跌市会持续破轨；需严格止损。\n"
        "⭐ 研究验证：3/3 股事件研究达标、成本不敏感、组合正收益。"
    ),
    "macd_divergence_swing": (
        "**MACD 底背离**：价格新低但 MACD 抬高 + 放量确认。\n"
        "适用：下跌末端的反转。\n"
        "风险：背离后可再背离，左侧信号。"
    ),
    "roe_pb_value": (
        "**低估值+高ROE**：PE/PB 低于阈值且 ROE>10% 的价值股。\n"
        "适用：弱市防御/价值回归。\n"
        "风险：价值陷阱（基本面恶化）；依赖财务数据。"
    ),
    "range_grid": (
        "**网格交易**：在箱体内网格低买高卖。\n"
        "适用：明确横盘区间（低迷期）。\n"
        "风险：单边突破会大幅回撤；仅适合区间市。"
    ),
    "volume_price_momentum": (
        "**量价动量**：价格动量>5% 且量能放大确认。\n"
        "适用：强势趋势中继。\n"
        "风险：追高；动量衰竭时反转快。"
    ),
    "oversold_bounce": (
        "**超跌反弹**：RSI 超卖 + 放量反转阳线。\n"
        "适用：急跌后的修复。\n"
        "风险：接飞刀，超卖可更超卖；需紧止损。"
    ),
    "insider_buying_follow": (
        "**内部人增持**：大股东/高管增持 + 质押率低。\n"
        "适用：事件驱动的底部信号。\n"
        "风险：依赖增持数据；增持可能仅维护股价。"
    ),
    "lockup_expiry_avoid": (
        "**解禁回避**：大额限售解禁临近 → 看空信号。\n"
        "适用：规避解禁抛压。\n"
        "风险：情绪面因素，基本面强时可能失效。"
    ),
    "block_trade_discount_alert": (
        "**大宗折价**：大宗交易大幅折价 → 看空信号。\n"
        "适用：大股东出货识别。\n"
        "风险：折价也可能因接盘方锁定。"
    ),
}


def get_strategy_tip(strategy_name: str) -> Optional[str]:
    """获取策略中文说明。"""
    return STRATEGY_TIPS.get(strategy_name)


def pick_best_strategy(
    kline_df,
    candidates: Optional[list[str]] = None,
    symbol: str = "000001",
) -> str:
    """快速检测最优策略（事件研究 5 日超额评分）。

    对候选策略逐一跑信号事件研究，选 5 日超额收益最高者。
    检测失败回退默认。耗时约每策略 1-2 秒。
    """
    import pandas as pd

    from pa_mcp.engine.strategies.base import StrategyRegistry
    from pa_mcp.research.event_study import signal_forward_returns

    if kline_df is None or kline_df.empty:
        return "bollinger_mean_reversion"

    registry = StrategyRegistry()
    registry.auto_discover()

    if candidates is None:
        candidates = list(STRATEGY_TIPS.keys())

    best_score = -999.0
    best_name = "bollinger_mean_reversion"

    for name in candidates:
        cls = registry.get(name)
        if cls is None:
            continue
        try:
            signals = cls.generate_signals(kline_df.copy())
        except Exception:
            continue
        if not signals:
            continue
        sig_df = pd.DataFrame([{
            "symbol": symbol,
            "date": getattr(s, "signal_time", None) or str(getattr(s, "timestamp", ""))[:10],
            "direction": getattr(s, "direction", "neutral").value
                        if hasattr(getattr(s, "direction", None), "value")
                        else str(getattr(s, "direction", "neutral")),
            "strategy_name": name,
        } for s in signals])
        results = signal_forward_returns(kline_df, sig_df, [5])
        if not results:
            continue
        r = results[0]
        if r.n_events < 10:
            continue
        score = r.excess_return_pct
        if score > best_score:
            best_score = score
            best_name = name

    return best_name
