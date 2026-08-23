# [AI:BEGIN]
# PA_MCP - Research: K 线几何特征（从 PA_Agent 搬运适配）
#
# 来源：PA_Agent/PA_Agent2 的 ai/kline_features.py（Al Brooks 价格行为
# 特征工程）。本项目 factors.py 覆盖指标类因子（MA/RSI/布林/ATR/量比），
# 但缺少"K 线形态"维度——实体比、影线、内包/外包、ii/iii、突破跟随等
# 确定性形态特征。这是喂给 LLM 的"看图能力"（数值化 K 线 + 形态标签）。
#
# 适配差异：
#   - PA 用 newest-first 的 KlineBar 序列；本项目为 date 升序 DataFrame
#   - PA 用 dataclass 输出；本项目输出 dict 列表（MCP 序列化友好）
#   - 指标复用本项目 engine/indicators（calc_ema / calc_atr）
#   - 纯函数、零网络、零 gradio 依赖，可单测
# [AI:END]

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

_GEOMETRY_COLS = ("open", "high", "low", "close")


def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "date" in d.columns:
        d = d.sort_values("date").reset_index(drop=True)
    for c in _GEOMETRY_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _round_or_none(v: Optional[float]) -> Optional[float]:
    if v is None or math.isnan(v):
        return None
    return round(v, 3)


def _overlap_ratio(bar: dict, prev: Optional[dict]) -> Optional[float]:
    if prev is None:
        return None
    hi = min(bar["high"], prev["high"])
    lo = max(bar["low"], prev["low"])
    overlap = max(0.0, hi - lo)
    denom = max(bar["high"], prev["high"]) - min(bar["low"], prev["low"])
    if denom <= 0:
        return None
    return overlap / denom


def _is_inside(bar: Optional[dict], prev: Optional[dict]) -> bool:
    if bar is None or prev is None:
        return False
    return bar["high"] <= prev["high"] and bar["low"] >= prev["low"]


def _is_outside(bar: Optional[dict], prev: Optional[dict]) -> bool:
    if bar is None or prev is None:
        return False
    return bar["high"] >= prev["high"] and bar["low"] <= prev["low"]


def _feature_for_row(d: pd.DataFrame, i: int, ema20: pd.Series,
                     atr14: pd.Series) -> dict[str, Any]:
    """单根 K 线的几何特征（i 为 date 升序索引）。"""
    bar = {"open": float(d["open"].iloc[i]), "high": float(d["high"].iloc[i]),
           "low": float(d["low"].iloc[i]), "close": float(d["close"].iloc[i])}
    prev = ({"open": float(d["open"].iloc[i - 1]), "high": float(d["high"].iloc[i - 1]),
             "low": float(d["low"].iloc[i - 1]), "close": float(d["close"].iloc[i - 1])}
            if i > 0 else None)
    prev2 = ({"open": float(d["open"].iloc[i - 2]), "high": float(d["high"].iloc[i - 2]),
              "low": float(d["low"].iloc[i - 2]), "close": float(d["close"].iloc[i - 2])}
             if i > 1 else None)
    prev3 = ({"open": float(d["open"].iloc[i - 3]), "high": float(d["high"].iloc[i - 3]),
              "low": float(d["low"].iloc[i - 3]), "close": float(d["close"].iloc[i - 3])}
             if i > 2 else None)

    full_range = bar["high"] - bar["low"]
    body = abs(bar["close"] - bar["open"])
    body_ratio = upper_wick = lower_wick = close_pos = None
    if full_range > 0:
        body_ratio = body / full_range
        upper_wick = (bar["high"] - max(bar["open"], bar["close"])) / full_range
        lower_wick = (min(bar["open"], bar["close"]) - bar["low"]) / full_range
        close_pos = max(0.0, min(1.0, (bar["close"] - bar["low"]) / full_range))

    atr = float(atr14.iloc[i]) if i < len(atr14) and not pd.isna(atr14.iloc[i]) else math.nan
    ema = float(ema20.iloc[i]) if i < len(ema20) and not pd.isna(ema20.iloc[i]) else math.nan

    range_atr = None
    if full_range > 0 and not math.isnan(atr) and atr > 0:
        range_atr = full_range / atr

    ema_rel = "unknown"
    if not math.isnan(ema):
        ema_rel = "above" if bar["close"] > ema else ("below" if bar["close"] < ema else "touch")

    # 分类（优先内外包，其次十字星/趋势棒）
    bar_type = "flat"
    if prev is not None:
        if bar["high"] <= prev["high"] and bar["low"] >= prev["low"]:
            bar_type = "inside"
        elif bar["high"] >= prev["high"] and bar["low"] <= prev["low"]:
            bar_type = "outside_bull" if bar["close"] >= bar["open"] else "outside_bear"
    if bar_type == "flat":
        if body_ratio is None or close_pos is None:
            bar_type = "flat"
        elif body_ratio <= 0.25:
            bar_type = "doji"
        elif bar["close"] > bar["open"] and close_pos >= 0.65:
            bar_type = "trend_bull"
        elif bar["close"] < bar["open"] and close_pos <= 0.35:
            bar_type = "trend_bear"
        else:
            bar_type = "other"

    # ii / iii 内包序列（i, i-1, i-2 连续内包）
    inside_seq = "none"
    if _is_inside(bar, prev) and _is_inside(prev, prev2) and _is_inside(prev2, prev3):
        inside_seq = "iii"
    elif _is_inside(bar, prev) and _is_inside(prev, prev2):
        inside_seq = "ii"

    # 内-外-内（i-o-i）蓄势形态
    ioi = (_is_inside(prev2, prev3) and _is_outside(prev, prev2)
           and _is_inside(bar, prev))

    # 微型双底/双顶（MDB/MDT）
    micro_double = "none"
    if prev is not None:
        tol = atr * 0.02 if not math.isnan(atr) and atr > 0 else 0.0
        if abs(bar["low"] - prev["low"]) <= tol:
            micro_double = "MDB"
        elif abs(bar["high"] - prev["high"]) <= tol:
            micro_double = "MDT"

    # EMA 跳空状态
    gap_bar = "none"
    if not math.isnan(ema):
        if bar["low"] > ema:
            gap_bar = "bull_gap"
        elif bar["high"] < ema:
            gap_bar = "bear_gap"

    # 突破前 5 根区间
    lookback = d.iloc[max(0, i - 5):i]
    breakout = "none"
    if not lookback.empty:
        broke_high = bar["high"] > float(lookback["high"].max())
        broke_low = bar["low"] < float(lookback["low"].min())
        breakout = "both" if (broke_high and broke_low) else (
            "up" if broke_high else ("down" if broke_low else "none"))

    # 跟随/失败（后续 1-2 根同向延续）
    follow = "pending"
    direction = 1 if bar["close"] > bar["open"] else (-1 if bar["close"] < bar["open"] else 0)
    if direction != 0:
        newer = d.iloc[i + 1:i + 3]
        same = opp = 0
        for _, r in newer.iterrows():
            c = float(r["close"]); o = float(r["open"])
            if direction > 0:
                same += int(c > bar["close"])
                opp += int(c < bar["open"])
            else:
                same += int(c < bar["close"])
                opp += int(c > bar["open"])
        follow = "yes" if same > 0 else ("failed" if opp > 0 else "no")

    return {
        "date": str(d["date"].iloc[i])[:10],
        "bar_type": bar_type,
        "body_ratio": _round_or_none(body_ratio),
        "upper_wick_ratio": _round_or_none(upper_wick),
        "lower_wick_ratio": _round_or_none(lower_wick),
        "close_position": _round_or_none(close_pos),
        "range_atr_ratio": _round_or_none(range_atr),
        "ema_relation": ema_rel,
        "overlap_prev_ratio": _round_or_none(_overlap_ratio(bar, prev)),
        "inside_sequence": inside_seq,
        "ioi_pattern": ioi,
        "micro_double": micro_double,
        "gap_bar": gap_bar,
        "breakout_prev5": breakout,
        "follow_through": follow,
    }


def compute_kline_geometry(df: pd.DataFrame, detail_bars: int = 30) -> list[dict[str, Any]]:
    """K 线几何特征（最近 detail_bars 根，date 升序输入）。

    Args:
        df: 日线（date/open/high/low/close 至少；volume 可选）
        detail_bars: 输出最近 N 根（默认 30；EMA/ATR 在更老数据上预热）

    Returns:
        [{"date", "bar_type", "body_ratio", ..., "follow_through"}] 最新在前；
        数据不足返回可计算的部分。
    """
    d = _ensure(df)
    if d.empty or len(d) < 2:
        return []

    from pa_mcp.engine.indicators.indicators import calc_atr, calc_ema

    ema_df = calc_ema(d, periods=[20])
    ema = ema_df["ema20"] if "ema20" in ema_df else pd.Series([math.nan] * len(d))
    atr_df = calc_atr(d)
    atr = atr_df["atr14"] if "atr14" in atr_df else pd.Series([math.nan] * len(d))

    ema20 = pd.Series(ema.to_numpy() if hasattr(ema, "to_numpy") else ema)
    atr14 = pd.Series(atr.to_numpy() if hasattr(atr, "to_numpy") else atr)

    out = [_feature_for_row(d, i, ema20, atr14) for i in range(len(d))]
    return list(reversed(out[-detail_bars:]))  # 最新在前


def format_geometry_text(features: list[dict[str, Any]]) -> str:
    """特征表 → 文本（注入 LLM prompt 的紧凑格式）。"""
    if not features:
        return "无K线形态数据"
    rows = []
    for f in features:
        bits = [
            f["date"][5:],
            f["bar_type"],
            f"实体{f['body_ratio'] or 0:.2f}",
            f"收位{f['close_position'] or 0:.2f}",
        ]
        if f["inside_sequence"] != "none":
            bits.append(f["inside_sequence"])
        if f["ioi_pattern"]:
            bits.append("IOI")
        if f["micro_double"] != "none":
            bits.append(f["micro_double"])
        if f["gap_bar"] != "none":
            bits.append(f["gap_bar"])
        if f["breakout_prev5"] != "none":
            bits.append(f"破{f['breakout_prev5']}")
        if f["follow_through"] != "pending":
            bits.append(f"跟{f['follow_through']}")
        if f["ema_relation"] != "unknown":
            bits.append(f"EMA{f['ema_relation']}")
        rows.append(" ".join(bits))
    return "\n".join(rows)
