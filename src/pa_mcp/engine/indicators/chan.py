# [AI:BEGIN]
# PA_MCP - Indicators: 缠论（缠中说禅）分型 / 笔 / 中枢 / 背驰
#
# 缠论核心结构（简化但忠实原义的实现）：
#   1. K线合并（包含处理）：上涨中取高高（高点高+低点高），下跌中取低低
#   2. 分型：顶分型（中间K线高点最高+低点最高）、底分型（反之）
#   3. 笔：相邻顶/底分型交替连接，顶底间至少间隔 1 根合并K线
#   4. 中枢：连续 3 笔的重叠区间 [max(低), min(高)]（前3笔重叠即成立）
#   5. 背驰：中枢前后两笔的 MACD 面积比较——后段面积 < 前段面积且
#      价格创新高/新低 → 盘整背驰信号（涨/跌动能衰竭）
#
# 纯确定性算法，无 LLM、无外部依赖（pandas）。输出结构供 UI 画图与 MCP 消费。
# [AI:END]

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MergedBar:
    """合并后 K 线。"""
    date: Any
    high: float
    low: float
    raw_indices: list[int] = field(default_factory=list)


@dataclass
class Fractal:
    """分型：top 顶分型 / bottom 底分型。"""
    kind: str
    bar_index: int          # 合并K线序号
    date: Any
    price: float            # 顶=高点，底=低点
    raw_index: int = -1     # 原始K线序号（画图用）


@dataclass
class Bi:
    """笔：一段走势（顶→底 或 底→顶）。"""
    direction: str          # up / down
    start_idx: int          # 合并K线序号
    end_idx: int
    start_price: float
    end_price: float
    start_date: Any = None
    end_date: Any = None


@dataclass
class Zhongshu:
    """中枢：连续 3 笔重叠区间。"""
    start_idx: int
    end_idx: int
    low: float              # max(低)  = 中枢下沿
    high: float             # min(高)  = 中枢上沿
    bi_count: int = 3


@dataclass
class ChanAnalysis:
    symbol: str
    merged_bars: list[MergedBar] = field(default_factory=list)
    fractals: list[Fractal] = field(default_factory=list)
    bi_list: list[Bi] = field(default_factory=list)
    zhongshu_list: list[Zhongshu] = field(default_factory=list)
    beichi_signal: str = "none"   # bullish / bearish / none
    beichi_detail: str = ""
    position: str = "unknown"     # 中枢内 / 中枢上沿 / 中枢下沿 / 无中枢

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "merged_bar_count": len(self.merged_bars),
            "fractal_count": len(self.fractals),
            "bi_count": len(self.bi_list),
            "zhongshu_count": len(self.zhongshu_list),
            "zhongshu_list": [
                {"start_idx": z.start_idx, "end_idx": z.end_idx,
                 "low": round(z.low, 3), "high": round(z.high, 3)}
                for z in self.zhongshu_list],
            "beichi_signal": self.beichi_signal,
            "beichi_detail": self.beichi_detail,
            "position": self.position,
            "last_bi": ({"direction": self.bi_list[-1].direction,
                         "start_price": self.bi_list[-1].start_price,
                         "end_price": self.bi_list[-1].end_price,
                         "start_date": str(self.bi_list[-1].start_date)[:10],
                         "end_date": str(self.bi_list[-1].end_date)[:10]}
                        if self.bi_list else None),
        }


def merge_kline(df: pd.DataFrame) -> list[MergedBar]:
    """K 线包含处理（缠论合并规则）。

    方向判定：前一根合并K线 高>低 → 上涨，取「高高」；否则取「低低」。
    包含关系：前K high >= 后K high 且 前K low <= 后K low（或反之）。
    """
    if df is None or df.empty:
        return []
    data = df.sort_values("date").reset_index(drop=True)
    merged: list[MergedBar] = []
    for idx, row in data.iterrows():
        bar = MergedBar(date=row["date"], high=float(row["high"]),
                        low=float(row["low"]), raw_indices=[idx])
        if not merged:
            merged.append(bar)
            continue
        prev = merged[-1]
        # 判断包含
        if (prev.high >= bar.high and prev.low <= bar.low) or \
                (prev.high <= bar.high and prev.low >= bar.low):
            rising = prev.high > prev.low
            if rising:
                prev.high = max(prev.high, bar.high)
                prev.low = max(prev.low, bar.low)
            else:
                prev.high = min(prev.high, bar.high)
                prev.low = min(prev.low, bar.low)
            prev.raw_indices.append(idx)
            prev.date = bar.date
        else:
            merged.append(bar)
    return merged


def find_fractals(merged: list[MergedBar]) -> list[Fractal]:
    """分型识别：顶分型 = 中间高点最高且中间低点最高；底分型反之。"""
    fractals: list[Fractal] = []
    n = len(merged)
    for i in range(1, n - 1):
        a, b, c = merged[i - 1], merged[i], merged[i + 1]
        if b.high > a.high and b.high > c.high and b.low > a.low and b.low > c.low:
            fractals.append(Fractal("top", i, b.date, b.high,
                                    b.raw_indices[-1]))
        elif b.low < a.low and b.low < c.low and b.high < a.high and b.high < c.high:
            fractals.append(Fractal("bottom", i, b.date, b.low,
                                    b.raw_indices[-1]))
    return fractals


def build_bi(merged: list[MergedBar], fractals: list[Fractal]) -> list[Bi]:
    """笔：交替连接顶/底分型，同型相邻取更极端者，间隔至少 1 根合并K线。"""
    if len(fractals) < 2:
        return []
    # 同型去重：连续同型分型保留更极端（顶取高、底取低）
    cleaned: list[Fractal] = []
    for f in fractals:
        if not cleaned:
            cleaned.append(f)
            continue
        last = cleaned[-1]
        if f.kind == last.kind:
            if f.kind == "top" and f.price >= last.price:
                cleaned[-1] = f
            elif f.kind == "bottom" and f.price <= last.price:
                cleaned[-1] = f
        else:
            cleaned.append(f)

    bi_list: list[Bi] = []
    for i in range(len(cleaned) - 1):
        a, b = cleaned[i], cleaned[i + 1]
        if a.kind == b.kind:
            continue
        # 间隔校验：顶底至少相隔 1 根合并K线（i+1 - i >= 1）
        if b.bar_index - a.bar_index < 1:
            continue
        if a.kind == "bottom":
            bi_list.append(Bi("up", a.bar_index, b.bar_index,
                              a.price, b.price, a.date, b.date))
        else:
            bi_list.append(Bi("down", a.bar_index, b.bar_index,
                              a.price, b.price, a.date, b.date))
    return bi_list


def build_zhongshu(bi_list: list[Bi]) -> list[Zhongshu]:
    """中枢：连续 3 笔重叠区间 [max(低), min(高)]。

    简化：取每 3 笔一组（第一笔高低点与后两笔比较），
    重叠区间 = [max(第2、3笔低点, 第一笔低点), min(第1、2笔高点, 第3笔高点)]。
    标准定义：Z = [max(三笔低点), min(三笔高点)]，且 max(低) < min(高) 才算重叠。
    """
    zones: list[Zhongshu] = []
    for i in range(len(bi_list) - 2):
        b1, b2, b3 = bi_list[i], bi_list[i + 1], bi_list[i + 2]
        lows = [min(b1.start_price, b1.end_price),
                min(b2.start_price, b2.end_price),
                min(b3.start_price, b3.end_price)]
        highs = [max(b1.start_price, b1.end_price),
                 max(b2.start_price, b2.end_price),
                 max(b3.start_price, b3.end_price)]
        z_low, z_high = max(lows), min(highs)
        if z_low < z_high:
            zones.append(Zhongshu(i, i + 2, z_low, z_high))
    # 合并重叠中枢（简化：保留全部，UI 画最近一个）
    return zones


def detect_beichi(bi_list: list[Bi], zones: list[Zhongshu],
                  macd_hist: Optional[pd.Series] = None) -> tuple[str, str]:
    """背驰检测：中枢后段笔的 MACD 面积 < 前段笔面积 且 价格创新高/低。

    简化实现（盘整背驰）：
      - 上涨背驰：中枢后的向上笔 创新高（end_price > 前一段高点），
        且该笔 MACD 柱面积 < 中枢前的向上笔面积 → bearish（涨势衰竭）
      - 下跌背驰：反之 → bullish（跌势衰竭）
    macd_hist 为 None 时用笔长度（笔幅度/笔内K数）近似动能。
    """
    if len(zones) < 1 or len(bi_list) < 5:
        return "none", "笔/中枢不足，无法判定背驰"
    z = zones[-1]
    if z.end_idx + 2 >= len(bi_list):
        return "none", "中枢后无足够笔"

    def _energy(bi: Bi, macd_hist) -> float:
        """笔动能：MACD 面积；无 MACD 时用幅度（保守近似）。"""
        if macd_hist is not None and len(macd_hist) > bi.end_idx:
            seg = macd_hist.iloc[bi.start_idx:bi.end_idx + 1]
            if not seg.empty and seg.notna().any():
                return float(seg.sum())
        return abs(bi.end_price - bi.start_price)

    # 中枢前的同向笔（b1 是下笔起点）：中枢前向上笔 = bi_list[z.start_idx - 1]（若存在）
    after_bi = bi_list[z.end_idx + 1]   # 中枢后第一笔（延续方向）
    before_idx = z.start_idx - 1
    if before_idx < 0:
        return "none", "中枢前无对比笔"
    before_bi = bi_list[before_idx]

    if after_bi.direction != before_bi.direction:
        return "none", "前后笔方向不一致，无法对比"

    e_before = _energy(before_bi, macd_hist)
    e_after = _energy(after_bi, macd_hist)

    if after_bi.direction == "up":
        new_high = after_bi.end_price > before_bi.end_price * 0.999
        if new_high and e_after < e_before * 0.8:
            return ("bearish",
                    f"上涨背驰：中枢后笔创新高（{after_bi.end_price:.2f}）但动能衰减"
                    f"（面积 {e_after:.4f} < 前段 {e_before:.4f} × 0.8），涨势衰竭")
    else:
        new_low = after_bi.end_price < before_bi.end_price * 1.001
        if new_low and e_after < e_before * 0.8:
            return ("bullish",
                    f"下跌背驰：中枢后笔创新低（{after_bi.end_price:.2f}）但动能衰减"
                    f"（面积 {e_after:.4f} < 前段 {e_before:.4f} × 0.8），跌势衰竭")
    return "none", f"无背驰（后段动能 {e_after:.4f} vs 前段 {e_before:.4f}）"


def chan_analysis(df: pd.DataFrame, symbol: str = "",
                  with_macd: bool = True) -> ChanAnalysis:
    """缠论分析主入口：合并 → 分型 → 笔 → 中枢 → 背驰。"""
    merged = merge_kline(df)
    if len(merged) < 5:
        return ChanAnalysis(symbol=symbol, merged_bars=merged)

    fractals = find_fractals(merged)
    bi_list = build_bi(merged, fractals)
    zones = build_zhongshu(bi_list)

    macd_hist = None
    if with_macd:
        try:
            from pa_mcp.engine.indicators.indicators import calc_macd
            m = calc_macd(df.sort_values("date").reset_index(drop=True))
            if "macd_hist" in m:
                macd_hist = m["macd_hist"]
        except Exception:
            pass

    signal, detail = detect_beichi(bi_list, zones, macd_hist)

    # 当前价格位置
    position = "unknown"
    if bi_list and zones and merged:
        last = merged[-1]
        z = zones[-1]
        if z.low <= last.low <= z.high:
            position = "中枢内"
        elif last.low < z.low:
            position = "中枢下方"
        else:
            position = "中枢上方"

    return ChanAnalysis(symbol=symbol, merged_bars=merged,
                        fractals=fractals, bi_list=bi_list,
                        zhongshu_list=zones, beichi_signal=signal,
                        beichi_detail=detail, position=position)


def format_chan(a: ChanAnalysis) -> str:
    """缠论分析 → 中文摘要。"""
    lines = [f"## 🌀 缠论分析 {a.symbol or ''}",
             f"- **结构**：合并K线 {len(a.merged_bars)} 根 / 分型 {len(a.fractals)} 个"
             f"（顶 {sum(1 for f in a.fractals if f.kind == 'top')} / "
             f"底 {sum(1 for f in a.fractals if f.kind == 'bottom')}）/ "
             f"笔 {len(a.bi_list)} 段 / 中枢 {len(a.zhongshu_list)} 个",
             f"- **当前位置**：{a.position}"]
    if a.bi_list:
        b = a.bi_list[-1]
        lines.append(f"- **最近一笔**：{'上涨' if b.direction == 'up' else '下跌'}"
                     f" {b.start_price:.2f} → {b.end_price:.2f}"
                     f"（{str(b.start_date)[:10]} ~ {str(b.end_date)[:10]}）")
    if a.zhongshu_list:
        z = a.zhongshu_list[-1]
        lines.append(f"- **最近中枢**：区间 [{z.low:.2f}, {z.high:.2f}]"
                     f"（{len(a.zhongshu_list)} 个中枢）")
    sig = a.beichi_signal
    if sig == "bearish":
        lines.append(f"⚠️ **{a.beichi_detail}**")
        lines.append("→ 缠论视角：涨势衰竭，警惕回调/中枢下移")
    elif sig == "bullish":
        lines.append(f"💡 **{a.beichi_detail}**")
        lines.append("→ 缠论视角：跌势衰竭，关注反转/中枢上移")
    else:
        lines.append(f"→ {a.beichi_detail}")
    lines.append("\n*缠论 = 缠中说禅《教你炒股票》结构分析体系（分型→笔→中枢→背驰）。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)
