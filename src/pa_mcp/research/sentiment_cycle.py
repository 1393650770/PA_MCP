# [AI:BEGIN]
# PA_MCP - Research: 游资情绪周期（涨停梯队 / 连板高度 / 晋级率 / 阶段判定）
#
# 借鉴著名游资方法论（龙头战法/情绪周期）与涨停板研究：
#   情绪四阶段：启动期 → 发酵期 → 高潮期 → 退潮期（+ 冰点期）
#   核心指标（从 kline_daily 日线可算，诚实标注近似）：
#     - 连板高度：个股连续涨停天数（pct_change >= 9.5 为涨停，10cm 近似）
#     - 梯队分布：首板 / 2板 / 3板 / 4板+ 数量
#     - 晋级率：今日≥2板数 ÷ 昨日涨停数（游资「晋级」概念）
#     - 情绪评分：涨停数 + 连板高度 + 晋级率加权（0-100）
#     - 阶段判定：晋级率骤降/高度回落 → 退潮；高度≥5 → 高潮；……
#   历史落库 sentiment_daily → 3 日趋势（退潮预警）
#
# 注意：日线只有收盘涨停，无盘中炸板数据 → 炸板率不计算（诚实标注），
# 晋级失败用「昨日涨停今日未涨停比例」近似。
# [AI:END]

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

LIMIT_UP_PCT = 9.5       # 涨停近似阈值（10cm；20cm 创业板科创板同判为 1 板）
LIMIT_DOWN_PCT = -9.5
HIGH_BOARD = 4           # 4板+ 视为高位连板
STAGE_UP_HEAT = 30       # 发酵期最小涨停数


class SentimentCycleAnalyzer:
    """游资情绪周期分析（数据：kline_daily 池内近似 + sentiment_daily 历史）。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    def _store(self):
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        path = self._store_path or get_settings().database.path
        store = DuckDBStore(path)
        store.connect()
        return store

    # ---- 连板计算（历史逐日） ----
    def _compute_streaks(self, store) -> pd.DataFrame:
        """全市场逐日连板高度：返回 date/board_height 明细。"""
        df = store.query_df(
            "SELECT symbol, date, pct_change FROM kline_daily "
            "ORDER BY symbol, date", [])
        if df.empty:
            return pd.DataFrame()
        df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
        rows = []
        prev: dict[str, int] = {}  # symbol → 昨日连板高度
        for _, r in df.iterrows():
            sym = r["symbol"]
            is_up = pd.notna(r["pct_change"]) and r["pct_change"] >= LIMIT_UP_PCT
            height = (prev.get(sym, 0) + 1) if is_up else 0
            prev[sym] = height
            rows.append({"date": str(r["date"])[:10], "symbol": sym,
                         "board_height": height, "is_limit_up": is_up,
                         "pct_change": float(r["pct_change"])
                         if pd.notna(r["pct_change"]) else 0.0})
        return pd.DataFrame(rows)

    # ---- 单日情绪统计 ----
    def _day_stats(self, streaks: pd.DataFrame, day: str,
                   prev_day: Optional[str]) -> dict[str, Any]:
        d = streaks[streaks["date"] == day]
        up = d[d["is_limit_up"]]
        total = len(up)
        heights = up["board_height"] if total else pd.Series(dtype=int)
        max_h = int(heights.max()) if total else 0
        n1 = int((heights == 1).sum()) if total else 0
        n2 = int((heights == 2).sum()) if total else 0
        n3 = int((heights == 3).sum()) if total else 0
        n4p = int((heights >= HIGH_BOARD).sum()) if total else 0
        down_count = int(
            ((d["is_limit_up"] == False) &  # noqa: E712
             (pd.to_numeric(d["pct_change"], errors="coerce") <= LIMIT_DOWN_PCT)).sum())

        # 晋级率：今日 ≥2 板 ÷ 昨日涨停数（游资晋级）
        promotion_rate = None
        if prev_day and prev_day in set(streaks["date"]):
            prev_up = streaks[(streaks["date"] == prev_day)
                              & streaks["is_limit_up"]]
            if len(prev_up) > 0:
                promotion_rate = round((n2 + n3 + n4p) / len(prev_up), 3)

        # 情绪评分：涨停数 40% + 连板高度 40% + 晋级率 20%（标准化）
        score = 0.0
        score += min(40, total / 2.0)                 # 80 家涨停 → 40 分
        score += min(40, max_h * 8)                   # 5 板 → 40 分
        if promotion_rate is not None:
            score += min(20, promotion_rate * 40)     # 50% 晋级 → 20 分
        score = round(min(100, score), 1)

        return {
            "date": day,
            "limit_up_count": total,
            "limit_down_count": down_count,
            "max_board_height": max_h,
            "board2_count": n2, "board3_count": n3, "board4p_count": n4p,
            "first_board_count": n1,
            "promotion_rate": promotion_rate,
            "sentiment_score": score,
        }

    # ---- 阶段判定（游资四阶段 + 冰点） ----
    @staticmethod
    def _stage(stats: dict[str, Any],
               prev_stats: Optional[dict[str, Any]]) -> tuple[str, str]:
        s = stats
        pr = s["promotion_rate"]
        max_h = s["max_board_height"]
        total = s["limit_up_count"]
        down = s["limit_down_count"]

        if total < 10 and down > 20:
            return "ice", "冰点期：涨停枯竭 + 跌停潮，空仓等情绪修复"
        if max_h >= 5 and total >= 50:
            return "climax", "高潮期：连板高度 ≥5 且涨停 ≥50，情绪亢奋防兑现"
        if pr is not None:
            if pr < 0.25 and max_h >= 3:
                return "recess", "退潮期：晋级率 <25%，连板高度回落，亏钱效应扩散"
        if prev_stats:
            if (max_h <= prev_stats["max_board_height"] - 2
                    and total <= prev_stats["limit_up_count"] * 0.7):
                return "recess", "退潮期：连板高度较前日降 ≥2 且涨停数萎缩"
        if max_h >= 3 and total >= STAGE_UP_HEAT and (pr is None or pr >= 0.4):
            return "fermenting", "发酵期：连板高度 3+ 且晋级率 ≥40%，赚钱效应扩散"
        if total >= 15:
            return "starting", "启动期：涨停家数回暖，连板高度尚低，试错期"
        return "ice", "冰点期：涨停不足，情绪冰点"

    # ---- 主分析 ----
    def analyze(self, target_date: Optional[str] = None,
                lookback: int = 5) -> dict[str, Any]:
        """情绪周期分析：指定日（默认最新）的情绪指标 + 阶段 + 近 N 日趋势。

        历史日（当日已有 sentiment_daily 记录）直接用缓存，否则现算。
        """
        store = self._store()
        try:
            latest = target_date or store.get_latest_date("kline_daily")
            if not latest:
                return {"error": "无行情数据（先运行调度装载）"}

            streaks = self._compute_streaks(store)
            if streaks.empty:
                return {"error": "无行情数据（先运行调度装载）"}
            days = sorted(streaks["date"].unique())
            if latest not in days:
                latest = days[-1]
            idx = days.index(latest)

            # 回溯填充缓存：lookback 窗口内所有日子确保有 sentiment_daily
            window_days = days[max(0, idx - lookback + 1): idx + 1]
            for wd in window_days:
                c = store.query_df(
                    "SELECT 1 AS x FROM sentiment_daily WHERE date = ?", [wd])
                if not c.empty:
                    continue
                w_idx = days.index(wd)
                w_prev = days[w_idx - 1] if w_idx > 0 else None
                self._save_day(store, self._day_stats(streaks, wd, w_prev))

            # 缓存优先（sentiment_daily）
            cached = store.query_df(
                "SELECT * FROM sentiment_daily WHERE date = ?", [latest])
            stats = None
            if not cached.empty:
                r = cached.iloc[0]
                stats = {k: (str(r[k])[:10] if k == "date"
                             else float(r[k]) if k in ("promotion_rate",
                                                       "sentiment_score")
                             else int(r[k]))
                         for k in ("date", "limit_up_count", "limit_down_count",
                                   "max_board_height", "board2_count",
                                   "board3_count", "board4p_count",
                                   "first_board_count", "promotion_rate",
                                   "sentiment_score") if k in r.index}
            else:
                prev_day = days[idx - 1] if idx > 0 else None
                stats = self._day_stats(streaks, latest, prev_day)
                self._save_day(store, stats)

            prev_stats = None
            if idx > 0:
                prev_cache = store.query_df(
                    "SELECT * FROM sentiment_daily WHERE date = ?",
                    [days[idx - 1]])
                if not prev_cache.empty:
                    r = prev_cache.iloc[0]
                    prev_stats = {k: (str(r[k])[:10] if k == "date"
                                      else float(r[k]) if k in (
                                          "promotion_rate", "sentiment_score")
                                      else int(r[k]))
                                  for k in ("date", "limit_up_count",
                                            "limit_down_count",
                                            "max_board_height",
                                            "board2_count", "board3_count",
                                            "board4p_count",
                                            "first_board_count",
                                            "promotion_rate",
                                            "sentiment_score")
                                  if k in r.index}
                else:
                    p_prev = days[idx - 2] if idx > 1 else None
                    prev_stats = self._day_stats(streaks, days[idx - 1], p_prev)

            stage, stage_zh = self._stage(stats, prev_stats)

            # 近 N 日趋势（情绪分/连板高度序列）
            trend = []
            for i in range(max(0, idx - lookback + 1), idx + 1):
                d = days[i]
                c = store.query_df(
                    "SELECT sentiment_score, max_board_height, "
                    "limit_up_count FROM sentiment_daily WHERE date = ?", [d])
                if not c.empty:
                    r = c.iloc[0]
                    trend.append({
                        "date": d,
                        "sentiment_score": float(r["sentiment_score"]),
                        "max_board_height": int(r["max_board_height"]),
                        "limit_up_count": int(r["limit_up_count"]),
                    })

            # 退潮预警：3 日内连板高度下降 ≥2
            warn = []
            if len(trend) >= 2 and \
                    trend[-1]["max_board_height"] <= trend[-2]["max_board_height"] - 2:
                warn.append(f"连板高度 {trend[-2]['max_board_height']} → "
                            f"{trend[-1]['max_board_height']}，退潮信号")
            if len(trend) >= 2 and trend[-1]["sentiment_score"] < 30:
                warn.append(f"情绪分 {trend[-1]['sentiment_score']} < 30，市场偏冷")

            return {
                **stats,
                "stage": stage,
                "stage_zh": stage_zh,
                "trend": trend,
                "warnings": warn,
                "limit_up_pct_threshold": LIMIT_UP_PCT,
                "note": "涨停判定为收盘 ≥9.5%（10cm 近似）；无盘中炸板数据，"
                        "晋级失败未单独统计",
            }
        finally:
            store.close()

    def _save_day(self, store, stats: dict[str, Any]) -> None:
        """当日情绪统计落库（幂等：已有则更新）。"""
        store.insert_df("sentiment_daily", pd.DataFrame([{
            "date": stats["date"],
            "limit_up_count": stats["limit_up_count"],
            "limit_down_count": stats["limit_down_count"],
            "max_board_height": stats["max_board_height"],
            "board2_count": stats["board2_count"],
            "board3_count": stats["board3_count"],
            "board4p_count": stats["board4p_count"],
            "first_board_count": stats["first_board_count"],
            "promotion_rate": stats["promotion_rate"],
            "sentiment_score": stats["sentiment_score"],
            "stage": "",
        }]))


_analyzer: Optional[SentimentCycleAnalyzer] = None


def get_sentiment_analyzer() -> SentimentCycleAnalyzer:
    """单例获取情绪周期分析器。"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentCycleAnalyzer()
    return _analyzer


def format_sentiment(result: dict[str, Any]) -> str:
    """情绪周期 → markdown（UI/MCP 共用）。"""
    if "error" in result:
        return f"情绪周期不可用：{result['error']}"
    stage_emoji = {"ice": "🧊", "starting": "🌱", "fermenting": "🔥",
                   "climax": "🌋", "recess": "🌊"}
    lines = [
        f"## 🌡️ 游资情绪周期（{result['date']}）",
        f"**阶段**：{stage_emoji.get(result['stage'], '')} **{result['stage_zh']}**"
        f"（情绪分 {result['sentiment_score']}）",
        "",
        "### 梯队分布",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 涨停家数 | {result['limit_up_count']} |",
        f"| 跌停家数 | {result['limit_down_count']} |",
        f"| 最大连板高度 | **{result['max_board_height']} 板** |",
        f"| 首板 / 2板 / 3板 / 4板+ | {result['first_board_count']} / "
        f"{result['board2_count']} / {result['board3_count']} / "
        f"{result['board4p_count']} |",
        f"| 晋级率（≥2板÷昨日涨停） | "
        f"{f'{result['promotion_rate']:.0%}' if result['promotion_rate'] is not None else '—'} |",
    ]
    if result.get("trend"):
        lines.append("\n### 近 N 日情绪趋势")
        lines.append("| 日期 | 情绪分 | 连板高度 | 涨停数 |")
        lines.append("|---|---|---|---|")
        for t in result["trend"]:
            lines.append(f"| {t['date']} | {t['sentiment_score']} | "
                         f"{t['max_board_height']} | {t['limit_up_count']} |")
    if result.get("warnings"):
        lines.append("\n### ⚠️ 预警")
        for w in result["warnings"]:
            lines.append(f"- {w}")
    lines.append(f"\n*{result.get('note', '')}研究参考，非投资建议。*")
    return "\n".join(lines)
