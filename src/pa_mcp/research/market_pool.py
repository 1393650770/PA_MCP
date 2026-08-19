# [AI:BEGIN]
# PA_MCP - Research: 全市场动态扫描池（解决"扫描总是那几只老票"）
#
# 问题背景：内置扫描池 = 东财板块成分（挂了 fallback 内置 8 板块）+ 白马
# 30 只 ≈ 60 只固定票。每天扫出来都是同一批 → 永远发现不了新票。
#
# 本模块用新浪全市场实时快照（~5500 只，并发 ~5s）按"今日表现"过滤出
# 动态池：涨幅区间剔除涨停/跌停 + 成交额门槛 + 换手率门槛。每天的池子
# 都是当日的强势票——真正"从市场里挖新票"。
#
# 设计要点：
#   - 复用 sentiment_cycle._fetch_market_snapshot（单一取数实现，不重复造轮子）
#   - TTL 缓存原始快照（5 分钟）：多个工具/UI 高频调用不重复抓全市场
#   - 过滤参数每次调用独立计算（缓存不绑定参数）
#   - 纯数据模块：不依赖 gradio/server，UI 与 MCP 工具共用
# [AI:END]

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---- TTL 缓存（全市场快照抓取 ~5s，禁止每次工具调用都重抓） ----
_snapshot_cache: Optional[list[dict]] = None
_snapshot_ts: float = 0.0
_snapshot_lock = threading.Lock()
_SNAPSHOT_TTL = 300.0  # 秒


def clear_pool_cache() -> None:
    """清空快照缓存（测试/收盘后手动刷新）。"""
    global _snapshot_cache, _snapshot_ts
    with _snapshot_lock:
        _snapshot_cache = None
        _snapshot_ts = 0.0


def _norm_symbol(raw: str) -> str:
    """'sh600000'/'sz000001' → '600000'（6 位代码）。"""
    s = (raw or "").strip().lower()
    if len(s) >= 6:
        s = s[-6:]
    return s if s.isdigit() and len(s) == 6 else ""


async def _snapshot_cached() -> list[dict]:
    """TTL 缓存的原始快照（~5500 行）。"""
    global _snapshot_cache, _snapshot_ts
    with _snapshot_lock:
        if _snapshot_cache is not None and time.time() - _snapshot_ts < _SNAPSHOT_TTL:
            return _snapshot_cache
    from pa_mcp.research.sentiment_cycle import SentimentCycleAnalyzer
    # 涨幅榜前 20 页（2000 只）已覆盖 2%+ 强势区间，全量 60 页留给情绪统计
    rows = await SentimentCycleAnalyzer._fetch_market_snapshot(max_pages=20)
    with _snapshot_lock:
        if rows:
            _snapshot_cache, _snapshot_ts = rows, time.time()
    return rows


async def build_market_scan_pool(
    limit: int = 60,
    min_pct: float = 2.0,
    max_pct: float = 9.0,
    min_amount_billion: float = 5.0,
    min_turnover: Optional[float] = None,
) -> list[dict[str, Any]]:
    """构建今日强势扫描池（全市场实时快照过滤，5 分钟缓存）。

    Args:
        limit: 返回数量上限（默认 60）
        min_pct/max_pct: 涨幅区间（默认 2%~9%，剔除涨停/跌停板）
        min_amount_billion: 成交额门槛（亿，默认 5 亿——过滤僵尸票）
        min_turnover: 换手率门槛（%，None = 不限）

    Returns:
        [{"symbol", "name", "change_pct", "amount_billion", "turnover",
          "price", "source"}]，按涨幅降序；失败返回 []（调用方降级）。
    """
    snapshot = await _snapshot_cached()
    picked: list[dict[str, Any]] = []
    for t in snapshot:
        try:
            chg = float(t.get("changepercent", 0) or 0)
            amt = float(t.get("amount", 0) or 0) / 1e8  # 元 → 亿
            turnover_raw = t.get("turnoverratio", None)
            turnover = float(turnover_raw) if turnover_raw not in (None, "") else None
            price = float(t.get("trade", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not (min_pct <= chg <= max_pct):
            continue
        if amt < min_amount_billion:
            continue
        if min_turnover is not None and (turnover is None or turnover < min_turnover):
            continue
        raw_sym = (t.get("symbol", "") or "").strip().lower()
        if raw_sym.startswith("bj"):
            continue  # 北交所：腾讯/新浪 K 线源均不支持，剔除
        sym = _norm_symbol(raw_sym)
        if not sym:
            continue
        picked.append({
            "symbol": sym,
            "name": t.get("name", ""),
            "change_pct": round(chg, 2),
            "amount_billion": round(amt, 2),
            "turnover": round(turnover, 2) if turnover is not None else None,
            "price": round(price, 2),
            "source": "sina_snapshot",
        })
    # 涨幅降序（今日强势优先）
    picked.sort(key=lambda x: x["change_pct"], reverse=True)
    return picked[:limit]
