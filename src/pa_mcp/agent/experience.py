# [AI:BEGIN]
# PA_MCP - Agent Layer: 经验库（分析历史记忆 + 检索增强）
#
# 借鉴 PA_Agent experience_reader 机制（按周期位置组织 + 时间倒序 top-N），
# 实现自研：
#   1. 每次 AI 分析（fast/deep）结束后自动落盘 analysis_history
#   2. 按 周期位置 / 符号 / 时间倒序 检索最近 top-N 案例
#   3. 检索结果格式化为文本注入下一次分析的 prompt（RAG 增强）
#   4. 事后回填实际收益（5d/20d）并标记 outcome（hit/miss/pending）
#      形成「历史案例 → 参考 → 预测 → 验证」闭环
# 存储：DuckDB（短连接模式，与其他连接无锁冲突）
# [AI:END]

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperienceEntry:
    """一条历史分析案例。"""
    symbol: str
    analyzed_date: str
    cycle_position: str = "unknown"
    direction: str = "neutral"
    strength_score: float = 50.0
    summary: str = ""
    conclusion: str = ""
    dimension_scores: dict[str, float] = field(default_factory=dict)
    key_risks: list[str] = field(default_factory=list)
    outcome: str = "pending"          # pending | hit | miss | ambiguous
    actual_return_5d: Optional[float] = None
    actual_return_20d: Optional[float] = None

    def to_prompt_text(self) -> str:
        """格式化供注入 prompt。"""
        dims = ", ".join(f"{k}={v:.0f}" for k, v in self.dimension_scores.items()) or "—"
        risks = "；".join(self.key_risks[:3]) or "—"
        outcome = self.outcome
        ret5 = f"{self.actual_return_5d:+.1f}%" if self.actual_return_5d is not None else "未验证"
        return (
            f"[{self.symbol} {self.analyzed_date} 周期={self.cycle_position} "
            f"方向={self.direction} 强度={self.strength_score:.0f}]\n"
            f"  结论：{self.conclusion or self.summary[:100]}\n"
            f"  维度：{dims}\n"
            f"  风险：{risks}\n"
            f"  事后：{outcome}（5日收益 {ret5}）"
        )


class ExperienceService:
    """经验库服务：落盘 / 检索 / 回填验证。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    def _store(self):
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        path = self._store_path or get_settings().database.path
        store = DuckDBStore(path)
        store.connect()
        return store

    # ---- 落盘 ----
    def save_analysis(self, symbol: str, result) -> int:
        """保存一次分析结论。result 须含 direction/strength/dimension_scores 等。

        兼容 AnalysisResult 或任意 dict（getattr 兜底）。
        """
        def _get(name: str, default=None):
            if isinstance(result, dict):
                return result.get(name, default)
            return getattr(result, name, default)

        score = _get("overall_strength_score") or _get("strength_score") or 50.0
        dims = _get("dimension_scores") or {}
        risks = _get("key_risks") or []
        summary = _get("summary") or ""
        conclusion = _get("conclusion") or ""
        if not summary:
            # 从 evidence 拼摘要
            evidence = _get("key_evidence") or []
            summary = "；".join(
                f"{e.get('dimension', '')}: {e.get('finding', '')}"
                for e in evidence[:3])[:800]
        if not conclusion and isinstance(result, dict):
            conclusion = str(result.get("disclaimer", ""))[:200]

        # 周期位置：尝试从结果附带特征获取，缺省 unknown
        cycle = str(_get("cycle_position") or "unknown").lower()
        if cycle not in (
                "spike", "micro_channel", "tight_channel", "normal_channel",
                "broad_channel", "trending_range", "trading_range", "extreme_range"):
            cycle = "unknown"

        store = self._store()
        try:
            max_id = store.query_df("SELECT COALESCE(MAX(id),0) AS m FROM analysis_history", [])
            new_id = int(max_id.iloc[0]["m"]) + 1 if not max_id.empty else 1
            row = pd.DataFrame([{
                "id": new_id,
                "symbol": str(symbol),
                "analyzed_date": date.today().isoformat(),
                "cycle_position": cycle,
                "direction": str(_get("direction") or "neutral"),
                "strength_score": float(score) if score is not None else 50.0,
                "summary": summary[:800],
                "conclusion": conclusion[:500],
                "dimension_scores": json.dumps(dims, ensure_ascii=False)[:500],
                "key_risks": json.dumps(risks, ensure_ascii=False)[:1000],
                "outcome": "pending",
            }])
            store.insert_df("analysis_history", row)
            return new_id
        finally:
            store.close()

    # ---- 检索（RAG 增强源） ----
    def search_experience(
        self,
        symbol: Optional[str] = None,
        cycle_position: Optional[str] = None,
        direction: Optional[str] = None,
        limit: int = 5,
    ) -> list[ExperienceEntry]:
        """检索最近分析案例：按时间倒序，支持按符号/周期/方向过滤。

        借鉴 PA_Agent experience_reader：按周期位置归类 + 时间倒序取 top-N。
        """
        store = self._store()
        try:
            conds: list[str] = []
            params: list = []
            if symbol:
                conds.append("symbol = ?"); params.append(symbol)
            if cycle_position and cycle_position != "unknown":
                conds.append("cycle_position = ?"); params.append(cycle_position)
            if direction and direction in ("bullish", "bearish", "neutral", "up", "down", "sideways"):
                conds.append("direction = ?"); params.append(direction)
            where = f"WHERE {' AND '.join(conds)}" if conds else ""
            sql = (
                f"SELECT * FROM analysis_history {where} "
                f"ORDER BY id DESC LIMIT ?"
            )
            params.append(limit)
            df = store.query_df(sql, params)
            entries: list[ExperienceEntry] = []
            for _, r in df.iterrows():
                try:
                    dims = json.loads(r["dimension_scores"]) if r["dimension_scores"] else {}
                except Exception:
                    dims = {}
                try:
                    risks = json.loads(r["key_risks"]) if r["key_risks"] else []
                except Exception:
                    risks = []
                entries.append(ExperienceEntry(
                    symbol=str(r["symbol"]),
                    analyzed_date=str(r["analyzed_date"])[:10],
                    cycle_position=str(r["cycle_position"] or "unknown"),
                    direction=str(r["direction"] or "neutral"),
                    strength_score=float(r["strength_score"]) if r["strength_score"] is not None else 50.0,
                    summary=str(r["summary"] or ""),
                    conclusion=str(r["conclusion"] or ""),
                    dimension_scores=dims if isinstance(dims, dict) else {},
                    key_risks=risks if isinstance(risks, list) else [],
                    outcome=str(r["outcome"] or "pending"),
                    actual_return_5d=float(r["actual_return_5d"]) if r["actual_return_5d"] is not None else None,
                    actual_return_20d=float(r["actual_return_20d"]) if r["actual_return_20d"] is not None else None,
                ))
            return entries
        finally:
            store.close()

    def format_experience(self, entries: list[ExperienceEntry],
                          limit: int = 5) -> str:
        """检索结果 → prompt 注入文本（无数据时返回空串）。"""
        if not entries:
            return ""
        lines = ["【历史参考案例（同一周期/方向的近期分析，含事后验证）】"]
        for e in entries[:limit]:
            lines.append(e.to_prompt_text())
        lines.append("请参考以上案例中成功/失败的经验教训，但注意市场环境可能已变化。")
        return "\n".join(lines)

    # ---- 回填验证 ----
    def evaluate_history(self, kline_provider=None) -> dict[str, Any]:
        """回填 pending 案例的 5d/20d 实际收益并标记 outcome。"""
        store = self._store()
        try:
            pending = store.query_df(
                "SELECT * FROM analysis_history WHERE outcome = 'pending' ORDER BY id", [])
            if pending.empty:
                return {"evaluated": 0, "hit_rate": None, "total": int(store.query_df(
                    "SELECT COUNT(*) AS c FROM analysis_history", []) .iloc[0]["c"] or 0)}

            klines: dict[str, pd.DataFrame] = {}
            for sym in pending["symbol"].unique():
                df = self._fetch_kline(str(sym), store, kline_provider)
                if df is not None and not df.empty:
                    klines[str(sym)] = df.sort_values("date").reset_index(drop=True)

            evaluated = 0
            for _, row in pending.iterrows():
                sym, adate = str(row["symbol"]), str(row["analyzed_date"])[:10]
                df = klines.get(sym)
                if df is None:
                    continue
                after = df[df["date"].astype(str).str[:10] >= adate]
                if len(after) < 2:
                    continue
                base = float(after["close"].iloc[0])
                r5 = None
                if len(after) >= 6:
                    r5 = (float(after["close"].iloc[5]) / base - 1) * 100
                r20 = None
                if len(after) >= 21:
                    r20 = (float(after["close"].iloc[20]) / base - 1) * 100
                outcome = "pending"
                ref = r5 if r5 is not None else r20
                direction = str(row["direction"] or "neutral")
                if ref is not None:
                    if direction in ("bullish", "up"):
                        outcome = "hit" if ref > 0 else "miss"
                    elif direction in ("bearish", "down"):
                        outcome = "hit" if ref < 0 else "miss"
                    else:
                        outcome = "hit" if abs(ref) <= 1.5 else "miss"
                    if abs(ref) <= 1.0 and direction in ("bullish", "bearish", "up", "down"):
                        outcome = "ambiguous"
                if r5 is None and r20 is None:
                    continue
                store.execute(
                    "UPDATE analysis_history SET outcome = ?, actual_return_5d = ?, "
                    "actual_return_20d = ? WHERE id = ?",
                    [outcome,
                     round(r5, 3) if r5 is not None else None,
                     round(r20, 3) if r20 is not None else None,
                     int(row["id"])])
                evaluated += 1

            ev = store.query_df(
                "SELECT outcome FROM analysis_history WHERE outcome != 'pending'", [])
            hit_rate = None
            if not ev.empty:
                hit_rate = round(float((ev["outcome"] == "hit").mean()), 3)
            total = int(store.query_df(
                "SELECT COUNT(*) AS c FROM analysis_history", []).iloc[0]["c"] or 0)
            return {"evaluated": evaluated, "hit_rate": hit_rate, "total": total}
        finally:
            store.close()

    def _fetch_kline(self, symbol: str, store, kline_provider=None) -> Optional[pd.DataFrame]:
        if kline_provider is not None:
            try:
                df = kline_provider(symbol)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
        try:
            df = store.query_df(
                "SELECT date, open, high, low, close, volume FROM kline_daily "
                "WHERE symbol = ? ORDER BY date", [symbol])
            if not df.empty:
                return df
        except Exception:
            pass
        return None

    # ---- 统计 ----
    def stats(self) -> dict[str, Any]:
        """经验库统计（供 UI 展示）。"""
        store = self._store()
        try:
            total = int(store.query_df(
                "SELECT COUNT(*) AS c FROM analysis_history", []).iloc[0]["c"] or 0)
            by_cycle = store.query_df(
                "SELECT cycle_position, COUNT(*) AS c FROM analysis_history "
                "GROUP BY cycle_position ORDER BY c DESC LIMIT 10", [])
            by_direction = store.query_df(
                "SELECT direction, COUNT(*) AS c FROM analysis_history "
                "GROUP BY direction", [])
            return {
                "total": total,
                "by_cycle": {str(r["cycle_position"]): int(r["c"])
                             for _, r in by_cycle.iterrows()},
                "by_direction": {str(r["direction"]): int(r["c"])
                                 for _, r in by_direction.iterrows()},
            }
        finally:
            store.close()


_service: Optional[ExperienceService] = None


def get_experience_service() -> ExperienceService:
    """单例获取经验库服务。"""
    global _service
    if _service is None:
        _service = ExperienceService()
    return _service
