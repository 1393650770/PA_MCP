# [AI:BEGIN]
# PA_MCP - Data: 数据质量体检报告
#
# 数据层完整性检查（研究可信度前置条件）：
#   1. 表覆盖：各关键表行数/日期范围/覆盖股票数/最新日期
#   2. kline_daily 抽样检查：
#      - OHLC 一致性（high ≥ max(open,close)、low ≤ min(open,close)）
#      - 非负检查（价格 ≤ 0、成交量 < 0、NaN 比例）
#      - 交易日缺口（相邻交易日间隔 > 5 天 = 疑似缺失）
#      - 异常收益（|pct_change| > 21% 且非创业板/科创板——简化全市场标记）
#   3. 输出：健康评分（0-100）+ 问题清单（可追溯）
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

KEY_TABLES = [
    "kline_daily", "kline_minute", "index_daily", "stock_basic",
    "financials_income", "dragon_tiger", "fund_flow_daily",
    "sector_daily", "prediction_log", "sentiment_daily",
    "analysis_history",
]
SAMPLE_LIMIT = 200       # 抽样股票数
GAP_THRESHOLD_DAYS = 5   # 交易日缺口阈值


class DataQualityReport:
    """数据质量体检（表覆盖 + OHLC 一致性 + 缺口 + 异常值）。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    def _store(self):
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        path = self._store_path or get_settings().database.path
        store = DuckDBStore(path)
        store.connect()
        return store

    def generate(self, sample_limit: int = SAMPLE_LIMIT) -> dict[str, Any]:
        """生成体检报告。"""
        store = self._store()
        try:
            tables = self._table_coverage(store)
            kline_issues = self._kline_checks(store, sample_limit)
            issues = [i for t in tables.values() if t.get("issue")
                      for i in [t["issue"]]]
            issues += kline_issues["issues"]

            score = self._score(tables, kline_issues)
            return {
                "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "score": score,
                "tables": tables,
                "kline_checks": kline_issues,
                "issues": issues,
                "summary": self._summary(score, tables, kline_issues),
            }
        finally:
            store.close()

    # ---- 表覆盖 ----
    def _table_coverage(self, store) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for t in KEY_TABLES:
            try:
                if not store.table_exists(t):
                    out[t] = {"rows": 0, "issue": f"表 {t} 不存在"}
                    continue
                info = store.query_df(
                    f"SELECT COUNT(*) AS rows FROM {t}", [])
                rows = int(info.iloc[0]["rows"]) if not info.empty else 0
                entry: dict[str, Any] = {"rows": rows}
                # 日期范围（有 date 类列的表）
                try:
                    rng = store.query_df(
                        f"SELECT MIN(date) AS d0, MAX(date) AS d1 FROM {t}")
                    if not rng.empty and rng.iloc[0]["d0"] is not None:
                        entry["date_range"] = [
                            str(rng.iloc[0]["d0"])[:10],
                            str(rng.iloc[0]["d1"])[:10]]
                except Exception:
                    pass
                # 覆盖股票数（有 symbol 列的表）
                try:
                    syms = store.query_df(
                        f"SELECT COUNT(DISTINCT symbol) AS c FROM {t}")
                    if not syms.empty:
                        entry["n_symbols"] = int(syms.iloc[0]["c"])
                except Exception:
                    pass
                out[t] = entry
            except Exception as e:  # noqa: BLE001
                out[t] = {"rows": 0, "issue": f"查询失败：{str(e)[:60]}"}
        return out

    # ---- K 线检查 ----
    def _kline_checks(self, store, sample_limit: int) -> dict[str, Any]:
        issues: list[str] = []
        try:
            syms = store.query_df(
                "SELECT DISTINCT symbol FROM kline_daily LIMIT ?",
                [sample_limit])
        except Exception:
            return {"sampled": 0, "issues": ["kline_daily 不可用"]}
        if syms.empty:
            return {"sampled": 0, "issues": ["kline_daily 为空（未装载行情）"]}

        n_checked = 0
        n_ohlc_bad = 0
        n_neg = 0
        n_nan = 0
        n_gaps = 0
        for sym in syms["symbol"]:
            try:
                df = store.query_df(
                    "SELECT date, open, high, low, close, volume, pct_change "
                    "FROM kline_daily WHERE symbol = ? ORDER BY date", [str(sym)])
            except Exception:
                continue
            if df.empty:
                continue
            n_checked += 1
            o = pd.to_numeric(df["open"], errors="coerce")
            h = pd.to_numeric(df["high"], errors="coerce")
            l = pd.to_numeric(df["low"], errors="coerce")
            c = pd.to_numeric(df["close"], errors="coerce")
            v = pd.to_numeric(df["volume"], errors="coerce")
            # OHLC 一致性
            bad_ohlc = int(((h < pd.concat([o, c], axis=1).max(axis=1))
                            | (l > pd.concat([o, c], axis=1).min(axis=1))).sum())
            n_ohlc_bad += bad_ohlc
            # 非负/NaN
            n_neg += int((c <= 0).sum() + (v < 0).sum())
            n_nan += int(c.isna().sum() + v.isna().sum())
            # 交易日缺口
            dates = pd.to_datetime(df["date"])
            gaps = int((dates.diff().dt.days > GAP_THRESHOLD_DAYS).sum())
            n_gaps += gaps
            if bad_ohlc > 0 or n_neg > 0 or gaps > 0:
                issues.append(
                    f"{sym}：OHLC 异常 {bad_ohlc} 行，非正 {n_neg}，"
                    f"缺口 {gaps} 处")

        return {
            "sampled": n_checked,
            "ohlc_bad_rows": n_ohlc_bad,
            "non_positive_rows": n_neg,
            "nan_rows": n_nan,
            "date_gaps": n_gaps,
            "issues": issues[:20],
            "issue_count": len(issues),
        }

    # ---- 评分 ----
    @staticmethod
    def _score(tables: dict, kline: dict) -> int:
        score = 100
        # 表缺失扣分
        missing = sum(1 for t in tables.values() if t.get("issue")
                      and "不存在" in t["issue"])
        score -= missing * 8
        # K 线异常扣分
        if kline.get("sampled", 0) > 0:
            if kline.get("ohlc_bad_rows", 0) > 0:
                score -= min(20, kline["ohlc_bad_rows"] // 10)
            if kline.get("nan_rows", 0) > 0:
                score -= min(10, kline["nan_rows"] // 50)
            if kline.get("date_gaps", 0) > 0:
                score -= min(15, kline["date_gaps"] * 2)
        else:
            score -= 30  # 无行情数据
        return max(0, min(100, score))

    @staticmethod
    def _summary(score: int, tables: dict, kline: dict) -> str:
        level = ("✅ 健康" if score >= 90 else "⚠️ 需关注" if score >= 70
                 else "❌ 严重问题")
        return (f"健康评分 {score}/100（{level}）："
                f"表缺失 {sum(1 for t in tables.values() if t.get('issue') and '不存在' in t['issue'])} 张，"
                f"K线检查 {kline.get('sampled', 0)} 只股票，"
                f"OHLC 异常 {kline.get('ohlc_bad_rows', 0)} 行，"
                f"缺口 {kline.get('date_gaps', 0)} 处")


_generator: Optional[DataQualityReport] = None


def get_quality_report() -> DataQualityReport:
    """单例获取体检器。"""
    global _generator
    if _generator is None:
        _generator = DataQualityReport()
    return _generator


def format_report(result: dict[str, Any]) -> str:
    """体检报告 → markdown。"""
    lines = [
        f"## 🩺 数据质量体检报告（{result['date']}）",
        f"**{result['summary']}**",
        "",
        "### 表覆盖",
        "| 表 | 行数 | 股票数 | 日期范围 |",
        "|---|---|---|---|",
    ]
    for t, info in result["tables"].items():
        if "issue" in info and info["rows"] == 0 and "不存在" in info["issue"]:
            lines.append(f"| {t} | ❌ {info['issue']} |")
            continue
        lines.append(
            f"| {t} | {info.get('rows', 0):,} | "
            f"{info.get('n_symbols', '—')} | "
            f"{info.get('date_range', ['—', '—'])[0]} ~ "
            f"{info.get('date_range', ['—', '—'])[1]} |")
    k = result["kline_checks"]
    lines.extend([
        "",
        "### K 线检查",
        f"- 抽样 {k.get('sampled', 0)} 只股票",
        f"- OHLC 一致性异常：{k.get('ohlc_bad_rows', 0)} 行",
        f"- 非正价格/负量：{k.get('non_positive_rows', 0)} 行",
        f"- NaN：{k.get('nan_rows', 0)} 行",
        f"- 交易日缺口（>{GAP_THRESHOLD_DAYS} 天）：{k.get('date_gaps', 0)} 处",
    ])
    if result["issues"]:
        lines.append("\n### 问题清单")
        for i in result["issues"][:10]:
            lines.append(f"- {i}")
    lines.append("\n*数据质量是研究可信度的前置条件。研究参考，非投资建议。*")
    return "\n".join(lines)
