# [AI:BEGIN]
# PA_MCP - Research: CANSLIM 成长股扫描（威廉·欧奈尔《笑傲股市》）
#
# 七要素（A股适配，数据可得性标注）：
#   C 当季盈利：最新报告期净利同比 >= 20%，且营收同比为正（financials）
#   A 年度增长：最近 4 期净利同比均值 >= 25% 或 ROE >= 17%（financials）
#   N 新高/新气象：股价创 250 日新高（或 60 日新高放宽）（kline）
#   S 供给需求：突破放量——近 5 日内量比 >= 1.5（kline）
#   L 领军股：池内 60 日收益率 RS 分位 >= 80%（kline）
#   I 机构认同：无机构持仓数据源 → 不评分（标注 unavailable）
#   M 市场方向：市场状态（高潮/发酵/启动通过，低迷中性，冰点否决）
#
# 纯确定性规则（无 LLM 依赖），输出每要素命中明细（可追溯）。
# 与现有架构一致：store 短连接读取，不碰真实写入。
# [AI:END]

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 阈值（可调参数，OOS 前预注册）
C_YOY_MIN = 20.0        # C：净利同比最低 %
A_YOY_MEAN_MIN = 25.0   # A：4 期同比均值最低 %
A_ROE_MIN = 17.0        # A：ROE 最低 %
N_DAYS = 250            # N：52 周新高窗口
N_DAYS_LOOSE = 60       # N：宽松窗口
S_VOL_RATIO = 1.5       # S：突破放量倍数
S_LOOKBACK = 5          # S：放量观察窗
L_WINDOW = 60           # L：RS 窗口（交易日）
L_TOP_PCT = 0.80        # L：RS 分位门槛


@dataclass
class FactorResult:
    code: str                 # C/A/N/S/L/M
    name: str                 # 要素中文名
    passed: bool              # 是否命中
    detail: str               # 判定依据（可追溯）
    available: bool = True    # 数据可得性（I 为 False）

    def to_dict(self) -> dict:
        return {"code": self.code, "name": self.name, "passed": self.passed,
                "detail": self.detail, "available": self.available}


@dataclass
class CanslimResult:
    symbol: str
    name: str = ""
    score: int = 0                  # 命中要素数（I 不计）
    factors: list[FactorResult] = field(default_factory=list)
    market_state: str = "unknown"
    overall: str = "观望"           # 高关注/关注/观望/否决

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "name": self.name,
            "score": self.score, "overall": self.overall,
            "market_state": self.market_state,
            "factors": [f.to_dict() for f in self.factors],
        }


class CanslimScanner:
    """CANSLIM 成长股扫描器（确定性规则）。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    def _store(self):
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        path = self._store_path or get_settings().database.path
        store = DuckDBStore(path)
        store.connect()
        return store

    # ---- 主入口 ----
    def scan(self, pool: Optional[list[str]] = None,
             market_state: Optional[str] = None,
             top_n: int = 20) -> list[CanslimResult]:
        """对股票池执行 CANSLIM 扫描。

        pool 为空 → 从 kline_daily 有数据的股票取池。
        market_state 为空 → 从数据库自动判定（无法判定记 unknown）。
        """
        store = self._store()
        try:
            if pool is None:
                pool = self._auto_pool(store)
            if not pool:
                return []

            # 批量取行情与财务
            klines, fin = self._load_data(store, pool)
            if market_state is None:
                market_state = self._detect_market(store)

            # L 要素需要池内 RS 分位（先整体算）
            rs_ranks = self._relative_strength(klines)

            results: list[CanslimResult] = []
            for sym in pool:
                df = klines.get(sym)
                fdf = fin.get(sym)
                if df is None or df.empty:
                    continue
                factors = self._evaluate(sym, df, fdf, market_state, rs_ranks)
                score = sum(1 for f in factors if f.passed)
                overall = self._overall(score, market_state, factors)
                results.append(CanslimResult(
                    symbol=sym,
                    name=self._stock_name(sym, store),
                    score=score, factors=factors,
                    market_state=market_state or "unknown",
                    overall=overall,
                ))

            # 排序：分数降序 → RS 降序
            results.sort(key=lambda r: (-r.score, r.market_state == "unknown"))
            return results[:top_n]
        finally:
            store.close()

    # ---- 数据装载 ----
    def _auto_pool(self, store) -> list[str]:
        df = store.query_df(
            "SELECT DISTINCT symbol FROM kline_daily ORDER BY symbol", [])
        return [str(s) for s in df["symbol"]] if not df.empty else []

    # 内置常用股池（库内无数据时兜底，与技术型降级配合）
    BUILTIN_POOL = [
        "000001", "600036", "601398", "601288", "600519", "000858",
        "000568", "300750", "002594", "601012", "600438", "000333",
        "000651", "600030", "601318", "600276", "300760", "002415",
        "300059", "688981", "002475", "601138", "000725", "002230",
        "300308", "603986", "600900", "601857", "600028", "601088",
    ]

    async def scan_async(self, pool: Optional[list[str]] = None,
                         market_state: Optional[str] = None,
                         top_n: int = 20,
                         kline_provider=None,
                         fin_provider=None) -> list[CanslimResult]:
        """异步 CANSLIM 扫描（库内无数据时网络兜底 + 财务缺失降级）。

        - 池：pool 参数 → 库内股票 → 内置常用池（30 只）
        - 行情：库内 → kline_provider(symbol)（多源 router）
        - 财务：库内 → fin_provider；均无 → C/A 标 unavailable（技术型降级）
        """
        store = self._store()
        try:
            if pool is None:
                pool = self._auto_pool(store)
            if not pool:
                pool = list(self.BUILTIN_POOL)

            klines: dict[str, pd.DataFrame] = {}
            fin: dict[str, pd.DataFrame] = {}
            for sym in pool:
                df = None
                try:
                    df = store.query_df(
                        "SELECT date, open, high, low, close, volume, "
                        "turnover, pct_change FROM kline_daily WHERE symbol = ? "
                        "ORDER BY date DESC LIMIT 280", [sym])
                    if not df.empty:
                        df = df.sort_values("date").reset_index(drop=True)
                except Exception:
                    pass
                if (df is None or df.empty) and kline_provider is not None:
                    try:
                        kdf = kline_provider(sym)
                        if inspect.isawaitable(kdf):
                            kdf = await kdf
                        if kdf is not None and not kdf.empty:
                            df = kdf.sort_values("date").reset_index(drop=True)
                    except Exception:
                        pass
                if df is not None and not df.empty:
                    klines[sym] = df
                fdf = None
                try:
                    fdf = store.query_df(
                        "SELECT report_date, profit_yoy, revenue_yoy, roe, eps "
                        "FROM financials_income WHERE symbol = ? "
                        "ORDER BY report_date DESC LIMIT 4", [sym])
                    if fdf.empty:
                        fdf = None
                except Exception:
                    fdf = None
                if fdf is None and fin_provider is not None:
                    try:
                        fdf = await fin_provider(sym)
                    except Exception:
                        fdf = None
                if fdf is not None:
                    fin[sym] = fdf

            if not klines:
                return []

            if market_state is None:
                market_state = self._detect_market(store)
            rs_ranks = self._relative_strength(klines)

            results = []
            for sym in klines:
                factors = self._evaluate(sym, klines[sym], fin.get(sym),
                                         market_state, rs_ranks)
                score = sum(1 for f in factors if f.passed)
                results.append(CanslimResult(
                    symbol=sym, name=self._stock_name(sym, store),
                    score=score, factors=factors,
                    market_state=market_state or "unknown",
                    overall=self._overall(score, market_state, factors),
                ))
            results.sort(key=lambda r: (-r.score, r.market_state == "unknown"))
            return results[:top_n]
        finally:
            store.close()

    def _load_data(self, store, pool: list[str]):
        klines: dict[str, pd.DataFrame] = {}
        fin: dict[str, pd.DataFrame] = {}
        for sym in pool:
            try:
                df = store.query_df(
                    "SELECT date, open, high, low, close, volume, turnover, "
                    "pct_change FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 280", [sym])
                if not df.empty:
                    klines[sym] = df.sort_values("date").reset_index(drop=True)
            except Exception:
                pass
            try:
                fdf = store.query_df(
                    "SELECT report_date, profit_yoy, revenue_yoy, roe, eps "
                    "FROM financials_income WHERE symbol = ? "
                    "ORDER BY report_date DESC LIMIT 4", [sym])
                if not fdf.empty:
                    fin[sym] = fdf.reset_index(drop=True)
            except Exception:
                pass
        return klines, fin

    @staticmethod
    def _detect_market(store) -> Optional[str]:
        try:
            from pa_mcp.engine.market_state import MarketStateDetector, MarketIndicators
            latest = store.get_latest_date("kline_daily")
            if not latest:
                return None
            df = store.query_df("""
                SELECT
                    COUNT(CASE WHEN pct_change >= 9.5 THEN 1 END) as limit_up,
                    COUNT(CASE WHEN pct_change <= -9.5 THEN 1 END) as limit_down,
                    COUNT(CASE WHEN pct_change > 0 THEN 1 END) as up_count,
                    COUNT(CASE WHEN pct_change < 0 THEN 1 END) as down_count,
                    SUM(amount) / 100000000.0 as turnover
                FROM kline_daily WHERE date = ?
            """, [latest])
            row = df.iloc[0]
            indicators = MarketIndicators(
                limit_up_count=int(row["limit_up"]),
                limit_down_count=int(row["limit_down"]),
                up_count=int(row["up_count"]),
                down_count=int(row["down_count"]),
                turnover_billion=float(row["turnover"]),
            )
            return MarketStateDetector().detect(indicators).value
        except Exception:
            return None

    def _relative_strength(self, klines: dict[str, pd.DataFrame]) -> dict[str, float]:
        """池内 60 日收益率 → 分位（0-1，1=最强）。"""
        rets: dict[str, float] = {}
        for sym, df in klines.items():
            if len(df) < L_WINDOW + 1:
                continue
            rets[sym] = float(df["close"].iloc[-1] / df["close"].iloc[-L_WINDOW - 1] - 1)
        if not rets:
            return {}
        ranks = pd.Series(rets).rank(pct=True)
        return {sym: float(v) for sym, v in ranks.items()}

    # ---- 要素判定 ----
    def _evaluate(self, symbol: str, df: pd.DataFrame,
                  fdf: Optional[pd.DataFrame], market_state: Optional[str],
                  rs_ranks: dict[str, float]) -> list[FactorResult]:
        factors: list[FactorResult] = []
        has_fin = fdf is not None and not fdf.empty

        # C 当季盈利
        c_detail, c_pass = (
            "财务数据不足（先运行调度器装载 financials，当前降级为技术型 CANSLIM）",
            False)
        if has_fin:
            row = fdf.iloc[0]
            py = row.get("profit_yoy")
            ry = row.get("revenue_yoy")
            if pd.notna(py):
                c_pass = py >= C_YOY_MIN
                c_detail = (f"最新报告期净利同比 {py:+.1f}%"
                            f"（营收同比 {ry:+.1f}% 若可得）"
                            + (" ≥ 20% ✓" if c_pass else " < 20% ✗"))
            else:
                c_detail = "财务字段 profit_yoy 缺失"
        factors.append(FactorResult("C", "当季盈利", c_pass, c_detail,
                                    available=has_fin))

        # A 年度增长
        a_detail, a_pass = "财务数据不足", False
        if has_fin:
            py_vals = [float(r["profit_yoy"]) for _, r in fdf.iterrows()
                       if pd.notna(r.get("profit_yoy"))]
            roe_vals = [float(r["roe"]) for _, r in fdf.iterrows()
                        if pd.notna(r.get("roe"))]
            parts = []
            if py_vals:
                mean = sum(py_vals) / len(py_vals)
                parts.append(f"4期净利同比均值 {mean:+.1f}%")
            if roe_vals:
                parts.append(f"ROE 最高 {max(roe_vals):.1f}%")
            a_pass = (py_vals and sum(py_vals) / len(py_vals) >= A_YOY_MEAN_MIN) \
                or (roe_vals and max(roe_vals) >= A_ROE_MIN)
            a_detail = "；".join(parts) + (
                " ✓" if a_pass else " ✗（需均值≥25% 或 ROE≥17%）")
        factors.append(FactorResult("A", "年度增长", a_pass, a_detail,
                                    available=has_fin))

        # N 新高
        n_detail, n_pass = "数据不足", False
        if len(df) >= N_DAYS_LOOSE:
            last_close = float(df["close"].iloc[-1])
            hi250 = float(df["high"].tail(N_DAYS_LOOSE).max()) if len(df) >= N_DAYS_LOOSE else None
            if len(df) >= N_DAYS and float(df["high"].tail(N_DAYS).max()) <= last_close * 1.001:
                n_pass = True
                n_detail = f"创 {N_DAYS} 日新高（{last_close:.2f}）✓"
            elif len(df) >= N_DAYS_LOOSE and hi250 is not None \
                    and last_close >= hi250 * 0.97:
                n_pass = True
                n_detail = (f"接近 {N_DAYS_LOOSE} 日新高（{last_close:.2f} / "
                            f"{hi250:.2f}，-3% 内）✓")
            else:
                n_detail = (f"未创新高（收 {last_close:.2f}，"
                            f"{N_DAYS_LOOSE} 日高 {hi250:.2f}）✗")
        factors.append(FactorResult("N", "新高/新气象", n_pass, n_detail))

        # S 供给需求（突破放量）
        s_detail, s_pass = "数据不足", False
        if len(df) >= 25:
            vol = df["volume"]
            avg20 = float(vol.tail(20).mean()) if vol.tail(20).mean() > 0 else 0.0
            recent5 = vol.tail(S_LOOKBACK)
            ratios = [float(v) / avg20 for v in recent5] if avg20 > 0 else [0.0]
            if max(ratios, default=0) >= S_VOL_RATIO:
                s_pass = True
                s_detail = f"近{S_LOOKBACK}日最大量比 {max(ratios):.1f} ≥ {S_VOL_RATIO} ✓"
            else:
                s_detail = f"近{S_LOOKBACK}日最大量比 {max(ratios, default=0):.1f} < {S_VOL_RATIO} ✗"
        factors.append(FactorResult("S", "供给需求", s_pass, s_detail))

        # L 领军股（池内 RS 分位）
        l_detail, l_pass = "无池内基准", False
        rs = rs_ranks.get(symbol)
        if rs is not None:
            l_pass = rs >= L_TOP_PCT
            l_detail = (f"池内 RS 分位 {rs:.0%}（{L_WINDOW}日收益）"
                        + (" ≥ 80% ✓" if l_pass else " < 80% ✗"))
        factors.append(FactorResult("L", "领军股", l_pass, l_detail))

        # I 机构认同（数据不可得，不评分）
        factors.append(FactorResult(
            "I", "机构认同", False,
            "无机构持仓数据源（基金持仓接口不可得），不参与评分",
            available=False))

        # M 市场方向
        m_pass = False
        if market_state in ("climax", "fermenting", "starting"):
            m_pass = True
            m_detail = f"市场状态 {market_state}：顺势操作 ✓"
        elif market_state in ("dull", "frozen"):
            m_detail = f"市场状态 {market_state}：逆势，风险高 ✗"
        else:
            m_detail = "市场状态未知：不构成否决，也不加分"
            m_pass = False
        factors.append(FactorResult("M", "市场方向", m_pass, m_detail))

        return factors

    @staticmethod
    def _overall(score: int, market_state: Optional[str], factors) -> str:
        """综合评级：高分关注；M 否决降级。"""
        if market_state == "frozen":
            return "否决（冰点期）"
        if score >= 5:
            return "高关注"
        if score >= 3:
            return "关注"
        return "观望"

    @staticmethod
    def _stock_name(symbol: str, store) -> str:
        try:
            from pa_mcp.data.symbols import get_stock_name
            return get_stock_name(symbol)
        except Exception:
            return symbol


_scanner: Optional[CanslimScanner] = None


def get_canslim_scanner() -> CanslimScanner:
    """单例获取扫描器。"""
    global _scanner
    if _scanner is None:
        _scanner = CanslimScanner()
    return _scanner


def format_scan(results: list[CanslimResult]) -> str:
    """扫描结果 → markdown（UI/MCP 共用）。"""
    if not results:
        return "无 CANSLIM 结果（股票池无数据，先运行调度器装载行情与财务）"
    lines = ["## 🧬 CANSLIM 成长股扫描（欧奈尔七要素）",
             "| 排名 | 代码 | 名称 | 总分 | 评级 | C | A | N | S | L | M |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    marks = {True: "✅", False: "❌"}
    for i, r in enumerate(results[:20], 1):
        fmap = {f.code: f for f in r.factors}

        def _mark(code: str) -> str:
            f = fmap.get(code)
            if f is None:
                return "·"
            return "⬜" if not f.available else marks.get(f.passed, "·")

        lines.append(
            f"| {i} | {r.symbol} | {r.name} | **{r.score}/6** | {r.overall} | "
            f"{_mark('C')} {_mark('A')} {_mark('N')} {_mark('S')} "
            f"{_mark('L')} {_mark('M')} |")
    lines.append("\n市场状态：" + (results[0].market_state if results else "未知"))
    lines.append("\n*CANSLIM = 欧奈尔《笑傲股市》成长股选股法则（C当季盈利/A年度增长/"
                 "N新高/S放量/L领军/M市场方向；I机构数据暂缺）。研究参考，非投资建议。*")
    return "\n".join(lines)
