# [AI:BEGIN]
# PA_MCP - Research: 因子工厂（因子注册 / 批量验证 / 打包）
#
# 机制借鉴开源 QuantSkills factor-skill-factory + Qlib 因子工作流：
#   1. FactorRegistry：装饰器注册因子（name/category/description/version）
#   2. 内置因子：复用已有指标库（MA/RSI/MACD/ATR/BOLL/量比/动量/缠论）
#      ——非必要不造轮子
#   3. evaluate_factor：单因子检验——IC（秩相关）+ 分层（Q1-Q5 组平均
#      未来收益）+ 单调性 + 覆盖率（量化界标准因子检验）
#   4. scan_factors：股票池批量验证 → 因子排行（有效因子清单）
#   5. pack_factor：因子定义打包为 JSON（可版本化/复用/跨会话）
# [AI:END]

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON_DEFAULT = 5      # 默认前瞻窗口（交易日）
IC_MIN_SAMPLES = 30      # IC 最小样本
QUANTILES = 5            # 分层数


# ---- 因子定义与注册表 ----

@dataclass
class FactorDefinition:
    name: str
    category: str            # momentum/mean_reversion/volatility/volume/trend/value
    description: str
    version: str = "v1"
    lookback: int = 20       # 参考信息窗口（说明用）
    fn: Optional[Callable] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "category": self.category,
                "description": self.description, "version": self.version,
                "lookback": self.lookback}


class FactorRegistry:
    """因子注册表（装饰器模式，类比 StrategyRegistry）。"""

    def __init__(self) -> None:
        self._factors: dict[str, FactorDefinition] = {}

    def register(self, fn: Callable, name: str, category: str,
                 description: str, version: str = "v1",
                 lookback: int = 20) -> FactorDefinition:
        """注册因子。"""
        fd = FactorDefinition(name=name, category=category,
                              description=description, version=version,
                              lookback=lookback, fn=fn)
        self._factors[name] = fd
        return fd

    def factor(self, name: str, category: str, description: str,
               version: str = "v1", lookback: int = 20):
        """装饰器用法：@registry.factor(...)"""
        def deco(fn: Callable) -> Callable:
            self.register(fn, name, category, description, version, lookback)
            return fn
        return deco

    def get(self, name: str) -> Optional[FactorDefinition]:
        return self._factors.get(name)

    def list_all(self) -> list[FactorDefinition]:
        return list(self._factors.values())

    def list_by_category(self, category: str) -> list[FactorDefinition]:
        return [f for f in self._factors.values()
                if f.category == category]


# ---- 内置因子（复用已有指标库） ----

_registry = FactorRegistry()


def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("date").reset_index(drop=True)


@_registry.factor("ma_gap20", "mean_reversion", "收盘相对 MA20 偏离（%）：负 = 超跌",
                  lookback=20)
def _f_ma_gap20(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    ma = d["close"].rolling(20).mean()
    return (d["close"] / ma - 1) * 100


@_registry.factor("rsi14", "mean_reversion", "RSI(14) 水平：高 = 超买", lookback=14)
def _f_rsi14(df: pd.DataFrame) -> pd.Series:
    from pa_mcp.engine.indicators.indicators import calc_rsi
    r = calc_rsi(_ensure(df))
    return r["rsi14"] if "rsi14" in r else pd.Series(dtype=float)


@_registry.factor("macd_hist", "momentum", "MACD 柱（归一化到价格 %）", lookback=26)
def _f_macd_hist(df: pd.DataFrame) -> pd.Series:
    from pa_mcp.engine.indicators.indicators import calc_macd
    m = calc_macd(_ensure(df))
    if "macd_hist" not in m:
        return pd.Series(dtype=float)
    return m["macd_hist"] / _ensure(df)["close"] * 100


@_registry.factor("atr_pct", "volatility", "ATR(14) 占价格 %（波动率）", lookback=14)
def _f_atr_pct(df: pd.DataFrame) -> pd.Series:
    from pa_mcp.engine.indicators.indicators import calc_atr
    a = calc_atr(_ensure(df))
    if "atr14" not in a:
        return pd.Series(dtype=float)
    return a["atr14"] / _ensure(df)["close"] * 100


@_registry.factor("boll_position", "mean_reversion", "布林 %B：>100 突破上轨",
                  lookback=20)
def _f_boll_pos(df: pd.DataFrame) -> pd.Series:
    from pa_mcp.engine.indicators.indicators import calc_bollinger
    b = calc_bollinger(_ensure(df))
    if not all(c in b for c in ("boll_up", "boll_low")):
        return pd.Series(dtype=float)
    d = _ensure(df)
    rng = (b["boll_up"] - b["boll_low"]).replace(0, np.nan)
    return (d["close"] - b["boll_low"]) / rng * 100


@_registry.factor("vol_ratio", "volume", "量比：当日量 / 20 日均量", lookback=20)
def _f_vol_ratio(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    avg = d["volume"].rolling(20).mean().replace(0, np.nan)
    return d["volume"] / avg


@_registry.factor("ret20", "momentum", "20 日收益率（%）", lookback=20)
def _f_ret20(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    return d["close"].pct_change(20) * 100


@_registry.factor("ret60", "momentum", "60 日收益率（%）", lookback=60)
def _f_ret60(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    return d["close"].pct_change(60) * 100


@_registry.factor("high_52w_dist", "trend", "距 52 周新高（%，0 = 新高）",
                  lookback=250)
def _f_high52_dist(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    hi = d["high"].rolling(250).max()
    return (d["close"] / hi - 1) * 100


@_registry.factor("chan_beichi", "mean_reversion",
                  "缠论背驰信号（+1 下跌背驰看多 / -1 上涨背驰看空 / 0 无）",
                  lookback=40)
def _f_chan_beichi(df: pd.DataFrame) -> pd.Series:
    """缠论背驰编码为因子：滚动窗口检测，信号日置 ±1（NaN 填充 0）。

    性能：窗口 40 根 + 步长 5（背驰需 ≥5 笔，40 根通常足够；
    全历史重算成本高，步长 5 权衡覆盖率与耗时）。
    """
    from pa_mcp.engine.indicators.chan import chan_analysis
    d = _ensure(df)
    n = len(d)
    out = pd.Series(0.0, index=d.index)
    window = min(40, max(30, n // 4))
    if n < window + 5:
        return out
    for i in range(window - 1, n, 5):
        win = d.iloc[i - window + 1:i + 1]
        try:
            a = chan_analysis(win, with_macd=True)
        except Exception:
            continue
        if a.beichi_signal == "bullish":
            out.iloc[i] = 1.0
        elif a.beichi_signal == "bearish":
            out.iloc[i] = -1.0
    return out


def get_factor_registry() -> FactorRegistry:
    """获取因子注册表（单例）。"""
    return _registry


# ---- 因子检验 ----

def evaluate_factor(factor: FactorDefinition, kline_df: pd.DataFrame,
                    horizon: int = HORIZON_DEFAULT) -> dict[str, Any]:
    """单因子检验（量化标准）：IC + 分层 + 单调性 + 覆盖率。

    - IC：因子值与未来 horizon 日收益的 Spearman 秩相关（全部样本）
    - 分层：因子值分位 Q1-Q5，各组未来收益均值 → 单调性 + 高低组差
    - 覆盖率：有效样本比例（NA 剔除后）
    """
    if factor.fn is None:
        return {"error": f"因子 {factor.name} 无实现函数"}
    if kline_df is None or kline_df.empty:
        return {"error": "无行情数据"}
    d = _ensure(kline_df)
    try:
        values = factor.fn(d)
    except Exception as e:  # noqa: BLE001
        return {"error": f"因子计算失败：{e}"}
    if values is None or len(values) != len(d):
        return {"error": "因子输出长度与行情不一致"}

    values = pd.Series(values.to_numpy(), index=d.index)
    fwd = d["close"].shift(-horizon) / d["close"] - 1  # 未来 horizon 收益
    valid = values.notna() & fwd.notna()
    n_valid = int(valid.sum())
    if n_valid < IC_MIN_SAMPLES:
        return {"error": f"有效样本 {n_valid} < {IC_MIN_SAMPLES}，无法检验"}

    v = values[valid].astype(float)
    r = fwd[valid].astype(float)
    # Spearman IC（rank + pearson，无 scipy）
    ic = float(v.rank().corr(r.rank())) if v.rank().corr(r.rank()) is not None \
        else 0.0
    # 分层
    q = pd.qcut(v, QUANTILES, labels=False, duplicates="drop")
    group_ret = r.groupby(q).mean()
    if len(group_ret) >= 2:
        spread = float(group_ret.iloc[-1] - group_ret.iloc[0])
        monotonic = all(
            group_ret.iloc[i] < group_ret.iloc[i + 1] or
            group_ret.iloc[i] > group_ret.iloc[i + 1]
            for i in range(len(group_ret) - 1))
    else:
        spread, monotonic = 0.0, False

    return {
        "factor": factor.name,
        "category": factor.category,
        "version": factor.version,
        "description": factor.description,
        "horizon": horizon,
        "n_samples": n_valid,
        "coverage_pct": round(n_valid / len(d) * 100, 1),
        "ic": round(ic, 4),
        "quantile_returns": [round(float(x), 4) for x in group_ret],
        "spread_pct": round(spread * 100, 3),      # Q5 - Q1 未来收益差
        "monotonic": bool(monotonic),
        "useful": abs(ic) >= 0.03,                 # |IC| ≥ 0.03 视为有信息
    }


def scan_factors(kline_df: pd.DataFrame,
                 horizon: int = HORIZON_DEFAULT) -> list[dict[str, Any]]:
    """批量验证全部注册因子 → 按 |IC| 排序的因子排行。"""
    results = []
    for fd in get_factor_registry().list_all():
        r = evaluate_factor(fd, kline_df, horizon=horizon)
        if "error" in r:
            r["factor"] = fd.name
            r["error"] = r["error"]
            results.append(r)
            continue
        results.append(r)
    results.sort(key=lambda x: abs(x.get("ic", 0)), reverse=True)
    return results


def pack_factor(name: str) -> Optional[dict[str, Any]]:
    """因子定义打包为 JSON（可版本化/复用）。"""
    fd = get_factor_registry().get(name)
    if fd is None:
        return None
    return {"factor": fd.to_dict(),
            "interface": "fn(kline_df: DataFrame) -> Series（与行情等长）",
            "packed_at_note": "定义打包（不含函数体）；跨会话复用需重新注册"}


# ---- 多因子选股（Qlib 风格：IC 方向调整的截面 z-score 合成） ----

def select_stocks_by_factors(
    klines: dict[str, pd.DataFrame],
    top_n: int = 10,
    horizon: int = HORIZON_DEFAULT,
    min_ic: float = 0.02,
    prediction_weight: float = 0.0,
) -> dict[str, Any]:
    """多因子截面选股（可选融合 AI 预测概率）。

    流程：
      1. 池内 pooled 数据计算每因子 IC 符号（全样本秩相关）
      2. 每股票最新一期因子值 → 截面 z-score
      3. 综合分 = 平均(|IC|-达标的 因子 z-score × IC 符号)（IC 加权方向）
      4. 可选：AI 预测概率融合——预测方向化概率（up=+p_up / down=-p_down
         / sideways≈0）截面 z-score，按 prediction_weight 加权并入综合分
      5. 按综合分排序输出 top N + 因子明细

    Args:
        klines: {symbol: DataFrame}（各股 ≥ 60 根）
        top_n: 返回数量
        horizon: 前瞻窗口（IC 计算用）
        min_ic: IC 门槛（低于该 |IC| 的因子不参与合成）
        prediction_weight: AI 预测权重 0-1（0 = 纯量化因子；
            0.3-0.5 = 混合；1 = 纯预测）
    """
    if not klines:
        return {"error": "无行情数据"}
    n_stock = len(klines)

    # 1) 每因子 pooled 截面 IC 符号（跨股票拼接秩相关——横截面选股
    #    的正确口径；逐股票 IC 平均会混入时序噪声反转，方向失真）
    factor_signs: dict[str, float] = {}
    factor_meta: dict[str, FactorDefinition] = {}
    for fd in get_factor_registry().list_all():
        try:
            all_v: list[pd.Series] = []
            all_r: list[pd.Series] = []
            for sym, df in klines.items():
                d = _ensure(df)
                values = fd.fn(d)
                if values is None or len(values) != len(d):
                    continue
                vals = pd.Series(values.to_numpy(), index=d.index)
                fwd = d["close"].shift(-horizon) / d["close"] - 1
                valid = vals.notna() & fwd.notna()
                if valid.sum() < 20:
                    continue
                all_v.append(vals[valid].astype(float))
                all_r.append(fwd[valid].astype(float))
            if len(all_v) < 3:
                continue
            V = pd.concat(all_v)
            R = pd.concat(all_r)
            ic = V.rank().corr(R.rank())
            if ic is not None and not pd.isna(ic) and abs(float(ic)) >= min_ic:
                factor_signs[fd.name] = 1.0 if ic > 0 else -1.0
                factor_meta[fd.name] = fd
        except Exception:  # noqa: BLE001
            continue

    if not factor_signs:
        return {"error": "无因子达到 IC 门槛，无法选股（数据不足或因子无信息）"}

    # 2+3) 截面 z-score 合成
    latest_values: dict[str, dict[str, float]] = {}  # symbol → {factor: value}
    for sym, df in klines.items():
        d = _ensure(df)
        entry = {}
        for fname in factor_signs:
            try:
                vals = factor_meta[fname].fn(d)
                if vals is not None and len(vals) == len(d):
                    v = float(vals.iloc[-1])
                    if not pd.isna(v):
                        entry[fname] = v
            except Exception:  # noqa: BLE001
                continue
        latest_values[sym] = entry

    # AI 预测概率融合（方向化概率，截面 z-score）
    pred_values: dict[str, float] = {}
    if prediction_weight > 0:
        try:
            import asyncio as _asyncio
            from pa_mcp.agent.prediction import get_prediction_service
            svc = get_prediction_service()
            for sym, df in klines.items():
                try:
                    r = _asyncio.run(svc.predict(
                        sym, df, horizon="5d", use_llm=False))
                    p = r.to_dict()
                    d = p["direction"]
                    if d == "up":
                        pred_values[sym] = p["prob_up"]
                    elif d == "down":
                        pred_values[sym] = -p["prob_down"]
                    else:
                        pred_values[sym] = 0.0
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pred_values = {}
        if len(pred_values) >= 3:
            vals = list(pred_values.values())
            mean, std = float(np.mean(vals)), float(np.std(vals))
            if std < 1e-12:
                pred_values = {}
            else:
                pred_values = {s: (v - mean) / std
                               for s, v in pred_values.items()}

    # z-score（截面）
    rows = []
    for sym, entry in latest_values.items():
        if len(entry) < max(2, len(factor_signs) // 2):
            continue
        scores = []
        details: dict[str, float] = {}
        for fname, sign in factor_signs.items():
            if fname not in entry:
                continue
            vals = [latest_values[s2][fname] for s2 in latest_values
                    if fname in latest_values[s2]]
            if len(vals) < 3:
                continue
            mean, std = float(np.mean(vals)), float(np.std(vals))
            if std < 1e-12:
                continue
            z = (entry[fname] - mean) / std
            scores.append(sign * z)
            details[fname] = round(sign * z, 3)
        # 预测融合
        if pred_values and sym in pred_values:
            pz = pred_values[sym]
            scores.append(pz * prediction_weight)
            details["prediction"] = round(pz * prediction_weight, 3)
        if not scores:
            continue
        rows.append({
            "symbol": sym,
            "score": round(float(np.mean(scores)), 4),
            "factor_details": details,
        })
    if not rows:
        return {"error": "合成失败（因子覆盖不足）"}

    rows.sort(key=lambda r: r["score"], reverse=True)
    top = rows[:top_n]
    method = (f"多因子截面选股：IC 方向调整的 z-score 等权合成"
              f"（{len(factor_signs)} 因子达标，|IC|≥{min_ic}）")
    if prediction_weight > 0 and pred_values:
        method += f" + AI 预测融合（权重 {prediction_weight:.0%}）"
    return {
        "method": method,
        "n_stock": n_stock,
        "n_scored": len(rows),
        "factors_used": sorted(factor_signs.keys()),
        "prediction_weight": prediction_weight,
        "prediction_used": bool(pred_values),
        "selection": top,
        "top_symbols": [r["symbol"] for r in top],
    }


# ---- 因子选股组合回测（选股 → 组合闭环，复用 PortfolioBacktestEngine） ----

def backtest_factor_selection(
    klines: dict[str, pd.DataFrame],
    top_n: int = 5,
    horizon: int = 5,
    train_window: int = 120,
    initial_cash: float = 100_000.0,
    prediction_weight: float = 0.0,
) -> dict[str, Any]:
    """滚动窗口因子选股组合回测。

    流程：
      1. 对齐日历，从 train_window 起每 horizon 天调仓一次
      2. 每个调仓日：用过去 train_window 天做 pooled 截面 IC → 选 top N
         （prediction_weight > 0 时融合确定性 AI 预测概率，控制成本）
      3. 信号：top N 发 bullish（买入）、池内其余发 bearish（卖出/调出）
         ——等权再平衡，延迟一天执行（引擎语义）
      4. 复用 PortfolioBacktestEngine（共享账本/单票10%/T+1/费用）
      5. 基准：全池等权持有（同样引擎，无调仓信号）
      6. 输出：组合收益/超额/最大回撤/年化/换手率

    Args:
        klines: {symbol: 日线（升序）}（各股 ≥ train_window + 2 根）
        top_n: 每期持仓数量
        horizon: 调仓周期（交易日）
        train_window: IC 训练窗口
        initial_cash: 初始资金
        prediction_weight: AI 预测融合权重 0-1（回测内用确定性预测）
    """
    from pa_mcp.portfolio.backtest import PortfolioBacktestEngine

    if len(klines) < 3 or top_n < 1:
        return {"error": "至少需要 3 只股票"}
    # 对齐日历
    aligned: dict[str, pd.DataFrame] = {}
    for sym, df in klines.items():
        d = _ensure(df)
        if len(d) < train_window + 2:
            continue
        aligned[sym] = d
    if len(aligned) < 3:
        return {"error": f"满足训练窗口的股票不足（{len(aligned)} < 3）"}
    calendar = sorted(set().union(
        *[set(df["date"].astype(str).str[:10]) for df in aligned.values()]))
    n = len(calendar)
    if n <= train_window + horizon:
        return {"error": f"日历 {n} 天不足（需 > {train_window + horizon}）"}

    # 滚动选股 → 信号
    signals_by_symbol: dict[str, list[dict]] = {s: [] for s in aligned}
    for t in range(train_window, n - 1, horizon):
        end_date = calendar[t]
        # 训练窗口切片
        train_klines = {}
        for sym, d in aligned.items():
            sub = d[d["date"].astype(str).str[:10] <= end_date]
            train_klines[sym] = sub.tail(train_window)
        sel = select_stocks_by_factors(
            train_klines, top_n=top_n, horizon=horizon,
            prediction_weight=prediction_weight)
        if "error" in sel:
            continue
        top = sel["top_symbols"]
        for sym in aligned:
            sig = {
                "date": end_date,
                "symbol": sym,
                "direction": "bullish" if sym in top else "bearish",
                "strength_score": 60.0 if sym in top else 40.0,
            }
            signals_by_symbol[sym].append(sig)

    sig_dfs = {s: pd.DataFrame(lst) if lst else pd.DataFrame(
        columns=["date", "symbol", "direction", "strength_score"])
        for s, lst in signals_by_symbol.items()}

    # 组合回测（因子选股）
    engine = PortfolioBacktestEngine(initial_cash=initial_cash)
    result = engine.run(aligned, sig_dfs)

    # 基准：全池等权（每调仓日全部 bullish，不做任何选股）
    bench_signals = {}
    for sym, d in aligned.items():
        bench_signals[sym] = pd.DataFrame([
            {"date": calendar[t], "symbol": sym, "direction": "bullish",
             "strength_score": 50.0}
            for t in range(train_window, n - 1, horizon)])
    bench_engine = PortfolioBacktestEngine(initial_cash=initial_cash)
    bench = bench_engine.run(aligned, bench_signals)

    return {
        "method": (f"滚动 {train_window} 日截面 IC 选股 → 每 {horizon} 日调仓 "
                   f"top {top_n}，等权组合回测"),
        "n_stock": len(aligned),
        "n_rebalances": len(next(iter(sig_dfs.values()))),
        "portfolio": {
            "total_return_pct": getattr(result, "total_return_pct", None),
            "annual_return_pct": getattr(result, "annual_return_pct", None),
            "max_drawdown_pct": getattr(result, "max_drawdown_pct", None),
            "sharpe_ratio": getattr(result, "sharpe_ratio", None),
            "total_trades": getattr(result, "total_trades", None),
            "total_fees": getattr(result, "total_fees", None),
            "nav_series": [{"date": str(r.get("trade_date", ""))[:10],
                            "nav": round(float(r["nav"]), 4)}
                           for r in getattr(result, "nav_series", [])],
        },
        "benchmark": {
            "total_return_pct": getattr(bench, "total_return_pct", None),
            "max_drawdown_pct": getattr(bench, "max_drawdown_pct", None),
        },
        "excess_return_pct": round(
            float(getattr(result, "total_return_pct", 0) or 0)
            - float(getattr(bench, "total_return_pct", 0) or 0), 2),
        "note": "基准 = 全池等权（无选股）。研究参考，非投资建议。",
    }


# ---- 预测权重敏感性分析 ----

def sensitivity_analysis(
    klines: dict[str, pd.DataFrame],
    top_n: int = 5,
    horizon: int = 5,
    train_window: int = 120,
    weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """预测权重敏感性：各权重下因子选股组合回测 → 最优权重。

    复用 backtest_factor_selection（预测融合用确定性模式，控制成本）——
    数据回答「AI 预测该占多大权重」。

    Returns:
        results: 每权重 {weight, total_return, excess, max_drawdown}
        best_weight: 超额收益最高的权重
    """
    rows = []
    for w in weights:
        try:
            r = backtest_factor_selection(
                klines, top_n=top_n, horizon=horizon,
                train_window=train_window, prediction_weight=w)
            if "error" in r:
                rows.append({"weight": w, "error": r["error"]})
                continue
            rows.append({
                "weight": w,
                "total_return_pct": r["portfolio"]["total_return_pct"],
                "excess_return_pct": r["excess_return_pct"],
                "max_drawdown_pct": r["portfolio"]["max_drawdown_pct"],
                "sharpe_ratio": r["portfolio"]["sharpe_ratio"],
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"weight": w, "error": str(e)[:60]})

    valid = [r for r in rows if "error" not in r]
    if not valid:
        return {"error": "全部权重回测失败", "results": rows}
    best = max(valid, key=lambda r: r["excess_return_pct"])
    return {
        "method": ("预测权重敏感性：各权重下因子选股组合回测对比"
                   "（预测用确定性模式，控制成本）"),
        "results": rows,
        "best_weight": best["weight"],
        "best_excess_pct": best["excess_return_pct"],
        "recommendation": (
            f"最优权重 {best['weight']:.0%}（超额 {best['excess_return_pct']:+.2f}%）"
            if best["excess_return_pct"] is not None else "数据不足"),
    }


def format_sensitivity(result: dict[str, Any]) -> str:
    """敏感性结果 → markdown。"""
    if "error" in result and not result.get("results"):
        return f"敏感性分析不可用：{result['error']}"
    lines = [
        f"## ⚖️ 预测权重敏感性分析",
        f"**方法**：{result.get('method', '')}",
        "",
        "| 权重 | 总收益% | 超额% | 回撤% | Sharpe |",
        "|---|---|---|---|---|",
    ]
    for r in result["results"]:
        if "error" in r:
            lines.append(f"| {r['weight']:.0%} | ❌ {r['error']} |")
            continue
        lines.append(f"| {r['weight']:.0%} | {r['total_return_pct']} | "
                     f"{r['excess_return_pct']:+.2f} | "
                     f"{r['max_drawdown_pct']} | {r['sharpe_ratio']} |")
    lines.append(f"\n**结论**：{result.get('recommendation', '—')}")
    lines.append("\n*权重 0 = 纯因子，1 = 纯预测。研究参考，非投资建议。*")
    return "\n".join(lines)


def format_portfolio_backtest(result: dict[str, Any]) -> str:
    """因子组合回测 → markdown。"""
    if "error" in result:
        return f"因子组合回测不可用：{result['error']}"
    p, b = result["portfolio"], result["benchmark"]
    return (
        f"## 🏆 因子选股组合回测\n"
        f"**方法**：{result['method']}\n"
        f"**样本**：{result['n_stock']} 只股票 × {result['n_rebalances']} 次调仓\n"
        f"- **组合**：总收益 {p['total_return_pct']}% | 年化 {p['annual_return_pct']}% | "
        f"最大回撤 {p['max_drawdown_pct']}% | Sharpe {p['sharpe_ratio']}\n"
        f"- **基准**（全池等权）：总收益 {b['total_return_pct']}% | "
        f"回撤 {b['max_drawdown_pct']}%\n"
        f"- **超额收益**：**{result['excess_return_pct']:+.2f}%**\n"
        f"- 交易 {p['total_trades']} 笔 | 费用 {p['total_fees']} 元\n"
        f"*{result['note']}*"
    )


def format_selection(result: dict[str, Any]) -> str:
    """选股结果 → markdown。"""
    if "error" in result:
        return f"因子选股不可用：{result['error']}"
    lines = [
        f"## 🎯 多因子选股",
        f"**方法**：{result['method']}",
        f"**池**：{result['n_scored']}/{result['n_stock']} 只完成评分",
        "",
        "| 排名 | 代码 | 综合分 | 因子明细（z-score×方向） |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(result["selection"], 1):
        det = "，".join(f"{k} {v:+.2f}" for k, v in
                        list(r["factor_details"].items())[:5])
        lines.append(f"| {i} | {r['symbol']} | **{r['score']:+.3f}** | {det} |")
    lines.append(f"\n*参与因子：{'、'.join(result['factors_used'])}。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)


def format_factor_report(result: dict[str, Any]) -> str:
    """单因子检验 → markdown。"""
    if "error" in result:
        return f"因子 {result.get('factor', '')}：{result['error']}"
    mono = "✅ 单调" if result["monotonic"] else "❌ 非单调"
    useful = "✅ 有信息量" if result["useful"] else "⚠️ 弱信息"
    q_str = " → ".join(f"{x:+.2f}%" for x in result["quantile_returns"])
    return (
        f"## 🧬 因子检验：{result['factor']}（{result['category']} · "
        f"{result['version']}）\n"
        f"**IC**：{result['ic']:+.4f}（{useful}）｜ 前瞻 {result['horizon']} 日\n"
        f"**分层收益**（Q1→Q5）：{q_str}\n"
        f"**高低组差**：{result['spread_pct']:+.2f}% ｜ {mono}\n"
        f"**样本**：{result['n_samples']}（覆盖率 {result['coverage_pct']}%）\n"
        f"*IC = 因子值与未来收益的秩相关（|IC|≥0.03 视为有信息）。"
        f"研究参考，非投资建议。*"
    )


def format_scan(results: list[dict[str, Any]]) -> str:
    """因子批量扫描 → markdown。"""
    if not results:
        return "无因子可检验"
    lines = ["## 🧬 因子批量扫描（按 |IC| 排序）",
             "| 排名 | 因子 | 类别 | IC | 高低组差% | 单调 | 样本 |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"| {i} | {r.get('factor', '?')} | — | ❌ {r['error'][:40]} |")
            continue
        lines.append(
            f"| {i} | {r['factor']} | {r['category']} | {r['ic']:+.4f} | "
            f"{r['spread_pct']:+.2f} | {'✅' if r['monotonic'] else '❌'} | "
            f"{r['n_samples']} |")
    lines.append("\n*IC 为因子值与未来收益的秩相关；高低组差 = Q5-Q1 未来收益。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)
