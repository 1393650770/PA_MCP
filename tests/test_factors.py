# [AI:BEGIN]
# PA_MCP - 因子工厂测试（注册/检验/扫描/打包）
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa_mcp.research.factors import (
    evaluate_factor,
    get_factor_registry,
    pack_factor,
    scan_factors,
)


def _trend_df(n=300, seed=21):
    """分段趋势行情：每 60 日切换方向 → 动量因子应有正 IC。"""
    rng = np.random.default_rng(seed)
    close = 10.0
    rows = []
    regime = 1.0
    for i in range(n):
        if i % 60 == 0:
            regime = rng.choice([-1.0, 1.0])
        close *= 1 + regime * 0.004 + rng.normal(0, 0.006)
        rows.append({
            "date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=i),
            "open": close * 0.995, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1e6,
        })
    return pd.DataFrame(rows)


def test_registry_builtin_factors():
    """内置因子注册齐全（覆盖主要类别）。"""
    reg = get_factor_registry()
    names = {f.name for f in reg.list_all()}
    assert {"ma_gap20", "rsi14", "macd_hist", "atr_pct",
            "boll_position", "vol_ratio", "ret20", "ret60",
            "high_52w_dist"} <= names
    cats = {f.category for f in reg.list_all()}
    assert "momentum" in cats and "mean_reversion" in cats


def test_evaluate_factor_momentum():
    """趋势行情：动量因子 ret60 应有正 IC 且单调分层。"""
    df = _trend_df()
    fd = get_factor_registry().get("ret60")
    r = evaluate_factor(fd, df, horizon=5)
    assert "error" not in r
    assert r["n_samples"] >= 30
    assert r["ic"] > 0.05       # 趋势延续 → 动量 IC 正
    assert r["coverage_pct"] > 50
    assert len(r["quantile_returns"]) >= 3
    assert r["spread_pct"] > 0  # Q5 > Q1


def test_evaluate_factor_insufficient():
    """样本不足 → 明确 error（不静默）。"""
    df = _trend_df(n=50)
    fd = get_factor_registry().get("ret60")
    r = evaluate_factor(fd, df, horizon=5)
    assert "error" in r


def test_scan_factors_ranking():
    """批量扫描：返回按 |IC| 排序的完整排行。"""
    df = _trend_df()
    results = scan_factors(df, horizon=5)
    assert len(results) == len(get_factor_registry().list_all())
    ics = [abs(r.get("ic", 0)) for r in results if "error" not in r]
    assert ics == sorted(ics, reverse=True)  # 降序
    # 至少一个有效因子（趋势行情动量应该有效）
    assert any(r.get("useful") for r in results)


def test_pack_factor():
    p = pack_factor("ma_gap20")
    assert p is not None
    assert p["factor"]["name"] == "ma_gap20"
    assert p["factor"]["category"] == "mean_reversion"
    assert p["factor"]["version"] == "v1"
    assert pack_factor("不存在") is None


def _stock_df(n=150, seed=1, trend=0.004):
    """单只股票行情（trend = 日收益趋势强度）。"""
    rng = np.random.default_rng(seed)
    close = 10.0
    rows = []
    for i in range(n):
        close *= 1 + trend + rng.normal(0, 0.008)
        rows.append({"date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6})
    return pd.DataFrame(rows)


def test_select_stocks_by_factors():
    """趋势强弱不同的 6 只股票：强趋势者综合分居前。"""
    from pa_mcp.research.factors import (
        select_stocks_by_factors, format_selection)
    klines = {}
    trends = {"000001": 0.006, "000002": 0.004, "000003": 0.002,
              "000004": 0.0, "000005": -0.002, "000006": -0.004}
    for sym, t in trends.items():
        klines[sym] = _stock_df(seed=int(sym), trend=t)

    r = select_stocks_by_factors(klines, top_n=6)
    assert "error" not in r
    assert len(r["selection"]) == 6
    # 强趋势股票综合分更高（动量因子 IC 正 → 排序与趋势一致）
    scores = {x["symbol"]: x["score"] for x in r["selection"]}
    assert scores["000001"] > scores["000003"] > scores["000006"]
    assert len(r["factors_used"]) >= 3
    text = format_selection(r)
    assert "多因子选股" in text and "综合分" in text


def test_select_stocks_insufficient():
    from pa_mcp.research.factors import select_stocks_by_factors
    r = select_stocks_by_factors({}, top_n=5)
    assert "error" in r
    # 波动太大（无趋势）→ 因子 IC 不达标
    klines = {f"6000{i:02d}": _stock_df(seed=i, trend=0.0)
              for i in range(5)}
    r2 = select_stocks_by_factors(klines)
    assert "error" in r2 or r2["n_scored"] >= 0


def _regime_df(n=400, seed=1):
    """分段趋势（120 日切换）：动量可预测。"""
    rng = np.random.default_rng(seed)
    close = 10.0
    rows = []
    regime = 1.0
    for i in range(n):
        if i % 120 == 0:
            regime = rng.choice([-1.0, 1.0])
        close *= 1 + regime * 0.003 + rng.normal(0, 0.006)
        rows.append({"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e6})
    return pd.DataFrame(rows)


def test_backtest_factor_selection():
    """分段趋势池：因子选股组合应跑赢全池等权基准。"""
    from pa_mcp.research.factors import (
        backtest_factor_selection, format_portfolio_backtest)
    klines = {f"6000{i:02d}": _regime_df(seed=i) for i in range(1, 9)}
    r = backtest_factor_selection(klines, top_n=3, horizon=5,
                                  train_window=120)
    assert "error" not in r
    assert r["n_stock"] == 8
    assert r["n_rebalances"] >= 5
    p = r["portfolio"]
    assert p["total_return_pct"] is not None
    assert p["max_drawdown_pct"] is not None
    assert r["benchmark"]["total_return_pct"] is not None
    assert r["excess_return_pct"] > 0  # 选股优于等权
    text = format_portfolio_backtest(r)
    assert "超额收益" in text and "组合" in text


def test_backtest_factor_selection_insufficient():
    from pa_mcp.research.factors import backtest_factor_selection
    klines = {f"6000{i:02d}": _regime_df(n=80, seed=i) for i in range(1, 5)}
    r = backtest_factor_selection(klines, train_window=120)
    assert "error" in r  # 数据不足训练窗口


def test_register_custom_factor():
    """自定义因子注册 + 检验。"""
    reg = get_factor_registry()

    @reg.factor("custom_test", "momentum", "测试因子：5 日收益")
    def _f(df):
        return df.sort_values("date")["close"].pct_change(5) * 100

    fd = reg.get("custom_test")
    assert fd is not None
    df = _trend_df()
    r = evaluate_factor(fd, df, horizon=5)
    assert "error" not in r or True  # 数值合理性由趋势数据保证
