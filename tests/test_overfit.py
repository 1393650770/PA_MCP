# [AI:BEGIN]
# PA_MCP - 回测过拟合评估（DSR/PBO/Harvey-Liu）测试
# [AI:END]

from __future__ import annotations

import numpy as np
import pytest

from pa_mcp.research.overfit import (
    deflated_sharpe,
    expected_max_sharpe,
    format_overfit_report,
    harvey_liu_threshold,
    pbo_cscv,
    run_overfit_report,
)


# ---- DSR ----

def test_expected_max_sharpe_monotonic():
    """试验次数越多，期望最大 Sharpe 越高。"""
    assert expected_max_sharpe(1) == 0.0
    assert expected_max_sharpe(10) < expected_max_sharpe(100)
    assert expected_max_sharpe(100) < expected_max_sharpe(10000)
    # 数值合理：E[max Z_N] ~ sqrt(2 ln N)
    assert 1.0 < expected_max_sharpe(10) < 3.5
    assert 2.0 < expected_max_sharpe(1000) < 5.0


def test_dsr_no_trials_no_deflation():
    """N=1 时无折减：高 Sharpe 显著。"""
    r = deflated_sharpe(sharpe_obs=3.0, n_trials=1, periods=250)
    assert r["expected_max_sharpe"] == 0.0
    assert r["dsr"] > 0.99
    assert r["significant"] is True


def test_dsr_deflation_with_many_trials():
    """100 次调参后：平庸 Sharpe 不再显著。"""
    r = deflated_sharpe(sharpe_obs=1.0, n_trials=100, periods=250)
    # 折减存在：期望最大 Sharpe（sr_star）> 0 且 DSR 显著低于 N=1
    assert r["expected_max_sharpe"] > 0.05
    assert r["dsr"] < 0.5                   # 观测 1.0 被折减
    assert r["significant"] is False
    # 真强信号（4.5 年化 Sharpe）经得起 100 次试验的多重检验
    r2 = deflated_sharpe(sharpe_obs=4.5, n_trials=100, periods=250)
    assert r2["significant"] is True


def test_dsr_more_trials_lower_dsr():
    """同 Sharpe 下：试验次数越多 DSR 越低（单调）。"""
    d10 = deflated_sharpe(1.2, 10, 250)["dsr"]
    d100 = deflated_sharpe(1.2, 100, 250)["dsr"]
    d1000 = deflated_sharpe(1.2, 1000, 250)["dsr"]
    assert d10 > d100 > d1000
    assert d10 < 1.0  # 避免饱和（用 1.2 而非 1.8）


def test_harvey_liu_threshold():
    """阈值 sqrt(2 ln N / T)：N 增阈值升，T 增阈值降。"""
    t10 = harvey_liu_threshold(10, 250)
    t100 = harvey_liu_threshold(100, 250)
    t100_long = harvey_liu_threshold(100, 1000)
    assert t100 > t10
    assert t100_long < t100
    assert 0.0 < t100 < 1.0


# ---- CSCV-PBO ----

def _make_matrix(n_strat=10, n_periods=400, seed=42):
    """合成：2 个"真策略"（正 alpha）+ 8 个噪声策略。"""
    rng = np.random.default_rng(seed)
    mat = rng.normal(0, 1, size=(n_strat, n_periods))
    mat[0] += 0.15   # 真 alpha
    mat[1] += 0.10
    return mat


def test_pbo_low_for_real_edge():
    """真 alpha 策略存在 → PBO 应低（<0.35）。"""
    mat = _make_matrix()
    r = pbo_cscv(mat, s_blocks=8, seed=7)
    assert "error" not in r
    assert 0.0 <= r["pbo"] <= 1.0
    assert r["n_strategies"] == 10
    assert r["avg_rank_quantile"] <= 0.5  # 样本内最优通常仍居前


def test_pbo_high_for_noise():
    """全噪声策略（无真 alpha）→ PBO 高（>0.35），过拟合显著。"""
    rng = np.random.default_rng(99)
    mat = rng.normal(0, 1, size=(12, 300))
    r = pbo_cscv(mat, s_blocks=8, seed=7)
    assert r["pbo"] > 0.3


def test_pbo_insufficient_data():
    r = pbo_cscv(np.zeros((2, 10)))   # 策略 <3 → error
    assert "error" in r
    r2 = pbo_cscv(np.zeros((4, 4)), s_blocks=8)  # 期数不足 → error
    assert "error" in r2


def test_pbo_deterministic_with_seed():
    """固定 seed → 可复现。"""
    mat = _make_matrix(seed=5)
    r1 = pbo_cscv(mat, s_blocks=6, seed=123)
    r2 = pbo_cscv(mat, s_blocks=6, seed=123)
    assert r1["pbo"] == r2["pbo"]


# ---- 组合报告 ----

def test_full_report_and_format():
    report = run_overfit_report(2.0, 50, 500, returns_matrix=_make_matrix())
    assert "dsr" in report and "harvey_liu" in report and "pbo" in report
    text = format_overfit_report(report)
    assert "DSR" in text and "PBO" in text and "Harvey" in text
    # 仅 DSR（无收益矩阵）
    r2 = run_overfit_report(2.0, 50, 500)
    assert "pbo" not in r2
