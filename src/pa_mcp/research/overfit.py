# [AI:BEGIN]
# PA_MCP - Research: 回测过拟合与多重检验评估
#
# 机制借鉴开源 QuantSkills skill-backtest-overfit（Bailey & López de Prado
# 学术框架），自研纯 numpy 实现（无 scipy 依赖）：
#   1. DSR（Deflated Sharpe Ratio，Bailey & López de Prado 2014）：
#      考虑试验次数 N 后的期望最大 Sharpe（SR*），观测 Sharpe 是否仍显著
#   2. CSCV-PBO（Probability of Backtest Overfitting，Bailey et al. 2015）：
#      时间序列分块 → 组合子集训练/测试 → 样本内最优策略在样本外
#      排名不佳的比例 = 过拟合概率
#   3. Harvey-Liu 折减（2015）：
#      多重检验下 Sharpe 显著性阈值 = sqrt(2 ln N / T)
#
# 用途：参数网格调优 / 多策略筛选后必须做——否则「调参 100 次挑出的
# 最佳 Sharpe」可能只是运气。
# [AI:END]

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni 常数


# ---- 正态 CDF（numpy 无 scipy 时的 erf 实现） ----

def _normal_cdf(x: float) -> float:
    """标准正态 CDF：0.5 * (1 + erf(x/√2))，erf 用 Abramowitz-Stegun 近似。"""
    if abs(x) > 6.5:
        return 1.0 if x > 0 else 0.0
    # Abramowitz-Stegun 7.1.26（误差 < 1.5e-7）
    sign = 1.0 if x >= 0 else -1.0
    ax = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                 - 0.284496736) * t + 0.254829592) * t * math.exp(-ax * ax)
    return 0.5 * (1.0 + sign * y)


def _normal_ppf(p: float) -> float:
    """标准正态分位数（Acklam 近似，误差 < 1.15e-9）。"""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


# ---- DSR：考虑多重检验的 Sharpe 显著性 ----

def expected_max_sharpe(n_trials: int) -> float:
    """E[max SR_n]：N 次独立试验的期望最大标准正态值（Bailey 2014 式 9-10）。"""
    if n_trials <= 1:
        return 0.0
    z1 = _normal_ppf(1.0 - 1.0 / n_trials)
    z2 = _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    return (1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2


def deflated_sharpe(
    sharpe_obs: float,
    n_trials: int,
    periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, Any]:
    """DSR：观测 Sharpe 在 N 次试验后是否仍显著（Bailey & López de Prado 2014）。

    Args:
        sharpe_obs: 观测**年化** Sharpe（UI/MCP 传入的直观单位）
        n_trials: 试验/调参次数（策略×参数组合×股票，诚实估计）
        periods: 样本交易日数（年化 Sharpe 的分母）
        skew: 收益偏度（默认 0 高斯）
        kurtosis: 收益峰度（默认 3 高斯）

    内部按论文约定转周期频率：sr_daily = 年化 / √T，V[SR] 用日频计算。
    """
    n_trials = max(1, int(n_trials))
    T = max(2.0, float(periods))

    # 年化 → 日频（论文 V[SR] 与观测 SR 须同频率）
    sr_daily = sharpe_obs / math.sqrt(T)

    # V[SR] = (1 - γ3·SR + (γ4-1)/4·SR²) / (T-1)
    var_sr = (1.0 - skew * sr_daily
              + (kurtosis - 1.0) / 4.0 * sr_daily ** 2) / (T - 1.0)
    var_sr = max(var_sr, 1e-12)

    sr_star_daily = math.sqrt(var_sr) * expected_max_sharpe(n_trials)
    # 折减阈值换算回年化（展示用）
    sr_star = sr_star_daily * math.sqrt(T)

    # DSR = Φ((SR - SR*)·√(T-1) / √(1 - γ3·SR + (γ4-1)/4·SR²))（日频口径）
    denom = math.sqrt(max(1e-12, (1.0 - skew * sr_daily
                                  + (kurtosis - 1.0) / 4.0 * sr_daily ** 2)))
    dsr = _normal_cdf((sr_daily - sr_star_daily) * math.sqrt(T - 1.0) / denom)

    return {
        "sharpe_obs": round(sharpe_obs, 4),
        "n_trials": n_trials,
        "periods": int(T),
        "expected_max_sharpe": round(sr_star, 4),
        "dsr": round(dsr, 4),
        "significant": dsr >= 0.95,
        "note": (f"考虑 {n_trials} 次试验后，期望最大（运气）Sharpe 为 "
                 f"{sr_star:.3f}（年化）；观测 {sharpe_obs:.3f} 的 DSR = {dsr:.1%}"
                 f"（{'显著 ✅' if dsr >= 0.95 else '不显著 ⚠️ 可能只是调参运气'}）"),
    }


# ---- Harvey-Liu 折减：多重检验 Sharpe 阈值 ----

def harvey_liu_threshold(n_trials: int, periods: int) -> float:
    """多重检验下 Sharpe 显著性阈值（Harvey & Liu 2015）。

    SR* = sqrt(2 ln N / T)。若观测 Sharpe 低于阈值，说明可能是运气。
    """
    n_trials = max(1, int(n_trials))
    T = max(2.0, float(periods))
    return round(math.sqrt(2.0 * math.log(n_trials) / T), 4)


# ---- CSCV-PBO：过拟合概率（Bailey et al. 2015） ----

def pbo_cscv(returns_matrix: np.ndarray, s_blocks: int = 8,
             seed: Optional[int] = None) -> dict[str, Any]:
    """组合对称交叉验证（CSCV）→ PBO。

    Args:
        returns_matrix: 策略收益矩阵（n_strategies × n_periods），
            每行一个策略/参数组合的收益序列
        s_blocks: 时间块数（默认 8）
        seed: 随机种子（可复现）

    Returns:
        PBO（过拟合概率 0-1）+ 诊断（样本内最优策略集合/样本外排名不佳比例）
    """
    mat = np.asarray(returns_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("returns_matrix 须为二维（策略 × 时间）")
    n_strat, n_periods = mat.shape
    if n_strat < 3 or n_periods < s_blocks:
        return {"error": f"数据不足：策略 {n_strat} 个、时间 {n_periods} 期"
                         f"（需 ≥3 策略、≥{s_blocks} 期）"}

    # 分块（按列）
    block_size = n_periods // s_blocks
    blocks = [mat[:, i * block_size:(i + 1) * block_size]
              for i in range(s_blocks - 1)]
    blocks.append(mat[:, (s_blocks - 1) * block_size:])  # 最后一块含余数

    # 块内 Sharpe（行 = 策略，列 = 块）
    sharpe_blocks = np.array([
        [float(blocks[b][s].mean() / (blocks[b][s].std() + 1e-12))
         for b in range(s_blocks)]
        for s in range(n_strat)
    ])

    # C(S, S/2) 组合（太多则抽样）
    half = s_blocks // 2
    combos = list(combinations(range(s_blocks), half))
    rng = np.random.default_rng(seed)
    if len(combos) > 64:
        combos = rng.choice(combos, 64, replace=False).tolist()

    w_star_list: list[float] = []
    for train_idx in combos:
        test_idx = [i for i in range(s_blocks) if i not in train_idx]
        train_sr = sharpe_blocks[:, list(train_idx)].mean(axis=1)
        test_sr = sharpe_blocks[:, test_idx].mean(axis=1)
        # 样本内最优策略在样本外的表现排名（分位，越小越好）
        best_in = int(np.argmax(train_sr))
        test_rank = np.argsort(np.argsort(-test_sr))[best_in]  # 0 = 样本外最佳
        w_star = (test_rank + 1) / n_strat  # 分位 0-1
        w_star_list.append(w_star)

    # PBO = P(W* > 50%)：样本内最优在样本外排名差于中位的概率
    #（纯噪声 → 均匀分布 → PBO ≈ 0.5；真 alpha → 样本内最优样本外仍优 → PBO 低）
    pbo = float(np.mean([w > 0.5 for w in w_star_list]))
    avg_w = float(np.mean(w_star_list))
    return {
        "pbo": round(pbo, 4),
        "avg_rank_quantile": round(avg_w, 4),
        "n_strategies": n_strat,
        "n_periods": n_periods,
        "s_blocks": s_blocks,
        "n_combos": len(combos),
        "note": (f"PBO = {pbo:.0%}：样本内最优策略在样本外跌出前半的概率"
                 f"（{'低 → 过拟合风险可控 ✅' if pbo < 0.25 else '中 → 警惕' if pbo < 0.5 else '高 → 过拟合严重 ⚠️'}）；"
                 f"平均排名分位 {avg_w:.0%}（越低越好）"),
    }


# ---- 组合报告 ----

def run_overfit_report(
    sharpe_obs: float,
    n_trials: int,
    periods: int,
    returns_matrix: Optional[np.ndarray] = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    s_blocks: int = 8,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """完整过拟合诊断：DSR + Harvey-Liu 阈值 + （可选）CSCV-PBO。"""
    report = {
        "dsr": deflated_sharpe(sharpe_obs, n_trials, periods, skew, kurtosis),
        "harvey_liu": {
            "threshold": harvey_liu_threshold(n_trials, periods),
            "passed": sharpe_obs >= harvey_liu_threshold(n_trials, periods),
        },
    }
    if returns_matrix is not None:
        report["pbo"] = pbo_cscv(returns_matrix, s_blocks=s_blocks, seed=seed)
    return report


def format_overfit_report(report: dict[str, Any]) -> str:
    """诊断报告 → markdown（UI/MCP 共用）。"""
    dsr = report["dsr"]
    hl = report["harvey_liu"]
    lines = [
        "## 🎲 回测过拟合诊断（多重检验修正）",
        f"- **观测 Sharpe**：{dsr['sharpe_obs']:.3f}（试验次数 {dsr['n_trials']}，"
        f"样本期 {dsr['periods']}）",
        f"- **期望最大 Sharpe（E[max SR]）**：{dsr['expected_max_sharpe']:.3f}"
        f"——调参 {dsr['n_trials']} 次后，运气最好能到的高度",
        f"- **DSR（Deflated Sharpe）**：**{dsr['dsr']:.1%}**"
        f"（{'✅ 显著' if dsr['significant'] else '⚠️ 不显著'}）",
        f"- **Harvey-Liu 阈值**：Sharpe ≥ {hl['threshold']:.3f}"
        f"（{'✅ 达标' if hl['passed'] else '⚠️ 未达标'}）",
        f"- **解读**：{dsr['note']}",
    ]
    if "pbo" in report:
        p = report["pbo"]
        if "error" in p:
            lines.append(f"\n- **CSCV-PBO**：{p['error']}")
        else:
            lines.extend([
                f"\n### CSCV-PBO（组合对称交叉验证）",
                f"- **PBO（过拟合概率）**：**{p['pbo']:.0%}**"
                f"（{p['n_strategies']} 策略 × {p['n_periods']} 期，"
                f"{p['s_blocks']} 块 × {p['n_combos']} 组合）",
                f"- **平均样本外排名分位**：{p['avg_rank_quantile']:.0%}",
                f"- **解读**：{p['note']}",
            ])
    lines.append("\n*多重检验修正：调参次数越多，碰出好 Sharpe 的概率越大，"
                 "DSR/PBO 就是为此设计的。研究参考，非投资建议。*")
    return "\n".join(lines)
