# [AI:BEGIN]
# PA_MCP - Agent Layer: 市场预测（LLM 未来走势预测）
#
# 设计借鉴 PA_Agent「未来走势预期 + 周期位置」机制（机制层借鉴，实现自研）：
#   1. 确定性特征抽取：不把原始 K 线全量丢给 LLM，先压缩为可解释的技术特征
#   2. 周期位置（cycle_position）：尖峰/通道/区间等市场结构枚举（确定性规则判定）
#   3. 结构化多场景预测：方向 + 概率分布 + 期望收益 + 关键价位 + 多情景
#   4. 落盘验证闭环：预测写入 prediction_log 表，到期后用真实收益回填，
#      计算命中率 / Brier 分数 / 方向收益 —— 预测可检验，不做纯算命
#   5. 无 LLM 时确定性统计降级（方向由趋势/动量/量能打分，概率由历史波动映射）
#   6. JSON 校验 + 语义校验 + 一次修复重试（借鉴 PA_Agent validation_retry 思路）
# [AI:END]

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

from pa_mcp.engine.indicators.indicators import calc_adx, calc_atr, calc_macd, calc_ma, calc_rsi

logger = logging.getLogger(__name__)

# ---- 周期位置枚举（A 股语境，借鉴 PA_Agent cycle 思路但值自研） ----

CYCLE_POSITIONS: tuple[str, ...] = (
    "spike",           # 尖峰行情（近期急涨急跌）
    "micro_channel",   # 微型通道（极窄振幅）
    "tight_channel",   # 窄通道
    "normal_channel",  # 正常通道
    "broad_channel",   # 宽通道
    "trending_range",  # 趋势型区间（缓慢趋势 + 区间结构）
    "trading_range",   # 震荡区间
    "extreme_range",   # 极端震荡（宽幅无方向）
)

CYCLE_POSITION_ZH: dict[str, str] = {
    "spike": "尖峰行情",
    "micro_channel": "微型通道",
    "tight_channel": "窄通道",
    "normal_channel": "正常通道",
    "broad_channel": "宽通道",
    "trending_range": "趋势型区间",
    "trading_range": "震荡区间",
    "extreme_range": "极端震荡",
    "unknown": "未知",
}

# 预测验证阈值
SIDEWAYS_THRESHOLD_PCT = 1.5   # |收益| <= 1.5% 视为 sideways 命中
AMBIGUOUS_THRESHOLD_PCT = 1.0  # 非 sideways 预测中 |收益| <= 1.0% 视为模糊
DEFAULT_SIDEWAYS_PROB = 0.15   # 确定性降级时 sideways 基准概率

PROMPT_VERSION = "pred-v1"


def cycle_zh(raw: str) -> str:
    """周期位置中文名。"""
    key = (raw or "unknown").strip().lower()
    return CYCLE_POSITION_ZH.get(key, raw or "未知")


# ---- 特征抽取（确定性，无未来函数） ----

def _series(x: pd.Series, default: float = 0.0) -> float:
    try:
        v = float(x.iloc[-1])
        return v if pd.notna(v) else default
    except Exception:
        return default


def extract_features(df: pd.DataFrame) -> dict[str, Any]:
    """从日 K 线抽取确定性技术特征（供 LLM 预测与确定性降级共用）。

    只用截至最后一根 bar 的数据，保证无未来函数。
    """
    if df is None or df.empty:
        return {"error": "no data"}
    data = df.sort_values("date").reset_index(drop=True)
    close = data["close"]

    ma = calc_ma(data)
    rsi = calc_rsi(data)
    macd = calc_macd(data)
    atr = calc_atr(data)
    adx = calc_adx(data)

    n = len(data)
    last_close = float(close.iloc[-1])
    ret20 = (last_close / close.iloc[-21] - 1) * 100 if n >= 21 else 0.0
    ret60 = (last_close / close.iloc[-61] - 1) * 100 if n >= 61 else ret20

    high20 = float(data["high"].tail(20).max())
    low20 = float(data["low"].tail(20).min())
    ma20 = _series(ma["ma20"]) if "ma20" in ma else last_close
    ma60 = _series(ma["ma60"]) if "ma60" in ma else last_close
    ma5 = _series(ma["ma5"]) if "ma5" in ma else last_close

    vol = data["volume"] if "volume" in data else None
    vol_ratio = 0.0
    if vol is not None and n >= 21:
        avg20 = float(vol.tail(20).mean()) if vol.tail(20).mean() > 0 else 0.0
        if avg20 > 0:
            vol_ratio = float(vol.iloc[-1]) / avg20

    atr_pct = _series(atr["atr14"]) / last_close * 100 if "atr14" in atr else 0.0
    adx_val = _series(adx["adx14"]) if "adx14" in adx else 20.0
    rsi14 = _series(rsi["rsi14"]) if "rsi14" in rsi else 50.0
    macd_hist = _series(macd["macd_hist"]) if "macd_hist" in macd else 0.0

    # 布林位置 %B
    boll_pos = 50.0
    try:
        from pa_mcp.engine.indicators.indicators import calc_bollinger
        boll = calc_bollinger(data)
        if "boll_mid" in boll and "boll_up" in boll and "boll_low" in boll:
            up = float(boll["boll_up"].iloc[-1]); low = float(boll["boll_low"].iloc[-1])
            if up > low:
                boll_pos = (last_close - low) / (up - low) * 100
    except Exception:
        pass

    # ---- 周期位置判定（确定性规则） ----
    amp20 = (high20 - low20) / last_close * 100 if last_close > 0 else 0.0
    spike = any(
        abs((float(close.iloc[-i]) / float(close.iloc[-i - 1]) - 1) * 100) >= 6.0
        for i in range(1, 4) if n >= i + 1
    )
    if spike:
        cycle_position = "spike"
    elif adx_val >= 30:
        if amp20 >= 25:
            cycle_position = "broad_channel"
        elif amp20 >= 12:
            cycle_position = "normal_channel"
        else:
            cycle_position = "tight_channel"
    elif adx_val >= 22:
        cycle_position = "trending_range"
    else:
        if amp20 >= 25:
            cycle_position = "extreme_range"
        else:
            cycle_position = "trading_range"

    features = {
        "last_close": round(last_close, 3),
        "ret20_pct": round(ret20, 2),
        "ret60_pct": round(ret60, 2),
        "ma5": round(ma5, 3), "ma20": round(ma20, 3), "ma60": round(ma60, 3),
        "ma_alignment": (
            "多头排列" if ma5 > ma20 > ma60 else
            "空头排列" if ma5 < ma20 < ma60 else "均线缠绕"),
        "rsi14": round(rsi14, 1),
        "macd_hist": round(macd_hist, 4),
        "adx14": round(adx_val, 1),
        "atr_pct": round(atr_pct, 2),
        "volume_ratio": round(vol_ratio, 2),
        "boll_position_pct": round(boll_pos, 1),
        "support_20d": round(low20, 3),
        "resistance_20d": round(high20, 3),
        "cycle_position": cycle_position,
        "cycle_position_zh": cycle_zh(cycle_position),
        "days": n,
    }
    return features


def format_features(features: dict[str, Any]) -> str:
    """特征字典 → LLM 可读文本。"""
    if not features or "error" in features:
        return "无数据"
    return (
        f"收盘 {features['last_close']}，20日涨跌 {features['ret20_pct']:+.1f}%"
        f"（60日 {features['ret60_pct']:+.1f}%）\n"
        f"均线：MA5 {features['ma5']} / MA20 {features['ma20']} / MA60 {features['ma60']}"
        f" → {features['ma_alignment']}\n"
        f"动量：RSI14 {features['rsi14']}，MACD柱 {features['macd_hist']}，"
        f"ADX14 {features['adx14']}，ATR {features['atr_pct']:.2f}%\n"
        f"量能：量比 {features['volume_ratio']}，布林位置 {features['boll_position_pct']}%\n"
        f"关键位：支撑 {features['support_20d']} / 压力 {features['resistance_20d']}\n"
        f"周期位置：{features['cycle_position_zh']}（{features['cycle_position']}）"
    )


# ---- 预测结果 DTO ----

@dataclass
class PredictionResult:
    symbol: str
    predict_date: str
    horizon: str
    direction: str = "sideways"
    probability: float = 0.5
    prob_up: float = 0.4
    prob_down: float = 0.4
    prob_sideways: float = 0.2
    expected_return_pct: float = 0.0
    expected_range_low: float = -3.0
    expected_range_high: float = 3.0
    cycle_position: str = "trading_range"
    cycle_forecast: str = "trading_range"
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    scenarios: list[dict] = field(default_factory=list)
    confidence: float = 0.5
    key_reasons: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    model: str = "deterministic"
    prompt_version: str = PROMPT_VERSION
    mode: str = "deterministic"  # llm | deterministic
    disclaimer: str = "研究参考，非投资建议。预测存在不确定性，请以实际行情为准。"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "predict_date": self.predict_date,
            "horizon": self.horizon,
            "direction": self.direction,
            "probability": self.probability,
            "probability_distribution": {
                "up": self.prob_up, "down": self.prob_down, "sideways": self.prob_sideways,
            },
            "expected_return_pct": self.expected_return_pct,
            "expected_range_pct": [self.expected_range_low, self.expected_range_high],
            "cycle_position": self.cycle_position,
            "cycle_position_zh": cycle_zh(self.cycle_position),
            "cycle_forecast": self.cycle_forecast,
            "cycle_forecast_zh": cycle_zh(self.cycle_forecast),
            "key_levels": {"support": self.support_levels, "resistance": self.resistance_levels},
            "scenarios": self.scenarios,
            "confidence": self.confidence,
            "key_reasons": self.key_reasons,
            "key_risks": self.key_risks,
            "model": self.model,
            "mode": self.mode,
            "disclaimer": self.disclaimer,
        }

    @classmethod
    def from_llm_json(cls, symbol: str, predict_date: str, horizon: str,
                      raw: dict[str, Any], model: str) -> "PredictionResult":
        """从 LLM JSON 构造（含越界 clamp 与必填校验）。"""
        def _clamp(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, float(v)))

        direction = str(raw.get("direction", "sideways")).lower()
        if direction not in ("up", "down", "sideways"):
            direction = "sideways"

        dist = raw.get("probability_distribution") or {}
        try:
            pu = _clamp(float(dist.get("up", 0.34)), 0.0, 1.0)
            pd_ = _clamp(float(dist.get("down", 0.33)), 0.0, 1.0)
            ps = _clamp(float(dist.get("sideways", 0.33)), 0.0, 1.0)
        except Exception:
            pu, pd_, ps = 0.4, 0.4, 0.2
        total = pu + pd_ + ps
        if total <= 0:
            pu, pd_, ps = 0.4, 0.4, 0.2
        else:
            pu, pd_, ps = pu / total, pd_ / total, ps / total

        prob_map = {"up": pu, "down": pd_, "sideways": ps}
        prob = _clamp(prob_map.get(direction, 0.5), 0.3, 0.95)

        try:
            exp_ret = float(raw.get("expected_return_pct", 0.0))
        except Exception:
            exp_ret = 0.0
        try:
            rng = raw.get("expected_range_pct") or raw.get("expected_range") or []
            lo = float(rng[0]) if len(rng) >= 1 else exp_ret - 3
            hi = float(rng[1]) if len(rng) >= 2 else exp_ret + 3
        except Exception:
            lo, hi = exp_ret - 3, exp_ret + 3
        exp_ret = _clamp(exp_ret, -25, 25)
        lo, hi = _clamp(lo, -30, 30), _clamp(hi, -30, 30)

        cycle = str(raw.get("cycle_position", "trading_range")).lower()
        if cycle not in CYCLE_POSITIONS:
            cycle = "trading_range"
        cf = str(raw.get("cycle_forecast", cycle)).lower()
        if cf not in CYCLE_POSITIONS:
            cf = cycle

        try:
            conf = _clamp(float(raw.get("confidence", 0.5)), 0.0, 1.0)
        except Exception:
            conf = 0.5

        return cls(
            symbol=symbol, predict_date=predict_date, horizon=horizon,
            direction=direction, probability=prob,
            prob_up=pu, prob_down=pd_, prob_sideways=ps,
            expected_return_pct=round(exp_ret, 2),
            expected_range_low=round(lo, 2), expected_range_high=round(hi, 2),
            cycle_position=cycle, cycle_forecast=cf,
            support_levels=[float(x) for x in (raw.get("support_levels") or []) if isinstance(x, (int, float))][:5],
            resistance_levels=[float(x) for x in (raw.get("resistance_levels") or []) if isinstance(x, (int, float))][:5],
            scenarios=raw.get("scenarios") or [],
            confidence=conf,
            key_reasons=[str(x) for x in (raw.get("key_reasons") or [])][:8],
            key_risks=[str(x) for x in (raw.get("key_risks") or [])][:8],
            model=model, mode="llm",
        )


# ---- LLM Prompt ----

PREDICTION_PROMPT = """你是 A 股量化研究员。基于以下确定性技术特征，对 {symbol} 做未来 {horizon} 个交易日的走势预测（不是投资建议）。

【当前特征】
{features}

【预测要求】
1. 只输出一个 JSON 对象，不要 markdown 代码块、不要注释
2. direction 只允许 "up" / "down" / "sideways"
3. probability_distribution 的 up/down/sideways 三个值相加必须等于 1
4. expected_return_pct 必须落在 expected_range_pct 区间内
5. cycle_position（当前周期）与 cycle_forecast（预测期末周期）从枚举中选：
   spike, micro_channel, tight_channel, normal_channel, broad_channel, trending_range, trading_range, extreme_range
6. scenarios 给 2-3 个情景，每个含 name（中文）、probability（相加为1）、target_pct（相对当前价）、description（触发条件，中文）
7. 关键价位 support_levels / resistance_levels 给具体价格数字
8. confidence 为 0 到 1 的置信度
9. key_reasons / key_risks 用中文，各 2-4 条

【JSON 格式】
{{
  "direction": "up",
  "probability": 0.62,
  "probability_distribution": {{"up": 0.62, "down": 0.20, "sideways": 0.18}},
  "expected_return_pct": 3.5,
  "expected_range_pct": [-1.5, 6.0],
  "cycle_position": "normal_channel",
  "cycle_forecast": "broad_channel",
  "support_levels": [10.2, 9.8],
  "resistance_levels": [11.5, 12.0],
  "scenarios": [
    {{"name": "放量突破", "probability": 0.4, "target_pct": 6.0, "description": "放量站上压力位后延续"}},
    {{"name": "区间震荡", "probability": 0.45, "target_pct": 1.0, "description": "在支撑压力间反复"}},
    {{"name": "跌破支撑", "probability": 0.15, "target_pct": -4.0, "description": "跌破支撑后加速下行"}}
  ],
  "confidence": 0.65,
  "key_reasons": ["均线多头排列", "量比放大"],
  "key_risks": ["大盘调整风险", "财报窗口波动"]
}}"""


# ---- 服务 ----

class PredictionService:
    """市场预测服务：LLM 预测 + 确定性降级 + 落盘 + 验证闭环。

    依赖注入保持解耦：store_path 可选（默认用全局配置）；LLM 通过
    get_llm_adapter() 惰性获取，未配置时自动降级为确定性预测。
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    # ---- 数据访问（短连接模式，避免与其他连接锁冲突） ----
    def _store(self):
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        path = self._store_path or get_settings().database.path
        store = DuckDBStore(path)
        store.connect()
        return store

    # ---- 核心预测 ----
    async def predict(self, symbol: str, kline_df: pd.DataFrame,
                      horizon: str = "5d", use_llm: bool = True) -> PredictionResult:
        """对 symbol 做未来 horizon 个交易日预测。

        Args:
            symbol: 股票代码
            kline_df: 日 K 线
            horizon: "5d"（短线）或 "20d"（中线）
            use_llm: True = LLM 优先（无配置自动降级）；False = 强制
                确定性统计预测（批量/敏感性分析时控制成本）
        """
        horizon = horizon if horizon in ("1d", "5d", "20d") else "5d"
        today = date.today().isoformat()
        features = extract_features(kline_df)
        if "error" in features:
            raise ValueError(f"无法抽取 {symbol} 的 K 线特征：{features['error']}")

        # 尝试 LLM（use_llm=False 时跳过），失败/未配置则确定性降级
        # ensure_llm_adapter 统一兜底：单例为空时主动读配置初始化
        try:
            from pa_mcp.agent.llm_port import LLMCallParams
            from pa_mcp.agent.llm_factory import ensure_llm_adapter
            adapter = ensure_llm_adapter() if use_llm else None
            if adapter is not None:
                return await self._predict_with_llm(
                    adapter, symbol, today, horizon, features, kline_df)
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 预测失败，降级为确定性预测: %s / %s", symbol, e)

        return self._predict_deterministic(symbol, today, horizon, features)

    async def _predict_with_llm(self, adapter, symbol: str, predict_date: str,
                                horizon: str, features: dict[str, Any],
                                kline_df: pd.DataFrame) -> PredictionResult:
        """LLM 预测 + JSON 校验 + 一次修复重试。"""
        from pa_mcp.agent.llm_port import LLMCallParams
        user_prompt = PREDICTION_PROMPT.format(
            symbol=symbol, horizon=horizon, features=format_features(features))
        # 板块轮动融合：注入所属板块 RS 上下文（best-effort，无数据跳过）
        sector_ctx = self._sector_context(symbol)
        if sector_ctx:
            user_prompt += f"\n\n【所属板块环境】\n{sector_ctx}"
        # 市场结构融合：注入指数缠论方向（大盘环境）
        mkt_ctx = await self._market_bias_context()
        if mkt_ctx:
            user_prompt += f"\n\n【大盘环境】\n{mkt_ctx}"
        params = LLMCallParams(
            system_prompt=(
                "你是有经验的 A 股量化研究员。只输出合法 JSON，不输出任何其他文本。"
                "预测是研究输出，不是投资建议。"
            ),
            user_prompt=user_prompt,
            mode="fast", max_tokens=1500,
        )
        raw = await adapter.chat_json(params)
        if not isinstance(raw, dict) or "error" in raw:
            raise ValueError(f"LLM 返回异常: {raw}")

        errors = self._validate_llm_json(raw)
        if errors:
            # 校验失败 → 反馈错误重试一次（借鉴 PA_Agent validation_retry 机制）
            logger.info("预测 JSON 校验失败，重试一次: %s", errors)
            retry_prompt = (
                f"{user_prompt}\n\n【校验错误，请修正后重新输出完整 JSON】\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n只输出修正后的 JSON。"
            )
            params2 = LLMCallParams(
                system_prompt=params.system_prompt, user_prompt=retry_prompt,
                mode="fast", max_tokens=1500,
            )
            raw2 = await adapter.chat_json(params2)
            if isinstance(raw2, dict) and "error" not in raw2:
                errors2 = self._validate_llm_json(raw2)
                if not errors2:
                    raw = raw2

        result = PredictionResult.from_llm_json(
            symbol, predict_date, horizon, raw, model=adapter.provider_name)
        # 回填收盘价上下文（供 UI 展示）
        result.support_levels = features.get("support_20d") and [
            features["support_20d"]] + [x for x in result.support_levels if x] or result.support_levels
        result.resistance_levels = features.get("resistance_20d") and [
            features["resistance_20d"]] + [x for x in result.resistance_levels if x] or result.resistance_levels
        return result

    async def _market_bias_context(self) -> str:
        """查询指数缠论结构方向（大盘环境，注入个股预测 prompt）。

        库内指数数据优先（不触发网络）；无数据返回空串。
        """
        try:
            from pa_mcp.research.market_structure import (
                MarketStructureAnalyzer)
            ms = await MarketStructureAnalyzer(
                self._store_path).analyze(use_network=False)
            if ms["index"]["rows"] <= 0:
                return ""
            j = ms["joint"]
            return (f"上证指数 {ms['index']['last_close']}"
                    f"（{ms['index']['last_date']}）：{j['bias']}——"
                    f"{j['structure']}。大盘方向应影响个股预测的置信与方向权重。")
        except Exception as e:  # noqa: BLE001
            logger.debug("market bias context unavailable: %s", e)
            return ""

    def _sector_context(self, symbol: str) -> str:
        """查询股票所属板块的 RS 强弱（板块轮动 → 个股预测上下文）。

        依赖 stock_basic（sector 映射）与 sector_daily（板块日线），
        任一缺失返回空串（best-effort，不阻塞预测）。
        """
        try:
            from pa_mcp.config import get_settings
            from pa_mcp.data.store import DuckDBStore
            path = self._store_path or get_settings().database.path
            store = DuckDBStore(path)
            store.connect()
            try:
                sb = store.query_df(
                    "SELECT sector FROM stock_basic WHERE symbol = ?", [symbol])
                if sb.empty or not sb.iloc[0]["sector"]:
                    return ""
                sector = str(sb.iloc[0]["sector"])
                # 板块 RS（20 日几何涨幅 + 加速）：查 sector_daily 中名称匹配
                df = store.query_df(
                    "SELECT date, close FROM sector_daily "
                    "WHERE name = ? ORDER BY date", [sector])
                if len(df) < 21:
                    return ""
                close = df["close"].astype(float)
                now = float(close.iloc[-1])
                rs = (now / float(close.iloc[-21]) - 1) * 100
                acc = ((now / float(close.iloc[-6])) ** 0.2 - 1) * 100 \
                    - ((now / float(close.iloc[-21])) ** (1 / 20) - 1) * 100
                state = ("强势（RS 为正且加速）" if rs > 0 and acc > 0
                         else "强势但减速" if rs > 0 else
                         "弱势（RS 为负）" if rs < 0 else "中性")
                return (f"所属板块：{sector}（20 日涨幅 {rs:+.1f}%，"
                        f"加速 {acc:+.3f}）→ {state}。"
                        f"板块强弱应影响个股预测的置信与方向权重。")
            finally:
                store.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("sector context unavailable: %s / %s", symbol, e)
            return ""

    @staticmethod
    def _validate_llm_json(raw: dict) -> list[str]:
        """预测 JSON 语义校验：返回错误列表（空 = 通过）。"""
        errors: list[str] = []
        d = str(raw.get("direction", "")).lower()
        if d not in ("up", "down", "sideways"):
            errors.append("direction 必须是 up/down/sideways")
        dist = raw.get("probability_distribution")
        if not isinstance(dist, dict):
            errors.append("probability_distribution 缺失")
        else:
            try:
                s = sum(float(dist.get(k, 0)) for k in ("up", "down", "sideways"))
                if abs(s - 1.0) > 0.01:
                    errors.append(f"probability_distribution 之和应为 1，实际 {s:.2f}")
            except Exception:
                errors.append("probability_distribution 含非数值")
        try:
            exp = float(raw.get("expected_return_pct", 0))
            rng = raw.get("expected_range_pct") or []
            if len(rng) >= 2 and not (float(rng[0]) <= exp <= float(rng[1])):
                errors.append("expected_return_pct 不在 expected_range_pct 内")
        except Exception:
            errors.append("expected_return_pct/expected_range_pct 格式错误")
        for k in ("cycle_position", "cycle_forecast"):
            v = str(raw.get(k, "")).lower()
            if v and v not in CYCLE_POSITIONS:
                errors.append(f"{k} 不在枚举中: {v}")
        scenarios = raw.get("scenarios") or []
        if scenarios:
            try:
                if abs(sum(float(s.get("probability", 0)) for s in scenarios) - 1.0) > 0.15:
                    errors.append("scenarios 概率之和偏离 1 过多")
            except Exception:
                errors.append("scenarios 格式错误")
        return errors

    def _predict_deterministic(self, symbol: str, predict_date: str,
                               horizon: str, features: dict[str, Any]) -> PredictionResult:
        """无 LLM 时的确定性统计预测（方向打分 + 概率映射）。"""
        h = {"1d": 1, "5d": 5, "20d": 20}.get(horizon, 5)
        ret20 = features["ret20_pct"]
        rsi = features["rsi14"]
        adx = features["adx14"]
        vol_ratio = features["volume_ratio"]
        align = features["ma_alignment"]

        # 方向打分：趋势 + 动量 + 量能
        score = 0.0
        reasons: list[str] = []
        if align == "多头排列":
            score += 2.0; reasons.append("均线多头排列")
        elif align == "空头排列":
            score -= 2.0; reasons.append("均线空头排列")
        score += max(-1.5, min(1.5, ret20 / 15))
        if rsi >= 65:
            score -= 0.8; reasons.append(f"RSI {rsi:.0f} 偏高，短线回调风险")
        elif rsi <= 35:
            score += 0.8; reasons.append(f"RSI {rsi:.0f} 偏低，超跌反弹可能")
        if vol_ratio >= 1.5:
            score += 0.5 if ret20 > 0 else -0.5
            reasons.append(f"量比 {vol_ratio:.1f} 放大")
        if adx >= 30:
            score += 0.5 if ret20 > 0 else -0.5
            reasons.append(f"ADX {adx:.0f} 趋势明确")

        # 概率映射（保守，向 0.5 收缩）
        if score >= 0.8:
            direction = "up"
        elif score <= -0.8:
            direction = "down"
        else:
            direction = "sideways"
        p_direction = 0.5 + min(0.35, abs(score) / 6)
        if direction == "sideways":
            p_side = 0.55
            pu = max(0.1, min(0.3, 0.22 + score / 10))
            pd_ = max(0.1, min(0.3, 0.22 - score / 10))
            ps = max(0.4, 1 - pu - pd_)
        else:
            ps = DEFAULT_SIDEWAYS_PROB
            p_other = (1 - ps) * (0.5 - abs(score) / 12)
            if direction == "up":
                pu, pd_ = p_direction, max(0.05, p_other)
            else:
                pu, pd_ = max(0.05, p_other), p_direction
            ps = max(0.05, 1 - pu - pd_)
        t = pu + pd_ + ps
        pu, pd_, ps = pu / t, pd_ / t, ps / t

        # 期望收益：动量延续 × 周期衰减，向 0 收缩
        expected = ret20 * (h / 20) * 0.25
        expected = max(-15, min(15, expected))
        vol_scale = features["atr_pct"] * (h ** 0.5) * 0.8
        lo, hi = expected - vol_scale * 1.2, expected + vol_scale * 1.2

        cp = features["cycle_position"]
        # 周期预测：spike 后大概率回归区间；区间内看方向
        if cp == "spike":
            cf = "trading_range" if abs(score) < 1.5 else cp
        elif cp in ("trading_range", "extreme_range") and direction != "sideways":
            cf = "normal_channel"
        else:
            cf = cp

        scenarios = [
            {"name": "顺势延续", "probability": round(pu if direction == "up" else 0.4, 2),
             "target_pct": round(expected * 1.3, 1),
             "description": f"维持{('上涨' if direction != 'down' else '下跌')}节奏，量能配合"},
            {"name": "横盘整理", "probability": round(ps, 2),
             "target_pct": round(expected * 0.2, 1),
             "description": "多空平衡，在支撑压力间反复"},
            {"name": "反向波动", "probability": round(pd_ if direction == "up" else 0.35, 2),
             "target_pct": round(-expected * 1.2, 1),
             "description": "突破失败或利空冲击，回踩支撑"},
        ]
        conf = 0.45 + min(0.25, abs(score) / 8 + features["adx14"] / 200)

        return PredictionResult(
            symbol=symbol, predict_date=predict_date, horizon=horizon,
            direction=direction, probability=round(p_direction, 2),
            prob_up=round(pu, 2), prob_down=round(pd_, 2), prob_sideways=round(ps, 2),
            expected_return_pct=round(expected, 2),
            expected_range_low=round(lo, 2), expected_range_high=round(hi, 2),
            cycle_position=cp, cycle_forecast=cf,
            support_levels=[features["support_20d"], round(features["support_20d"] * 0.97, 2)],
            resistance_levels=[features["resistance_20d"], round(features["resistance_20d"] * 1.03, 2)],
            scenarios=scenarios, confidence=round(conf, 2),
            key_reasons=reasons or [f"{h}日动量 {ret20:+.1f}%", f"ADX {adx:.0f}"],
            key_risks=["无 LLM 配置，此为确定性统计预测（非 AI 解读）",
                       "大盘系统性风险不可通过个股特征预测"],
            model="deterministic", mode="deterministic",
        )

    # ---- 落盘 ----
    def save_prediction(self, result: PredictionResult) -> int:
        """写入 prediction_log 表，返回记录 id。

        注意：id 为 NOT NULL 主键且 fill_defaults 会用 None 填充缺失列，
        因此显式计算 id（COALESCE(MAX)+1），避免插入 NULL 主键。
        """
        store = self._store()
        try:
            max_id = store.query_df("SELECT COALESCE(MAX(id), 0) AS m FROM prediction_log", [])
            new_id = int(max_id.iloc[0]["m"]) + 1 if not max_id.empty else 1
            row = pd.DataFrame([{
                "id": new_id,
                "symbol": result.symbol,
                "predict_date": result.predict_date,
                "horizon": result.horizon,
                "direction": result.direction,
                "probability": result.probability,
                "prob_up": result.prob_up,
                "prob_down": result.prob_down,
                "prob_sideways": result.prob_sideways,
                "expected_return_pct": result.expected_return_pct,
                "expected_range_low": result.expected_range_low,
                "expected_range_high": result.expected_range_high,
                "cycle_position": result.cycle_position,
                "cycle_forecast": result.cycle_forecast,
                "support_levels": json.dumps(result.support_levels, ensure_ascii=False),
                "resistance_levels": json.dumps(result.resistance_levels, ensure_ascii=False),
                "scenarios": json.dumps(result.scenarios, ensure_ascii=False),
                "confidence": result.confidence,
                "key_reasons": json.dumps(result.key_reasons, ensure_ascii=False),
                "key_risks": json.dumps(result.key_risks, ensure_ascii=False),
                "model": result.model,
                "prompt_version": result.prompt_version,
                "mode": result.mode,
                "status": "pending",
            }])
            store.insert_df("prediction_log", row)
            return new_id
        finally:
            store.close()

    # ---- 评估（回填真实收益 + 命中判定） ----
    async def evaluate_predictions(self, kline_provider=None,
                                   today: Optional[str] = None) -> dict[str, Any]:
        """回填已到期预测的真实收益并计算命中率/Brier 分数。

        kline_provider: 可选回调 kline_provider(symbol) -> DataFrame（可为 async），
        默认从 kline_daily 表读取。到期标准：
        status='pending' 且 predict_date + horizon <= 最新可用行情日。
        """
        today = today or date.today().isoformat()
        store = self._store()
        try:
            pending = store.query_df(
                "SELECT * FROM prediction_log WHERE status = 'pending' ORDER BY id",
                [])
            if pending.empty:
                return self._summary(store)

            # 拉取所需股票最新行情（按到期预测分组，避免重复拉取）
            needed = {
                row["symbol"]: row["horizon"]
                for _, row in pending.iterrows()
            }
            klines: dict[str, pd.DataFrame] = {}
            latest_dates: dict[str, str] = {}
            for sym, hor in needed.items():
                df = await self._fetch_kline(sym, store, kline_provider)
                if df is None or df.empty:
                    continue
                klines[sym] = df.sort_values("date").reset_index(drop=True)
                latest_dates[sym] = str(df["date"].iloc[-1])[:10]

            evaluated = 0
            for _, row in pending.iterrows():
                sym, hor = row["symbol"], row["horizon"]
                df = klines.get(sym)
                if df is None:
                    continue
                h_days = {"1d": 1, "5d": 5, "20d": 20}.get(hor, 5)
                predict_dt = str(row["predict_date"])[:10]
                # 找预测日之后的行情
                after = df[df["date"].astype(str).str[:10] >= predict_dt]
                if len(after) < 2:
                    continue
                base_close = float(after["close"].iloc[0])
                # 取 horizon 个交易日后（不超过最新）
                target = after.iloc[min(h_days, len(after) - 1)]
                actual = (float(target["close"]) / base_close - 1) * 100
                status = self._judge(row["direction"], actual)
                store.execute(
                    "UPDATE prediction_log SET status = ?, actual_return_pct = ?, "
                    "evaluated_date = ? WHERE id = ?",
                    [status, round(actual, 3), today, int(row["id"])])
                evaluated += 1

            return self._summary(store)
        finally:
            store.close()

    @staticmethod
    def _judge(direction: str, actual: float) -> str:
        """命中判定。"""
        d = (direction or "sideways").lower()
        if d == "sideways":
            return "hit" if abs(actual) <= SIDEWAYS_THRESHOLD_PCT else "miss"
        if abs(actual) <= AMBIGUOUS_THRESHOLD_PCT:
            return "ambiguous"
        if d == "up":
            return "hit" if actual > 0 else "miss"
        if d == "down":
            return "hit" if actual < 0 else "miss"
        return "ambiguous"

    async def _fetch_kline(self, symbol: str, store,
                           kline_provider=None) -> Optional[pd.DataFrame]:
        """取个股日线（外部回调优先（支持 async），否则查表）。"""
        if kline_provider is not None:
            try:
                df = kline_provider(symbol)
                if inspect.isawaitable(df):
                    df = await df
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

    @staticmethod
    def _summary(store) -> dict[str, Any]:
        """评估汇总（全部预测 + 已评估部分，含 Brier 概率校准分数）。"""
        total = store.query_df("SELECT COUNT(*) AS c FROM prediction_log", [])
        ev_df = store.query_df(
            "SELECT direction, status, actual_return_pct, probability, "
            "prob_up, prob_down, prob_sideways, expected_return_pct, "
            "predict_date, mode, horizon "
            "FROM prediction_log WHERE status != 'pending'", [])
        n = int(ev_df.shape[0])
        out: dict[str, Any] = {
            "total_predictions": int(total.iloc[0]["c"]) if not total.empty else 0,
            "evaluated": n,
            "hit_rate": 0.0,
            "brier_score": None,
            "baseline_brier": None,      # 气候学基准（按历史频率预测）
            "brier_skill_score": None,   # 1 - brier/baseline（>0 = 优于基准）
            "return_correlation": None,  # 预测期望收益 vs 实际收益 Pearson 相关
            "ic": None,                  # Spearman IC：期望收益排序 vs 实际收益排序
            "icir": None,                # ICIR：滚动窗口 IC 均值/标准差（Qlib 标准）
            "calibration_bins": [],      # 概率校准分桶（预测概率 vs 实际命中率）
            "by_mode": {},               # AI(llm) vs 统计(deterministic) 对比
            "by_direction": {},
            "avg_actual_return_pct": None,
            "direction_agreement_pct": None,
        }
        if n == 0 or ev_df.empty:
            return out
        hits = ev_df[ev_df["status"] == "hit"]
        out["hit_rate"] = round(len(hits) / n, 3)
        out["avg_actual_return_pct"] = round(float(ev_df["actual_return_pct"].mean()), 3)
        # 方向一致率：预测方向与收益符号一致（不含 sideways）
        dir_df = ev_df[ev_df["direction"].isin(["up", "down"])]
        if not dir_df.empty:
            agree = dir_df.apply(
                lambda r: (r["direction"] == "up" and r["actual_return_pct"] > 0)
                or (r["direction"] == "down" and r["actual_return_pct"] < 0), axis=1)
            out["direction_agreement_pct"] = round(float(agree.mean()), 3)

        # Brier 分数（三分类概率校准）：
        # 实际类别：up = 涨超阈值、down = 跌超阈值、sideways = 其余
        # Brier_i = Σ_c (p_c - y_c)²，范围 [0,2]，越小越准
        prob_cols = ("prob_up", "prob_down", "prob_sideways")
        if ev_df[list(prob_cols)].notna().all().all():
            y_up = (ev_df["actual_return_pct"] > SIDEWAYS_THRESHOLD_PCT).astype(float)
            y_down = (ev_df["actual_return_pct"] < -SIDEWAYS_THRESHOLD_PCT).astype(float)
            y_side = ((ev_df["actual_return_pct"].abs() <= SIDEWAYS_THRESHOLD_PCT)
                      ).astype(float)
            brier = float(((ev_df["prob_up"] - y_up) ** 2
                           + (ev_df["prob_down"] - y_down) ** 2
                           + (ev_df["prob_sideways"] - y_side) ** 2).mean())
            out["brier_score"] = round(brier, 4)
            # 气候学基准：用样本频率作为恒定预测
            f_up, f_down = float(y_up.mean()), float(y_down.mean())
            f_side = float(y_side.mean())
            base = float(((f_up - y_up) ** 2 + (f_down - y_down) ** 2
                          + (f_side - y_side) ** 2).mean())
            out["baseline_brier"] = round(base, 4)
            if base > 0:
                out["brier_skill_score"] = round(1 - brier / base, 4)

        # 预测期望收益 vs 实际收益 相关性（Pearson）
        exp = pd.to_numeric(ev_df["expected_return_pct"], errors="coerce")
        act = pd.to_numeric(ev_df["actual_return_pct"], errors="coerce")
        valid = exp.notna() & act.notna()
        if valid.sum() >= 3:
            corr = exp[valid].corr(act[valid], method="pearson")
            if pd.notna(corr):
                out["return_correlation"] = round(float(corr), 4)

        # ---- IC / ICIR（Qlib 标准：预测排序 vs 实际收益排序的秩相关） ----
        # Spearman 用 rank+Pearson 实现（避免 scipy 依赖）
        def _spearman(a: pd.Series, b: pd.Series) -> float:
            va = a.notna() & b.notna()
            if va.sum() < 3:
                return float("nan")
            ra = a[va].rank()
            rb = b[va].rank()
            return float(ra.corr(rb)) if ra.corr(rb) is not None else float("nan")

        if valid.sum() >= 8:
            ic = _spearman(exp, act)
            if not pd.isna(ic):
                out["ic"] = round(ic, 4)
            # 滚动窗口 IC：按预测日分组（每窗 ≥3 条才计入）
            work = ev_df[valid].copy()
            work["wk"] = pd.to_datetime(work["predict_date"]).dt.to_period("W")
            win_ics: list[float] = []
            for _, g in work.groupby("wk"):
                e = pd.to_numeric(g["expected_return_pct"], errors="coerce")
                a = pd.to_numeric(g["actual_return_pct"], errors="coerce")
                ic_w = _spearman(e, a)
                if not pd.isna(ic_w):
                    win_ics.append(ic_w)
            if len(win_ics) >= 2:
                mean_ic = sum(win_ics) / len(win_ics)
                std_ic = (sum((x - mean_ic) ** 2 for x in win_ics)
                          / len(win_ics)) ** 0.5
                if std_ic > 0:
                    out["icir"] = round(mean_ic / std_ic, 3)

        # ---- 概率校准分桶（预测概率 vs 实际命中率，检验过度自信） ----
        # 只评估有方向的预测（up/down），用主方向概率分桶
        dirn = ev_df[ev_df["direction"].isin(["up", "down"])].copy()
        if not dirn.empty:
            dirn["prob"] = pd.to_numeric(dirn["probability"], errors="coerce")
            dirn = dirn[dirn["prob"].notna()]
            if len(dirn) >= 4:
                buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
                for lo, hi in buckets:
                    sub = dirn[(dirn["prob"] >= lo) & (dirn["prob"] < hi)]
                    if sub.empty:
                        continue
                    hit = float((sub["status"] == "hit").mean())
                    out["calibration_bins"].append({
                        "prob_range": f"{lo:.0%}-{min(hi, 1.0):.0%}",
                        "n": int(sub.shape[0]),
                        "actual_hit_rate": round(hit, 3),
                        "mid_prob": round((lo + min(hi, 1.0)) / 2, 3),
                        "overconfident": hit < (lo + min(hi, 1.0)) / 2 - 0.1,
                    })

        # ---- 模式对比：AI(llm) vs 统计(deterministic) ----
        by_mode: dict[str, dict] = {}
        for m in ("llm", "deterministic"):
            sub = ev_df[ev_df["mode"] == m]
            if sub.empty:
                continue
            entry: dict[str, Any] = {
                "count": int(sub.shape[0]),
                "hit_rate": round(float((sub["status"] == "hit").mean()), 3),
                "avg_return_pct": round(float(sub["actual_return_pct"].mean()), 3),
            }
            sub_prob = sub[["prob_up", "prob_down", "prob_sideways"]]
            if sub_prob.notna().all().all():
                y_up = (sub["actual_return_pct"] > SIDEWAYS_THRESHOLD_PCT).astype(float)
                y_down = (sub["actual_return_pct"] < -SIDEWAYS_THRESHOLD_PCT).astype(float)
                y_side = (sub["actual_return_pct"].abs() <= SIDEWAYS_THRESHOLD_PCT).astype(float)
                b = float(((sub["prob_up"] - y_up) ** 2
                           + (sub["prob_down"] - y_down) ** 2
                           + (sub["prob_sideways"] - y_side) ** 2).mean())
                entry["brier_score"] = round(b, 4)
            by_mode[m] = entry
        out["by_mode"] = by_mode

        # ---- 按预测周期分组（1d/5d/20d 命中率对比） ----
        by_horizon: dict[str, dict] = {}
        for h in ("1d", "5d", "20d"):
            sub = ev_df[ev_df["horizon"] == h]
            if sub.empty:
                continue
            entry = {
                "count": int(sub.shape[0]),
                "hit_rate": round(float((sub["status"] == "hit").mean()), 3),
                "avg_return_pct": round(float(sub["actual_return_pct"].mean()), 3),
            }
            sub_prob = sub[["prob_up", "prob_down", "prob_sideways"]]
            if sub_prob.notna().all().all():
                y_up = (sub["actual_return_pct"] > SIDEWAYS_THRESHOLD_PCT).astype(float)
                y_down = (sub["actual_return_pct"] < -SIDEWAYS_THRESHOLD_PCT).astype(float)
                y_side = (sub["actual_return_pct"].abs() <= SIDEWAYS_THRESHOLD_PCT).astype(float)
                b = float(((sub["prob_up"] - y_up) ** 2
                           + (sub["prob_down"] - y_down) ** 2
                           + (sub["prob_sideways"] - y_side) ** 2).mean())
                entry["brier_score"] = round(b, 4)
            by_horizon[h] = entry
        out["by_horizon"] = by_horizon

        by_dir: dict[str, dict] = {}
        for d in ("up", "down", "sideways"):
            sub = ev_df[ev_df["direction"] == d]
            if not sub.empty:
                by_dir[d] = {
                    "count": int(sub.shape[0]),
                    "hit_rate": round(len(sub[sub["status"] == "hit"]) / sub.shape[0], 3),
                    "avg_return_pct": round(float(sub["actual_return_pct"].mean()), 3),
                }
        out["by_direction"] = by_dir
        return out

    # ---- 预测驱动的仓位建议（借鉴 ai-hedge-fund Risk Manager 思路） ----
    async def position_sizing(self, symbol: str,
                              account_value: float = 100000.0,
                              horizon: str = "5d",
                              base_position_pct: Optional[float] = None,
                              kline_df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
        """预测 → 仓位翻译器：预测概率 × 历史命中率校准 → 仓位建议。

        逻辑（可追溯）：
          1. 预测（复用 predict()，含板块上下文）
          2. 校准：该股票历史同方向命中率（prediction_log 已评估记录），
             无历史时用全局命中率；再按概率分桶校准（过度自信桶降权）
          3. 仓位 = base（分析建议或方向默认）× 校准系数，受 20% 硬上限
          4. 输出完整推导链（prob/校准/系数/最终建议）

        Args:
            symbol: 股票代码
            account_value: 账户资金（用于展示金额）
            horizon: 预测周期
            base_position_pct: 基础仓位（分析师建议）；None = 按方向回退
            kline_df: 可选 K 线（缺省自动拉取）
        """
        if kline_df is None:
            from pa_mcp.config import get_settings
            from pa_mcp.data.store import DuckDBStore
            store = DuckDBStore(get_settings().database.path)
            store.connect()
            try:
                kline_df = store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 160", [symbol])
            finally:
                store.close()

        result = await self.predict(symbol, kline_df, horizon=horizon)
        p = result.to_dict()
        direction = p["direction"]
        prob = p["probability"]

        # 历史校准：同方向已评估预测的命中率
        store = self._store()
        try:
            hist = store.query_df(
                "SELECT direction, status, probability FROM prediction_log "
                "WHERE symbol = ? AND direction = ? AND status != 'pending'",
                [symbol, direction])
            if hist.empty:
                hist = store.query_df(
                    "SELECT direction, status, probability FROM prediction_log "
                    "WHERE direction = ? AND status != 'pending'",
                    [direction])
            hist_hit = float((hist["status"] == "hit").mean()) \
                if not hist.empty else 0.5
            n_hist = int(hist.shape[0])
        finally:
            store.close()

        # 概率分桶校准：桶内实际命中率（若低于桶中值 → 过度自信 → 降权）
        bucket_hit = None
        for lo, hi in ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)):
            if lo <= prob < hi:
                bucket_hit = self._bucket_hit_rate(lo, hi)
                break

        # 校准系数：历史命中率/0.5（相对随机的比值）再乘桶校准
        hist_factor = hist_hit / 0.5 if hist_hit > 0 else 0.8
        bucket_factor = 1.0
        if bucket_hit is not None:
            mid = min(lo + (hi - lo) / 2, 1.0)
            bucket_factor = max(0.5, min(1.2, bucket_hit / mid * 1.2))

        # 基础仓位：base 或按方向回退（与决策树一致）
        if base_position_pct is None or base_position_pct <= 0:
            base_position_pct = (10.0 if direction == "up"
                                 else 0.0 if direction == "down" else 3.0)
        suggested = base_position_pct * hist_factor * bucket_factor

        # 多周期共振校准：强共振上调（趋势确认）/ 分歧收缩
        resonance_factor = 1.0
        resonance_note = ""
        try:
            from pa_mcp.research.resonance import ResonanceAnalyzer
            import asyncio as _asyncio
            res = _asyncio.run(ResonanceAnalyzer().analyze(
                symbol, kline_df=kline_df))
            if "error" not in res:
                strength = res["strength"]
                if strength >= 0.7 and res["signal"] == direction:
                    resonance_factor = 1.3
                    resonance_note = f"（强共振 {res['resonance']} 上调）"
                elif strength < 0.4:
                    resonance_factor = 0.7
                    resonance_note = "（共振分歧收缩）"
                suggested *= resonance_factor
        except Exception:
            pass

        # 综合信号校准（优先级高于共振）：强融合上调 / 弱融合收缩
        consensus_factor = 1.0
        consensus_note = ""
        try:
            from pa_mcp.research.consensus import ConsensusAnalyzer
            import asyncio as _asyncio
            con = _asyncio.run(ConsensusAnalyzer().analyze(
                symbol, kline_df=kline_df))
            if "error" not in con:
                strength = con["strength"]
                if strength >= 0.6 and con["signal"] == direction:
                    consensus_factor = 1.4
                    consensus_note = f"（综合信号{con['level']}强度上调）"
                elif strength < 0.4:
                    consensus_factor = 0.6
                    consensus_note = "（综合信号分歧收缩）"
                suggested *= consensus_factor
        except Exception:
            pass

        suggested = max(0.0, min(20.0, suggested))  # RiskGuard 硬上限
        suggested = round(suggested, 1)

        amount = account_value * suggested / 100
        return {
            "symbol": symbol,
            "horizon": horizon,
            "direction": direction,
            "probability": prob,
            "base_position_pct": base_position_pct,
            "hist_hit_rate": round(hist_hit, 3),
            "hist_samples": n_hist,
            "bucket_hit_rate": bucket_hit,
            "hist_factor": round(hist_factor, 3),
            "bucket_factor": round(bucket_factor, 3),
            "resonance_factor": round(resonance_factor, 3),
            "resonance_note": resonance_note,
            "consensus_factor": round(consensus_factor, 3),
            "consensus_note": consensus_note,
            "suggested_position_pct": suggested,
            "suggested_amount": round(amount, 2),
            "explanation": (
                f"预测{direction}({prob:.0%}) × 历史命中率{hist_hit:.0%}"
                f"（{n_hist}样本）{'× 概率桶校准' if bucket_hit is not None else ''}"
                f"× 共振校准 {resonance_factor:.1f}{resonance_note}"
                f"× 综合信号校准 {consensus_factor:.1f}{consensus_note}"
                f" → 建议仓位 ≤{suggested}%（RiskGuard 20% 上限内）"),
            "disclaimer": "研究参考，非投资建议。仓位须结合自身风险承受能力。",
        }

    def _bucket_hit_rate(self, lo: float, hi: float) -> Optional[float]:
        """全局概率桶实际命中率（evaluate 同款分桶）。"""
        store = self._store()
        try:
            df = store.query_df(
                "SELECT direction, status, probability FROM prediction_log "
                "WHERE status != 'pending' AND direction IN ('up','down')", [])
            if df.empty:
                return None
            df["prob"] = pd.to_numeric(df["probability"], errors="coerce")
            sub = df[(df["prob"] >= lo) & (df["prob"] < hi)]
            if sub.empty:
                return None
            return round(float((sub["status"] == "hit").mean()), 3)
        finally:
            store.close()

    # ---- 历史查询 ----
    def prediction_history(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        """某股票最近预测记录（含评估结果）。"""
        store = self._store()
        try:
            df = store.query_df(
                "SELECT id, symbol, predict_date, horizon, direction, probability, "
                "expected_return_pct, cycle_position, cycle_forecast, confidence, "
                "mode, model, status, actual_return_pct, evaluated_date "
                "FROM prediction_log WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                [symbol, limit])
            rows: list[dict[str, Any]] = []
            for _, r in df.iterrows():
                rows.append({
                    "id": int(r["id"]),
                    "symbol": r["symbol"],
                    "predict_date": str(r["predict_date"])[:10],
                    "horizon": r["horizon"],
                    "direction": r["direction"],
                    "probability": float(r["probability"]) if r["probability"] is not None else None,
                    "expected_return_pct": float(r["expected_return_pct"]) if r["expected_return_pct"] is not None else None,
                    "cycle_position": r["cycle_position"],
                    "cycle_forecast": r["cycle_forecast"],
                    "confidence": float(r["confidence"]) if r["confidence"] is not None else None,
                    "mode": r["mode"],
                    "model": r["model"],
                    "status": r["status"],
                    "actual_return_pct": float(r["actual_return_pct"]) if r["actual_return_pct"] is not None else None,
                    "evaluated_date": str(r["evaluated_date"])[:10] if r["evaluated_date"] is not None else None,
                })
            return rows
        finally:
            store.close()


_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """单例获取预测服务。"""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service
