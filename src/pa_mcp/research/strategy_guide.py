# [AI:BEGIN]
# PA_MCP - Research: 策略速查（新手该用哪个）
#
# 整合 11 个策略 + 8 大牛方法为新手可用的分层选择体系：
#   1. 按市场状态推荐（高潮/发酵/启动/低迷/冰点 → 适配策略）
#   2. 每策略标注：新手难度（⭐~⭐⭐⭐）/一句话说明/风险
#   3. 新手默认推荐（最稳健起步）
#   4. 方法归属（CANSLIM/缠论/海龟/利弗莫尔等）
#
# 数据源：策略注册表 + tips 中文说明 + 市场状态路由。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 策略 → 新手信息（难度 1-3 星，越少越易上手）
STRATEGY_GUIDE: dict[str, dict] = {
    "bollinger_mean_reversion": {
        "difficulty": 1,
        "method": "均值回归",
        "one_liner": "跌到布林下轨买入，回到中轨卖出——最简单的低吸策略",
        "risk": "下跌趋势中会接飞刀；需配合大盘状态",
        "default_for": ["starting", "dull"],
    },
    "ma_golden_cross": {
        "difficulty": 1,
        "method": "趋势跟随",
        "one_liner": "短期均线上穿长期均线买入——趋势起步的信号",
        "risk": "震荡市频繁假金叉",
        "default_for": ["fermenting", "starting"],
    },
    "oversold_bounce": {
        "difficulty": 1,
        "method": "超跌反弹",
        "one_liner": "跌过头了赌反弹——适合市场恐慌时低吸",
        "risk": "超跌可以更超跌；轻仓试错",
        "default_for": ["starting", "dull"],
    },
    "platform_breakout": {
        "difficulty": 2,
        "method": "平台突破",
        "one_liner": "横盘很久后放量突破买入——主力吸筹完成信号",
        "risk": "假突破多；需放量确认",
        "default_for": ["fermenting", "climax"],
    },
    "volume_price_momentum": {
        "difficulty": 2,
        "method": "量价动量",
        "one_liner": "放量上涨时跟随——强者恒强",
        "risk": "追高风险；高潮期慎用",
        "default_for": ["fermenting", "climax"],
    },
    "turtle": {
        "difficulty": 2,
        "method": "海龟交易",
        "one_liner": "突破 20 日新高买入，跌破离场——经典趋势跟踪",
        "risk": "震荡市连续假突破磨损",
        "default_for": ["fermenting"],
    },
    "livermore_pivot": {
        "difficulty": 3,
        "method": "利弗莫尔关键点",
        "one_liner": "只在趋势确认的关键点进场——耐心等待最小阻力线",
        "risk": "信号少；需严格纪律",
        "default_for": ["fermenting"],
    },
    "first_board_breakout": {
        "difficulty": 3,
        "method": "首板",
        "one_liner": "首次涨停买入——游资风格，短线",
        "risk": "炸板风险高；仅适合进取型",
        "default_for": ["fermenting", "climax"],
    },
    "dragon_second_wave": {
        "difficulty": 3,
        "method": "龙虎榜",
        "one_liner": "龙虎榜后第二波——机构/游资席位验证",
        "risk": "依赖龙虎榜数据；情绪退潮失效",
        "default_for": ["fermenting", "climax"],
    },
    "range_grid": {
        "difficulty": 2,
        "method": "网格交易",
        "one_liner": "区间内高抛低吸——震荡市自动赚钱",
        "risk": "单边行情网格会卖飞或深套",
        "default_for": ["dull"],
    },
    "roe_pb_value": {
        "difficulty": 2,
        "method": "价值投资",
        "one_liner": "高 ROE + 低 PB 的便宜好公司——长期持有",
        "risk": "价值陷阱；需要耐心",
        "default_for": ["dull", "frozen"],
    },
}

# 新手默认推荐（无脑起步的选择）
BEGINNER_DEFAULT = "bollinger_mean_reversion"


def strategy_guide(market_state: Optional[str] = None) -> dict[str, Any]:
    """策略速查：按市场状态推荐 + 新手难度。

    Args:
        market_state: 市场状态（climax/fermenting/starting/dull/frozen），
            缺省自动检测
    """
    from pa_mcp.engine.strategies.tips import STRATEGY_TIPS

    if market_state is None:
        market_state = _detect_market_state()

    recommended = [
        name for name, info in STRATEGY_GUIDE.items()
        if market_state in info.get("default_for", [])]
    # 未注册的策略排除
    from pa_mcp.engine.strategies.base import StrategyRegistry
    registry = StrategyRegistry()
    registry.auto_discover()
    registered = set(registry.list_all())
    recommended = [s for s in recommended if s in registered]

    all_info = []
    for name, info in STRATEGY_GUIDE.items():
        stars = "⭐" * info["difficulty"]
        all_info.append({
            "strategy": name,
            "difficulty": info["difficulty"],
            "difficulty_stars": stars,
            "method": info["method"],
            "one_liner": info["one_liner"],
            "risk": info["risk"],
            "tip": (STRATEGY_TIPS.get(name) or "").split("\n")[0],
        })

    state_zh = {"climax": "高潮期", "fermenting": "发酵期",
                "starting": "启动期", "dull": "低迷期",
                "frozen": "冰点期", "unknown": "未知"}

    return {
        "market_state": market_state,
        "market_state_zh": state_zh.get(market_state, market_state),
        "beginner_default": BEGINNER_DEFAULT,
        "beginner_default_reason": (
            "布林均值回归：规则最简单、信号最多、无需高频盯盘"
            "——新手最稳的起点"),
        "recommended": recommended,
        "all_strategies": all_info,
        "report": _render(market_state, recommended, all_info),
    }


def _detect_market_state() -> str:
    try:
        from pa_mcp.engine.market_state import MarketStateDetector, MarketIndicators
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        store = DuckDBStore(get_settings().database.path)
        store.connect()
        try:
            latest = store.get_latest_date("kline_daily")
            if not latest:
                return "unknown"
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
            ind = MarketIndicators(
                limit_up_count=int(row["limit_up"]),
                limit_down_count=int(row["limit_down"]),
                up_count=int(row["up_count"]),
                down_count=int(row["down_count"]),
                turnover_billion=float(row["turnover"]),
            )
            return MarketStateDetector().detect(ind).value
        finally:
            store.close()
    except Exception:
        return "unknown"


def _render(market_state: str, recommended: list[str],
            all_info: list[dict]) -> str:
    from pa_mcp.engine.strategies.tips import STRATEGY_TIPS
    state_zh = {"climax": "高潮期", "fermenting": "发酵期",
                "starting": "启动期", "dull": "低迷期",
                "frozen": "冰点期", "unknown": "未知"}
    lines = [
        f"## 📖 策略速查（当前市场：{state_zh.get(market_state, market_state)}）",
        "",
        f"**新手默认**：bollinger_mean_reversion ⭐（规则最简单、信号最多）",
        "",
        f"**当前市场推荐**：{'、'.join(recommended) if recommended else '（无明确推荐，观望为主）'}",
        "",
        "### 全部策略（难度：⭐易 → ⭐⭐⭐难）",
        "| 策略 | 难度 | 方法 | 一句话 | 风险 |",
        "|---|---|---|---|---|",
    ]
    for info in sorted(all_info, key=lambda x: x["difficulty"]):
        tip_first = (info["tip"] or info["one_liner"]).replace("**", "")[:30]
        lines.append(f"| {info['strategy']} | {info['difficulty_stars']} | "
                     f"{info['method']} | {tip_first} | {info['risk'][:30]} |")
    lines.append("\n*方法归属：CANSLIM/欧奈尔 → roe_pb_value；缠论 → 结构分析；"
                 "海龟 → turtle；利弗莫尔 → livermore_pivot；"
                 "游资 → first_board/dragon_second_wave。研究参考，非投资建议。*")
    return "\n".join(lines)


_service: Optional[dict] = None


def get_strategy_guide(market_state: Optional[str] = None) -> dict[str, Any]:
    """获取策略速查。"""
    return strategy_guide(market_state)
