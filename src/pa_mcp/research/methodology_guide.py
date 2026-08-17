# [AI:BEGIN]
# PA_MCP - Research: 新手决策地图（方法论整合）
#
# 把四类资产（策略 strategy / 大牛方法 method / 分析方法 analysis / LLM 能力 llm）
# 编入统一注册表 METHOD_CATALOG，按新手四步走导航：
#   ① 看环境 → ② 选方法 → ③ 做验证 → ④ 增强解读
# UI 面板 / MCP 工具 / 文档共用同一份数据源，保证口径一致。
#
# 设计要点：
# - strategy 类条目通过 ref_strategy 外键运行时从 STRATEGY_GUIDE 合并
#   （单一数据源：难度/说明/风险/适用市场只维护在 strategy_guide.py 一处）。
# - _routing_compare 对齐两套策略路由表（STRATEGY_GUIDE.default_for vs
#   orchestrator.MARKET_STATE_STRATEGY_ROUTING.strategies）——
#   两套路由第一次对齐展示，输出差异标注。
# - llm_required=True 的条目强制带 llm_cost_hint（测试校验），新手可见成本。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 市场状态中文映射（与 strategy_guide / orchestrator 保持一致）
STATE_ZH: dict[str, str] = {
    "climax": "高潮期", "fermenting": "发酵期", "starting": "启动期",
    "dull": "低迷期", "frozen": "冰点期", "unknown": "未知",
}

# 新手四步定义（step 序号、标题、目标、门槛说明）
METHOD_STEPS: list[dict[str, str]] = [
    {"step": 1, "title": "① 看环境",
     "goal": "今天是牛市还是熊市？情绪、结构、轮动在哪一阶段？",
     "gate": "环境判断是一切决策的前提——高潮期追高、冰点期抄底都是新手最容易踩的坑。"},
    {"step": 2, "title": "② 选方法",
     "goal": "当前市场状态适合哪些策略和大牛方法？",
     "gate": "不同状态适配不同方法（见下方推荐）；难度 ⭐ 越多越难，新手从 ⭐ 开始。"},
    {"step": 3, "title": "③ 做验证",
     "goal": "选中的信号到底靠不靠谱？",
     "gate": "事件研究 / Walk-Forward / 过拟合诊断是纯确定性检验——先验证再信，别被回测图骗了。"},
    {"step": 4, "title": "④ 增强解读",
     "goal": "让 AI 深度解读、预测未来、沉淀经验。",
     "gate": "LLM 只解读数据不编造；未配置 LLM 时此步可跳过，前三步不受影响。"},
]

# 四类资产编目注册表（31 条 = 11 strategy + 8 method + 6 analysis + 6 llm）
# id 前缀 = 类别，防重名且天然分组。
# strategy 类条目只写入口联动字段，难度/说明/风险/适用市场运行时从
# STRATEGY_GUIDE 合并（见 _merge_strategy_entry）。
# ui_entry.tab 必须 ∈ 已知 10 个 Tab 名（测试校验）；None = 仅 MCP/自动触发。
METHOD_CATALOG: dict[str, dict] = {
    # ============ strategy（11）：信号来源 ============
    "strategy.bollinger_mean_reversion": {
        "category": "strategy", "name_zh": "布林均值回归", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "bollinger_mean_reversion",
    },
    "strategy.ma_golden_cross": {
        "category": "strategy", "name_zh": "均线金叉", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "ma_golden_cross",
    },
    "strategy.oversold_bounce": {
        "category": "strategy", "name_zh": "超跌反弹", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "oversold_bounce",
    },
    "strategy.platform_breakout": {
        "category": "strategy", "name_zh": "平台突破", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "platform_breakout",
    },
    "strategy.volume_price_momentum": {
        "category": "strategy", "name_zh": "量价动量", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "volume_price_momentum",
    },
    "strategy.turtle": {
        "category": "strategy", "name_zh": "海龟趋势跟踪", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "turtle", "related": ["method.turtle"],
    },
    "strategy.livermore_pivot": {
        "category": "strategy", "name_zh": "利弗莫尔关键点", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "livermore_pivot", "related": ["method.livermore"],
    },
    "strategy.first_board_breakout": {
        "category": "strategy", "name_zh": "首板突破", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "first_board_breakout",
    },
    "strategy.dragon_second_wave": {
        "category": "strategy", "name_zh": "龙虎榜第二波", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "dragon_second_wave",
    },
    "strategy.range_grid": {
        "category": "strategy", "name_zh": "网格交易", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "range_grid",
    },
    "strategy.roe_pb_value": {
        "category": "strategy", "name_zh": "价值投资（ROE×PB）", "step": 2,
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测"},
        "mcp_tool": ["scan_market"], "llm_required": False,
        "ref_strategy": "roe_pb_value",
    },
    # ============ method（8）：大牛方法 ============
    "method.canslim": {
        "category": "method", "name_zh": "CANSLIM（欧奈尔成长股）", "step": 2,
        "difficulty": 2,
        "one_liner": "基本面+技术面七要素筛选成长股（C盈利/A增长/N新高/S放量/L领军/M市场方向）",
        "risk": "依赖财务数据完整性；成长股波动大",
        "market_states": ["fermenting", "starting"],
        "ui_entry": {"tab": "📡 市场扫描", "button": "🧬 CANSLIM 成长股扫描"},
        "mcp_tool": ["scan_canslim"], "llm_required": False,
    },
    "method.chan": {
        "category": "method", "name_zh": "缠论结构（缠中说禅）", "step": 2,
        "difficulty": 3,
        "one_liner": "K线合并→分型→笔→中枢→背驰，判断结构位置与动能衰竭",
        "risk": "简化实现（非完整缠论）；结构判断主观性强",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "📊 数据看板", "button": "🌀 缠论结构分析"},
        "mcp_tool": ["chan_analysis", "chan_beichi_event_study"],
        "llm_required": False,
    },
    "method.turtle": {
        "category": "method", "name_zh": "海龟交易（Richard Dennis）", "step": 2,
        "difficulty": 2,
        "one_liner": "唐奇安 20 日突破入场 + ATR 波动率目标仓位（1 单位=账户×1%÷ATR）",
        "risk": "震荡市连续假突破磨损；A 股仅做多",
        "market_states": ["fermenting"],
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "🐢 海龟仓位计算"},
        "mcp_tool": ["turtle_position_size"], "llm_required": False,
        "related": ["strategy.turtle"],
    },
    "method.livermore": {
        "category": "method", "name_zh": "利弗莫尔关键点（《股票大作手回忆录》）", "step": 2,
        "difficulty": 3,
        "one_liner": "枢轴关键点突破 + 站稳 MA60（只做上升趋势）+ 放量确认（无量=假突破）",
        "risk": "趋势确认严格 → 信号偏少；需严格纪律",
        "market_states": ["fermenting"],
        "ui_entry": {"tab": "🛠️ 策略回测", "button": "运行回测（策略选 livermore_pivot）"},
        "mcp_tool": ["get_decision_tree"], "llm_required": False,
        "related": ["strategy.livermore_pivot"],
    },
    "method.graham": {
        "category": "method", "name_zh": "格雷厄姆价值筛选", "step": 2,
        "difficulty": 2,
        "one_liner": "防御性 7 条标准 + 成长公式内在价值/安全边际",
        "risk": "价值陷阱；需要耐心持有",
        "market_states": ["dull", "frozen"],
        "ui_entry": {"tab": "📡 市场扫描", "button": "📗 格雷厄姆价值筛选"},
        "mcp_tool": ["graham_screen"], "llm_required": False,
    },
    "method.value_momentum": {
        "category": "method", "name_zh": "价值×动量复合（Asness 2013）", "step": 2,
        "difficulty": 2,
        "one_liner": "格雷厄姆评分 × 60 日动量 → 四象限（「便宜且走强」最佳）",
        "risk": "动量因子在震荡市失效",
        "market_states": ["fermenting", "starting"],
        "ui_entry": {"tab": "📡 市场扫描", "button": "⚖️ 价值×动量复合选股"},
        "mcp_tool": ["value_momentum_screen"], "llm_required": False,
    },
    "method.sector_rotation": {
        "category": "method", "name_zh": "板块轮动（RS 排名）", "step": 1,
        "difficulty": 2,
        "one_liner": "东财板块 20 日几何动量排名 + 加速检测 + 轮入/轮出信号",
        "risk": "东财 push2 接口偶发断连；板块数据需先装载",
        "market_states": ["fermenting", "starting", "climax"],
        "ui_entry": {"tab": "📡 市场扫描", "button": "🔄 板块轮动预测"},
        "mcp_tool": ["predict_sector_rotation"], "llm_required": False,
    },
    "method.sentiment_cycle": {
        "category": "method", "name_zh": "游资情绪周期", "step": 1,
        "difficulty": 2,
        "one_liner": "涨停梯队/连板高度/晋级率 → 情绪分 + 四阶段判定（启动/发酵/高潮/退潮）",
        "risk": "收盘涨停判定（≥9.5% 近似）；无盘中炸板数据",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "📡 市场扫描", "button": "🌡️ 游资情绪周期"},
        "mcp_tool": ["sentiment_cycle"], "llm_required": False,
    },
    # ============ analysis（6）：验证方法 ============
    "analysis.event_study": {
        "category": "analysis", "name_zh": "信号事件研究", "step": 3,
        "difficulty": 1,
        "one_liner": "信号后 5/10/20 日收益 vs 无条件基准 → 预测力判定（has_edge）",
        "risk": "需足够事件样本（n≥20）；基准选择影响结论",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "🧪 研究评估", "button": "📊 信号事件研究"},
        "mcp_tool": ["research_event_study"], "llm_required": False,
    },
    "analysis.walk_forward": {
        "category": "analysis", "name_zh": "Walk-Forward 样本外检验", "step": 3,
        "difficulty": 2,
        "one_liner": "滚动窗口样本外回测（15 folds），is_promotable 决定信号能否晋级",
        "risk": "需足够历史长度；区间选择影响结论",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "🧪 研究评估", "button": "运行 Walk-Forward"},
        "mcp_tool": ["research_walk_forward"], "llm_required": False,
    },
    "analysis.overfit": {
        "category": "analysis", "name_zh": "回测过拟合诊断（DSR/PBO）", "step": 3,
        "difficulty": 2,
        "one_liner": "Deflated Sharpe Ratio + 组合清洗检验——警惕回测图骗人",
        "risk": "需要收益序列样本；trial 数量估计影响 DSR",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "🧪 研究评估", "button": "🎲 回测过拟合诊断"},
        "mcp_tool": ["backtest_overfit_diagnosis"], "llm_required": False,
    },
    "analysis.factor_scan": {
        "category": "analysis", "name_zh": "因子扫描（IC/分层）", "step": 3,
        "difficulty": 1,
        "one_liner": "10 因子注册表逐一检验 IC/分层收益/单调性，找出有用因子",
        "risk": "横截面因子需 ≥5 只股票；短样本 IC 噪声大",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "📡 市场扫描", "button": "🧬 因子批量扫描"},
        "mcp_tool": ["factor_scan", "factor_library"], "llm_required": False,
    },
    "analysis.factor_neutralize": {
        "category": "analysis", "name_zh": "因子中性化（风格正交）", "step": 3,
        "difficulty": 3,
        "one_liner": "逐日 OLS 残差化（收益~市值+板块）→ 纯个股 alpha，去风格暴露",
        "risk": "市值用静态快照；同板块效果最佳",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "🧪 研究评估", "button": "🧮 因子中性化"},
        "mcp_tool": ["factor_neutralize"], "llm_required": False,
    },
    "analysis.strategy_compare": {
        "category": "analysis", "name_zh": "全策略事件研究对比", "step": 3,
        "difficulty": 1,
        "one_liner": "全部注册策略同台事件研究 → 有效判定可追溯",
        "risk": "历史样本内结论，需 OOS 复核",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "📚 研究总览", "button": "🏁 全策略对比"},
        "mcp_tool": ["strategy_compare"], "llm_required": False,
    },
    # ============ llm（6）：增强解读 ============
    "llm.agent_analyze": {
        "category": "llm", "name_zh": "AI 多维度分析（fast/deep/debate）", "step": 4,
        "difficulty": 3,
        "one_liner": "fast 单次调用五维分析；deep=5 分析师并行+PM 合成；debate 再+3 位投资大师辩论",
        "risk": "LLM 只解读数据不编造；未配置 LLM 时无此项",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "💬 AI 对话", "button": "对话（深度分析）"},
        "mcp_tool": ["agent_analyze_stock"], "llm_required": True,
        "llm_cost_hint": "fast≈8K tokens；deep≈50K；debate≈50K+5 次额外调用",
    },
    "llm.predict_market": {
        "category": "llm", "name_zh": "市场预测（1d/5d/20d）", "step": 4,
        "difficulty": 2,
        "one_liner": "确定性特征 → LLM 预测方向/概率/期望收益/场景，落盘回填验证成绩单",
        "risk": "预测可验证≠保证收益；短周期更准",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "🔮 市场预测", "button": "预测"},
        "mcp_tool": ["predict_market", "predict_market_multi"], "llm_required": True,
        "llm_cost_hint": "单票单周期 1 次调用；批量对比建议 limit 5 只内",
    },
    "llm.ai_report": {
        "category": "llm", "name_zh": "AI 市场研究报告", "step": 4,
        "difficulty": 1,
        "one_liner": "聚合全部确定性研究结果 → LLM 综述（总结/关注/风险/思路），不编造",
        "risk": "无 LLM 时模板降级",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "📚 研究总览", "button": "📋 AI 研究报告"},
        "mcp_tool": ["ai_market_report"], "llm_required": True,
        "llm_cost_hint": "单次综述≈2-4K tokens；每周/每决策点调用一次即可",
    },
    "llm.experience_search": {
        "category": "llm", "name_zh": "经验库（RAG 回放）", "step": 4,
        "difficulty": 1,
        "one_liner": "每次分析自动沉淀，后续分析自动参考相似历史案例（含事后验证）",
        "risk": "周期位置缺失时标 unknown；自动生效无需手动调用",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": None, "mcp_tool": ["agent_experience_search"],
        "llm_required": False,
    },
    "llm.memory_status": {
        "category": "llm", "name_zh": "长期记忆（决策回放）", "step": 4,
        "difficulty": 1,
        "one_liner": "决策→收益回填→胜率/盈亏比→贝叶斯策略权重→认知偏差检测",
        "risk": "回填需 ≥5 天；偏差阈值固定（60 天/5 次）",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "🔮 市场预测", "button": "🧠 长期记忆状态"},
        "mcp_tool": ["agent_memory_status"], "llm_required": False,
    },
    "llm.position_size": {
        "category": "llm", "name_zh": "预测→仓位建议", "step": 4,
        "difficulty": 2,
        "one_liner": "预测概率 × 历史命中率 × 概率桶校准 → 建议仓位（≤20% 硬上限）",
        "risk": "历史样本少时校准弱；仓位纪律优先于一切信号",
        "market_states": ["climax", "fermenting", "starting", "dull", "frozen"],
        "ui_entry": {"tab": "🔮 市场预测", "button": "💼 预测→仓位建议"},
        "mcp_tool": ["predict_position_size"], "llm_required": False,
    },
}

# 类别中文名（编目表头用）
CATEGORY_ZH: dict[str, str] = {
    "strategy": "策略", "method": "大牛方法", "analysis": "分析方法", "llm": "LLM 能力",
}


def _merge_strategy_entry(entry_id: str) -> Optional[dict[str, Any]]:
    """strategy 类条目与 STRATEGY_GUIDE 运行时合并（ref_strategy 外键）。

    难度/一句话/风险/适用市场只维护在 strategy_guide.py 一处，
    此处合并后返回完整条目；外键失效时记日志并返回 None。
    """
    entry = METHOD_CATALOG[entry_id]
    ref = entry.get("ref_strategy")
    if not ref:
        return None
    from pa_mcp.research.strategy_guide import STRATEGY_GUIDE
    guide = STRATEGY_GUIDE.get(ref)
    if not guide:
        logger.warning("methodology_guide: ref_strategy %r 不在 STRATEGY_GUIDE", ref)
        return None
    merged = dict(entry)
    merged["difficulty"] = guide.get("difficulty", 2)
    merged["method"] = guide.get("method", "")
    merged["one_liner"] = guide.get("one_liner", "")
    merged["risk"] = guide.get("risk", "")
    merged["market_states"] = list(guide.get("default_for", []))
    merged["beginner_default"] = (ref == _beginner_default_name())
    return merged


def _beginner_default_name() -> str:
    """新手默认策略名（从 strategy_guide 读取，惰性）。"""
    from pa_mcp.research.strategy_guide import BEGINNER_DEFAULT
    return BEGINNER_DEFAULT


def _full_entry(entry_id: str) -> Optional[dict[str, Any]]:
    """取某条目的完整视图（strategy 类合并，其余原样）。"""
    entry = METHOD_CATALOG[entry_id]
    if entry.get("category") == "strategy":
        return _merge_strategy_entry(entry_id)
    return entry


def _catalog_view(market_state: str) -> list[dict[str, Any]]:
    """展开全目录：合并字段、按 step 分组排序、按市场状态过滤。

    返回按 (step, difficulty) 升序的条目列表，每条含 id/category_zh。
    """
    rows: list[dict[str, Any]] = []
    for entry_id, entry in METHOD_CATALOG.items():
        full = _full_entry(entry_id)
        if full is None:
            continue
        if market_state not in full.get("market_states", []):
            continue
        row = dict(full)
        row["id"] = entry_id
        row["category_zh"] = CATEGORY_ZH.get(entry["category"], entry["category"])
        rows.append(row)
    rows.sort(key=lambda r: (r.get("step", 9), r.get("difficulty", 3)))
    return rows


def _routing_compare(market_state: str) -> dict[str, Any]:
    """对齐两套策略路由表，输出差异标注。

    - strategy_guide.STRATEGY_GUIDE.default_for：策略速查的推荐
    - orchestrator.MARKET_STATE_STRATEGY_ROUTING.strategies：市场诊断的路由
    两套表此前从未对齐展示；此处输出并集与 overlap/guide_only/routing_only。
    """
    from pa_mcp.research.strategy_guide import STRATEGY_GUIDE
    try:
        from pa_mcp.agent.orchestrator import MARKET_STATE_STRATEGY_ROUTING
    except Exception:  # noqa: BLE001  orchestrator 导入失败时降级为空
        MARKET_STATE_STRATEGY_ROUTING = {}

    guide_set = {name for name, info in STRATEGY_GUIDE.items()
                 if market_state in info.get("default_for", [])}
    routing = MARKET_STATE_STRATEGY_ROUTING.get(market_state) or {}
    routing_set = set(routing.get("strategies", [])) if isinstance(routing, dict) else set()

    overlap = sorted(guide_set & routing_set)
    guide_only = sorted(guide_set - routing_set)
    routing_only = sorted(routing_set - guide_set)
    return {
        "market_state": market_state,
        "overlap": overlap,
        "guide_only": guide_only,
        "routing_only": routing_only,
        "all": sorted(guide_set | routing_set),
        "consistency": "一致" if not guide_only and not routing_only
                      else "有差异（见差异列表）",
    }


def _llm_status() -> tuple[bool, Optional[str]]:
    """LLM 配置状态（configured, provider_name）。

    与 chat_reply 相同的兜底模式：单例为空时主动从 config/llm_config.json
    初始化（PROJECT_ROOT 绝对路径，不依赖 cwd），再判断。
    未配置/初始化失败返回 (False, None)，调用方各自降级。
    """
    try:
        from pa_mcp.agent.llm_factory import ensure_llm_adapter
        adapter = ensure_llm_adapter()  # 统一兜底：空单例主动读配置
        if adapter is not None:
            name = getattr(adapter, "provider_name", None)
            if name == "openai_compatible":
                # 通用 OpenAI 兼容适配器：读配置显示真实供应商名（如 doubao）
                name = _config_provider_name()
            return True, name
        return False, None
    except Exception:  # noqa: BLE001
        return False, None


def _config_provider_name() -> Optional[str]:
    """从 llm_config.json 读 active_provider 作为显示名（只读，失败返回 None）。"""
    try:
        from pa_mcp.config import PROJECT_ROOT
        config_file = PROJECT_ROOT / "config" / "llm_config.json"
        if not config_file.exists():
            return None
        import json
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f).get("active_provider") or None
    except Exception:  # noqa: BLE001
        return None


def _llm_configured() -> bool:
    """LLM 是否已配置（惰性；未配置返回 False）。"""
    return _llm_status()[0]


def _render_report(market_state: str, state_zh: str,
                   steps: list[dict[str, Any]], catalog: list[dict[str, Any]],
                   routing: dict[str, Any], beginner_default: str,
                   beginner_reason: str, llm_ok: bool,
                   llm_provider: Optional[str] = None) -> str:
    """渲染新手决策地图 markdown（UI/MCP 共用）。"""
    llm_status = f"✅ 已配置（{llm_provider}，第 ④ 步可用）" if llm_ok else \
        "❌ 未配置（第 ④ 步可跳过，前三步不受影响）"
    lines = [
        f"## 🗺️ 新手决策地图（当前市场：**{state_zh}**）",
        "",
        f"**新手默认起步**：{beginner_default} —— {beginner_reason}",
        f"**LLM 配置**：{llm_status}",
    ]
    if market_state == "unknown":
        lines += [
            "",
            "> ⚠️ 市场状态未能自动检测（数据未就绪）。建议先运行数据更新",
            "> （MCP: `run_daily_update`；UI: 「🔌 数据源健康」→ 数据体检），再回来查看。",
        ]
    lines.append("")
    for step in steps:
        lines.append(f"### {step['title']}")
        lines.append(step["goal"])
        rec = step.get("recommended", [])
        if rec:
            lines.append("")
            lines.append("**当前市场推荐**：")
            for r in rec:
                stars = "⭐" * r.get("difficulty", 1)
                llm_tag = "🔮" if r.get("llm_required") else ""
                ui = r.get("ui_entry")
                ui_txt = (f"「{ui['tab']}」{ui['button']}" if ui else "（MCP/自动）")
                lines.append(f"- {r['name_zh']} {stars} {llm_tag} —— "
                             f"{r.get('one_liner', '')[:60]}")
                lines.append(f"  - 入口：{ui_txt}；MCP：`{', '.join(r.get('mcp_tool', []))}`")
        else:
            lines.append("")
            lines.append("（当前状态无特别推荐，以观望为主）")
        lines.append("")
        lines.append(f"*门槛：{step['gate']}*")
        lines.append("")
    # 两套路由一致性
    if market_state != "unknown":
        lines.append("### 🧭 两套策略路由对齐")
        lines.append(f"策略速查 与 市场诊断路由：**{routing['consistency']}**")
        if routing["routing_only"]:
            lines.append(f"- 诊断路由额外推荐：`{'、'.join(routing['routing_only'])}`"
                         "（未在策略速查中评级）")
        if routing["guide_only"]:
            lines.append(f"- 策略速查额外推荐：`{'、'.join(routing['guide_only'])}`")
        lines.append("")
    # 全量编目表
    lines.append("### 📖 全部编目（当前状态可用）")
    lines.append("| 类别 | 资产 | 难度 | 一句话 | LLM | UI 入口 |")
    lines.append("|---|---|---|---|---|---|")
    for r in catalog:
        stars = "⭐" * r.get("difficulty", 1)
        ui = r.get("ui_entry")
        ui_txt = (f"「{ui['tab']}」{ui['button']}" if ui else "（MCP/自动）")
        llm_txt = "🔮 需要" if r.get("llm_required") else "—"
        lines.append(f"| {r.get('category_zh', '')} | {r['name_zh']} | {stars} | "
                     f"{r.get('one_liner', '')[:40]} | {llm_txt} | {ui_txt} |")
    lines.append("")
    lines.append("---")
    lines.append("*研究参考，非投资建议。仓位纪律（RiskGuard ≤20% 单票上限 / 回撤分级）优先于一切信号。*")
    return "\n".join(lines)


def render_tab_button_guide(tab: str) -> str:
    """渲染某 Tab 内按钮的用法速查（markdown，纯函数无 gradio 依赖）。

    从 METHOD_CATALOG 过滤 ui_entry.tab == tab 的条目（strategy 类经
    _full_entry 合并字段），按 (step, difficulty) 排序输出：
    | 入口按钮 | 用途 | 难度 | 风险 |。tab 为空则全量；无匹配返回 ""。

    Args:
        tab: Tab 名（如 "📡 市场扫描"），与 KNOWN_TABS 同源
    """
    rows: list[dict[str, Any]] = []
    for entry_id, entry in METHOD_CATALOG.items():
        ui = entry.get("ui_entry")
        if not ui:
            continue
        if tab and ui.get("tab") != tab:
            continue
        full = _full_entry(entry_id)
        if full is None:
            continue
        rows.append({
            "button": ui.get("button", ""),
            "one_liner": full.get("one_liner", ""),
            "difficulty": full.get("difficulty", 1),
            "risk": full.get("risk", ""),
        })
    if not rows:
        return ""
    rows.sort(key=lambda r: r["difficulty"])
    lines = ["| 入口按钮 | 用途 | 难度 | 风险 |", "|---|---|---|---|"]
    for r in rows:
        stars = "⭐" * r["difficulty"]
        lines.append(f"| {r['button']} | {r['one_liner'][:44]} | {stars} | "
                     f"{r['risk'][:36]} |")
    lines.append("\n*难度 ⭐ 越多越难；悬停按钮可见一句提示。研究参考，非投资建议。*")
    return "\n".join(lines)


def get_methodology_guide(market_state: Optional[str] = None) -> dict[str, Any]:
    """新手决策地图：四步研究路径 + 四类资产编目（UI / MCP 共用主入口）。

    Args:
        market_state: 市场状态（climax/fermenting/starting/dull/frozen），
            缺省自动检测（DuckDB 涨停/成交额指标，失败返回 unknown）
    """
    from pa_mcp.research.strategy_guide import strategy_guide as sg

    if market_state is None:
        # 复用 strategy_guide 的状态检测与推荐（单一实现）
        guide = sg()
        market_state = guide["market_state"]
        beginner_default = guide["beginner_default"]
        beginner_reason = guide["beginner_default_reason"]
    else:
        guide = sg(market_state)
        beginner_default = guide["beginner_default"]
        beginner_reason = guide["beginner_default_reason"]

    state_zh = STATE_ZH.get(market_state, market_state)
    catalog = _catalog_view(market_state)
    routing = _routing_compare(market_state)
    llm_ok, llm_provider = _llm_status()

    # 每步推荐（按 step 分组，取当前状态可用条目）
    steps: list[dict[str, Any]] = []
    for sdef in METHOD_STEPS:
        step_no = int(sdef["step"])
        rec = [r for r in catalog if r.get("step") == step_no]
        steps.append({
            **sdef,
            "recommended": rec,
        })

    report = _render_report(market_state, state_zh, steps, catalog, routing,
                            beginner_default, beginner_reason, llm_ok,
                            llm_provider)

    return {
        "market_state": market_state,
        "market_state_zh": state_zh,
        "steps": steps,
        "catalog": catalog,
        "beginner_default": beginner_default,
        "beginner_default_reason": beginner_reason,
        "llm_configured": llm_ok,
        "llm_provider": llm_provider,
        "routing_consistency": routing,
        "report": report,
    }
