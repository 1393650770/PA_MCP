# 新手决策地图（methodology_guide）测试
#
# 纯函数 + monkeypatch，不依赖网络/DB/真实 LLM——
# 市场状态一律显式传入，不触发 _detect_market_state 的 DuckDB 读取。

from __future__ import annotations

import pytest

from pa_mcp.research.methodology_guide import (
    METHOD_CATALOG, METHOD_STEPS,
    _catalog_view, _merge_strategy_entry, _routing_compare,
    get_methodology_guide, render_tab_button_guide,
)
from pa_mcp.research.strategy_guide import STRATEGY_GUIDE

# ---- 91 个 MCP 工具静态快照（来源：server.py @mcp.tool 清单）
# 注意：工具集变更时需同步更新此集合（新增工具时补一行即可）。
TOOL_SNAPSHOT = {
    # 数据
    "get_realtime_quote", "get_kline", "get_market_overview", "search_stock",
    "get_stock_info", "scan_limit_up", "scan_volume_surge", "get_major_events",
    "get_valuation_snapshot", "get_stock_capital_flow", "get_stock_name",
    "get_data_source_health", "get_market_sentiment", "review_daily_limit_up",
    "review_dragon_tiger",
    # Agent / 分析
    "agent_portfolio_review", "agent_earnings_analysis", "agent_analyze_stock",
    "agent_market_state", "agent_market_diagnosis", "agent_morning_brief",
    "agent_compare_stocks", "agent_sector_analysis", "agent_memory_status",
    "agent_experience_search",
    # 回测 / 研究
    "scan_market", "research_event_study", "research_walk_forward",
    "portfolio_backtest", "get_strategy_info", "list_strategies",
    "research_event_study_sector", "chan_beichi_backtest",
    "chan_beichi_event_study", "chan_analysis", "strategy_compare",
    "value_momentum_backtest", "value_momentum_screen", "graham_screen",
    "backtest_overfit_diagnosis", "scan_canslim", "turtle_position_size",
    "get_decision_tree",
    # 因子
    "factor_library", "evaluate_factor", "factor_prediction_sensitivity",
    "factor_portfolio_backtest", "factor_stock_selection", "factor_scan",
    "factor_neutralize",
    # 预测
    "predict_market", "predict_position_size", "predict_market_multi",
    "prediction_history", "evaluate_predictions",
    # 组合 / 自选 / 预警
    "portfolio_strategy_signals", "portfolio_ai_analysis",
    "portfolio_risk_dashboard", "portfolio_summary", "portfolio_add",
    "portfolio_remove", "watchlist_add", "watchlist_remove", "watchlist_show",
    "watchlist_overview", "watch_price_alert", "watch_volume_alert",
    "list_alerts",
    # 整合 / 市场结构
    "watchlist_consensus", "consensus_event_study", "signal_consensus",
    "watchlist_resonance", "resonance_event_study", "predict_resonance",
    "one_click_analysis", "trading_actions", "ai_market_report",
    "analyze_timeframe_alignment", "calc_vwap", "pa_help",
    # 系统
    "export_research_data", "run_daily_update", "data_quality_report",
    "regime_matrix", "market_structure", "sentiment_cycle",
    "predict_sector_rotation", "sector_rotation_status", "sector_leaders",
    "evaluate_sector_predictions",
}

KNOWN_TABS = {
    "📊 数据看板", "💬 AI 对话", "🔀 多股对比", "📡 市场扫描", "🧪 研究评估",
    "📦 组合构建", "🛠️ 策略回测", "📚 研究总览", "🔮 市场预测", "💼 组合管理",
}

MARKET_STATES = ["climax", "fermenting", "starting", "dull", "frozen"]

CATEGORY_COUNTS = {"strategy": 11, "method": 8, "analysis": 6, "llm": 6}


def test_catalog_covers_four_categories():
    """四类条目数达标，id 前缀 == category（防重名且天然分组）。"""
    counts = {}
    for entry_id, entry in METHOD_CATALOG.items():
        cat = entry["category"]
        counts[cat] = counts.get(cat, 0) + 1
        assert entry_id.startswith(cat + "."), f"{entry_id} 前缀与 category 不符"
        assert entry_id.split(".", 1)[1], f"{entry_id} 缺资产名"
    for cat, expected in CATEGORY_COUNTS.items():
        assert counts.get(cat, 0) >= expected, \
            f"{cat} 条目数不足：{counts.get(cat, 0)} < {expected}"


def test_strategy_entries_ref_guide_complete():
    """双向完整性：STRATEGY_GUIDE 全部策略均被 ref_strategy 引用，
    且每条 ref_strategy 都真实存在（防孤儿/悬空外键）。"""
    refs = set()
    for entry_id, entry in METHOD_CATALOG.items():
        if entry["category"] != "strategy":
            continue
        ref = entry.get("ref_strategy")
        assert ref, f"{entry_id} 缺 ref_strategy"
        assert ref in STRATEGY_GUIDE, f"{entry_id} 引用不存在的策略 {ref}"
        refs.add(ref)
    assert refs == set(STRATEGY_GUIDE.keys()), \
        f"STRATEGY_GUIDE 中未被引用的策略：{set(STRATEGY_GUIDE) - refs}"


def test_merged_strategy_entry_matches_guide():
    """合并后字段与 STRATEGY_GUIDE 逐字段一致（防两处数据漂移）。"""
    for entry_id, entry in METHOD_CATALOG.items():
        if entry["category"] != "strategy":
            continue
        merged = _merge_strategy_entry(entry_id)
        assert merged is not None
        ref = STRATEGY_GUIDE[entry["ref_strategy"]]
        assert merged["difficulty"] == ref["difficulty"]
        assert merged["one_liner"] == ref["one_liner"]
        assert merged["risk"] == ref["risk"]
        assert merged["market_states"] == list(ref["default_for"])


def test_mcp_tool_names_valid():
    """编目引用的 MCP 工具名必须真实存在（工具集变更时同步 TOOL_SNAPSHOT）。"""
    for entry_id, entry in METHOD_CATALOG.items():
        tools = entry.get("mcp_tool")
        assert tools, f"{entry_id} 缺 mcp_tool"
        assert isinstance(tools, list) and tools, f"{entry_id} mcp_tool 须为非空列表"
        for t in tools:
            assert t in TOOL_SNAPSHOT, f"{entry_id} 引用不存在的工具 {t}"


def test_ui_entries_have_known_tabs():
    """ui_entry.tab 必须 ∈ 已知 10 个 Tab（None = 仅 MCP/自动触发）。"""
    for entry_id, entry in METHOD_CATALOG.items():
        ui = entry.get("ui_entry")
        if ui is None:
            continue
        assert ui.get("tab") in KNOWN_TABS, \
            f"{entry_id} 的 Tab「{ui.get('tab')}」不在已知 Tab 集合中"
        assert ui.get("button"), f"{entry_id} 缺 ui_entry.button"


@pytest.mark.parametrize("state", MARKET_STATES)
def test_recommendations_by_market_state(state):
    """5 个真实市场状态：每步 recommended 非空、条目适用该状态、按 step 升序。"""
    result = get_methodology_guide(market_state=state)
    assert result["market_state"] == state
    assert result["market_state_zh"]
    steps = result["steps"]
    assert len(steps) == len(METHOD_STEPS)
    for s in steps:
        assert s["recommended"], f"{state}: step {s['step']} 推荐为空"
        for r in s["recommended"]:
            assert state in r.get("market_states", []), \
                f"{state}: {r['id']} 不适用该状态"
    step_nos = [s["step"] for s in steps]
    assert step_nos == sorted(step_nos)


def test_unknown_state_degrades_gracefully():
    """unknown 状态：catalog 为空，report 给数据未就绪提示而非报错。"""
    result = get_methodology_guide(market_state="unknown")
    assert result["market_state"] == "unknown"
    assert result["catalog"] == []
    assert "数据未就绪" in result["report"]


def test_routing_consistency_union():
    """routing_consistency：并集 = overlap + guide_only + routing_only（恒等式）。"""
    for state in MARKET_STATES:
        rc = _routing_compare(state)
        assert set(rc["all"]) == set(rc["overlap"]) | set(rc["guide_only"]) \
            | set(rc["routing_only"]), f"{state} 并集不恒等"
        assert len(rc["overlap"]) + len(rc["guide_only"]) + \
            len(rc["routing_only"]) == len(rc["all"]), f"{state} 集合不相交"


def test_report_markdown_render():
    """report 含四步标题、编目表头、市场状态中文、新手默认、LLM 配置状态行。"""
    result = get_methodology_guide(market_state="fermenting")
    report = result["report"]
    for s in result["steps"]:
        assert s["title"] in report, f"缺步骤标题 {s['title']}"
    assert "全部编目" in report
    assert "| 类别 | 资产 | 难度 | 一句话 | LLM | UI 入口 |" in report
    assert "发酵期" in report
    assert result["beginner_default"] in report
    assert "LLM 配置" in report


def test_llm_flags(monkeypatch):
    """llm_required=True 的条目强制带 cost_hint；
    get_llm_adapter 未配置时 llm_configured=False 且 report 提示第 4 步可跳过。"""
    # 编目完整性：LLM 条目必有成本提示
    for entry_id, entry in METHOD_CATALOG.items():
        if entry.get("llm_required"):
            assert entry.get("llm_cost_hint"), f"{entry_id} 缺 llm_cost_hint"

    # 未配置 LLM（单例为空 + 兜底初始化也失败）：降级提示
    monkeypatch.setattr(
        "pa_mcp.agent.llm_port.get_llm_adapter", lambda: None)
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.init_llm_adapter", lambda _path: None)
    result = get_methodology_guide(market_state="dull")
    assert result["llm_configured"] is False
    assert "未配置" in result["report"]
    assert "可跳过" in result["report"]

    # 已配置 LLM（单例已有 adapter）：状态翻转
    class _FakeAdapter:
        provider_name = "fake"
    monkeypatch.setattr(
        "pa_mcp.agent.llm_port.get_llm_adapter", lambda: _FakeAdapter())
    result2 = get_methodology_guide(market_state="dull")
    assert result2["llm_configured"] is True
    assert result2["llm_provider"] == "fake"
    assert "已配置（fake" in result2["report"]


def test_llm_lazy_init_fallback(monkeypatch):
    """单例为空但配置存在：_llm_status 主动调用 init_llm_adapter 兜底，
    并返回 provider 名（与 chat_reply 相同的兜底模式）。"""
    class _FakeAdapter:
        provider_name = "doubao"

    calls: list[str] = []

    def _init(path):
        calls.append(path)
        return _FakeAdapter()

    monkeypatch.setattr(
        "pa_mcp.agent.llm_port.get_llm_adapter", lambda: None)
    monkeypatch.setattr(
        "pa_mcp.agent.llm_factory.init_llm_adapter", _init)
    result = get_methodology_guide(market_state="dull")
    assert result["llm_configured"] is True
    assert result["llm_provider"] == "doubao"
    assert "已配置（doubao" in result["report"]
    assert calls, "单例为空时未触发 init_llm_adapter 兜底"
    # 兜底使用 PROJECT_ROOT 绝对路径（不依赖 cwd）
    assert "config" in calls[0] and "llm_config.json" in calls[0]


def test_render_tab_button_guide_filters_by_tab():
    """tab 过滤正确：市场扫描命中编目条目，不含其他 Tab 的条目。"""
    md = render_tab_button_guide("📡 市场扫描")
    assert md
    assert "CANSLIM" in md
    assert "格雷厄姆" in md
    # 「📚 研究总览」专属条目不误入（strategy_compare/ai_report）
    assert "全策略对比" not in md
    assert "AI 市场研究报告" not in md
    # 表头与格式
    assert "| 入口按钮 | 用途 | 难度 | 风险 |" in md
    assert "⭐⭐" in md  # 难度星


def test_render_tab_button_guide_strategy_merged():
    """strategy 类条目经 _full_entry 合并后字段完整（防 ref_strategy 缺失）。"""
    md = render_tab_button_guide("🛠️ 策略回测")
    assert md
    assert "运行回测" in md
    # 合并自 STRATEGY_GUIDE 的 one_liner 字段
    assert "布林下轨" in md or "布林" in md


def test_render_tab_button_guide_all_and_unknown():
    """空 tab = 全量；未知 tab = 空串。"""
    md_all = render_tab_button_guide("")
    assert md_all and "| 入口按钮 | 用途 | 难度 | 风险 |" in md_all
    assert render_tab_button_guide("不存在的 Tab") == ""


def test_catalog_view_sorted_and_filtered():
    """_catalog_view：按 (step, difficulty) 升序，且只含该状态可用条目。"""
    rows = _catalog_view("climax")
    assert rows, "climax 状态目录不应为空"
    keys = [(r["step"], r["difficulty"]) for r in rows]
    assert keys == sorted(keys), "目录未按 step/difficulty 升序"
    for r in rows:
        assert "climax" in r.get("market_states", [])
        assert r["id"].startswith(r["category"] + ".")
        assert r["category_zh"]
