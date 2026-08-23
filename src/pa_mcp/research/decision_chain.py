# [AI:BEGIN]
# PA_MCP - Research: 研究决策链（二元决策树适配版）
#
# 来源：PA_Agent 的 ai/decision_tree.py + decision_nodes.py 设计思想
# （方法论文档 → 可执行节点索引 → AI 携带 gate_trace → 程序校验），
# 但按本项目研究方法论适配（四步：看环境→选方法→做验证→增强解读，
# 与 methodology_guide.METHOD_CATALOG 对齐）。
#
# 核心机制：
#   1. 决策链节点定义（数据驱动，A 股研究版，比 PA 的 90 节点精简）
#   2. AI 分析输出必须携带 trace=[{node_id, answer, reason}]，程序逐节点
#      校验：id 合法、章节顺序、结局一致性（防跳步/自相矛盾）
#   3. 闸门短路：§1 环境不确定 → 直接合成"等待"结果，不浪费后续调用
#   4. 确定性评估器：无 LLM 时用统计规则沿链评估（降级）
#
# 纯数据模块，不依赖 gradio/server；可单测。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---- 决策链节点定义 ----

# 每节点: id/章节/title/question/answers(是|否|中性|等待)/outcome 含义
CHAIN_NODES: list[dict[str, Any]] = [
    # §0 数据前置（总原则）
    {"id": "0.1", "chapter": "数据前置", "title": "数据是否可靠",
     "question": "行情/情绪/板块数据是否新鲜可用（无断连、无陈旧）？",
     "answers": ["是", "否"], "note": "否 → 无法分析，等待"},
    {"id": "0.2", "chapter": "数据前置", "title": "分析条件是否充分",
     "question": "K 线历史 ≥120 根？（仅限个股分析条件，市场情绪/大盘数据在 §1 环境层判定）",
     "answers": ["是", "否"], "note": "否 → 等待"},
    # §1 市场环境（看环境）
    {"id": "1.1", "chapter": "市场环境", "title": "市场状态可识别",
     "question": "当前市场状态（低迷/正常/高潮/恐慌）可明确判断？",
     "answers": ["是", "否"], "note": "否 → 状态不明，观望为主"},
    {"id": "1.2", "chapter": "市场环境", "title": "情绪阶段判定",
     "question": "市场情绪处于冰点/发酵/启动/高潮哪个阶段？（有涨停与赚钱效应数据）",
     "answers": ["冰点", "发酵", "启动", "高潮", "未知"]},
    # §2 策略选择（选方法）
    {"id": "2.1", "chapter": "策略选择", "title": "策略适配路由",
     "question": "当前环境适配的策略方向？（均值回归/趋势/价值/动量）",
     "answers": ["均值回归", "趋势", "价值", "动量", "未知"]},
    # §3 信号验证（做验证）
    {"id": "3.1", "chapter": "信号验证", "title": "买入信号存在",
     "question": "标的近 10 日是否有适配策略的买入信号？",
     "answers": ["是", "否"]},
    {"id": "3.2", "chapter": "信号验证", "title": "信号历史验证",
     "question": "该信号历史 5 日胜率 ≥50% 或样本充足？",
     "answers": ["是", "否", "样本不足"]},
    # §4 风险收益（增强解读）
    {"id": "4.1", "chapter": "风险收益", "title": "风险收益比",
     "question": "止损至止盈的空间比 ≥1.5（盈亏比合格）？",
     "answers": ["是", "否", "无法计算"]},
    {"id": "4.2", "chapter": "风险收益", "title": "交易者方程",
     "question": "胜率×回报 > 败率×风险（期望值为正）？",
     "answers": ["是", "否", "无法计算"]},
    # §5 最终裁定
    {"id": "5.1", "chapter": "最终裁定", "title": "行动裁定",
     "question": "综合结论：开仓/观察/放弃？",
     "answers": ["观察", "开仓", "放弃"],
     "outcomes": {"观察": "wait", "开仓": "trade", "放弃": "reject"}},
]

# 结局编码（校验用）
OUTCOME_ORDER = {"wait": 0, "reject": 1, "observe": 2, "trade": 3}
VALID_ANSWERS = {"是", "否", "中性", "等待", "不适用", "冰点", "发酵", "启动",
                 "高潮", "未知", "均值回归", "趋势", "价值", "动量",
                 "样本不足", "无法计算", "观察", "开仓", "放弃"}


def chain_nodes() -> list[dict[str, Any]]:
    """节点定义（副本，防外部修改）。"""
    return [dict(n) for n in CHAIN_NODES]


def chain_text() -> str:
    """决策链 → prompt 文本（要求 AI 按链输出 trace）。"""
    lines = ["## 研究决策链（输出 JSON 时须携带 trace 数组）",
             "按 §0→§5 顺序逐节点回答，node_id 必须按顺序递增："]
    for n in CHAIN_NODES:
        lines.append(f"- **{n['id']} {n['title']}**（{n['chapter']}）："
                     f"{n['question']} 答案: {'/'.join(n['answers'])}")
        if n.get("note"):
            lines.append(f"  ⚠ {n['note']}")
    lines.append("trace 格式: [{\"node_id\":\"1.1\",\"answer\":\"是\","
                 "\"reason\":\"一句话依据\"}, ...]")
    return "\n".join(lines)


# ---- 轨迹校验（防跳步/自相矛盾） ----

_NODE_IDS = [n["id"] for n in CHAIN_NODES]
_VALID_ANSWERS: set[str] = set()

def _init_valid_answers() -> set[str]:
    global _VALID_ANSWERS
    if not _VALID_ANSWERS:
        for n in CHAIN_NODES:
            _VALID_ANSWERS.update(n["answers"])
        _VALID_ANSWERS.update(VALID_ANSWERS)
    return _VALID_ANSWERS


def validate_trace(trace: Any) -> list[str]:
    """校验 AI 输出的决策链轨迹。

    Returns:
        错误列表（空 = 通过）。
    """
    _init_valid_answers()
    errors: list[str] = []
    if not isinstance(trace, list) or not trace:
        return ["trace 缺失或为空（AI 未按决策链输出）"]

    seen_ids: set[str] = set()
    last_rank = -1
    for item in trace:
        if not isinstance(item, dict):
            errors.append(f"trace 元素非对象: {item!r}")
            continue
        nid = str(item.get("node_id", ""))
        if nid not in _NODE_IDS:
            errors.append(f"未知节点 {nid!r}（合法: {_NODE_IDS}）")
            continue
        # 章节顺序（0.1 < 0.2 < 1.1 < ... 数字序）
        rank = _node_rank(nid)
        if rank < last_rank:
            errors.append(f"节点顺序错误：{nid} 出现在更前章节之后")
        last_rank = max(last_rank, rank)
        if nid in seen_ids:
            errors.append(f"节点 {nid} 重复出现")
        seen_ids.add(nid)

        ans = str(item.get("answer", ""))
        if ans not in _VALID_ANSWERS:
            errors.append(f"节点 {nid} 答案 {ans!r} 非法（合法: "
                          f"{'/'.join(sorted(_VALID_ANSWERS))}）")

    # 结局一致性：§5.1 的 answer 必须是 观察/开仓/放弃
    final = [t for t in trace if isinstance(t, dict)
             and str(t.get("node_id", "")) == "5.1"]
    if final and str(final[-1].get("answer", "")) not in ("观察", "开仓", "放弃"):
        errors.append(f"§5.1 结局必须为 观察/开仓/放弃，得到 "
                      f"{final[-1].get('answer')!r}")
    return errors


def _node_rank(nid: str) -> int:
    """节点排序秩（按章节数字序）。"""
    try:
        major, _, minor = nid.partition(".")
        return int(major) * 10 + int(minor or 0)
    except ValueError:
        return 999


# ---- 闸门短路 ----

def gate_short_circuit(stage_result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """数据闸门检查：数据不可靠 → 直接合成"等待"结果。

    借鉴 PA 的闸门短路（环境不明不强行交易）。适配研究系统定位：
      - §0 数据闸门：硬短路（数据不可用，分析无意义）
      - §1 环境闸门：软标记（market_unclear=True，由调用方压降置信度
        而非阻断——情绪接口暂不可用时个股研究仍可进行）

    Returns:
        硬短路结果 dict，或 None（继续阶段二）；软标记放 stage_result["market_unclear"]。
    """
    trace = stage_result.get("trace") or []
    gate_answers: dict[str, str] = {}
    for t in trace:
        if isinstance(t, dict) and t.get("node_id") in ("0.1", "0.2", "1.1"):
            gate_answers[str(t["node_id"])] = str(t.get("answer", ""))

    if gate_answers.get("0.1") == "否" or gate_answers.get("0.2") == "否":
        return {
            "action": "wait", "mode": "gate_short_circuit",
            "summary": "数据不可靠或条件不足——等待数据可用再分析",
            "reason": gate_answers,
            "short_circuit_at": "0.x",
        }
    if gate_answers.get("1.1") == "否":
        stage_result["market_unclear"] = True
    return None


# ---- 确定性评估器（无 LLM 降级） ----

def evaluate_chain_rule(facts: dict[str, Any]) -> dict[str, Any]:
    """无 LLM 时沿决策链做确定性评估（统计规则）。

    facts 期望字段：
        data_ok: bool             # 数据可用
        hist_ok: bool             # K线≥120
        market_recognizable: bool # 市场状态可识别
        sentiment_phase: str      # 冰点/发酵/启动/高潮/未知
        strategy_fit: str         # 均值回归/趋势/价值/动量
        has_signal: bool          # 有买入信号
        win_rate: Optional[float] # 信号历史胜率
        rr_ratio: Optional[float] # 盈亏比
    """
    trace: list[dict] = []
    steps: list[str] = []

    def step(nid: str, answer: str, reason: str) -> None:
        trace.append({"node_id": nid, "answer": answer, "reason": reason})
        steps.append(f"{nid} {answer}：{reason}")

    data_ok = bool(facts.get("data_ok", False))
    hist_ok = bool(facts.get("hist_ok", False))
    step("0.1", "是" if data_ok else "否", "数据源状态" + ("正常" if data_ok else "异常"))
    step("0.2", "是" if hist_ok else "否", "历史长度" + ("充足" if hist_ok else "不足"))
    if not (data_ok and hist_ok):
        return {"action": "wait", "mode": "rule", "trace": trace,
                "summary": "数据或历史不足，等待", "steps": steps}

    mkt = bool(facts.get("market_recognizable", False))
    step("1.1", "是" if mkt else "否", "市场状态" + ("可识别" if mkt else "不明"))
    if not mkt:
        return {"action": "wait", "mode": "rule", "trace": trace,
                "summary": "市场状态不明，观望", "steps": steps}

    phase = str(facts.get("sentiment_phase", "未知"))
    step("1.2", phase, "情绪阶段")
    strat = str(facts.get("strategy_fit", "未知"))
    step("2.1", strat, "环境适配策略")

    has_sig = bool(facts.get("has_signal", False))
    step("3.1", "是" if has_sig else "否", "近期信号" + ("存在" if has_sig else "不存在"))
    if not has_sig:
        return {"action": "wait", "mode": "rule", "trace": trace,
                "summary": "无买入信号，等待", "steps": steps}

    wr = facts.get("win_rate")
    wr_ok = wr is not None and wr >= 50.0
    step("3.2", "是" if wr_ok else ("样本不足" if wr is None else "否"),
         f"信号胜率 {wr if wr is not None else '样本不足'}")
    if not wr_ok and wr is not None:
        return {"action": "reject", "mode": "rule", "trace": trace,
                "summary": f"信号历史胜率 {wr:.0f}% < 50%，放弃", "steps": steps}

    rr = facts.get("rr_ratio")
    rr_ok = rr is not None and rr >= 1.5
    step("4.1", "是" if rr_ok else ("无法计算" if rr is None else "否"),
         f"盈亏比 {rr if rr is not None else '无法计算'}")
    if rr is not None and not rr_ok:
        return {"action": "reject", "mode": "rule", "trace": trace,
                "summary": f"盈亏比 {rr:.2f} < 1.5，放弃", "steps": steps}

    # 交易者方程：胜率*回报 > 败率*风险（用胜率和盈亏比近似）
    ev_ok = False
    if wr is not None and rr is not None:
        ev_ok = (wr / 100.0) * rr > (1 - wr / 100.0)
    step("4.2", "是" if ev_ok else ("无法计算" if (wr is None or rr is None) else "否"),
         "期望值" + ("为正" if ev_ok else "为负或不可算"))
    if wr is not None and rr is not None and not ev_ok:
        return {"action": "reject", "mode": "rule", "trace": trace,
                "summary": "期望值为负，放弃", "steps": steps}

    action = "开仓" if ev_ok else "观察"
    step("5.1", action, "综合裁定")
    return {"action": {"开仓": "trade", "观察": "observe"}.get(action, "observe"),
            "mode": "rule", "trace": trace, "summary": "确定性规则评估通过",
            "steps": steps}
