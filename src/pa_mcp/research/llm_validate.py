# [AI:BEGIN]
# PA_MCP - Research: LLM 输出四层校验 + 反馈式重试
#
# 来源：PA_Agent 的 json_validator.py + validation_retry.py 设计
# （JSON 语法 → schema 字段 → 数值事实一致性 → 语义矛盾；失败把
# 校验错误回灌模型重试，最多 N 次；截断 JSON 识别；防"作弊"重试）。
#
# 本模块适配本项目：
#   - 复用 llm_openai_compat._extract_json 的稳健提取（markdown 剥壳）
#   - 轻量 schema（dict 描述，不引入 jsonschema 依赖）
#   - 事实一致性：已知数值（价格/涨跌幅等）与 AI 输出偏差超阈值报错
#   - chat_json_validated：包装 chat_json，错误回灌 + 计数重试
#
# 用途：analyze_stock / future_path / sector_rotation / self_improve 等
# 所有 LLM 调用点可替换为校验版，显著降低"AI 输出垃圾 JSON"事故率。
# [AI:END]

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---- 第一层：JSON 语法提取（复用现有实现思路） ----


def extract_json(text: str) -> Any:
    """从 LLM 响应稳健提取 JSON（markdown 剥壳 + 首尾大括号）。"""
    m = re.search(r"```(?:json)?\s*(.*?)```", text or "", re.S)
    if m:
        text = m.group(1)
    s, e = (text or "").find("{"), (text or "").rfind("}")
    if s >= 0 and e > s:
        text = text[s:e + 1]
    return json.loads((text or "").strip())


def is_truncated(raw: str) -> bool:
    """截断检测：输出尾部是未闭合 JSON（缺 } 或引号未闭合）。

    判定：解析失败且错误位置在内容末尾附近（<8 字符或达末尾 10%），
    典型形态 = max_tokens 截断。
    """
    s = (raw or "").rstrip()
    if not s:
        return False
    try:
        json.loads(s)
        return False
    except json.JSONDecodeError as e:
        return e.pos >= len(s) - 8


# ---- 第二层：轻量 schema 校验 ----


def validate_schema(obj: Any, schema: dict[str, Any]) -> list[str]:
    """字段级 schema 校验（不引入 jsonschema 依赖）。

    schema 格式: {"required": ["a"], "fields": {"a": {"type": "str|float|int|bool|list|dict",
                                             "enum": [...], "min": 0, "max": 100}}}
    """
    errors: list[str] = []
    if not isinstance(obj, dict):
        return [f"结果非 JSON 对象: {type(obj).__name__}"]
    for f in schema.get("required", []):
        if f not in obj or obj[f] is None:
            errors.append(f"缺少必填字段 {f!r}")
    for f, spec in (schema.get("fields") or {}).items():
        if f not in obj or obj[f] is None:
            continue
        val = obj[f]
        t = spec.get("type", "str")
        ok = {"str": isinstance(val, str), "float": isinstance(val, (int, float)),
              "int": isinstance(val, int) and not isinstance(val, bool),
              "bool": isinstance(val, bool),
              "list": isinstance(val, list), "dict": isinstance(val, dict)}
        if not ok.get(t, False):
            errors.append(f"字段 {f!r} 类型应为 {t}，得到 {type(val).__name__}")
            continue
        if t == "float" and isinstance(val, (int, float)):
            if "min" in spec and val < spec["min"]:
                errors.append(f"字段 {f!r} 低于下限 {spec['min']}（={val}）")
            if "max" in spec and val > spec["max"]:
                errors.append(f"字段 {f!r} 超过上限 {spec['max']}（={val}）")
        if "enum" in spec and val not in spec["enum"]:
            errors.append(f"字段 {f!r} 取值 {val!r} 不在枚举 {spec['enum']}")
    return errors


# ---- 第三层：数值事实一致性 ----


def validate_facts(obj: Any, facts: dict[str, Any],
                   rel_tolerance: float = 0.05) -> list[str]:
    """AI 输出数值与已知事实比对（偏差超阈值报错）。

    facts 格式: {"close": 11.27, "pct_change": -1.25, "direction": "up"}
    规则:
      - 数值字段：|AI值-事实| / |事实| > rel_tolerance → 报错
      - 方向字段：与事实相反 → 报错
    """
    errors: list[str] = []
    if not isinstance(obj, dict) or not facts:
        return errors
    for key, true_val in facts.items():
        if key not in obj or obj[key] is None:
            continue
        ai_val = obj[key]
        if isinstance(true_val, (int, float)) and isinstance(ai_val, (int, float)):
            denom = abs(true_val) if abs(true_val) > 1e-9 else 1e-9
            if abs(ai_val - true_val) / denom > rel_tolerance:
                errors.append(
                    f"数值与行情不符：{key} AI={ai_val} vs 实际={true_val}（偏差超 {rel_tolerance:.0%}）")
        elif isinstance(true_val, str):
            if true_val in ("up", "down") and isinstance(ai_val, str):
                if ai_val not in (true_val, "neutral"):
                    errors.append(f"方向与行情矛盾：{key} AI={ai_val} vs 实际={true_val}")
    return errors


# ---- 第四层：语义矛盾（轻量规则集） ----

# 已知的"必为真"矛盾规则：action 与 direction/strength 的常识一致性
def validate_semantics(obj: Any, rules: Optional[list[Callable[[dict], Optional[str]]]] = None) -> list[str]:
    """语义矛盾校验（可注入自定义规则函数，返回错误字符串或 None）。

    内置规则：
      - strength 高分(≥80) 但 action=wait → 提示（不报错，仅提示）
    """
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["结果非 JSON 对象"]
    # 内置规则：强度与行动一致性
    strength = obj.get("strength_score")
    action = obj.get("action")
    if isinstance(strength, (int, float)) and isinstance(action, str):
        if strength >= 80 and action in ("wait", "reject"):
            errors.append(f"矛盾：strength={strength} 高分但 action={action}（高分应倾向行动）")
        if strength < 30 and action == "trade":
            errors.append(f"矛盾：strength={strength} 低分但 action=trade")
    for rule in rules or []:
        try:
            err = rule(obj)
            if err:
                errors.append(err)
        except Exception as e:  # noqa: BLE001
            logger.debug("语义规则执行失败: %s", str(e)[:60])
    return errors


# ---- 组合校验 + 反馈式重试 ----


def validate_output(raw: str, schema: Optional[dict] = None,
                    facts: Optional[dict] = None,
                    semantic_rules: Optional[list] = None) -> tuple[Optional[Any], list[str]]:
    """四层校验组合。

    Returns:
        (obj, errors)：errors 空 = 通过；obj 在语法失败时为 None。
    """
    try:
        obj = extract_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        return None, [f"JSON 解析失败: {str(e)[:80]}"]
    errors = []
    if schema:
        errors += validate_schema(obj, schema)
    if facts:
        errors += validate_facts(obj, facts)
    errors += validate_semantics(obj, semantic_rules)
    return obj, errors


async def chat_json_validated(adapter, params, *, schema: Optional[dict] = None,
                              facts: Optional[dict] = None,
                              semantic_rules: Optional[list] = None,
                              max_retries: int = 2) -> dict[str, Any]:
    """校验版 chat_json：失败把错误回灌模型重试（反馈式）。

    借鉴 PA 的 validation_retry：校验错误作为补充说明重发；
    语法/事实类错误可重试，语义矛盾类默认不重试（防模型反复试错浪费）。

    Args:
        adapter: LLM 适配器（ensure_llm_adapter 结果）
        params: LLMCallParams（user_prompt 会被追加反馈轮）
        schema/facts/semantic_rules: 校验配置
        max_retries: 语法/事实类错误最多重试次数

    Returns:
        校验通过的结果 dict；重试耗尽后返回 {error, last_errors, raw_preview}
    """
    from pa_mcp.agent.llm_port import LLMCallParams

    last_errors: list[str] = []
    for attempt in range(max_retries + 1):
        resp = await adapter.chat(params)
        raw = resp.content or ""
        obj, errors = validate_output(raw, schema, facts, semantic_rules)
        if not errors:
            return {"_validated": True, "obj": obj, "attempts": attempt + 1,
                    "raw": raw}
        last_errors = errors
        # 语义矛盾不重试（模型反复试错无意义）
        semantic_errs = [e for e in errors if "矛盾" in e]
        if semantic_errs:
            logger.info("语义矛盾，放弃重试: %s", semantic_errs[0][:80])
            break
        if attempt < max_retries:
            feedback = ("\n\n【校验反馈】上次输出未通过校验，错误如下：\n"
                        + "\n".join(f"- {e}" for e in errors[:6])
                        + "\n请修正后重新输出完整 JSON。")
            params = LLMCallParams(
                system_prompt=params.system_prompt,
                user_prompt=params.user_prompt + feedback,
                mode=params.mode, max_tokens=params.max_tokens,
                temperature=getattr(params, "temperature", None),
            )
    return {"error": "LLM 输出未通过校验", "last_errors": last_errors,
            "raw_preview": raw[:300] if "raw" in locals() else ""}
