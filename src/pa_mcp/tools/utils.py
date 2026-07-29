# [AI:BEGIN]
# PA_MCP - MCP Tool Utilities: annotations, error formatting, progress
# [AI:END]

from __future__ import annotations

from typing import Any, Callable, Optional


# ---- Tool Annotations ----

READ_ONLY = {"readOnlyHint": True}
DESTRUCTIVE = {"destructiveHint": True}
IDEMPOTENT = {"idempotentHint": True}


# ---- Error Response Formatting ----

def format_error(what: str, expected: str, example: str) -> str:
    """3-part error message: what went wrong + what's expected + example.

    Following MCP best practices: single, actionable correction.

    Usage:
        err = format_error(
            "Symbol 'APPL' not found",
            "6-digit A-share stock code like 000001",
            "Try: get_stock_info(symbol='000001')"
        )
    """
    return f"{what}. {expected}. {example}."


def not_found_error(entity: str, identifier: str) -> str:
    """Frame 'not found' positively: suggest closest match, not 'nothing found'.

    Anti-pattern avoided: "No results found for query X" (negative framing).
    Correct: "Here are the closest matches for query X" (actionable).
    """
    return format_error(
        f"{entity} '{identifier}' not found",
        f"Provide a valid {entity} identifier",
        f"Try searching with a shorter keyword, e.g. search_stock(keyword='{identifier[:3]}')"
    )


# ---- Tool Annotation Decorator Factory ----

def tool_kwargs(
    name: str,
    read_only: bool = False,
    destructive: bool = False,
    idempotent: bool = False,
    extra_annotations: Optional[dict[str, bool]] = None,
) -> dict[str, Any]:
    """Build keyword arguments for @mcp.tool() decorator.

    Args:
        name: Tool name (with pa_ prefix)
        read_only: Tool only reads data, no writes
        destructive: Tool may perform destructive operations
        idempotent: Repeated calls with same args have no extra effect
        extra_annotations: Additional annotations to merge
    """
    annotations = {}
    if read_only:
        annotations["readOnlyHint"] = True
    if destructive:
        annotations["destructiveHint"] = True
    if idempotent:
        annotations["idempotentHint"] = True
    if extra_annotations:
        annotations.update(extra_annotations)

    result: dict[str, Any] = {"name": name}
    if annotations:
        result["annotations"] = annotations
    return result
