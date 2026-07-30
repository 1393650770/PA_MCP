# [AI:BEGIN]
# PA_MCP - Agent Layer: LLM Client (Backward Compatibility Wrapper)
#
# DEPRECATED: This file is kept for backward compatibility only.
# New code should use:
#   from pa_mcp.agent.llm_port import get_llm_adapter, LLMCallParams
#   from pa_mcp.agent.llm_factory import init_llm_adapter
#
# Internally, this wrapper delegates to the active port adapter.
# The old multi-provider HTTP client that sent everything to
# /chat/completions has been replaced with proper per-provider adapters
# (Anthropic Messages API, OpenAI-compatible, etc.).
# [AI:END]

from __future__ import annotations

import warnings
from typing import Any, Optional

import structlog

from pa_mcp.agent.llm_port import get_llm_adapter, LLMCallParams, LLMResponse

logger = structlog.get_logger(__name__)


# Re-export for backward compatibility
__all__ = ["LLMResponse", "LLMClient", "get_llm_client"]


class LLMClient:
    """Backward-compatible wrapper around the adapter-based LLM system.

    DEPRECATED: Prefer using get_llm_adapter() directly with LLMCallParams.

    This class delegates to the active provider adapter (Anthropic SDK,
    OpenAI-compatible, etc.) rather than making raw HTTP calls.
    """

    def __init__(self, config_path: str = "config/llm_config.json") -> None:
        warnings.warn(
            "LLMClient is deprecated. Use get_llm_adapter() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Trigger adapter initialization if not already done
        adapter = get_llm_adapter()
        if adapter is None:
            from pa_mcp.agent.llm_factory import init_llm_adapter
            init_llm_adapter(config_path)

    def chat(
        self, system_prompt: str, user_prompt: str,
        mode: str = "fast", temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat completion request via the active adapter."""
        adapter = get_llm_adapter()
        if adapter is None:
            import json
            return LLMResponse(
                content=json.dumps({
                    "error": "No LLM provider configured. Copy config/llm_config.example.json to config/llm_config.json and fill in API keys.",
                    "strength_score": 50,
                    "direction": "neutral",
                    "message": "LLM_NOT_CONFIGURED",
                }),
                model="none",
                provider="none",
            )

        import asyncio

        params = LLMCallParams(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode=mode,
            temperature=temperature,
        )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running in async context — create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, adapter.chat(params))
                    return future.result(timeout=120)
            else:
                return asyncio.run(adapter.chat(params))
        except RuntimeError:
            return asyncio.run(adapter.chat(params))

    def chat_json(
        self, system_prompt: str, user_prompt: str,
        mode: str = "fast",
    ) -> dict[str, Any]:
        """Chat and parse response as JSON. Falls back gracefully."""
        response = self.chat(system_prompt, user_prompt, mode)
        import json
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning(
                "Failed to parse LLM response as JSON",
                content_preview=response.content[:200],
            )
            return {
                "error": "Failed to parse JSON from LLM response",
                "raw_content_preview": response.content[:500],
                "strength_score": 50,
                "direction": "neutral",
            }


# Global client instance (deprecated)
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client (deprecated).

    Prefer get_llm_adapter() for new code.
    """
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
