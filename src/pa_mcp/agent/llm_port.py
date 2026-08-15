# [AI:BEGIN]
# PA_MCP - Agent Layer: LLM Port (Provider-Neutral Abstract Interface)
#
# Architecture:
#   LLMPort (abstract)  ←  defines the contract
#     ├── AnthropicAdapter  (official anthropic SDK, Messages API)
#     ├── OpenAICompatibleAdapter  (OpenAI-compatible /chat/completions)
#     └── future adapters (DeepSeek, Zhipu, Qwen, etc.)
#
# Application code depends only on LLMPort — never on a specific SDK.
# The Factory in config root selects the adapter.
# [AI:END]

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: str
    model: str
    provider: str
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    finish_reason: str = "stop"
    request_id: str = ""


@dataclass
class LLMCallParams:
    """Unified call parameters for all providers.

    Individual adapters map these to their native API shape.
    If a provider doesn't support a param (e.g. temperature on
    Claude Opus 5), the adapter should omit or error accordingly.
    """
    system_prompt: str = ""
    user_prompt: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: Optional[float] = None  # None = use provider default
    timeout_seconds: int = 60
    mode: str = "fast"  # "fast" or "deep"
    response_format: Optional[dict[str, Any]] = None  # JSON schema for structured output


class LLMPort(abc.ABC):
    """Abstract port for LLM providers.

    All application code depends on this interface.
    Concrete adapters implement provider-specific logic.
    """

    @abc.abstractmethod
    async def chat(self, params: LLMCallParams) -> LLMResponse:
        """Send a message and get a text response."""
        ...

    @abc.abstractmethod
    async def chat_json(self, params: LLMCallParams) -> dict[str, Any]:
        """Send a message and parse response as JSON dict."""
        ...

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'anthropic', 'openai')."""
        ...


# ---- Adapter Factory ----

_adapter: Optional[LLMPort] = None


def register_adapter(adapter: Optional[LLMPort]) -> None:
    """Set the global LLM adapter (called once at startup).

    None 会清除当前 adapter（测试隔离/热切换时使用）。
    """
    global _adapter
    _adapter = adapter
    if adapter is None:
        logger.info("LLM adapter cleared")
        return
    logger.info("LLM adapter registered", provider=adapter.provider_name)


def get_llm_adapter() -> Optional[LLMPort]:
    """Get the current LLM adapter, or None if not configured."""
    return _adapter


# Re-export for backward compatibility
LLMClient = None  # Deprecated — use get_llm_adapter() instead
