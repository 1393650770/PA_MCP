# [AI:BEGIN]
# PA_MCP - Agent Layer: Anthropic SDK Adapter
#
# Uses the official anthropic Python SDK (Messages API) for Claude.
# Does NOT use OpenAI-compatible /chat/completions shim.
#
# Requires: pip install anthropic
# [AI:END]

from __future__ import annotations

import json
import time
from typing import Any, Optional

import structlog

from pa_mcp.agent.llm_port import LLMPort, LLMCallParams, LLMResponse

logger = structlog.get_logger(__name__)


class AnthropicAdapter(LLMPort):
    """LLM adapter using the official Anthropic Python SDK.

    Authenticates via ANTHROPIC_API_KEY env var (default),
    ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile.
    """

    def __init__(
        self,
        fast_model: str = "claude-sonnet-5",
        deep_model: str = "claude-opus-5",
        max_tokens: int = 4096,
        timeout_seconds: int = 60,
        api_key: Optional[str] = None,
    ) -> None:
        self._fast_model = fast_model
        self._deep_model = deep_model
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._api_key = api_key  # None = use SDK default credential resolution

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_model(self, mode: str) -> str:
        if mode == "deep":
            return self._deep_model
        return self._fast_model

    async def chat(self, params: LLMCallParams) -> LLMResponse:
        """Send a message via Anthropic Messages API."""
        import anthropic

        model = params.model or self._get_model(params.mode)
        max_tokens = params.max_tokens or self._max_tokens

        client_kwargs: dict[str, Any] = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key

        # Use async client
        client = anthropic.AsyncAnthropic(**client_kwargs)

        t0 = time.monotonic()

        try:
            # Build system prompt
            system = params.system_prompt or None

            # Build messages (Anthropic Messages API format)
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": params.user_prompt},
            ]

            # Request params
            request_params: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "timeout": params.timeout_seconds or self._timeout,
            }

            if system:
                request_params["system"] = system

            # Only send temperature if explicitly requested (Claude Opus 5
            # rejects non-default temperature — the SDK itself handles this
            # by not sending it, but we explicitly omit it by default).
            if params.temperature is not None:
                # Note: Claude Opus 5/4.8/4.7 reject temperature entirely.
                # We set it only when the user explicitly asks, and trust
                # the SDK to raise if the model doesn't support it.
                try:
                    request_params["temperature"] = params.temperature
                except Exception:
                    pass

            # Structured output via output_config.format
            if params.response_format:
                request_params["output_config"] = {
                    "format": params.response_format,
                }

            # Use adaptive thinking by default (Claude 4.6+)
            request_params["thinking"] = {"type": "adaptive"}

            # Make the API call
            response = await client.messages.create(**request_params)

            latency = (time.monotonic() - t0) * 1000

            # Extract text from content blocks
            text_parts: list[str] = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    text_parts.append(block.text)

            content = "\n".join(text_parts)

            # Usage
            usage = response.usage
            tokens_input = usage.input_tokens if usage else 0
            tokens_output = usage.output_tokens if usage else 0

            return LLMResponse(
                content=content,
                model=response.model,
                provider="anthropic",
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                latency_ms=latency,
                finish_reason=str(response.stop_reason) if response.stop_reason else "stop",
                request_id=getattr(response, "_request_id", ""),
            )

        except anthropic.APIStatusError as e:
            logger.error(
                "Anthropic API error",
                status=e.status_code,
                message=e.message,
                request_id=getattr(e, "request_id", ""),
            )
            return LLMResponse(
                content=json.dumps({
                    "error": f"Anthropic API error ({e.status_code}): {e.message}",
                    "strength_score": 50,
                    "direction": "neutral",
                    "message": "LLM_ERROR",
                }),
                model=model,
                provider="anthropic",
            )
        except Exception as e:
            logger.error("Anthropic call failed", error=str(e))
            return LLMResponse(
                content=json.dumps({
                    "error": f"Anthropic call failed: {str(e)}",
                    "strength_score": 50,
                    "direction": "neutral",
                    "message": "LLM_ERROR",
                }),
                model=model,
                provider="anthropic",
            )
        finally:
            await client.close()

    async def chat_json(self, params: LLMCallParams) -> dict[str, Any]:
        """Chat and parse response as JSON. Falls back gracefully."""
        response = await self.chat(params)
        try:
            content = response.content
            # Extract JSON from response (may be wrapped in markdown)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning(
                "Failed to parse Anthropic response as JSON",
                content_preview=response.content[:200],
            )
            return {
                "error": "Failed to parse JSON from LLM response",
                "raw_content_preview": response.content[:500],
                "strength_score": 50,
                "direction": "neutral",
            }
