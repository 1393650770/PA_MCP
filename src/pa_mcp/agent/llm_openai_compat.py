# [AI:BEGIN]
# PA_MCP - Agent Layer: OpenAI-Compatible Adapter
#
# Uses OpenAI-compatible /chat/completions protocol.
# Works with OpenAI, DeepSeek, Zhipu (智谱), Qwen (通义千问),
# and any other provider implementing the OpenAI-compatible API.
# [AI:END]

from __future__ import annotations

import json
import time
from typing import Any, Optional

import structlog

from pa_mcp.agent.llm_port import LLMPort, LLMCallParams, LLMResponse

logger = structlog.get_logger(__name__)


class OpenAICompatibleAdapter(LLMPort):
    """LLM adapter for any OpenAI-compatible /chat/completions API.

    Uses urllib (stdlib) — no external HTTP dependency needed.
    Replace with httpx/requests if preferred.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        fast_model: str = "gpt-4o-mini",
        deep_model: str = "gpt-4o",
        max_tokens: int = 4096,
        timeout_seconds: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._fast_model = fast_model
        self._deep_model = deep_model
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    def _get_model(self, mode: str) -> str:
        if mode == "deep":
            return self._deep_model
        return self._fast_model

    async def chat(self, params: LLMCallParams) -> LLMResponse:
        """Send a chat completion request."""
        import urllib.request
        import urllib.error

        model = params.model or self._get_model(params.mode)
        temperature = params.temperature if params.temperature is not None else 0.1

        request_body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": params.system_prompt},
                {"role": "user", "content": params.user_prompt},
            ],
            "max_tokens": params.max_tokens or self._max_tokens,
            "temperature": temperature,
        }

        if params.response_format:
            request_body["response_format"] = params.response_format

        url = f"{self._base_url}/chat/completions"
        data = json.dumps(request_body).encode("utf-8")

        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")

        t0 = time.monotonic()
        timeout = params.timeout_seconds or self._timeout

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                latency = (time.monotonic() - t0) * 1000

                choice = result["choices"][0]
                content = choice["message"]["content"]
                usage = result.get("usage", {})

                return LLMResponse(
                    content=content,
                    model=result.get("model", model),
                    provider="openai_compatible",
                    tokens_input=usage.get("prompt_tokens", 0),
                    tokens_output=usage.get("completion_tokens", 0),
                    latency_ms=latency,
                    finish_reason=choice.get("finish_reason", "stop"),
                )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            logger.error("OpenAI-compatible API error", status=e.code, body=error_body[:500])
            return LLMResponse(
                content=json.dumps({
                    "error": f"HTTP {e.code}: {error_body[:300]}",
                    "strength_score": 50,
                    "direction": "neutral",
                    "message": "LLM_ERROR",
                }),
                model=model,
                provider="openai_compatible",
            )
        except Exception as e:
            logger.error("OpenAI-compatible call failed", error=str(e))
            return LLMResponse(
                content=json.dumps({
                    "error": f"LLM call failed: {str(e)}",
                    "strength_score": 50,
                    "direction": "neutral",
                    "message": "LLM_ERROR",
                }),
                model=model,
                provider="openai_compatible",
            )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从 LLM 响应中稳健提取 JSON。

        容忍三种常见形态：纯 JSON、```json 包裹、开头无 ``` 但结尾有 ```、
        前后夹带解释文字。提取第一个 { 到最后一个 } 再解析。
        """
        import re
        # 1) markdown 代码块剥壳（```json ... ``` 或 ``` ... ```）
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if m:
            text = m.group(1)
        # 2) 提取首尾大括号之间的内容（容忍前后杂质/结尾多余 ```）
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            text = text[s:e + 1]
        return json.loads(text.strip())

    async def chat_json(self, params: LLMCallParams) -> dict[str, Any]:
        """Chat and parse response as JSON dict."""
        response = await self.chat(params)
        try:
            return self._extract_json(response.content)
        except (json.JSONDecodeError, IndexError, ValueError):
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
