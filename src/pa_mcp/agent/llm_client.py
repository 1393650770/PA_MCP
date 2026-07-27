# [AI:BEGIN]
# PA_MCP - Agent Layer: LLM Client
# Multi-provider LLM client with unified API, retry, and response caching.
# Configure via config/llm_config.json (gitignored).
# [AI:END]

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class ProviderConfig:
    api_key: str
    base_url: str
    fast_model: str
    deep_model: str
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout_seconds: int = 60


class LLMConfig:
    """Loads and manages LLM provider configuration from JSON."""

    def __init__(self, config_path: str = "config/llm_config.json") -> None:
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._active_provider: str = "anthropic"
        self._providers: dict[str, ProviderConfig] = {}
        self._token_budget: dict[str, int] = {}
        self._retry_config: dict[str, Any] = {}
        self._cache_config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load config from JSON file, fall back to example if missing."""
        config_file = self.config_path
        if not config_file.exists():
            example = Path("config/llm_config.example.json")
            if example.exists():
                logger.warning(
                    "LLM config not found, using example. Copy llm_config.example.json to llm_config.json and fill in API keys.",
                    path=str(config_file),
                )
                config_file = example
            else:
                logger.error("No LLM config found. Agent layer will not work.")
                return

        with open(config_file, "r", encoding="utf-8") as f:
            self._config = json.load(f)

        self._active_provider = self._config.get("active_provider", "anthropic")

        # Parse provider configs
        providers_raw = self._config.get("providers", {})
        for name, cfg in providers_raw.items():
            self._providers[name] = ProviderConfig(
                api_key=cfg.get("api_key", ""),
                base_url=cfg.get("base_url", ""),
                fast_model=cfg.get("models", {}).get("fast", ""),
                deep_model=cfg.get("models", {}).get("deep", ""),
                max_tokens=cfg.get("max_tokens", 4096),
                temperature=cfg.get("temperature", 0.1),
                timeout_seconds=cfg.get("timeout_seconds", 60),
            )

        # Token budget
        self._token_budget = self._config.get("token_budget", {
            "fast_analysis_max": 8000,
            "deep_analysis_max": 50000,
            "daily_scan_budget": 200000,
            "per_request_timeout": 90,
        })

        self._retry_config = self._config.get("retry", {
            "max_retries": 3,
            "backoff_base_seconds": 2,
            "backoff_multiplier": 2,
        })

        self._cache_config = self._config.get("cache", {
            "enable_response_cache": True,
            "cache_ttl_minutes": 60,
            "max_cached_responses": 500,
        })

        logger.info(
            "LLM config loaded",
            active_provider=self._active_provider,
            available_providers=list(self._providers.keys()),
        )

    @property
    def active_provider(self) -> Optional[ProviderConfig]:
        return self._providers.get(self._active_provider)

    @property
    def token_budget(self) -> dict[str, int]:
        return self._token_budget

    @property
    def retry_config(self) -> dict[str, Any]:
        return self._retry_config

    @property
    def cache_enabled(self) -> bool:
        return self._cache_config.get("enable_response_cache", True)


class LLMClient:
    """Multi-provider LLM client with unified chat interface.

    Supports: Anthropic, OpenAI-compatible, DeepSeek, Zhipu (智谱), Qwen (通义千问)

    Usage:
        client = LLMClient()
        response = client.chat(
            system_prompt="You are a financial analyst.",
            user_prompt="Analyze stock 000001",
            mode="fast",
        )
    """

    def __init__(self, config_path: str = "config/llm_config.json") -> None:
        self.config = LLMConfig(config_path)
        self._response_cache: dict[str, tuple[float, LLMResponse]] = {}

    def chat(
        self, system_prompt: str, user_prompt: str,
        mode: str = "fast", temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat completion request to the active LLM provider.

        Args:
            system_prompt: System instruction
            user_prompt: User message content
            mode: 'fast' (cheap model) or 'deep' (powerful model)
            temperature: Override default temperature
        """
        provider = self.config.active_provider
        if provider is None:
            return LLMResponse(
                content=json.dumps({
                    "error": "No LLM provider configured. Copy config/llm_config.example.json to config/llm_config.json and fill in API keys.",
                    "strength_score": 50,
                    "direction": "neutral",
                    "message": "LLM_NOT_CONFIGURED",
                }),
                model="none", provider="none",
            )

        model = provider.deep_model if mode == "deep" else provider.fast_model
        temp = temperature if temperature is not None else provider.temperature

        # Check cache
        cache_key = self._make_cache_key(provider.base_url, model, system_prompt, user_prompt, temp)
        if self.config.cache_enabled:
            cached = self._check_cache(cache_key)
            if cached:
                logger.debug("LLM cache hit", model=model)
                return cached

        # Build request body (OpenAI-compatible format for max compatibility)
        request_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": provider.max_tokens,
            "temperature": temp,
        }

        # Try with retries
        t0 = time.monotonic()
        last_error: Optional[str] = None

        max_retries = self.config.retry_config.get("max_retries", 3)
        backoff = self.config.retry_config.get("backoff_base_seconds", 2)
        multiplier = self.config.retry_config.get("backoff_multiplier", 2)

        for attempt in range(max_retries + 1):
            try:
                result = self._call_api(
                    provider.base_url, provider.api_key,
                    request_body, provider.timeout_seconds,
                )
                # Cache the result
                if self.config.cache_enabled:
                    self._cache_response(cache_key, result)
                return result
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    wait = backoff * (multiplier ** attempt)
                    logger.warning(
                        "LLM call failed, retrying",
                        attempt=attempt + 1, wait_seconds=wait,
                        error=last_error,
                    )
                    time.sleep(wait)

        logger.error("LLM call exhausted retries", error=last_error)
        return LLMResponse(
            content=json.dumps({
                "error": f"LLM call failed after {max_retries} retries: {last_error}",
                "strength_score": 50,
                "direction": "neutral",
                "message": "LLM_ERROR",
            }),
            model=model, provider=self.config._active_provider,
        )

    def _call_api(
        self, base_url: str, api_key: str,
        body: dict, timeout: int,
    ) -> LLMResponse:
        """Make the actual HTTP call to the LLM API."""
        import urllib.request
        import urllib.error

        url = f"{base_url.rstrip('/')}/chat/completions"
        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                latency = (time.monotonic() - t0) * 1000

                choice = result["choices"][0]
                content = choice["message"]["content"]
                usage = result.get("usage", {})

                return LLMResponse(
                    content=content,
                    model=result.get("model", body["model"]),
                    provider=self.config._active_provider,
                    tokens_input=usage.get("prompt_tokens", 0),
                    tokens_output=usage.get("completion_tokens", 0),
                    latency_ms=latency,
                    finish_reason=choice.get("finish_reason", "stop"),
                )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"HTTP {e.code}: {error_body[:500]}") from e
        except Exception as e:
            raise RuntimeError(str(e)) from e

    def _make_cache_key(self, *args: Any) -> str:
        raw = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _check_cache(self, key: str) -> Optional[LLMResponse]:
        if key in self._response_cache:
            timestamp, response = self._response_cache[key]
            ttl_seconds = self.config._cache_config.get("cache_ttl_minutes", 60) * 60
            if time.time() - timestamp < ttl_seconds:
                response.cached = True
                return response
            del self._response_cache[key]
        return None

    def _cache_response(self, key: str, response: LLMResponse) -> None:
        max_size = self.config._cache_config.get("max_cached_responses", 500)
        if len(self._response_cache) >= max_size:
            # Evict oldest
            oldest_key = min(self._response_cache, key=lambda k: self._response_cache[k][0])
            del self._response_cache[oldest_key]
        self._response_cache[key] = (time.time(), response)

    def chat_json(
        self, system_prompt: str, user_prompt: str,
        mode: str = "fast",
    ) -> dict[str, Any]:
        """Chat and parse response as JSON. Falls back gracefully."""
        response = self.chat(system_prompt, user_prompt, mode)
        try:
            # Extract JSON from response (may be wrapped in markdown)
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse LLM response as JSON", content_preview=response.content[:200])
            return {
                "error": "Failed to parse JSON from LLM response",
                "raw_content_preview": response.content[:500],
                "strength_score": 50,
                "direction": "neutral",
            }


# Global client instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
