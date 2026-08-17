# [AI:BEGIN]
# PA_MCP - Agent Layer: LLM Adapter Factory
#
# Reads llm_config.json and creates the appropriate adapter.
# The rest of the application calls get_llm_adapter() without
# knowing which provider is active.
#
# Usage:
#   from pa_mcp.agent.llm_factory import init_llm_adapter
#   init_llm_adapter()  # called once at startup
#   adapter = get_llm_adapter()
# [AI:END]

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import structlog

from pa_mcp.agent.llm_port import LLMPort, get_llm_adapter, register_adapter

logger = structlog.get_logger(__name__)


def ensure_llm_adapter() -> Optional[LLMPort]:
    """获取 LLM adapter；单例为空时主动从 config/llm_config.json 初始化。

    所有 LLM 触点统一用本函数（chat/prediction/sector_rotation/
    methodology_guide/ai_report/orchestrator…），避免「配置了 LLM 却
    静默降级确定性」：进程内单例只在 init_llm_adapter 后被设置，而多数
    入口（UI 按钮/MCP 工具）启动时并不初始化。

    仍可能返回 None（未配置或初始化失败），调用方照常降级。
    """
    from pa_mcp.agent.llm_port import get_llm_adapter
    adapter = get_llm_adapter()
    if adapter is None:
        from pa_mcp.config import PROJECT_ROOT
        adapter = init_llm_adapter(
            str(PROJECT_ROOT / "config" / "llm_config.json"))
    return adapter


def init_llm_adapter(config_path: Optional[str] = None) -> Optional[LLMPort]:
    """Initialize the LLM adapter from config.

    Reads config/llm_config.json (or the example if missing).
    Creates the correct adapter based on active_provider.

    Returns the adapter, or None if no config/keys are found.
    """
    if config_path is None:
        config_path = "config/llm_config.json"

    config_file = Path(config_path)
    if not config_file.exists():
        example = Path("config/llm_config.example.json")
        if example.exists():
            logger.warning(
                "LLM config not found, using example. "
                "Copy llm_config.example.json to llm_config.json and fill in API keys.",
                path=str(config_file),
            )
            config_file = example
        else:
            logger.error("No LLM config found. Agent analysis will not work.")
            return None

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    active_provider = config.get("active_provider", "anthropic")
    providers = config.get("providers", {})

    if active_provider not in providers:
        logger.error(
            "Active provider not found in config",
            active_provider=active_provider,
            available=list(providers.keys()),
        )
        return None

    provider_cfg = providers[active_provider]
    api_key = provider_cfg.get("api_key", "") or os.environ.get(
        f"{active_provider.upper()}_API_KEY", ""
    )
    base_url = provider_cfg.get("base_url", "")
    models = provider_cfg.get("models", {})

    if active_provider == "anthropic":
        from pa_mcp.agent.llm_anthropic import AnthropicAdapter

        if base_url and "/chat/completions" in base_url:
            logger.error(
                "Anthropic provider configured with OpenAI-compatible base_url. "
                "Anthropic uses the Messages API, not /chat/completions. "
                "Remove base_url from config or switch to anthropic SDK adapter. "
                "Falling back to OpenAI-compatible adapter — this will NOT work with Anthropic."
            )
            # Fall through to OpenAI-compatible below

        adapter = AnthropicAdapter(
            fast_model=models.get("fast", "claude-sonnet-5"),
            deep_model=models.get("deep", "claude-opus-5"),
            max_tokens=provider_cfg.get("max_tokens", 4096),
            timeout_seconds=provider_cfg.get("timeout_seconds", 60),
            api_key=api_key if api_key else None,  # None = use SDK default creds
        )
    else:
        # OpenAI-compatible 协议通用回退：
        # OpenAI, DeepSeek, Zhipu, Qwen, 豆包(doubao/ark), 以及任意
        # 配置了 base_url 的 OpenAI 兼容供应商
        from pa_mcp.agent.llm_openai_compat import OpenAICompatibleAdapter

        if not base_url:
            logger.warning(
                f"No base_url for provider '{active_provider}'. "
                "Using default. Set base_url in llm_config.json."
            )

        adapter = OpenAICompatibleAdapter(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            fast_model=models.get("fast", "gpt-4o-mini"),
            deep_model=models.get("deep", "gpt-4o"),
            max_tokens=provider_cfg.get("max_tokens", 4096),
            timeout_seconds=provider_cfg.get("timeout_seconds", 60),
        )

    register_adapter(adapter)
    logger.info(
        "LLM adapter initialized",
        provider=active_provider,
        fast_model=models.get("fast", ""),
        deep_model=models.get("deep", ""),
    )
    return adapter
