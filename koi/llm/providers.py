"""Env-driven pydantic-ai Model factory: picks between direct Anthropic or OpenAI-compatible."""
import os
from typing import Optional

from pydantic_ai.models import Model

_DEFAULT_MODELS = {
    "openrouter": "deepseek/deepseek-chat",
    "anthropic": "claude-sonnet-4-6",
}


def build_model(
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Model:
    provider = (provider or os.environ.get("KOI_LLM_PROVIDER", "openrouter")).lower()
    if provider not in _DEFAULT_MODELS:
        raise ValueError(
            f"unknown KOI_LLM_PROVIDER: {provider!r} "
            f"(expected one of: {sorted(_DEFAULT_MODELS)})"
        )
    model_id = (
        model_id
        or os.environ.get("KOI_AGENT_MODEL")
        or _DEFAULT_MODELS[provider]
    )

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when KOI_LLM_PROVIDER=anthropic"
            )
        return AnthropicModel(model_id, provider=AnthropicProvider(api_key=key))

    # openrouter — also covers any OpenAI-compatible endpoint (vLLM, self-hosted, real OpenAI)
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    base_url = base_url or os.environ.get("KOI_BASE_URL", "https://openrouter.ai/api/v1")
    key = (
        api_key
        or os.environ.get("KOI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "KOI_API_KEY (or OPENROUTER_API_KEY) is required for KOI_LLM_PROVIDER=openrouter"
        )
    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(base_url=base_url, api_key=key),
    )
