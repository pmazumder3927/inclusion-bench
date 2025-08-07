from __future__ import annotations

from typing import Dict

from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .openrouter_provider import OpenRouterProvider

PROVIDERS: Dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openrouter": OpenRouterProvider,
}
