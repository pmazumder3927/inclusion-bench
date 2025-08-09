from __future__ import annotations

from typing import Dict

PROVIDERS: Dict[str, type] = {}

# Structured providers
try:
    from .structured_openai import StructuredOpenAIProvider
    PROVIDERS["openai"] = StructuredOpenAIProvider
except Exception:
    pass

try:
    from .structured_anthropic import StructuredAnthropicProvider
    PROVIDERS["anthropic"] = StructuredAnthropicProvider
except Exception:
    pass