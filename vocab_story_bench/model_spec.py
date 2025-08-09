from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # openai | anthropic | openrouter
    model: str
    label: Optional[str] = None
    params: Optional[dict[str, Any]] = None

    @property
    def display_label(self) -> str:
        return self.label or f"{self.provider}:{self.model}"

    @staticmethod
    def parse_inline(token: str) -> "ModelSpec":
        # format: provider:model or provider:model:label
        parts = token.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid model token: {token}")
        if len(parts) == 2:
            provider, model = parts
            label = None
        else:
            provider, model, label = parts[0], parts[1], ":".join(parts[2:])
        return ModelSpec(
            provider=provider.strip(),
            model=model.strip(),
            label=(label.strip() if label else None),
            params=None,
        )
