from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelSpec:
    model: str  # OpenRouter model ID (e.g., "openai/gpt-4o", "qwen/qwq-32b:free")
    label: Optional[str] = None  # Short label for display
    params: Optional[dict[str, Any]] = None  # Model parameters like temperature

    @property
    def display_label(self) -> str:
        return self.label or self.model

    @staticmethod
    def parse_inline(token: str) -> "ModelSpec":
        # format: model or model:label
        parts = token.split(":", 1)
        if len(parts) == 1:
            model = parts[0]
            label = None
        else:
            model, label = parts
        return ModelSpec(
            model=model.strip(),
            label=(label.strip() if label else None),
            params=None,
        )