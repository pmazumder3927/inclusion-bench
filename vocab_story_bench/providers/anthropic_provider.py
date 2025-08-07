from __future__ import annotations

import os
from typing import Tuple

import anthropic


class AnthropicProvider:
    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, model: str, system: str, user: str, max_output_tokens: int = 400) -> str:
        resp = self.client.messages.create(
            model=model,
            system=system,
            max_tokens=max_output_tokens,
            temperature=0.7,
            messages=[
                {"role": "user", "content": user},
            ],
        )
        # anthropic returns content as list of blocks
        parts = []
        for block in resp.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()
