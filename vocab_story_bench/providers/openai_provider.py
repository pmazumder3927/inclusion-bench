from __future__ import annotations

import os
from typing import Tuple

from openai import OpenAI


class OpenAIProvider:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

    def generate(self, model: str, system: str, user: str, max_output_tokens: int = 400) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_completion_tokens=max_output_tokens,
        )
        return resp.choices[0].message.content.strip()
