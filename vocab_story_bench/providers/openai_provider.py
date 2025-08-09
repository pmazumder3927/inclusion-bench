from __future__ import annotations

import os
from typing import Any, Tuple

from openai import OpenAI


class OpenAIProvider:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        model: str,
        system: str,
        user: str,
        max_output_tokens: int = 400,
        params: dict[str, Any] | None = None,
    ) -> str:
        params = params or {}
        # Use the unified Responses API for all OpenAI chat-capable models
        args: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_output_tokens": max_output_tokens,
        }
        # Optional sampling/behavior params if provided
        if "temperature" in params:
            args["temperature"] = params["temperature"]
        if "top_p" in params:
            args["top_p"] = params["top_p"]
        if "presence_penalty" in params:
            args["presence_penalty"] = params["presence_penalty"]
        if "frequency_penalty" in params:
            args["frequency_penalty"] = params["frequency_penalty"]
        # GPT-5 style reasoning controls
        if "reasoning_effort" in params:
            args["reasoning"] = {"effort": params["reasoning_effort"]}
        if "verbosity" in params:
            args["verbosity"] = params["verbosity"]

        resp = self.client.responses.create(**args)

        # Prefer SDK helper when available
        text: str | None = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        # Fallbacks for older/newer SDK response shapes
        try:
            first_item = resp.output[0]
            block = first_item.content[0]
            if getattr(block, "type", "") == "output_text" and getattr(block, "text", None):
                return block.text.strip()
        except Exception:
            pass
        # Final fallback – attempt chat.completions structure if present
        try:
            return resp.choices[0].message.content.strip()
        except Exception:
            raise RuntimeError("OpenAI response did not contain text output")
