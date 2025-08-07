from __future__ import annotations

import os
from typing import Any

import httpx


class OpenRouterProvider:
    def __init__(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        self.x_title = os.getenv("OPENROUTER_X_TITLE")

    def generate(self, model: str, system: str, user: str, max_output_tokens: int = 400) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_output_tokens,
            "temperature": 0.7,
        }
        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=120) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
