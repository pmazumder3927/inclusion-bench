"""Anthropic provider with structured output support."""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from anthropic import Anthropic


class StructuredAnthropicProvider:
    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = Anthropic(api_key=api_key)

    def generate_structured(
        self,
        model: str,
        system: str,
        user: str,
        response_format: Dict[str, Any],
        max_output_tokens: int = 400,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate with structured output support for Anthropic models."""
        params = params or {}
        
        # Add JSON instruction to the user prompt
        user_with_json = (
            f"{user}\n\n"
            "Output your response as a JSON object with this structure:\n"
            '{"words": ["word1", "word2", "word3", ...]}\n'
            "where each word is a separate string in the array."
        )
        
        # Build the request
        args: Dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": user_with_json}],
            "max_tokens": max_output_tokens,
        }
        
        # Add optional parameters
        if "temperature" in params:
            args["temperature"] = params["temperature"]
        if "top_p" in params:
            args["top_p"] = params["top_p"]
        if "top_k" in params:
            args["top_k"] = params["top_k"]
        
        # Make the API call
        response = self.client.messages.create(**args)
        
        # Extract the content
        if response.content:
            # Anthropic returns a list of content blocks
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
                elif isinstance(block, dict) and 'text' in block:
                    return block['text']
        
        # Fallback to empty array
        return '{"words": []}'