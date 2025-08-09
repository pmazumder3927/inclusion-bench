"""OpenAI provider with structured output support."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


class StructuredOpenAIProvider:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

    def generate_structured(
        self,
        model: str,
        system: str,
        user: str,
        response_format: Dict[str, Any],
        max_output_tokens: int = 400,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate with structured output support."""
        params = params or {}
        
        # For GPT-5, use responses API with structured output
        if model.startswith("gpt-5"):
            return self._generate_gpt5_structured(
                model, system, user, response_format, max_output_tokens, params
            )
        
        # For GPT-4 models, use chat completions with response_format
        return self._generate_gpt4_structured(
            model, system, user, response_format, max_output_tokens, params
        )
    
    def _generate_gpt5_structured(
        self,
        model: str,
        system: str,
        user: str,
        response_format: Dict[str, Any],
        max_output_tokens: int,
        params: Dict[str, Any],
    ) -> str:
        """Generate structured output for GPT-5 models."""
        # GPT-5 needs extra tokens for reasoning
        gpt5_extra_tokens = 1500
        
        # Build the prompt with JSON instruction
        full_prompt = f"{system}\n\n{user}\n\nRespond with a JSON object containing a 'words' array."
        
        args: Dict[str, Any] = {
            "model": model,
            "input": full_prompt,
            "max_output_tokens": max_output_tokens + gpt5_extra_tokens,
        }
        
        if "temperature" in params:
            args["temperature"] = params["temperature"]
        if "reasoning_effort" in params:
            args["reasoning"] = {"effort": params["reasoning_effort"]}
        
        try:
            r = self.client.responses.create(**args)
            
            # Extract text from GPT-5 response
            outputs = getattr(r, "output", None)
            if outputs and isinstance(outputs, list):
                for item in outputs:
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            content = item.content
                            if isinstance(content, list):
                                for block in content:
                                    if hasattr(block, 'text') and block.text:
                                        return block.text.strip()
        except Exception:
            pass
        
        # Fallback
        return '{"words": []}'
    
    def _generate_gpt4_structured(
        self,
        model: str,
        system: str,
        user: str,
        response_format: Dict[str, Any],
        max_output_tokens: int,
        params: Dict[str, Any],
    ) -> str:
        """Generate structured output for GPT-4 models."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        args: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_output_tokens,
        }
        
        # Add response_format for models that support it
        if model in ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]:
            args["response_format"] = response_format
        else:
            # For older models, add JSON instruction to prompt
            messages[1]["content"] += "\n\nRespond with a JSON object containing a 'words' array."
        
        if "temperature" in params:
            args["temperature"] = params["temperature"]
        if "top_p" in params:
            args["top_p"] = params["top_p"]
        
        resp = self.client.chat.completions.create(**args)
        content = resp.choices[0].message.content
        
        if content is None:
            return '{"words": []}'
        
        return content.strip()