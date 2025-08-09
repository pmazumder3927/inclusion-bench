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
        # Prefer the Responses API (works across GPT-5 and newer) and read `text`
        # GPT-5 needs extra tokens for reasoning before the actual message
        gpt5_extra_tokens = 1500 if model.startswith("gpt-5") else 0
        args: dict[str, Any] = {
            "model": model,
            "input": f"{system}\n\n{user}" if system else user,
            "max_output_tokens": max_output_tokens + gpt5_extra_tokens,
        }
        if "temperature" in params:
            args["temperature"] = params["temperature"]
        if "top_p" in params:
            args["top_p"] = params["top_p"]
        if "reasoning_effort" in params:
            args["reasoning"] = {"effort": params["reasoning_effort"]}

        try:
            r = self.client.responses.create(**args)
            
            # Check for direct text attribute first (newest SDK format)
            # Note: r.text might be a config object, not the actual text
            if hasattr(r, 'text') and isinstance(r.text, str) and r.text.strip():
                return r.text.strip()
            
            # Check for message in the response (newer format)
            if hasattr(r, 'message') and r.message:
                if hasattr(r.message, 'content') and r.message.content:
                    return r.message.content.strip()
                if isinstance(r.message, str):
                    return r.message.strip()
            
            # Check for output/outputs attribute containing structured data
            outputs = getattr(r, "output", None) or getattr(r, "outputs", None)
            if outputs:
                # Handle list of outputs - GPT-5 returns reasoning first, then message
                if isinstance(outputs, list):
                    # First look for message type items (GPT-5 pattern)
                    for item in outputs:
                        if hasattr(item, 'type') and item.type == 'message':
                            if hasattr(item, 'content') and item.content:
                                content = item.content
                                if isinstance(content, str) and content.strip():
                                    return content.strip()
                                if isinstance(content, list):
                                    for block in content:
                                        # Handle ResponseOutputText objects
                                        if hasattr(block, 'text') and block.text:
                                            return block.text.strip()
                                        # Handle dict format
                                        if isinstance(block, dict) and 'text' in block:
                                            return block['text'].strip()
                    
                    # Fallback: check any item with text
                    for item in outputs:
                        # Check if item has direct text
                        if hasattr(item, 'text') and item.text and isinstance(item.text, str):
                            return item.text.strip()
                        
                        # Check for content blocks directly
                        if hasattr(item, 'content') and item.content:
                            content = item.content
                            if isinstance(content, list):
                                for block in content:
                                    if hasattr(block, 'text') and block.text:
                                        return block.text.strip()
                
                # Handle single output object
                elif hasattr(outputs, 'text') and outputs.text:
                    return outputs.text.strip()
            
            # Try to extract from model_dump if available
            if hasattr(r, 'model_dump'):
                data = r.model_dump()
                # Look for text in various possible locations
                if 'text' in data and data['text']:
                    return data['text'].strip()
                if 'message' in data and isinstance(data['message'], dict):
                    if 'content' in data['message'] and data['message']['content']:
                        return data['message']['content'].strip()
                if 'output' in data or 'outputs' in data:
                    outputs_data = data.get('output') or data.get('outputs')
                    if isinstance(outputs_data, list):
                        for item in outputs_data:
                            if isinstance(item, dict):
                                if 'text' in item and item['text']:
                                    return item['text'].strip()
                                if item.get('type') == 'message' and 'content' in item:
                                    content = item['content']
                                    if isinstance(content, str):
                                        return content.strip()
                                    if isinstance(content, list):
                                        for block in content:
                                            if isinstance(block, dict) and 'text' in block:
                                                return block['text'].strip()
            
        except Exception:
            # Fall back to chat completions
            pass

        # Final fallback – Chat Completions works for GPT-4.* and may proxy GPT-5
        # GPT-5 models require the responses API, so if we're here something went wrong
        if model.startswith("gpt-5"):
            # Try chat completions anyway as a last resort, but without temperature
            chat_args: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": max_output_tokens,
            }
            # Don't include temperature for GPT-5
            if "top_p" in params:
                chat_args["top_p"] = params["top_p"]
            if "presence_penalty" in params:
                chat_args["presence_penalty"] = params["presence_penalty"]
            if "frequency_penalty" in params:
                chat_args["frequency_penalty"] = params["frequency_penalty"]
            
            try:
                resp = self.client.chat.completions.create(**chat_args)
                content = resp.choices[0].message.content
                if content is None:
                    return ""
                return content.strip()
            except Exception:
                # If chat completions also fails for GPT-5, return empty
                return ""
        
        chat_args: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_output_tokens,
        }
        if "temperature" in params:
            chat_args["temperature"] = params["temperature"]
        if "top_p" in params:
            chat_args["top_p"] = params["top_p"]
        if "presence_penalty" in params:
            chat_args["presence_penalty"] = params["presence_penalty"]
        if "frequency_penalty" in params:
            chat_args["frequency_penalty"] = params["frequency_penalty"]

        resp = self.client.chat.completions.create(**chat_args)
        content = resp.choices[0].message.content
        if content is None:
            return ""
        return content.strip()
