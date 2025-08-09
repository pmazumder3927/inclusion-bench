"""Unified OpenRouter provider for all models."""
import json
import os
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class UnifiedProvider:
    """Single provider for all models through OpenRouter."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        self.base_url = "https://openrouter.ai/api/v1"
        self.referrer = os.getenv("OPENROUTER_HTTP_REFERRER")
        self.title = os.getenv("OPENROUTER_X_TITLE")
    
    def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate structured output using any model through OpenRouter."""
        
        # Add JSON instruction to system prompt
        structured_system = f"""{system_prompt}

CRITICAL: You must respond with ONLY a valid JSON object in this exact format:
{{"words": ["word1", "word2", "word3", ...]}}

Rules:
1. The response must be parseable JSON with a "words" key containing an array of strings
2. Do NOT include any text before or after the JSON object
3. Do NOT include markdown code blocks or backticks
4. Start your response with {{ and end with }}
5. Each word in the array must be a separate string"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/inclusion-bench",
            "X-Title": "Vocabulary Inclusion Benchmark"
        }
        
        # Build the request
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": structured_system},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add any additional parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "verbosity"]:
            if key in kwargs:
                data[key] = kwargs[key]
        
        # Some models support JSON mode
        if self._supports_json_mode(model):
            data["response_format"] = {"type": "json_object"}
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=120
        )
        
        if response.status_code != 200:
            error_msg = f"OpenRouter API error: {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f" - {error_data['error']}"
            except:
                error_msg += f" - {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        
        # Handle potential error in response
        if "error" in result:
            raise Exception(f"OpenRouter error: {result['error']}")
        
        if "choices" not in result or len(result["choices"]) == 0:
            raise Exception(f"No response from model: {result}")
        
        content = result["choices"][0]["message"]["content"]
        
        # Clean up thinking tags if present
        if "<think>" in content:
            # Extract content after thinking tags
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
        
        # Try to extract JSON if there's other text
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end]
        
        return content
    
    def _supports_json_mode(self, model: str) -> bool:
        """Check if model supports response_format parameter."""
        # Models that support JSON mode
        json_capable = [
            "openai/gpt-4",
            "openai/gpt-3.5",
            "anthropic/claude-3",
            "google/gemini",
            "mistralai/mistral-large",
            "mistralai/mixtral"
        ]
        
        return any(model.startswith(prefix) for prefix in json_capable)