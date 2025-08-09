#!/usr/bin/env python3
"""Debug script to test GPT-5 API directly."""

import json
import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found")

# Simple test vocabulary
vocabulary = ["the", "cat", "dog", "sat", "on", "mat", "and", "ran", "jumped", "happy", "was", "very", "a", "big", "small"]
targets = ["cat", "mat", "happy"]

# Build prompt
system_prompt = """You are a creative writer that generates stories as arrays of words. 
You must follow all lexical constraints strictly and output in the exact JSON format requested.

CRITICAL: You must respond with ONLY a valid JSON object in this exact format:
{"words": ["word1", "word2", "word3", ...]}

Rules:
1. The response must be parseable JSON with a "words" key containing an array of strings
2. Do NOT include any text before or after the JSON object
3. Do NOT include markdown code blocks or backticks
4. Start your response with { and end with }
5. Each word in the array must be a separate string"""

user_prompt = f"""Vocabulary list (use ONLY these words):
{', '.join(vocabulary)}

Target words (must appear in the story): {', '.join(targets)}
Story length: approximately 30 words.

Generate a coherent story following these rules:
1. Use ONLY words from the vocabulary list
2. Include ALL target words
3. Output as a JSON array of words
4. Each word should be a separate string in the array"""

# Make request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/inclusion-bench",
    "X-Title": "Vocabulary Inclusion Benchmark"
}

data = {
    "model": "openai/gpt-oss-120b",  # Testing GPT-OSS
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "max_tokens": 2000,  # Much higher limit
    "temperature": 0.2
}

print("Making request to OpenRouter API...")
print(f"Model: {data['model']}")
print(f"Vocabulary size: {len(vocabulary)}")
print(f"Target words: {targets}")
print()

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=data,
    timeout=30
)

print(f"Status code: {response.status_code}")
print()

if response.status_code == 200:
    result = response.json()
    if "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"]
        print("Raw response:")
        print(content)
        print()
        
        # Try to parse JSON
        try:
            # Clean up thinking tags if present
            if "<think>" in content:
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
            
            # Try to extract JSON if there's other text
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]
            
            parsed = json.loads(content)
            print("Parsed JSON:")
            print(json.dumps(parsed, indent=2))
            
            if "words" in parsed:
                words = parsed["words"]
                print(f"\nExtracted {len(words)} words")
                print(f"Words: {' '.join(words[:10])}..." if len(words) > 10 else f"Words: {' '.join(words)}")
                
                # Check vocabulary compliance
                vocab_set = set(vocabulary)
                invalid_words = [w for w in words if w not in vocab_set]
                if invalid_words:
                    print(f"Invalid words (not in vocabulary): {invalid_words}")
                
                # Check targets
                found_targets = [t for t in targets if t in words]
                print(f"Found targets: {found_targets}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
    else:
        print("No choices in response:")
        print(json.dumps(result, indent=2))
else:
    print(f"Error: {response.status_code}")
    try:
        error_data = response.json()
        print(json.dumps(error_data, indent=2))
    except:
        print(response.text)