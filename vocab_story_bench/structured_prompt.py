"""Structured output prompts for the vocabulary benchmark."""
from typing import List, Tuple, Dict, Any
import json


def build_structured_prompt(
    vocabulary: List[str],
    targets: List[str],
    language: str,
    story_length: int,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (system, user, response_format) for structured output.
    Conforms to OpenRouter Structured Outputs with JSON Schema.
    """
    vocab_str = ", ".join(vocabulary)
    targets_str = ", ".join(targets)

    system = (
        "You are a creative writer that generates stories as arrays of words. "
        "You must follow all lexical constraints strictly and output in the exact JSON format requested."
    )
    
    user = (
        f"Language: {language if language else 'unspecified'}\n"
        f"Vocabulary list (use ONLY these words):\n"
        f"{vocab_str}\n\n"
        f"Target words (must appear in the story): {targets_str}\n"
        f"Story length: approximately {story_length} words.\n\n"
        "Generate a coherent story following these rules:\n"
        "1. Use ONLY words from the vocabulary list\n"
        "2. Include ALL target words\n"
        "3. Output as a JSON array of words\n"
        "4. Each word should be a separate string in the array"
    )
    
    # JSON schema for structured output (OpenRouter structured outputs)
    # Reference: https://openrouter.ai/docs/features/structured-outputs
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "story_words",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "words": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Array of words forming the story"
                    }
                },
                "required": ["words"],
                "additionalProperties": False
            }
        }
    }
    
    return system, user, response_format


def parse_structured_response(response: str) -> List[str]:
    """Parse the structured JSON response to get word array."""
    # First, try to parse as JSON
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "words" in data:
            words = data["words"]
            # Ensure we have a list of strings
            if isinstance(words, list):
                return [str(w) for w in words]
            elif isinstance(words, str):
                # If words is a string, split it
                return words.split()
        elif isinstance(data, list):
            return [str(w) for w in data]
    except json.JSONDecodeError:
        pass
    
    # If the response looks like it contains JSON, try to extract it
    if '{"words"' in response or '["' in response:
        # Try to find JSON within the text
        import re
        json_pattern = r'(\{[^{}]*"words"\s*:\s*\[[^\]]*\]\}|\[[^\]]*\])'
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "words" in data:
                    return data["words"]
                elif isinstance(data, list):
                    return data
            except:
                continue
    
    # Final fallback: split as plain text
    # Remove any JSON-like characters and split
    cleaned = response.replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace('"', '').replace(',', ' ')
    words = cleaned.split()
    # Filter out empty strings and "words:" variations
    filtered = []
    for w in words:
        # Skip empty or words: variations
        if not w or w.lower() in ["words:", "words", "word:"]:
            continue
        # Remove "words:" prefix if attached to a word
        if w.startswith("words:"):
            w = w[6:]
        if w:
            filtered.append(w)
    return filtered