from __future__ import annotations

import os
from typing import Iterable, List, Set

from openai import OpenAI


def discover_openai_chat_models(prefixes: Iterable[str]) -> List[str]:
    """Return OpenAI model ids that start with any given prefix.
    Requires OPENAI_API_KEY. Filters out embeddings and TTS models heuristically.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    client = OpenAI(api_key=api_key)
    pref = [p.strip() for p in prefixes if p and p.strip()]
    if not pref:
        return []

    models = client.models.list()
    found: Set[str] = set()
    for m in models.data:
        mid = getattr(m, "id", "")
        if not mid:
            continue
        if any(mid.startswith(p) for p in pref):
            lname = mid.lower()
            if any(x in lname for x in ["embedding", "embeddings", "tts", "audio", "whisper", "realtime", "omni"]):
                continue
            found.add(mid)
    return sorted(found)
