from typing import List, Tuple


def build_prompt(
    language: str,
    vocabulary: List[str],
    target_words: List[str],
    desired_length_words: int,
) -> Tuple[str, str]:
    """
    Returns (system, user) messages.
    """
    vocab_str = ", ".join(vocabulary)
    targets_str = ", ".join(target_words)

    system = (
        "You write coherent short stories that strictly follow lexical constraints. "
        "Output only the story text, without headings or explanations."
    )
    user = (
        f"Language: {language if language else 'unspecified'}\n"
        f"Vocabulary list (use ONLY these words; no others are allowed):\n"
        f"{vocab_str}\n\n"
        f"Target words (must appear as standalone words): {targets_str}\n"
        f"Story length: about {desired_length_words} words.\n\n"
        "Rules:\n"
        "- Use ONLY words from the vocabulary.\n"
        "- Include ALL target words exactly as standalone words.\n"
        "- Do NOT include any other text besides the story."
    )
    return system, user
