from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Set, Dict

import regex as re

_WORD_RE = re.compile(r"\p{L}+")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def tokenize_words(text: str) -> List[str]:
    """Tokenize into sequences of Unicode letters; hyphens and punctuation are separators."""
    norm = normalize_text(text)
    return _WORD_RE.findall(norm)


@dataclass
class ValidationResult:
    only_vocab: bool
    all_targets_present: bool
    oov_words: Set[str]
    missing_targets: Set[str]
    total_tokens: int


def validate_story(
    story: str,
    vocabulary: Iterable[str],
    target_words: Iterable[str],
) -> ValidationResult:
    vocab_set: Set[str] = {normalize_text(w).strip() for w in vocabulary if w.strip()}
    target_set: Set[str] = {normalize_text(w).strip() for w in target_words if w.strip()}

    tokens = tokenize_words(story)
    token_set: Set[str] = set(tokens)

    oov = {t for t in token_set if t and t not in vocab_set}
    missing = {t for t in target_set if t not in token_set}

    return ValidationResult(
        only_vocab=len(oov) == 0,
        all_targets_present=len(missing) == 0,
        oov_words=oov,
        missing_targets=missing,
        total_tokens=len(tokens),
    )
