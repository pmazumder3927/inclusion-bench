from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set, Tuple

from dotenv import load_dotenv

from .model_spec import ModelSpec
from .runner import build_models_from_inline, build_models_from_yaml, load_lines, run_benchmark
from .discovery import discover_openai_chat_models


def main() -> None:
    # Load environment variables from .env if present
    load_dotenv()

    ap = argparse.ArgumentParser(description="Vocab-only story benchmark")
    ap.add_argument("--vocab-file", required=True, help="Path to newline-separated vocabulary list")
    ap.add_argument("--targets-file", required=False, help="Path to newline-separated target words")
    ap.add_argument("--targets", nargs="*", help="Inline target words (use instead of --targets-file)")
    ap.add_argument("--language", default="", help="Language hint for the story")
    ap.add_argument("--story-words", type=int, default=150, help="Approximate desired story length in words")
    ap.add_argument("--trials", type=int, default=3, help="Trials per model")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+", help="Inline models like 'openai:gpt-4.1' 'anthropic:claude-sonnet-4-20250514'")
    group.add_argument("--models-config", help="YAML file listing models")

    # Dynamic discovery options
    ap.add_argument("--openai-prefixes", nargs="*", default=[], help="Auto-include all OpenAI models starting with these prefixes (e.g., gpt-5)")

    ap.add_argument("--output-dir", default=None, help="Directory to write outputs; defaults to outputs/{timestamp}")

    args = ap.parse_args()

    vocabulary = load_lines(args.vocab_file)
    if args.targets_file:
        target_words = load_lines(args.targets_file)
    else:
        target_words = args.targets or []

    if args.models:
        models: List[ModelSpec] = build_models_from_inline(args.models)
    else:
        models = build_models_from_yaml(args.models_config)

    # Discover and append OpenAI models by prefix
    if args.openai_prefixes:
        discovered = discover_openai_chat_models(args.openai_prefixes)
        for mid in discovered:
            models.append(ModelSpec(provider="openai", model=mid, label=mid))

    # Deduplicate
    seen: Set[Tuple[str, str]] = set()
    deduped: List[ModelSpec] = []
    for m in models:
        key = (m.provider, m.model)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    result = run_benchmark(
        models=deduped,
        language=args.language,
        vocabulary=vocabulary,
        target_words=target_words,
        desired_length_words=args.story_words,
        trials=args.trials,
        output_dir=args.output_dir or f"outputs",
    )

    out_dir = result["output_dir"]
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
