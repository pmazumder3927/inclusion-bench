from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set, Tuple

from dotenv import load_dotenv

from .model_spec import ModelSpec
from .runner import build_models_from_inline, build_models_from_yaml, load_lines, run_benchmark
from .discovery import discover_openai_chat_models
from .data_fetch import ensure_top_n_for_lang
from .aggregate import merge_runs, scan_run_roots
from .visualize import save_dashboard


def main() -> None:
    # Load environment variables from .env if present
    load_dotenv()

    ap = argparse.ArgumentParser(description="Vocab-only story benchmark")

    subparsers = ap.add_subparsers(dest="command")

    # Fetch vocab data
    fetch = subparsers.add_parser("fetch", help="Download frequency lists and build top-N vocab files")
    fetch.add_argument("--langs", nargs="+", required=True, help="Language codes (e.g., en es fr de it pt ru zh ja ko ar hi)")
    fetch.add_argument("--top", type=int, default=2000, help="Top-N words to keep")
    fetch.add_argument("--data-dir", default="data/vocab", help="Destination directory for data")

    # Run benchmark
    bench = subparsers.add_parser("run", help="Run the benchmark")
    bench.add_argument("--vocab-file", required=True, help="Path to newline-separated vocabulary list")
    bench.add_argument("--targets-file", required=False, help="Path to newline-separated target words")
    bench.add_argument("--targets", nargs="*", help="Inline target words (use instead of --targets-file)")
    bench.add_argument("--language", default="", help="Language hint for the story")
    bench.add_argument("--story-words", type=int, default=150, help="Approximate desired story length in words")
    bench.add_argument("--trials", type=int, default=3, help="Trials per model")

    group = bench.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+", help="Inline models like 'openai:gpt-4.1' 'anthropic:claude-sonnet-4-20250514'")
    group.add_argument("--models-config", help="YAML file listing models")

    # Dynamic discovery options
    bench.add_argument("--openai-prefixes", nargs="*", default=[], help="Auto-include all OpenAI models starting with these prefixes (e.g., gpt-5)")

    bench.add_argument("--output-dir", default=None, help="Directory to write outputs; defaults to outputs/{timestamp}")

    # Batch benchmark
    batch = subparsers.add_parser("batch", help="Run multiple vocab sizes/languages and build an HTML dashboard")
    batch.add_argument("--langs", nargs="+", required=True, help="Language codes (e.g., en es fr)")
    batch.add_argument("--sizes", nargs="+", type=int, required=True, help="Vocab sizes (e.g., 1000 2000 3000)")
    batch.add_argument("--data-dir", default="data/vocab", help="Data directory where fetch placed files")
    batch.add_argument("--targets", nargs="+", required=True, help="Targets to include (applies to each language; pass language-specific ones by running separate batches)")
    batch.add_argument("--language-label", default="", help="Language label shown to models (applies to each run)")
    batch_group = batch.add_mutually_exclusive_group(required=True)
    batch_group.add_argument("--models", nargs="+", help="Inline models like 'openai:gpt-4.1-mini'")
    batch_group.add_argument("--models-config", help="YAML file listing models")
    batch.add_argument("--story-words", type=int, default=150)
    batch.add_argument("--trials", type=int, default=1)
    batch.add_argument("--out-root", default="runs/batch")

    # Overall dashboard
    overall = subparsers.add_parser("overall", help="Build a single HTML dashboard from multiple run roots")
    overall.add_argument("--roots", nargs="+", required=True, help="Run roots like runs/full_en runs/full_es ...")
    overall.add_argument("--out", required=True, help="Path to output HTML dashboard")

    args = ap.parse_args()

    if args.command == "fetch":
        out_paths = []
        for lang in args.langs:
            p = ensure_top_n_for_lang(lang, args.top, args.data_dir)
            out_paths.append(str(p))
        print("\n".join(out_paths))
        return

    if args.command == "batch":
        if args.models:
            models = build_models_from_inline(args.models)
        else:
            models = build_models_from_yaml(args.models_config)
        produced_dirs: List[str] = []
        for lang in args.langs:
            for size in args.sizes:
                vocab_path = Path(args.data_dir) / "top" / f"{lang}_top{size}.txt"
                out_dir = Path(args.out_root) / f"{lang}_top{size}"
                res = run_benchmark(
                    models=models,
                    language=args.language_label or lang,
                    vocabulary=load_lines(vocab_path),
                    target_words=args.targets,
                    desired_length_words=args.story_words,
                    trials=args.trials,
                    output_dir=out_dir,
                )
                produced_dirs.append(str(out_dir))
        # Aggregate and visualize
        df = merge_runs(produced_dirs)
        dashboard_path = Path(args.out_root) / "dashboard.html"
        save_dashboard(df, dashboard_path)
        print(f"Dashboard saved to {dashboard_path}")
        return

    if args.command == "overall":
        run_dirs = scan_run_roots(args.roots)
        df = merge_runs(run_dirs)
        save_dashboard(df, args.out)
        print(f"Overall dashboard saved to {args.out}")
        return

    # default or 'run'
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
