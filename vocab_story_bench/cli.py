#!/usr/bin/env python3
"""Beautiful CLI for the vocabulary benchmark with structured outputs."""
from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path
from typing import List, Dict, Optional
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from rich.text import Text
from dotenv import load_dotenv

from .model_spec import ModelSpec
from .parallel_runner import ParallelBenchmarkRunner
from .data_fetch import ensure_top_n_for_lang
from .visualizations import create_dashboard_from_jsonl


console = Console()

# All supported languages
ALL_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"]


def print_banner():
    """Display a minimal banner."""
    console.print()
    console.print("[bold #ff6b3d]vocab[/bold #ff6b3d][bold white]-bench[/bold white]", style="bold")
    console.print("[#71717a]vocabulary inclusion benchmark[/#71717a]")
    console.print()


def load_vocabulary(language: str, size: int, data_dir: str = "data/vocab") -> List[str]:
    """Load vocabulary for a language and size."""
    vocab_path = Path(data_dir) / "top" / f"{language}_top{size}.txt"
    if not vocab_path.exists():
        # Try to fetch it
        console.print(f"[#71717a]fetching {language} top {size}...[/#71717a]")
        ensure_top_n_for_lang(language, size, data_dir)
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def select_random_targets(vocabulary: List[str], n: int = 5, seed: Optional[int] = None) -> List[str]:
    """Select n random words from vocabulary as targets."""
    if seed is not None:
        random.seed(seed)
    
    # Select from middle of vocabulary (avoid most common and least common)
    start_idx = min(100, len(vocabulary) // 4)
    end_idx = min(len(vocabulary), 500)
    
    if end_idx - start_idx < n:
        # If vocabulary is too small, just select from all
        candidates = vocabulary
    else:
        candidates = vocabulary[start_idx:end_idx]
    
    return random.sample(candidates, min(n, len(candidates)))


def interactive_mode():
    """Interactive mode for configuring and running benchmarks."""
    print_banner()
    
    # Language selection
    console.print("\n[bold #ff6b3d]→ Languages[/bold #ff6b3d]")
    console.print(f"[#71717a]available: {', '.join(ALL_LANGUAGES)} (or 'all')[/#71717a]")
    languages_input = Prompt.ask(
        "[#71717a]languages[/#71717a]",
        default="en es fr"
    )
    
    if languages_input.lower() == "all":
        languages = ALL_LANGUAGES.copy()
        console.print(f"[#71717a]selected: all ({len(languages)} languages)[/#71717a]\n")
    else:
        languages = languages_input.split()
        console.print(f"[#71717a]selected: {', '.join(languages)}[/#71717a]\n")
    
    # Vocabulary sizes
    console.print("\n[bold #4a9eff]→ Vocabulary Sizes[/bold #4a9eff]")
    console.print("[#71717a]options: 1000, 2000, 3000, 4000, 5000[/#71717a]")
    sizes_input = Prompt.ask(
        "[#71717a]sizes[/#71717a]",
        default="1000 2000 3000"
    )
    vocab_sizes = [int(s) for s in sizes_input.split()]
    console.print(f"[#71717a]selected: {', '.join(map(str, vocab_sizes))}[/#71717a]\n")
    
    # Model selection
    console.print("\n[bold #7c77c6]→ Models[/bold #7c77c6]")
    console.print("[#71717a]1. use config file[/#71717a]")
    console.print("[#71717a]2. enter manually[/#71717a]")
    
    model_choice = Prompt.ask("[#71717a]choice[/#71717a]", choices=["1", "2"])
    
    models = []
    if model_choice == "1":
        config_path = Prompt.ask("[#71717a]config path[/#71717a]", default="configs/models.yaml")
        models = load_models_from_yaml(config_path)
    else:
        console.print("[#71717a]enter models (openrouter format, empty line to finish):[/#71717a]")
        while True:
            model_input = Prompt.ask("[#71717a]model[/#71717a]", default="")
            if not model_input:
                break
            models.append(ModelSpec.parse_inline(model_input))
    
    console.print(f"[#71717a]{len(models)} models loaded[/#71717a]\n")
    
    # Target words configuration
    console.print("\n[bold #ff375f]→ Target Words[/bold #ff375f]")
    console.print("[#71717a]auto-selected from each language's vocabulary[/#71717a]")
    
    num_targets = Prompt.ask("[#71717a]target words per language[/#71717a]", default="5")
    # Basic bounds to keep prompts reasonable
    try:
        num_targets = int(num_targets)
    except Exception:
        num_targets = 5
    if num_targets < 1:
        num_targets = 1
    if num_targets > 10:
        num_targets = 10
    use_custom_targets = Confirm.ask("[#71717a]custom targets?[/#71717a]", default=False)
    
    target_words = {}
    
    # First, set up automatic targets for all languages
    console.print("\n[#71717a]Selecting random target words from vocabularies...[/#71717a]")
    for lang in languages:
        try:
            # Load the smallest vocabulary size to select targets from
            vocab = load_vocabulary(lang, vocab_sizes[0])
            targets = select_random_targets(vocab, num_targets, seed=42)  # Use seed for reproducibility
            target_words[lang] = targets
            console.print(f"  [#71717a]{lang}:[/#71717a] {', '.join(targets)}")
        except Exception as e:
            console.print(f"  [#71717a]error: {lang} - {e}[/#71717a]")
            target_words[lang] = []
    
    # Allow custom overrides if requested
    if use_custom_targets:
        console.print("\n[#71717a]enter custom targets (or press enter to keep):[/#71717a]")
        for lang in languages:
            current = ', '.join(target_words.get(lang, []))
            custom = Prompt.ask(f"[#71717a]{lang} targets[/#71717a]", default=current)
            if custom:
                target_words[lang] = custom.split(', ') if ', ' in custom else custom.split()
    
    console.print(f"\n[#71717a]configured {len(languages)} languages[/#71717a]\n")
    
    # Additional settings
    console.print("\n[bold #ffd60a]→ Settings[/bold #ffd60a]")
    trials = int(Prompt.ask("[#71717a]trials per model[/#71717a]", default="3"))
    story_length = int(Prompt.ask("[#71717a]story length (words)[/#71717a]", default="150"))
    max_parallel = int(Prompt.ask("[#71717a]max parallel[/#71717a]", default="8"))
    
    # Confirm settings
    console.print("\n[bold #30d158]→ Summary[/bold #30d158]")
    
    summary_table = Table(
        show_header=False,
        box=None,
        padding=(0, 1)
    )
    summary_table.add_column("Setting", style="#71717a")
    summary_table.add_column("Value", style="white")
    
    summary_table.add_row("Languages", f"{len(languages)} languages")
    summary_table.add_row("Vocabulary Sizes", ", ".join(map(str, vocab_sizes)))
    summary_table.add_row("Models", f"{len(models)} models")
    summary_table.add_row("Target Words", f"{num_targets} per language")
    summary_table.add_row("Trials per Model", str(trials))
    summary_table.add_row("Story Length", f"{story_length} words")
    summary_table.add_row("Max Parallel", str(max_parallel))
    
    console.print(summary_table)
    console.print()
    
    if not Confirm.ask("[#71717a]run?[/#71717a]", default=True):
        console.print("[#71717a]cancelled[/#71717a]")
        return
    
    # Run benchmark
    run_benchmark_with_config(
        languages, vocab_sizes, models, target_words,
        trials, story_length, max_parallel,
        output_dir=Path("outputs"),
    )


def run_benchmark_with_config(
    languages: List[str],
    vocab_sizes: List[int],
    models: List[ModelSpec],
    target_words: Dict[str, List[str]],
    trials: int,
    story_length: int,
    max_parallel: int,
    output_dir: Path | str = Path("outputs"),
):
    """Run the benchmark with the given configuration."""
    console.print("\n[bold #4a9eff]→ Starting benchmark...[/bold #4a9eff]\n")
    
    # Ensure vocabulary data exists and load vocabularies
    console.print("[#71717a]Checking vocabulary data...[/#71717a]")
    vocabularies = {}
    for lang in languages:
        vocabularies[lang] = {}
        for size in vocab_sizes:
            try:
                ensure_top_n_for_lang(lang, size, "data/vocab")
                vocab = load_vocabulary(lang, size)
                vocabularies[lang][size] = vocab
            except Exception as e:
                console.print(f"[#71717a]error fetching {lang} top {size}: {e}[/#71717a]")
                return
    
    # Run benchmark
    runner = ParallelBenchmarkRunner(output_dir=str(output_dir))
    results = runner.run_parallel(
        models=models,
        languages=languages,
        vocab_sizes=vocab_sizes,
        vocabularies=vocabularies,
        targets=target_words,
        trials=trials,
        story_length=story_length,
        max_parallel=max_parallel
    )
    
    # Display completion message
    console.print("\n[bold #30d158]✓ Complete[/bold #30d158]")
    console.print(f"[#71717a]results:[/#71717a] {results}")


def load_models_from_yaml(path: str) -> List[ModelSpec]:
    """Load models from a YAML configuration file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    models = []
    for item in data.get("models", []):
        models.append(
            ModelSpec(
                model=item["model"],
                label=item.get("label"),
                params=item.get("params")
            )
        )
    return models


def main():
    """Main CLI entry point."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Beautiful Vocabulary Inclusion Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Fetch command
    fetch = subparsers.add_parser("fetch", help="Download frequency lists and build top-N vocab files")
    fetch.add_argument("--langs", nargs="+", help="Language codes (e.g., en es fr). Use 'all' for all languages")
    fetch.add_argument("--top", type=int, default=2000, help="Top-N words to keep")
    fetch.add_argument("--data-dir", default="data/vocab", help="Destination directory for data")
    
    # Run command
    run = subparsers.add_parser("run", help="Run the benchmark")
    run.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    run.add_argument(
        "--languages", "-l",
        nargs="+",
        help="Languages to test (e.g., en es fr). Use 'all' for all languages"
    )
    
    run.add_argument(
        "--vocab-sizes", "-v",
        nargs="+",
        type=int,
        help="Vocabulary sizes (e.g., 1000 2000 3000)"
    )
    
    run.add_argument(
        "--models-config", "-m",
        help="Path to models configuration file"
    )
    
    run.add_argument(
        "--num-targets",
        type=int,
        default=5,
        help="Number of target words to select from vocabulary (default: 5)"
    )
    
    run.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per model (default: 3)"
    )
    
    run.add_argument(
        "--story-length",
        type=int,
        default=150,
        help="Approximate story length in words (default: 150)"
    )
    
    run.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Max parallel model executions (default: 8)"
    )
    
    run.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("outputs"),
        help="Output directory (default: outputs)"
    )

    # Viz command: build dashboard from an existing details.jsonl
    viz = subparsers.add_parser("viz", help="Generate dashboard HTML from a details.jsonl file")
    viz.add_argument("details", type=Path, help="Path to details.jsonl")
    viz.add_argument("--out", type=Path, default=Path("dashboard.html"), help="Output HTML file path")
    
    # Also add --interactive at top level for backward compatibility
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Handle fetch command
    if args.command == "fetch":
        langs = args.langs
        if not langs:
            console.print("[#71717a]error: --langs required[/#71717a]")
            sys.exit(1)
        
        if langs == ["all"]:
            langs = ALL_LANGUAGES
            
        console.print(f"[#71717a]fetching {len(langs)} languages...[/#71717a]")
        out_paths = []
        for lang in langs:
            try:
                p = ensure_top_n_for_lang(lang, args.top, args.data_dir)
                out_paths.append(str(p))
                console.print(f"  [#71717a]{lang}:[/#71717a] {p}")
            except Exception as e:
                console.print(f"  [#71717a]error {lang}:[/#71717a] {e}")
        
        console.print(f"\n[#71717a]fetched {len(out_paths)} files[/#71717a]")
        return
    if args.command == "viz":
        # Create dashboard from a JSONL file
        try:
            out_file = args.out
            out_file.parent.mkdir(parents=True, exist_ok=True)
            create_dashboard_from_jsonl(args.details, out_file)
            console.print(f"\n[#71717a]dashboard:[/#71717a] {out_file}")
        except Exception as e:
            console.print(f"[#71717a]error:[/#71717a] {e}")
        return
    
    # If interactive mode or no arguments provided
    if args.interactive or (args.command == "run" and hasattr(args, 'interactive') and args.interactive) or len(sys.argv) == 1:
        interactive_mode()
    elif args.command == "run":
        # Command-line mode
        if not args.languages:
            console.print("[#71717a]error: --languages required[/#71717a]")
            sys.exit(1)
        
        if not args.vocab_sizes:
            console.print("[#71717a]error: --vocab-sizes required[/#71717a]")
            sys.exit(1)
        
        if not args.models_config:
            console.print("[#71717a]error: --models-config required[/#71717a]")
            sys.exit(1)
        
        # Handle "all" languages
        languages = args.languages
        if languages == ["all"]:
            languages = ALL_LANGUAGES
        
        # Load models
        models = load_models_from_yaml(args.models_config)
        
        # Set up target words - automatically select from vocabulary
        console.print(f"\n[#71717a]selecting {args.num_targets} target words...[/#71717a]")
        target_words = {}
        for lang in languages:
            try:
                vocab = load_vocabulary(lang, args.vocab_sizes[0])
                targets = select_random_targets(vocab, args.num_targets, seed=42)
                target_words[lang] = targets
                console.print(f"  [#7c77c6]{lang}:[/#7c77c6] {', '.join(targets)}")
            except Exception as e:
                console.print(f"  [#71717a]error {lang}:[/#71717a] {e}")
                target_words[lang] = []
        
        # Run benchmark
        print_banner()
        run_benchmark_with_config(
            languages,
            args.vocab_sizes,
            models,
            target_words,
            args.trials,
            args.story_length,
            args.parallel,
            args.output_dir
        )
    else:
        # No command specified, run interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()

