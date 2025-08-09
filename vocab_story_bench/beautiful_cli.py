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
from rich.prompt import Prompt, IntPrompt, Confirm
from rich import print as rprint
from dotenv import load_dotenv

from .model_spec import ModelSpec
from .parallel_runner import ParallelBenchmarkRunner
from .data_fetch import ensure_top_n_for_lang


console = Console()

# All supported languages
ALL_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"]


def print_banner():
    """Display a beautiful banner."""
    banner = Panel.fit(
        "[bold cyan]🎯 Vocabulary Inclusion Benchmark[/bold cyan]\n"
        "[dim]Test language models with vocabulary constraints[/dim]",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(banner)
    console.print()


def load_vocabulary(language: str, size: int, data_dir: str = "data/vocab") -> List[str]:
    """Load vocabulary for a language and size."""
    vocab_path = Path(data_dir) / "top" / f"{language}_top{size}.txt"
    if not vocab_path.exists():
        # Try to fetch it
        console.print(f"[yellow]Vocabulary file not found, fetching {language} top {size}...[/yellow]")
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
    console.print("[bold yellow]Step 1: Select Languages[/bold yellow]")
    console.print(f"Available languages: {', '.join(ALL_LANGUAGES)}")
    console.print("Enter 'all' to select all languages")
    languages_input = Prompt.ask(
        "Enter languages (space-separated)",
        default="en es fr"
    )
    
    if languages_input.lower() == "all":
        languages = ALL_LANGUAGES.copy()
        console.print(f"✅ Selected all {len(languages)} languages\n")
    else:
        languages = languages_input.split()
        console.print(f"✅ Selected languages: {', '.join(languages)}\n")
    
    # Vocabulary sizes
    console.print("[bold yellow]Step 2: Select Vocabulary Sizes[/bold yellow]")
    sizes_input = Prompt.ask(
        "Enter vocabulary sizes (space-separated)",
        default="1000 2000"
    )
    vocab_sizes = [int(s) for s in sizes_input.split()]
    console.print(f"✅ Selected sizes: {', '.join(map(str, vocab_sizes))}\n")
    
    # Model selection
    console.print("[bold yellow]Step 3: Select Models[/bold yellow]")
    console.print("Options:")
    console.print("  1. Use predefined models from config file")
    console.print("  2. Enter models manually")
    
    model_choice = IntPrompt.ask("Choose option", choices=["1", "2"])
    
    models = []
    if model_choice == 1:
        config_path = Prompt.ask("Enter config file path", default="configs/models.yaml")
        models = load_models_from_yaml(config_path)
    else:
        console.print("Enter models in OpenRouter format (e.g., 'openai/gpt-4o' or 'qwen/qwq-32b:free')\nOne per line, empty line to finish:")
        while True:
            model_input = Prompt.ask("Model (or press Enter to finish)", default="")
            if not model_input:
                break
            models.append(ModelSpec.parse_inline(model_input))
    
    console.print(f"✅ Selected {len(models)} models\n")
    
    # Target words configuration
    console.print("[bold yellow]Step 4: Configure Target Words[/bold yellow]")
    console.print("Target words will be automatically selected from each language's vocabulary")
    
    num_targets = IntPrompt.ask("Number of target words per language", default=5, min_value=1, max_value=10)
    use_custom_targets = Confirm.ask("Do you want to specify custom targets for any language?", default=False)
    
    target_words = {}
    
    # First, set up automatic targets for all languages
    console.print("\n[dim]Selecting random target words from vocabularies...[/dim]")
    for lang in languages:
        try:
            # Load the smallest vocabulary size to select targets from
            vocab = load_vocabulary(lang, vocab_sizes[0])
            targets = select_random_targets(vocab, num_targets, seed=42)  # Use seed for reproducibility
            target_words[lang] = targets
            console.print(f"  {lang}: {', '.join(targets)}")
        except Exception as e:
            console.print(f"  [red]Error loading vocabulary for {lang}: {e}[/red]")
            target_words[lang] = []
    
    # Allow custom overrides if requested
    if use_custom_targets:
        console.print("\n[yellow]Enter custom targets (or press Enter to keep automatic selection):[/yellow]")
        for lang in languages:
            current = ', '.join(target_words.get(lang, []))
            custom = Prompt.ask(f"Targets for {lang}", default=current)
            if custom:
                target_words[lang] = custom.split(', ') if ', ' in custom else custom.split()
    
    console.print(f"\n✅ Configured target words for {len(languages)} languages\n")
    
    # Additional settings
    console.print("[bold yellow]Step 5: Additional Settings[/bold yellow]")
    trials = IntPrompt.ask("Trials per model", default=3)
    story_length = IntPrompt.ask("Approximate story length (words)", default=150)
    max_parallel = IntPrompt.ask("Max parallel models", default=8)
    
    # Confirm settings
    console.print("\n[bold green]Configuration Summary:[/bold green]")
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Setting", style="cyan")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Languages", f"{len(languages)} languages")
    summary_table.add_row("Vocabulary Sizes", ", ".join(map(str, vocab_sizes)))
    summary_table.add_row("Models", f"{len(models)} models")
    summary_table.add_row("Target Words", f"{num_targets} per language")
    summary_table.add_row("Trials per Model", str(trials))
    summary_table.add_row("Story Length", f"{story_length} words")
    summary_table.add_row("Max Parallel", str(max_parallel))
    
    console.print(summary_table)
    console.print()
    
    if not Confirm.ask("Run benchmark with these settings?", default=True):
        console.print("[red]Benchmark cancelled.[/red]")
        return
    
    # Run benchmark
    run_benchmark_with_config(
        languages, vocab_sizes, models, target_words,
        trials, story_length, max_parallel
    )


def run_benchmark_with_config(
    languages: List[str],
    vocab_sizes: List[int],
    models: List[ModelSpec],
    target_words: Dict[str, List[str]],
    trials: int,
    story_length: int,
    max_parallel: int
):
    """Run the benchmark with the given configuration."""
    console.print("\n[bold blue]🚀 Starting Benchmark[/bold blue]\n")
    
    # Ensure vocabulary data exists and load vocabularies
    console.print("[dim]Checking vocabulary data...[/dim]")
    vocabularies = {}
    for lang in languages:
        vocabularies[lang] = {}
        for size in vocab_sizes:
            try:
                ensure_top_n_for_lang(lang, size, "data/vocab")
                vocab = load_vocabulary(lang, size)
                vocabularies[lang][size] = vocab
            except Exception as e:
                console.print(f"[red]Error fetching vocabulary for {lang} top {size}: {e}[/red]")
                return
    
    # Run benchmark
    runner = ParallelBenchmarkRunner()
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
    console.print("\n[bold green]✅ Benchmark Complete![/bold green]")
    console.print(f"Results saved to: {results}")


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
            console.print("[red]Error: --langs is required for fetch command[/red]")
            sys.exit(1)
        
        if langs == ["all"]:
            langs = ALL_LANGUAGES
            
        console.print(f"[cyan]Fetching vocabulary data for {len(langs)} language(s)...[/cyan]")
        out_paths = []
        for lang in langs:
            try:
                p = ensure_top_n_for_lang(lang, args.top, args.data_dir)
                out_paths.append(str(p))
                console.print(f"  ✅ {lang}: {p}")
            except Exception as e:
                console.print(f"  ❌ {lang}: {e}")
        
        console.print(f"\n[green]Completed fetching {len(out_paths)} vocabulary files[/green]")
        return
    
    # If interactive mode or no arguments provided
    if args.interactive or (args.command == "run" and hasattr(args, 'interactive') and args.interactive) or len(sys.argv) == 1:
        interactive_mode()
    elif args.command == "run":
        # Command-line mode
        if not args.languages:
            console.print("[red]Error: --languages is required[/red]")
            sys.exit(1)
        
        if not args.vocab_sizes:
            console.print("[red]Error: --vocab-sizes is required[/red]")
            sys.exit(1)
        
        if not args.models_config:
            console.print("[red]Error: --models-config is required[/red]")
            sys.exit(1)
        
        # Handle "all" languages
        languages = args.languages
        if languages == ["all"]:
            languages = ALL_LANGUAGES
        
        # Load models
        models = load_models_from_yaml(args.models_config)
        
        # Set up target words - automatically select from vocabulary
        console.print(f"\n[cyan]Selecting {args.num_targets} target words for each language...[/cyan]")
        target_words = {}
        for lang in languages:
            try:
                vocab = load_vocabulary(lang, args.vocab_sizes[0])
                targets = select_random_targets(vocab, args.num_targets, seed=42)
                target_words[lang] = targets
                console.print(f"  {lang}: {', '.join(targets)}")
            except Exception as e:
                console.print(f"  [red]Error with {lang}: {e}[/red]")
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
            args.parallel
        )
    else:
        # No command specified, run interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()