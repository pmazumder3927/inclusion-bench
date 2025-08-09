"""Parallel benchmark runner with structured outputs."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from threading import Lock

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich import print as rprint

from .model_spec import ModelSpec
from .structured_prompt import build_structured_prompt, parse_structured_response
from .providers.structured_openai import StructuredOpenAIProvider
from .providers.structured_anthropic import StructuredAnthropicProvider
from .reports import ensure_dir, append_jsonl, write_json


@dataclass
class StructuredTrialResult:
    """Result from a single trial with structured output."""
    model_label: str
    provider: str
    model: str
    language: str
    vocab_size: int
    trial_index: int
    words: List[str]  # Array of words instead of story text
    validation: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Configuration for parallel benchmark execution."""
    languages: List[str]
    vocab_sizes: List[int]
    models: List[ModelSpec]
    target_words: Dict[str, List[str]]  # Per-language targets
    trials_per_model: int = 3
    max_parallel_models: int = 8
    max_parallel_languages: int = 4
    story_length: int = 150
    output_dir: Path = Path("outputs")


class ParallelBenchmarkRunner:
    """Runs benchmarks in parallel across languages and models."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.console = Console()
        self.results_lock = Lock()
        self.all_results: List[StructuredTrialResult] = []
        
        # Initialize providers
        self.providers = {}
        
        # Initialize OpenAI provider if API key is available
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.providers["openai"] = StructuredOpenAIProvider()
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not initialize OpenAI provider: {e}[/yellow]")
        
        # Initialize Anthropic provider if API key is available
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.providers["anthropic"] = StructuredAnthropicProvider()
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not initialize Anthropic provider: {e}[/yellow]")
    
    def validate_words(
        self,
        words: List[str],
        vocabulary: Set[str],
        targets: Set[str]
    ) -> Dict[str, Any]:
        """Validate word array against vocabulary and targets."""
        word_set = set(w.lower() for w in words)
        vocab_set = set(v.lower() for v in vocabulary)
        target_set = set(t.lower() for t in targets)
        
        oov_words = [w for w in word_set if w not in vocab_set]
        missing_targets = [t for t in target_set if t not in word_set]
        
        return {
            "only_vocab": len(oov_words) == 0,
            "all_targets_present": len(missing_targets) == 0,
            "oov_words": oov_words,
            "missing_targets": missing_targets,
            "total_words": len(words),
            "unique_words": len(word_set),
            "vocabulary_coverage": len(word_set & vocab_set) / len(vocab_set) if vocab_set else 0
        }
    
    def run_single_trial(
        self,
        model: ModelSpec,
        language: str,
        vocabulary: List[str],
        targets: List[str],
        vocab_size: int,
        trial_index: int
    ) -> StructuredTrialResult:
        """Execute a single trial for a model."""
        start_time = dt.datetime.now()
        
        try:
            provider = self.providers.get(model.provider)
            if not provider:
                raise ValueError(f"Unknown provider: {model.provider}")
            
            # Build structured prompt
            system, user, response_format = build_structured_prompt(
                language,
                vocabulary,
                targets,
                self.config.story_length
            )
            
            # Generate structured response
            response = provider.generate_structured(
                model.model,
                system,
                user,
                response_format,
                max_output_tokens=self.config.story_length * 2,
                params=model.params
            )
            
            # Parse response to get word array
            words = parse_structured_response(response)
            
            # Validate
            validation = self.validate_words(
                words,
                set(vocabulary),
                set(targets)
            )
            
            execution_time = (dt.datetime.now() - start_time).total_seconds()
            
            return StructuredTrialResult(
                model_label=model.display_label,
                provider=model.provider,
                model=model.model,
                language=language,
                vocab_size=vocab_size,
                trial_index=trial_index,
                words=words,
                validation=validation,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = (dt.datetime.now() - start_time).total_seconds()
            return StructuredTrialResult(
                model_label=model.display_label,
                provider=model.provider,
                model=model.model,
                language=language,
                vocab_size=vocab_size,
                trial_index=trial_index,
                words=[],
                validation={
                    "only_vocab": False,
                    "all_targets_present": False,
                    "oov_words": [],
                    "missing_targets": targets,
                    "total_words": 0,
                    "unique_words": 0,
                    "vocabulary_coverage": 0
                },
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def run_model_trials(
        self,
        model: ModelSpec,
        language: str,
        vocabulary: List[str],
        targets: List[str],
        vocab_size: int,
        progress: Progress,
        task_id: int
    ) -> List[StructuredTrialResult]:
        """Run all trials for a single model."""
        results = []
        
        for trial in range(self.config.trials_per_model):
            result = self.run_single_trial(
                model, language, vocabulary, targets, vocab_size, trial
            )
            results.append(result)
            
            with self.results_lock:
                self.all_results.append(result)
            
            progress.update(task_id, advance=1)
        
        return results
    
    def load_vocabulary(self, language: str, size: int) -> List[str]:
        """Load vocabulary for a language and size."""
        vocab_path = Path("data/vocab/top") / f"{language}_top{size}.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    
    def run_parallel(self) -> Dict[str, Any]:
        """Run benchmarks in parallel across all configurations."""
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_dir(self.config.output_dir / f"parallel_{timestamp}")
        
        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        
        # Calculate total tasks
        total_tasks = (
            len(self.config.languages) *
            len(self.config.vocab_sizes) *
            len(self.config.models) *
            self.config.trials_per_model
        )
        
        # Create layout for beautiful display
        layout = Layout()
        layout.split_column(
            Layout(Panel("🚀 Parallel Vocabulary Benchmark", style="bold blue"), size=3),
            Layout(progress, size=10),
            Layout(name="status", size=3)
        )
        
        self.console.print(layout)
        
        with Live(layout, refresh_per_second=4, console=self.console):
            with progress:
                overall_task = progress.add_task(
                    "[cyan]Overall Progress",
                    total=total_tasks
                )
                
                # Run tasks in parallel
                with ThreadPoolExecutor(max_workers=self.config.max_parallel_models) as executor:
                    futures = []
                    
                    for language in self.config.languages:
                        for vocab_size in self.config.vocab_sizes:
                            # Load vocabulary
                            try:
                                vocabulary = self.load_vocabulary(language, vocab_size)
                            except FileNotFoundError as e:
                                self.console.print(f"[red]Error: {e}[/red]")
                                continue
                            
                            # Get targets for this language
                            targets = self.config.target_words.get(
                                language,
                                self.config.target_words.get("default", [])
                            )
                            
                            for model in self.config.models:
                                task_name = f"{model.display_label} | {language} | {vocab_size} words"
                                task_id = progress.add_task(
                                    task_name,
                                    total=self.config.trials_per_model
                                )
                                
                                future = executor.submit(
                                    self.run_model_trials,
                                    model,
                                    language,
                                    vocabulary,
                                    targets,
                                    vocab_size,
                                    progress,
                                    task_id
                                )
                                futures.append(future)
                    
                    # Wait for completion and update overall progress
                    for future in as_completed(futures):
                        results = future.result()
                        progress.update(
                            overall_task,
                            advance=len(results)
                        )
                        
                        # Update status
                        completed = progress.tasks[overall_task].completed
                        layout["status"].update(
                            Panel(f"✅ Completed: {completed}/{total_tasks}", style="green")
                        )
        
        # Save results
        self.save_results(output_dir)
        
        # Generate summary
        summary = self.generate_summary()
        
        # Display final results table
        self.display_results_table(summary)
        
        return {
            "output_dir": str(output_dir),
            "summary": summary,
            "total_trials": len(self.all_results)
        }
    
    def save_results(self, output_dir: Path):
        """Save all results to files."""
        # Save detailed results as JSONL
        details_path = output_dir / "details.jsonl"
        for result in self.all_results:
            append_jsonl(details_path, asdict(result))
        
        # Save summary
        summary = self.generate_summary()
        summary_path = output_dir / "summary.json"
        write_json(summary_path, summary)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        summary = {}
        
        # Group by model, language, vocab_size
        grouped = {}
        for result in self.all_results:
            key = (result.model_label, result.language, result.vocab_size)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Calculate statistics for each group
        for (model, lang, size), results in grouped.items():
            success_rate = sum(1 for r in results if r.success) / len(results)
            pass_rate = sum(
                1 for r in results
                if r.validation["only_vocab"] and r.validation["all_targets_present"]
            ) / len(results)
            
            avg_execution_time = sum(r.execution_time for r in results) / len(results)
            avg_word_count = sum(r.validation["total_words"] for r in results) / len(results)
            avg_coverage = sum(r.validation["vocabulary_coverage"] for r in results) / len(results)
            
            summary[f"{model}_{lang}_{size}"] = {
                "model": model,
                "language": lang,
                "vocab_size": size,
                "trials": len(results),
                "success_rate": round(success_rate, 3),
                "pass_rate": round(pass_rate, 3),
                "avg_execution_time": round(avg_execution_time, 2),
                "avg_word_count": round(avg_word_count, 1),
                "avg_vocabulary_coverage": round(avg_coverage, 3)
            }
        
        return summary
    
    def display_results_table(self, summary: Dict[str, Any]):
        """Display a beautiful results table."""
        table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
        
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Language", style="yellow")
        table.add_column("Vocab Size", justify="right", style="green")
        table.add_column("Pass Rate", justify="right", style="blue")
        table.add_column("Avg Words", justify="right")
        table.add_column("Avg Time (s)", justify="right")
        table.add_column("Coverage", justify="right", style="magenta")
        
        for key, stats in summary.items():
            pass_rate_color = "green" if stats["pass_rate"] > 0.8 else "yellow" if stats["pass_rate"] > 0.5 else "red"
            
            table.add_row(
                stats["model"],
                stats["language"],
                str(stats["vocab_size"]),
                f"[{pass_rate_color}]{stats['pass_rate']:.1%}[/{pass_rate_color}]",
                str(int(stats["avg_word_count"])),
                f"{stats['avg_execution_time']:.1f}",
                f"{stats['avg_vocabulary_coverage']:.1%}"
            )
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")