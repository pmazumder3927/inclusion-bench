"""Parallel benchmark runner with unified OpenRouter provider."""
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
from .unified_provider import UnifiedProvider
from .reports import ensure_dir, append_jsonl, write_json
from .visualizations import create_dashboard_html, display_terminal_charts


@dataclass
class StructuredTrialResult:
    """Result from a single trial with structured output."""
    model_label: str
    model: str
    language: str
    vocab_size: int
    trial_index: int
    words: List[str]  # Array of words instead of story text
    validation: Dict[str, Any]
    execution_time: float
    success: bool
    error: Optional[str] = None


class ParallelBenchmarkRunner:
    """Runs benchmarks in parallel across models and languages."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.console = Console()
        self.provider = UnifiedProvider()
        self.results_lock = Lock()
        self.all_results = []
    
    def validate_words(
        self,
        words: List[str],
        vocabulary: Set[str],
        targets: List[str]
    ) -> Dict[str, Any]:
        """Validate words against vocabulary and targets."""
        validation = {
            "total_words": len(words),
            "only_vocab": all(word.lower() in vocabulary for word in words),
            "all_targets_present": all(
                any(target.lower() in word.lower() for word in words)
                for target in targets
            ),
            "vocabulary_coverage": len(set(w.lower() for w in words) & vocabulary) / len(vocabulary),
            "unique_words": len(set(words)),
            "target_words_found": [t for t in targets if any(t.lower() in w.lower() for w in words)]
        }
        
        validation["pass"] = validation["only_vocab"] and validation["all_targets_present"]
        
        return validation
    
    def run_single_trial(
        self,
        model_spec: ModelSpec,
        language: str,
        vocabulary: List[str],
        targets: List[str],
        vocab_size: int,
        trial_index: int,
        story_length: int = 150,
        task_id: Optional[int] = None,
        progress: Optional[Progress] = None
    ) -> StructuredTrialResult:
        """Run a single trial for a model-language combination."""
        import time
        start_time = time.time()
        
        try:
            # Build structured prompt
            system_prompt, user_prompt = build_structured_prompt(
                vocabulary=vocabulary,
                targets=targets,
                language=language,
                story_length=story_length
            )
            
            # Generate structured output
            response = self.provider.generate_structured(
                model=model_spec.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=story_length * 3,  # Buffer for JSON overhead
                **model_spec.params
            )
            
            # Parse structured response
            words = parse_structured_response(response)
            
            # Validate
            vocab_set = set(v.lower() for v in vocabulary)
            validation = self.validate_words(words, vocab_set, targets)
            
            execution_time = time.time() - start_time
            
            result = StructuredTrialResult(
                model_label=model_spec.label,
                model=model_spec.model,
                language=language,
                vocab_size=vocab_size,
                trial_index=trial_index,
                words=words,
                validation=validation,
                execution_time=execution_time,
                success=True
            )
            
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = StructuredTrialResult(
                model_label=model_spec.label,
                model=model_spec.model,
                language=language,
                vocab_size=vocab_size,
                trial_index=trial_index,
                words=[],
                validation={"pass": False, "error": str(e)},
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
            
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            
            return result
    
    def run_model_language_combination(
        self,
        model_spec: ModelSpec,
        language: str,
        vocabulary: List[str],
        targets: List[str],
        vocab_size: int,
        trials: int,
        story_length: int,
        progress: Progress,
        task_id: int
    ) -> List[StructuredTrialResult]:
        """Run all trials for a specific model-language combination."""
        results = []
        
        for trial_index in range(trials):
            result = self.run_single_trial(
                model_spec=model_spec,
                language=language,
                vocabulary=vocabulary,
                targets=targets,
                vocab_size=vocab_size,
                trial_index=trial_index,
                story_length=story_length,
                task_id=task_id,
                progress=progress
            )
            
            results.append(result)
            
            # Save result immediately
            with self.results_lock:
                self.all_results.append(result)
        
        return results
    
    def run_parallel(
        self,
        models: List[ModelSpec],
        languages: List[str],
        vocab_sizes: List[int],
        vocabularies: Dict[str, Dict[int, List[str]]],
        targets: Dict[str, List[str]],
        trials: int = 3,
        story_length: int = 150,
        max_parallel: int = 8
    ) -> str:
        """Run benchmarks in parallel across models and languages."""
        
        # Create output directory with timestamp
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"parallel_{timestamp}"
        ensure_dir(run_dir)
        
        # Calculate total tasks
        total_tasks = len(models) * len(languages) * len(vocab_sizes) * trials
        
        # Create rich progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True
        ) as progress:
            
            # Create overall progress task
            overall_task = progress.add_task(
                f"[bold cyan]🚀 Running {total_tasks} trials",
                total=total_tasks
            )
            
            # Create tasks for each model-language combination
            model_tasks = {}
            for model in models:
                for lang in languages:
                    for vocab_size in vocab_sizes:
                        task_name = f"{model.label} | {lang} | {vocab_size} words"
                        task_id = progress.add_task(
                            task_name,
                            total=trials,
                            visible=True
                        )
                        model_tasks[(model.label, lang, vocab_size)] = task_id
            
            # Run combinations in parallel
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = []
                
                for model in models:
                    for lang in languages:
                        for vocab_size in vocab_sizes:
                            if lang not in vocabularies or vocab_size not in vocabularies[lang]:
                                self.console.print(f"[yellow]Skipping {lang}-{vocab_size}: vocabulary not found[/yellow]")
                                continue
                            
                            vocabulary = vocabularies[lang][vocab_size]
                            lang_targets = targets.get(lang, [])
                            task_id = model_tasks[(model.label, lang, vocab_size)]
                            
                            future = executor.submit(
                                self.run_model_language_combination,
                                model,
                                lang,
                                vocabulary,
                                lang_targets,
                                vocab_size,
                                trials,
                                story_length,
                                progress,
                                task_id
                            )
                            futures.append((future, model, lang, vocab_size))
                
                # Process results as they complete
                for future, model, lang, vocab_size in futures:
                    try:
                        results = future.result(timeout=300)
                        
                        # Write results to JSONL
                        for result in results:
                            append_jsonl(
                                run_dir / "details.jsonl",
                                asdict(result)
                            )
                        
                        # Update overall progress
                        progress.update(overall_task, advance=trials)
                        
                    except Exception as e:
                        self.console.print(f"[red]Error with {model.label} {lang}: {e}[/red]")
                        progress.update(overall_task, advance=trials)
        
        # Generate summary
        summary = self.generate_summary(self.all_results)
        write_json(run_dir / "summary.json", summary)
        
        # Display results table
        self.display_results_table(self.all_results)
        
        # Display terminal charts
        display_terminal_charts(self.all_results)
        
        # Generate HTML dashboard
        dashboard_path = run_dir / "dashboard.html"
        create_dashboard_html(self.all_results, dashboard_path)
        self.console.print(f"\n📊 [bold green]Interactive dashboard generated![/bold green]")
        self.console.print(f"   Open in browser: [cyan]file://{dashboard_path.absolute()}[/cyan]")
        
        return str(run_dir)
    
    def generate_summary(self, results: List[StructuredTrialResult]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not results:
            return {}
        
        # Group by model and language
        model_stats = {}
        language_stats = {}
        
        for result in results:
            # Model stats
            if result.model_label not in model_stats:
                model_stats[result.model_label] = {
                    "trials": 0,
                    "passed": 0,
                    "total_words": 0,
                    "vocab_coverage": [],
                    "execution_times": []
                }
            
            model_stats[result.model_label]["trials"] += 1
            if result.success and result.validation.get("pass", False):
                model_stats[result.model_label]["passed"] += 1
            if result.success:
                model_stats[result.model_label]["total_words"] += result.validation.get("total_words", 0)
                model_stats[result.model_label]["vocab_coverage"].append(
                    result.validation.get("vocabulary_coverage", 0)
                )
            model_stats[result.model_label]["execution_times"].append(result.execution_time)
            
            # Language stats
            if result.language not in language_stats:
                language_stats[result.language] = {
                    "trials": 0,
                    "passed": 0,
                    "vocab_coverage": []
                }
            
            language_stats[result.language]["trials"] += 1
            if result.success and result.validation.get("pass", False):
                language_stats[result.language]["passed"] += 1
            if result.success:
                language_stats[result.language]["vocab_coverage"].append(
                    result.validation.get("vocabulary_coverage", 0)
                )
        
        # Calculate averages
        for model, stats in model_stats.items():
            stats["pass_rate"] = stats["passed"] / stats["trials"] if stats["trials"] > 0 else 0
            stats["avg_words"] = stats["total_words"] / stats["trials"] if stats["trials"] > 0 else 0
            stats["avg_vocab_coverage"] = (
                sum(stats["vocab_coverage"]) / len(stats["vocab_coverage"])
                if stats["vocab_coverage"] else 0
            )
            stats["avg_execution_time"] = (
                sum(stats["execution_times"]) / len(stats["execution_times"])
                if stats["execution_times"] else 0
            )
            # Remove raw lists
            del stats["vocab_coverage"]
            del stats["execution_times"]
        
        for lang, stats in language_stats.items():
            stats["pass_rate"] = stats["passed"] / stats["trials"] if stats["trials"] > 0 else 0
            stats["avg_vocab_coverage"] = (
                sum(stats["vocab_coverage"]) / len(stats["vocab_coverage"])
                if stats["vocab_coverage"] else 0
            )
            del stats["vocab_coverage"]
        
        return {
            "total_trials": len(results),
            "models": model_stats,
            "languages": language_stats,
            "timestamp": dt.datetime.now().isoformat()
        }
    
    def display_results_table(self, results: List[StructuredTrialResult]):
        """Display results in a beautiful table."""
        if not results:
            return
        
        # Aggregate results
        aggregated = {}
        for result in results:
            key = (result.model_label, result.language, result.vocab_size)
            if key not in aggregated:
                aggregated[key] = {
                    "trials": 0,
                    "passed": 0,
                    "total_words": 0,
                    "vocab_coverage": [],
                    "execution_times": []
                }
            
            aggregated[key]["trials"] += 1
            if result.success and result.validation.get("pass", False):
                aggregated[key]["passed"] += 1
            if result.success:
                aggregated[key]["total_words"] += result.validation.get("total_words", 0)
                aggregated[key]["vocab_coverage"].append(
                    result.validation.get("vocabulary_coverage", 0)
                )
            aggregated[key]["execution_times"].append(result.execution_time)
        
        # Create table
        table = Table(title="📊 Benchmark Results", expand=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Language", style="magenta")
        table.add_column("Vocab Size", style="yellow")
        table.add_column("Pass Rate", style="green")
        table.add_column("Vocab Coverage", style="blue")
        table.add_column("Avg Words", style="white")
        table.add_column("Avg Time", style="dim")
        
        for (model, lang, vocab_size), stats in sorted(aggregated.items()):
            pass_rate = stats["passed"] / stats["trials"] if stats["trials"] > 0 else 0
            avg_coverage = (
                sum(stats["vocab_coverage"]) / len(stats["vocab_coverage"])
                if stats["vocab_coverage"] else 0
            )
            avg_words = stats["total_words"] / stats["trials"] if stats["trials"] > 0 else 0
            avg_time = (
                sum(stats["execution_times"]) / len(stats["execution_times"])
                if stats["execution_times"] else 0
            )
            
            # Color code pass rate
            if pass_rate >= 0.8:
                pass_rate_str = f"[bold green]{pass_rate:.1%}[/bold green]"
            elif pass_rate >= 0.5:
                pass_rate_str = f"[yellow]{pass_rate:.1%}[/yellow]"
            else:
                pass_rate_str = f"[red]{pass_rate:.1%}[/red]"
            
            table.add_row(
                model,
                lang,
                str(vocab_size),
                pass_rate_str,
                f"{avg_coverage:.1%}",
                f"{avg_words:.0f}",
                f"{avg_time:.1f}s"
            )
        
        self.console.print("\n")
        self.console.print(table)