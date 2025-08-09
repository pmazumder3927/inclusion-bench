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
import hashlib

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
    # Core identifiers
    model_label: str
    model: str
    language: str
    vocab_size: int
    trial_index: int

    # Prompt/generation context
    story_length: int
    targets: List[str]
    model_params: Optional[Dict[str, Any]]

    # Output and validation
    words: List[str]  # Array of words instead of story text
    validation: Dict[str, Any]

    # Timing and meta
    execution_time: float
    started_at: str
    ended_at: str
    run_id: Optional[str]
    vocabulary_hash: Optional[str]
    vocabulary_count: Optional[int]

    # Status (non-defaults must come before any default fields)
    success: bool
    error: Optional[str] = None

    # Optional debug preview (must come after non-defaults)
    raw_preview: Optional[str] = None


class ParallelBenchmarkRunner:
    """Runs benchmarks in parallel across models and languages."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.console = Console()
        self.provider = UnifiedProvider()
        self.results_lock = Lock()
        self.all_results = []
        self.current_run_id: Optional[str] = None
        self.run_context: Dict[str, Any] = {}
    
    def validate_words(
        self,
        words: List[str],
        vocabulary: Set[str],
        targets: List[str]
    ) -> Dict[str, Any]:
        """Validate words against vocabulary and targets."""
        total_words = len(words)
        lower_words = [w.lower() for w in words]
        unique_words = len(set(lower_words))
        only_vocab = all(w in vocabulary for w in lower_words)

        # Find OOV details
        oov_indices = [i for i, w in enumerate(lower_words) if w not in vocabulary]
        oov_words = sorted(list({lower_words[i] for i in oov_indices}))

        # Target details (substring match to align with existing logic)
        target_positions: Dict[str, List[int]] = {}
        for t in targets:
            t_lower = t.lower()
            positions = [i for i, w in enumerate(lower_words) if t_lower in w]
            if positions:
                target_positions[t] = positions

        all_targets_present = all(
            any(t.lower() in w for w in lower_words) for t in targets
        ) if targets else True

        # Coverage and diversity
        vocab_overlap = len(set(lower_words) & vocabulary)
        vocabulary_coverage = (vocab_overlap / len(vocabulary)) if vocabulary else 0.0
        percent_in_vocab = (sum(1 for w in lower_words if w in vocabulary) / total_words) if total_words > 0 else 0.0
        targets_present_ratio = (len(target_positions) / len(targets)) if targets else 1.0
        type_token_ratio = (unique_words / total_words) if total_words > 0 else 0.0

        # Repetition metrics
        max_repeat_run = 0
        current_run = 0
        last_w = None
        for w in lower_words:
            if w == last_w:
                current_run += 1
            else:
                current_run = 1
                last_w = w
            if current_run > max_repeat_run:
                max_repeat_run = current_run

        # Bigram repetition rate
        bigrams = list(zip(lower_words, lower_words[1:])) if total_words >= 2 else []
        unique_bigrams = len(set(bigrams)) if bigrams else 0
        repeated_bigrams = len(bigrams) - unique_bigrams if bigrams else 0
        repeated_bigrams_rate = (repeated_bigrams / len(bigrams)) if bigrams else 0.0

        missing_targets = [t for t in targets if t not in target_positions]

        validation: Dict[str, Any] = {
            "total_words": total_words,
            "unique_words": unique_words,
            "type_token_ratio": type_token_ratio,
            "only_vocab": only_vocab,
            "percent_in_vocab": percent_in_vocab,
            "oov_words": oov_words,
            "num_oov": len(oov_indices),
            "oov_indices": oov_indices,
            "all_targets_present": all_targets_present,
            "targets_present_ratio": targets_present_ratio,
            "target_positions": target_positions,
            "missing_targets": missing_targets,
            "target_words_found": list(target_positions.keys()),
            "vocabulary_coverage": vocabulary_coverage,
            "max_repeat_run": max_repeat_run,
            "unique_bigrams": unique_bigrams,
            "repeated_bigrams_rate": repeated_bigrams_rate,
        }

        # Simple pass/fail and reasons to aid visualization
        fail_reasons: List[str] = []
        if not only_vocab:
            fail_reasons.append("oov_words_present")
        if not all_targets_present:
            fail_reasons.append("missing_targets")
        validation["pass"] = (len(fail_reasons) == 0)
        validation["fail_reasons"] = fail_reasons

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
        started_at_iso = dt.datetime.utcnow().isoformat()
        
        try:
            # Build structured prompt + JSON schema response_format
            system_prompt, user_prompt, response_format = build_structured_prompt(
                vocabulary=vocabulary,
                targets=targets,
                language=language,
                story_length=story_length
            )
            
            # Generate structured output
            # Set a conservative max_tokens based on desired story length
            # Each word token roughly ~1.3 tokens avg; add headroom for JSON and overhead
            est_tokens = int(story_length * 2.0 + 200)
            response = self.provider.generate_structured(
                model=model_spec.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                max_tokens=25000,
                **(model_spec.params or {})
            )
            
            # Parse structured response
            words = parse_structured_response(response)
            raw_preview = response[:500] if isinstance(response, str) else None
            
            # Validate
            vocab_set = set(v.lower() for v in vocabulary)
            validation = self.validate_words(words, vocab_set, targets)
            
            execution_time = time.time() - start_time
            ended_at_iso = dt.datetime.utcnow().isoformat()

            # Vocab meta
            vocab_hash = hashlib.sha256("\n".join(vocabulary).encode("utf-8")).hexdigest() if vocabulary else None
            
            result = StructuredTrialResult(
                model_label=model_spec.label,
                model=model_spec.model,
                language=language,
                vocab_size=vocab_size,
                trial_index=trial_index,
                story_length=story_length,
                targets=targets,
                model_params=(model_spec.params or {}),
                words=words,
                validation=validation,
                execution_time=execution_time,
                started_at=started_at_iso,
                ended_at=ended_at_iso,
                run_id=self.current_run_id,
                vocabulary_hash=vocab_hash,
                vocabulary_count=len(vocabulary) if vocabulary else None,
                raw_preview=raw_preview,
                # Attach short raw preview to aid debugging when words are empty
                # This keeps details.jsonl compact while still useful
                success=True
            )
            
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            ended_at_iso = dt.datetime.utcnow().isoformat()
            
            result = StructuredTrialResult(
                model_label=model_spec.label,
                model=model_spec.model,
                language=language,
                vocab_size=vocab_size,
                trial_index=trial_index,
                story_length=story_length,
                targets=targets,
                model_params=(model_spec.params or {}),
                words=[],
                validation={"pass": False, "error": str(e), "fail_reasons": ["exception"]},
                execution_time=execution_time,
                started_at=started_at_iso,
                ended_at=ended_at_iso,
                run_id=self.current_run_id,
                vocabulary_hash=None,
                vocabulary_count=len(vocabulary) if vocabulary else None,
                raw_preview=None,
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
        self.current_run_id = run_dir.name
        # Persist run context for summary
        self.run_context = {
            "run_id": self.current_run_id,
            "created_at": dt.datetime.utcnow().isoformat(),
            "output_dir": str(run_dir),
            "languages": languages,
            "vocab_sizes": vocab_sizes,
            "models": [
                {
                    "label": m.label,
                    "model": m.model,
                    "params": (m.params or {})
                }
                for m in models
            ],
            "trials": trials,
            "story_length": story_length,
            "targets": targets,
        }
        
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
        """Generate rich summary statistics from results for comprehensive visualization."""
        if not results:
            return {"run": self.run_context, "total_trials": 0, "models": {}, "languages": {}, "timestamp": dt.datetime.utcnow().isoformat()}

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            data_sorted = sorted(data)
            k = (len(data_sorted) - 1) * p
            f = int(k)
            c = min(f + 1, len(data_sorted) - 1)
            if f == c:
                return float(data_sorted[int(k)])
            d0 = data_sorted[f] * (c - k)
            d1 = data_sorted[c] * (k - f)
            return float(d0 + d1)

        # Group by model, language, and combination
        model_stats: Dict[str, Dict[str, Any]] = {}
        language_stats: Dict[str, Dict[str, Any]] = {}
        combo_stats: Dict[str, Dict[str, Any]] = {}

        for result in results:
            coverage = result.validation.get("vocabulary_coverage", 0) if result.success else 0
            words_cnt = result.validation.get("total_words", 0) if result.success else 0
            passed = 1 if (result.success and result.validation.get("pass", False)) else 0
            oov = result.validation.get("num_oov", 0) if result.validation else 0
            ttr = result.validation.get("type_token_ratio", 0) if result.validation else 0

            # Model stats
            ms = model_stats.setdefault(result.model_label, {
                "trials": 0,
                "passed": 0,
                "total_words": 0,
                "vocab_coverage_list": [],
                "execution_times": [],
                "oov_list": [],
                "ttr_list": [],
            })
            ms["trials"] += 1
            ms["passed"] += passed
            ms["total_words"] += words_cnt
            ms["vocab_coverage_list"].append(coverage)
            ms["execution_times"].append(result.execution_time)
            ms["oov_list"].append(oov)
            ms["ttr_list"].append(ttr)

            # Language stats
            ls = language_stats.setdefault(result.language, {
                "trials": 0,
                "passed": 0,
                "vocab_coverage_list": [],
            })
            ls["trials"] += 1
            ls["passed"] += passed
            ls["vocab_coverage_list"].append(coverage)

            # Combination (model|language|vocab_size)
            key = f"{result.model_label}|{result.language}|{result.vocab_size}"
            cs = combo_stats.setdefault(key, {
                "model": result.model_label,
                "language": result.language,
                "vocab_size": result.vocab_size,
                "trials": 0,
                "passed": 0,
                "vocab_coverage_list": [],
                "execution_times": [],
                "words_list": [],
                "percent_in_vocab_list": [],
                "targets_present_ratio_list": [],
                "oov_list": [],
                "ttr_list": [],
            })
            cs["trials"] += 1
            cs["passed"] += passed
            cs["vocab_coverage_list"].append(coverage)
            cs["execution_times"].append(result.execution_time)
            cs["words_list"].append(words_cnt)
            cs["percent_in_vocab_list"].append(result.validation.get("percent_in_vocab", 0))
            cs["targets_present_ratio_list"].append(result.validation.get("targets_present_ratio", 0))
            cs["oov_list"].append(oov)
            cs["ttr_list"].append(ttr)

        # Calculate aggregates
        for stats in model_stats.values():
            stats["pass_rate"] = stats["passed"] / stats["trials"] if stats["trials"] else 0
            stats["avg_words"] = stats["total_words"] / stats["trials"] if stats["trials"] else 0
            stats["avg_vocab_coverage"] = sum(stats["vocab_coverage_list"]) / len(stats["vocab_coverage_list"]) if stats["vocab_coverage_list"] else 0
            stats["avg_execution_time"] = sum(stats["execution_times"]) / len(stats["execution_times"]) if stats["execution_times"] else 0
            stats["p50_time"] = percentile(stats["execution_times"], 0.50)
            stats["p95_time"] = percentile(stats["execution_times"], 0.95)
            stats["avg_oov"] = sum(stats["oov_list"]) / len(stats["oov_list"]) if stats["oov_list"] else 0
            stats["avg_ttr"] = sum(stats["ttr_list"]) / len(stats["ttr_list"]) if stats["ttr_list"] else 0
            # Remove raw lists to keep summary light
            del stats["vocab_coverage_list"]
            del stats["execution_times"]
            del stats["oov_list"]
            del stats["ttr_list"]

        for stats in language_stats.values():
            stats["pass_rate"] = stats["passed"] / stats["trials"] if stats["trials"] else 0
            stats["avg_vocab_coverage"] = sum(stats["vocab_coverage_list"]) / len(stats["vocab_coverage_list"]) if stats["vocab_coverage_list"] else 0
            del stats["vocab_coverage_list"]

        for stats in combo_stats.values():
            stats["pass_rate"] = stats["passed"] / stats["trials"] if stats["trials"] else 0
            stats["avg_vocab_coverage"] = sum(stats["vocab_coverage_list"]) / len(stats["vocab_coverage_list"]) if stats["vocab_coverage_list"] else 0
            stats["avg_execution_time"] = sum(stats["execution_times"]) / len(stats["execution_times"]) if stats["execution_times"] else 0
            stats["p50_time"] = percentile(stats["execution_times"], 0.50)
            stats["p95_time"] = percentile(stats["execution_times"], 0.95)
            stats["avg_words"] = sum(stats["words_list"]) / len(stats["words_list"]) if stats["words_list"] else 0
            stats["avg_percent_in_vocab"] = sum(stats["percent_in_vocab_list"]) / len(stats["percent_in_vocab_list"]) if stats.get("percent_in_vocab_list") else 0
            stats["avg_targets_present_ratio"] = sum(stats["targets_present_ratio_list"]) / len(stats["targets_present_ratio_list"]) if stats.get("targets_present_ratio_list") else 0
            stats["avg_oov"] = sum(stats["oov_list"]) / len(stats["oov_list"]) if stats["oov_list"] else 0
            stats["avg_ttr"] = sum(stats["ttr_list"]) / len(stats["ttr_list"]) if stats["ttr_list"] else 0
            # Trim heavy raw lists
            del stats["vocab_coverage_list"]
            del stats["execution_times"]
            del stats["words_list"]
            if "percent_in_vocab_list" in stats:
                del stats["percent_in_vocab_list"]
            if "targets_present_ratio_list" in stats:
                del stats["targets_present_ratio_list"]
            del stats["oov_list"]
            del stats["ttr_list"]

        return {
            "run": self.run_context,
            "total_trials": len(results),
            "models": model_stats,
            "languages": language_stats,
            "by_model_language_vocab": combo_stats,
            "timestamp": dt.datetime.utcnow().isoformat(),
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
                    "success": 0,
                    "total_words": 0,
                    "vocab_coverage": [],
                    "percent_in_vocab": [],
                    "targets_present_ratio": [],
                    "execution_times": []
                }
            
            aggregated[key]["trials"] += 1
            if result.success and result.validation.get("pass", False):
                aggregated[key]["passed"] += 1
            if result.success:
                aggregated[key]["success"] += 1
            if result.success:
                aggregated[key]["total_words"] += result.validation.get("total_words", 0)
                aggregated[key]["vocab_coverage"].append(
                    result.validation.get("vocabulary_coverage", 0)
                )
                aggregated[key]["percent_in_vocab"].append(
                    result.validation.get("percent_in_vocab", 0)
                )
                aggregated[key]["targets_present_ratio"].append(
                    result.validation.get("targets_present_ratio", 0)
                )
            aggregated[key]["execution_times"].append(result.execution_time)
        
        # Create table
        table = Table(title="📊 Benchmark Results", expand=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Language", style="magenta")
        table.add_column("Vocab Size", style="yellow")
        table.add_column("Compliance", style="green")
        table.add_column("Targets", style="blue")
        table.add_column("Success", style="white")
        table.add_column("Avg Words", style="white")
        table.add_column("Avg Time", style="dim")
        
        for (model, lang, vocab_size), stats in sorted(aggregated.items()):
            pass_rate = stats["passed"] / stats["trials"] if stats["trials"] > 0 else 0
            success_rate = stats["success"] / stats["trials"] if stats["trials"] > 0 else 0
            avg_compliance = (
                sum(stats["percent_in_vocab"]) / len(stats["percent_in_vocab"]) if stats["percent_in_vocab"] else 0
            )
            avg_targets = (
                sum(stats["targets_present_ratio"]) / len(stats["targets_present_ratio"]) if stats["targets_present_ratio"] else 0
            )
            avg_words = stats["total_words"] / stats["trials"] if stats["trials"] > 0 else 0
            avg_time = (
                sum(stats["execution_times"]) / len(stats["execution_times"])
                if stats["execution_times"] else 0
            )
            
            # Color code compliance
            comp_color = "green" if avg_compliance >= 0.9 else "yellow" if avg_compliance >= 0.7 else "red"
            targets_color = "green" if avg_targets >= 0.9 else "yellow" if avg_targets >= 0.7 else "red"
            success_color = "green" if success_rate >= 0.9 else "yellow" if success_rate >= 0.7 else "red"
            
            table.add_row(
                model,
                lang,
                str(vocab_size),
                f"[{comp_color}]{avg_compliance:.1%}[/{comp_color}]",
                f"[{targets_color}]{avg_targets:.1%}[/{targets_color}]",
                f"[{success_color}]{success_rate:.1%}[/{success_color}]",
                f"{avg_words:.0f}",
                f"{avg_time:.1f}s"
            )
        
        self.console.print("\n")
        self.console.print(table)