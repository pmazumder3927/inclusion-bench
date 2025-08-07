from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml
from tqdm import tqdm

from .model_spec import ModelSpec
from .prompt import build_prompt
from .providers import PROVIDERS
from .reports import append_jsonl, ensure_dir, write_csv, write_json
from .validator import ValidationResult, validate_story


@dataclass
class TrialResult:
    model_label: str
    provider: str
    model: str
    trial_index: int
    story: str
    validation: ValidationResult
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def load_lines(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_models_from_inline(tokens: List[str]) -> List[ModelSpec]:
    return [ModelSpec.parse_inline(t) for t in tokens]


def build_models_from_yaml(path: str | Path) -> List[ModelSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    models: List[ModelSpec] = []
    for item in data.get("models", []):
        models.append(
            ModelSpec(
                provider=item["provider"],
                model=item["model"],
                label=item.get("label"),
            )
        )
    return models


def run_benchmark(
    *,
    models: List[ModelSpec],
    language: str,
    vocabulary: List[str],
    target_words: List[str],
    desired_length_words: int,
    trials: int,
    output_dir: str | Path,
) -> Dict[str, any]:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(output_dir or f"outputs/{ts}")

    details_path = out_dir / "details.jsonl"
    summary_path = out_dir / "summary.json"
    summary_csv_path = out_dir / "summary.csv"

    per_model_records: Dict[str, List[TrialResult]] = {}

    for spec in models:
        provider_cls = PROVIDERS.get(spec.provider)
        if not provider_cls:
            raise ValueError(f"Unknown provider: {spec.provider}")
        provider = provider_cls()

        system, user = build_prompt(language, vocabulary, target_words, desired_length_words)

        model_results: List[TrialResult] = []
        for trial in tqdm(range(trials), desc=f"{spec.display_label}"):
            story = provider.generate(
                spec.model,
                system=system,
                user=user,
                max_output_tokens=max(128, desired_length_words * 3),
            )
            val = validate_story(story, vocabulary, target_words)
            tr = TrialResult(
                model_label=spec.display_label,
                provider=spec.provider,
                model=spec.model,
                trial_index=trial,
                story=story,
                validation=val,
            )
            model_results.append(tr)

            append_jsonl(
                details_path,
                {
                    "model": spec.display_label,
                    "provider": spec.provider,
                    "trial": trial,
                    "story": story,
                    "validation": {
                        "only_vocab": val.only_vocab,
                        "all_targets_present": val.all_targets_present,
                        "oov_words": sorted(val.oov_words),
                        "missing_targets": sorted(val.missing_targets),
                        "total_tokens": val.total_tokens,
                    },
                },
            )

        per_model_records[spec.display_label] = model_results

    # Aggregate summary
    summary_rows = []
    for label, results in per_model_records.items():
        total = len(results)
        passes = sum(1 for r in results if r.validation.only_vocab and r.validation.all_targets_present)
        avg_oov = sum(len(r.validation.oov_words) for r in results) / total if total else 0.0
        avg_missing = sum(len(r.validation.missing_targets) for r in results) / total if total else 0.0
        summary_rows.append(
            {
                "model": label,
                "trials": total,
                "pass_rate": round(passes / total, 4) if total else 0.0,
                "avg_oov_types": round(avg_oov, 2),
                "avg_missing_targets": round(avg_missing, 2),
            }
        )

    write_json(summary_path, summary_rows)
    write_csv(summary_csv_path, summary_rows)

    return {
        "output_dir": str(out_dir),
        "summary": summary_rows,
        "details": str(details_path),
    }
