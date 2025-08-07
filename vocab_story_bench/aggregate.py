from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json
import pandas as pd


def _parse_lang_size(run_dir: Path) -> tuple[str | None, int | None]:
    name = run_dir.name
    # Expect pattern like: <lang>_top<size>
    if "_top" in name:
        lang, top_part = name.split("_top", 1)
        try:
            size = int(top_part)
        except ValueError:
            size = None
        return lang, size
    return None, None


def load_run_summary(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    p = run_dir / "summary.json"
    if not p.exists():
        return pd.DataFrame()
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("summary", []) if isinstance(data, dict) else data
    df = pd.DataFrame(rows)
    df["run_dir"] = str(run_dir)
    lang, size = _parse_lang_size(run_dir)
    if lang is not None:
        df["language"] = lang
    if size is not None:
        df["vocab_size"] = size
    return df


def merge_runs(run_dirs: List[str | Path]) -> pd.DataFrame:
    dfs = [load_run_summary(d) for d in run_dirs]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def scan_run_roots(roots: List[str | Path]) -> List[str]:
    run_dirs: List[str] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        # children directories under root
        for child in root_path.iterdir():
            if child.is_dir() and (child / "summary.json").exists():
                run_dirs.append(str(child))
    return run_dirs
