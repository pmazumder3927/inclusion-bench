from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import httpx

BASE_URL = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/{lang}/{file}"

# Map user-friendly codes to repo directory codes
LANG_MAP = {
    "zh": "zh_cn",  # default to Simplified Chinese
}


def _resolve_lang(lang: str) -> str:
    return LANG_MAP.get(lang.lower(), lang.lower())


def _try_download(url: str) -> Optional[bytes]:
    with httpx.Client(timeout=120) as client:
        r = client.get(url)
        if r.status_code == 200:
            return r.content
        return None


def download_language_any(lang: str, dest_dir: str | Path) -> Path:
    """Download language list, preferring full then 50k. Returns local path.
    Raises if nothing available.
    """
    lang_dir = _resolve_lang(lang)
    candidates = [
        f"{lang_dir}_full.txt",
        f"{lang_dir}_50k.txt",
    ]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for fname in candidates:
        url = BASE_URL.format(lang=lang_dir, file=fname)
        blob = _try_download(url)
        if blob is not None:
            dest_path = dest_dir / fname
            dest_path.write_bytes(blob)
            return dest_path

    raise FileNotFoundError(f"No frequency list found for language '{lang}' (tried {candidates})")


def build_top_n(lang: str, full_path: str | Path, n: int, out_dir: str | Path) -> Path:
    out_path = Path(out_dir) / f"{_resolve_lang(lang)}_top{n}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    words: List[str] = []
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            words.append(token)
            if len(words) >= n:
                break
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))
    return out_path


def ensure_top_n_for_lang(lang: str, n: int, data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    full_dir = data_dir / "full"
    top_dir = data_dir / "top"
    out_name = f"{_resolve_lang(lang)}_top{n}.txt"
    top_path = top_dir / out_name
    if top_path.exists():
        return top_path

    # Try to reuse existing downloaded file if present
    candidates = [full_dir / f"{_resolve_lang(lang)}_full.txt", full_dir / f"{_resolve_lang(lang)}_50k.txt"]
    full_path: Optional[Path] = None
    for p in candidates:
        if p.exists():
            full_path = p
            break
    if full_path is None:
        full_path = download_language_any(lang, full_dir)

    return build_top_n(lang, full_path, n, top_dir)
