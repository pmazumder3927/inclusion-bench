## Vocab-Only Story Benchmark

Benchmark LLMs on a strict lexical control task: given a vocabulary (e.g., the top 500–5000 words of a language), write a short story that uses only words from that vocabulary and must include a set of target words.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

### Table of contents
- Quickstart
- Data: frequency lists → vocab files
- Running benchmarks (single and batch)
- Models configuration
- Outputs and dashboards
- Validation rules
- Troubleshooting
- License

### Quickstart
1) Create a virtualenv and install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Set API keys (copy `.env.example` → `.env` and fill, or export env vars)

```bash
cp .env.example .env
# then edit .env to add your keys
```

3) Fetch vocab data and run a demo

```bash
# Fetch frequency lists and build top-N vocab
python -m vocab_story_bench.cli fetch \
  --langs en es fr de it pt ru zh ja ko ar hi \
  --top 3000 \
  --data-dir data/vocab

# Run a single benchmark over multiple models from YAML
python -m vocab_story_bench.cli run \
  --vocab-file data/vocab/top/en_top3000.txt \
  --targets hope river child smile stone \
  --models-config configs/models.yaml \
  --story-words 180 \
  --trials 3 \
  --output-dir runs/demo
```

### Data: frequency lists → vocab files
We use the HermitDave OpenSubtitles 2018 frequency lists and convert them into top-N vocabulary files. The `fetch` command handles download and conversion. Language codes are standard ISO (e.g., `en`, `es`, `fr`, …). Note: `zh` maps to Simplified Chinese (`zh_cn`).

```bash
python -m vocab_story_bench.cli fetch \
  --langs en es fr de it pt ru zh ja ko ar hi \
  --top 2000 \
  --data-dir data/vocab
```

This produces files like `data/vocab/top/en_top2000.txt` (one token per line).

### Running benchmarks
You can run a single benchmark or a batch across multiple languages and vocab sizes.

- Single run (inline models example):

```bash
python -m vocab_story_bench.cli run \
  --vocab-file data/vocab/top/en_top2000.txt \
  --targets river child stone \
  --models openai:gpt-4.1 anthropic:claude-sonnet-4-20250514 openrouter:qwen/qwen3-14b \
  --story-words 150 \
  --trials 2 \
  --output-dir runs/en_top2000_demo
```

- Batch runs with dashboard:

```bash
python -m vocab_story_bench.cli batch \
  --langs en es \
  --sizes 1000 2000 3000 \
  --data-dir data/vocab \
  --targets hope river child smile stone \
  --language-label English \
  --models-config configs/models.yaml \
  --story-words 150 \
  --trials 1 \
  --out-root runs/batch_demo
```

An interactive Plotly dashboard will be saved to `runs/batch_demo/dashboard.html`.

- Build an overall dashboard across multiple run roots:

```bash
python -m vocab_story_bench.cli overall \
  --roots runs/full_en runs/full_es runs/full_de \
  --out runs/overall_dashboard.html
```

### Models configuration
You can specify models inline (e.g., `openai:gpt-4.1`) or via a YAML file. See `configs/models.yaml` for examples across OpenAI, Anthropic, and OpenRouter providers. Labels are optional and used for display.

```yaml
models:
  - provider: openai
    model: gpt-5
    label: gpt5
  - provider: anthropic
    model: claude-sonnet-4-20250514
    label: claude-sonnet-4
  - provider: openrouter
    model: qwen/qwen3-14b
    label: qwen3-14b
```

OpenAI models can also be auto-discovered by prefix using `--openai-prefixes gpt-5 gpt-4.1`.

### API keys
Put keys in a `.env` file (recommended) or export as environment variables:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
OPENROUTER_HTTP_REFERER=https://example.com
OPENROUTER_X_TITLE=Vocab Story Benchmark
```

### Outputs and dashboards
Each run directory contains:
- `summary.json` and `summary.csv`: per-model pass rates and averages
- `details.jsonl`: trial-level stories and validation info
- For batch runs: `dashboard.html` with interactive plots

Columns in `summary.csv` include `model`, `trials`, `pass_rate`, `avg_oov_types`, `avg_missing_targets`. Batch/overall dashboards visualize pass rate by model/language, OOV vs missing targets, and pass rate vs vocab size.

### Validation rules
- Tokenization is Unicode-aware and treats punctuation and hyphens as separators.
- Text is normalized using NFKC and lowercased.
- A story passes only if it uses only vocabulary words and includes all target words exactly as standalone tokens.

See `vocab_story_bench/validator.py` for details.

### Troubleshooting
- Missing API key: providers will raise errors like `OPENAI_API_KEY is not set`. Ensure `.env` is loaded (we auto-load via `python-dotenv`).
- OpenRouter headers: set `OPENROUTER_HTTP_REFERER` and `OPENROUTER_X_TITLE` if required by your account/org policy.
- HTTP 429 / rate limits: reduce `--trials`, remove high-cost models, or retry later.
- Empty dashboards: ensure your run directories contain `summary.json` files.

### License
MIT
