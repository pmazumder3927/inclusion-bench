# Vocab-Only Story Benchmark

Benchmark LLMs on the task: given an input vocabulary (top 500–5000 words of a language), write a story that ONLY uses words from that vocabulary and MUST include a given list of target words.

### What this does

- Loads a vocabulary file and target words
- Prompts multiple models (OpenAI, Anthropic/Claude, OpenRouter like Qwen)
- Validates the generated story:
  - Only uses allowed words (after simple normalization)
  - Contains all target words
- Produces JSON/CSV/Markdown reports with pass rates and error details

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API keys

You can put keys in a `.env` file (recommended) or export env vars.

`.env` example (copy from `.env.example`):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
OPENROUTER_HTTP_REFERER=https://example.com
OPENROUTER_X_TITLE=Vocab Story Benchmark
```

### Get real vocab data (multilingual)

We use HermitDave OpenSubtitles 2018 frequency lists. Fetch top-N vocab files for multiple languages:

```bash
python -m vocab_story_bench.cli fetch \
  --langs en es fr de it pt ru zh ja ko ar hi \
  --top 3000 \
  --data-dir data/vocab
```

### Run a single benchmark

```bash
python -m vocab_story_bench.cli run \
  --vocab-file data/vocab/top/en_top3000.txt \
  --targets hope river child smile stone \
  --models-config configs/models.yaml \
  --story-words 180 \
  --trials 3 \
  --output-dir runs/demo
```

### Batch runs + dashboard

Run multiple vocab sizes and languages in one go and produce an HTML dashboard:

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

This writes per-run summaries and `runs/batch_demo/dashboard.html` with interactive plots (pass rate by model, OOV vs missing targets).

### Output

- JSON and CSV summaries per run directory
- Per-model JSONL details with trial-level results
- HTML dashboard for batch runs

### Notes on validation

- Tokenization is Unicode-aware and strips punctuation/hyphens. Words are compared after lowercase normalization.
- Hyphens are treated as separators. Apostrophes are removed in tokens.
- If you need different normalization rules, see `vocab_story_bench/validator.py`.

### License

MIT
