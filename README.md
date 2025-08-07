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

If you prefer exporting:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

### Inputs

- Vocabulary file: newline-separated words, e.g. `examples/en_vocab_sample.txt`
- Target words: newline-separated or CLI list, e.g. `examples/en_targets.txt`

All matching is case-insensitive with simple normalization. The story must use only words in the vocabulary and must include all target words as standalone tokens.

### Quick start

```bash
python -m vocab_story_bench.cli \
  --vocab-file examples/en_vocab_sample.txt \
  --targets-file examples/en_targets.txt \
  --models-config configs/models.yaml \
  --story-words 180 \
  --trials 3 \
  --output-dir runs/demo
```

### Auto-include all GPT‑5 variants

Use dynamic discovery to include all available GPT‑5 models in your OpenAI account:

```bash
python -m vocab_story_bench.cli \
  --vocab-file examples/en_vocab_sample.txt \
  --targets-file examples/en_targets.txt \
  --models-config configs/models.yaml \
  --openai-prefixes gpt-5 \
  --story-words 180 \
  --trials 3 \
  --output-dir runs/gpt5
```

You can pass multiple prefixes: `--openai-prefixes gpt-5 gpt-4.1`.

### Output

- JSON and CSV summaries in the `--output-dir` (default: `outputs/{timestamp}/`)
- Per-model JSONL details with trial-level results

### Notes on validation

- Tokenization is Unicode-aware and strips punctuation/hyphens. Words are compared after lowercase normalization.
- Hyphens are treated as separators. Apostrophes are removed in tokens.
- If you need different normalization rules, see `vocab_story_bench/validator.py`.

### Limitations

- Very large vocabularies may approach context limits. You can try reducing `--story-words` or using models with larger context windows.

### License

MIT
