# vocab-bench
![results](https://github.com/pmazumder3927/inclusion-bench/blob/master/examples/results.png)
benchmark language models on vocabulary-constrained generation. models output structured json word arrays for precise validation.

## setup

```bash
git clone https://github.com/yourusername/inclusion-bench.git
cd inclusion-bench
pip install -r requirements.txt
```

set your openrouter api key:
```bash
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

## usage

### interactive mode

```bash
python -m vocab_story_bench --interactive
```

walks you through language selection, vocabulary sizes, model configuration, and benchmark execution.

### command line

```bash
# fetch vocabulary data
python -m vocab_story_bench fetch --langs en es fr --top 3000

# run benchmark
python -m vocab_story_bench run \
  --languages en es fr \
  --vocab-sizes 1000 2000 \
  --models-config configs/models.yaml \
  --trials 3

# visualize existing results
python -m vocab_story_bench viz outputs/parallel_*/details.jsonl
```

## models

configure models in `configs/models.yaml`:

```yaml
models:
  # gpt-5 with reasoning levels
  - model: openai/gpt-5
    label: gpt5-low
    params:
      reasoning_effort: low

  - model: openai/gpt-5
    label: gpt5-high
    params:
      reasoning_effort: high

  # claude 4
  - model: anthropic/claude-opus-4.1
    label: claude-4-opus

  # grok
  - model: x-ai/grok-4
    label: grok-4

  # open models
  - model: qwen/qwen3-8b:free
    label: qwen3-8b
```

all models accessed through openrouter's unified api. see [openrouter.ai/models](https://openrouter.ai/models) for available models.

## how it works

1. models receive a constrained vocabulary (e.g., top 1000 words in a language)
2. they generate stories using only those words
3. output is structured json: `{"words": ["the", "child", "found", "..."]}`
4. validation checks vocabulary compliance and target word inclusion
5. results aggregated across languages and models

## languages

12 languages supported: `en`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `zh`, `ja`, `ko`, `ar`, `hi`

vocabulary data fetched from frequency lists for each language.

## output

benchmark runs create:

```
outputs/parallel_20240809_143022/
├── details.jsonl       # trial-by-trial results
├── summary.json        # aggregated metrics
└── dashboard.html      # interactive visualization
```

### metrics

- **compliance**: percentage of words within allowed vocabulary
- **target inclusion**: ratio of required words present
- **pass rate**: trials meeting all constraints
- **execution time**: median response time
- **coverage**: vocabulary utilization

### visualization

the dashboard provides:
- overall model performance comparison
- vocabulary compliance across sizes
- target inclusion heatmap by model and language
- performance distribution visualizations
- efficiency vs accuracy scatter plots

## cli reference

### fetch
```bash
python -m vocab_story_bench fetch --langs LANGS [--top N]
```
download frequency lists and build vocabulary files.

### run
```bash
python -m vocab_story_bench run [OPTIONS]
```
- `--languages`: languages to test (or 'all')
- `--vocab-sizes`: vocabulary sizes to test
- `--models-config`: path to models yaml
- `--trials`: trials per model (default: 3)
- `--parallel`: max parallel executions (default: 8)

### viz
```bash
python -m vocab_story_bench viz DETAILS_PATH [--out OUTPUT]
```
generate dashboard from existing results.

## architecture

```
input → cli → parallel runner
               ↓
         thread pool
         ↙    ↓    ↘
    model 1  model 2  model 3
         ↓     ↓     ↓
    structured json outputs
         ↓     ↓     ↓
      validation engine
         ↘    ↓    ↙
      results aggregation
              ↓
     dashboard generation
```

## extending

### adding models

edit `configs/models.yaml` to add any openrouter-supported model.

### custom providers

implement the structured generation interface:

```python
class YourProvider:
    def generate_structured(self, model, system_prompt, user_prompt, 
                          response_format, **kwargs):
        # return json string
        return '{"words": [...]}'
```

## performance notes

- parallel execution across models and languages
- structured outputs eliminate parsing ambiguity
- configurable parallelization (default: 8 concurrent)
- typical benchmark: ~100 trials complete in 2-3 minutes

## license

mit
