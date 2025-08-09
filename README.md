# Vocabulary Inclusion Benchmark

A beautiful, high-performance benchmark for testing language models with vocabulary constraints. Models generate structured outputs (word arrays) for precise validation of vocabulary inclusion/exclusion.

## Features

- **Structured Output**: Models generate JSON arrays of words for precise validation
- **Parallel Execution**: Run benchmarks across multiple languages and models simultaneously  
- **Beautiful CLI**: Interactive mode with rich formatting and real-time progress tracking
- **Multi-Provider Support**: OpenAI (GPT-5, GPT-4) and Anthropic (Claude) models
- **Comprehensive Metrics**: Pass rates, vocabulary coverage, execution times, and more

## Installation

```bash
git clone https://github.com/yourusername/inclusion-bench.git
cd inclusion-bench
pip install -r requirements.txt
```

## Quick Start

### Interactive Mode (Recommended)

The easiest way to run benchmarks:

```bash
python -m vocab_story_bench --interactive
```

This will guide you through:
1. Language selection (or 'all' for all 12 languages)
2. Vocabulary size configuration (1000, 2000, 3000, etc.)
3. Model selection (from config or manual entry)
4. Target word configuration (automatically selected from each language's vocabulary)
5. Benchmark execution with live progress

### Command Line Mode

```bash
# Fetch vocabulary data for all languages
python -m vocab_story_bench fetch --langs all --top 3000

# Or fetch specific languages
python -m vocab_story_bench fetch --langs en es fr --top 3000

# Run benchmark with automatic target selection
python -m vocab_story_bench run \
  --languages en es fr \
  --vocab-sizes 1000 2000 \
  --models-config configs/models.yaml \
  --num-targets 5 \
  --trials 3

# Run benchmark for all languages
python -m vocab_story_bench run \
  --languages all \
  --vocab-sizes 1000 \
  --models-config configs/models.yaml
```

## Configuration

### Models Configuration (configs/models.yaml)

```yaml
models:
  - provider: openai
    model: gpt-5
    label: gpt5
    params:
      reasoning_effort: medium
      temperature: 0.7
  
  - provider: anthropic
    model: claude-3-5-sonnet-20241022
    label: claude-3.5-sonnet
    params:
      temperature: 0.7
```

### Environment Variables (.env)

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## How It Works

1. **Vocabulary Constraints**: Models receive a limited vocabulary list and must generate stories using only those words
2. **Target Words**: Automatically selected from each language's vocabulary (5 words by default from the middle range of frequency)
3. **Structured Output**: Models generate JSON arrays of words: `{"words": ["the", "child", "found", "..."]}`
4. **Validation**: Each word is validated against the vocabulary and target requirements
5. **Parallel Processing**: Multiple models and languages are tested simultaneously for efficiency
6. **Language Support**: 12 languages supported (en, es, fr, de, it, pt, ru, zh, ja, ko, ar, hi)

## Output Format

### Results Directory Structure
```
outputs/parallel_20240809_143022/
├── details.jsonl          # Detailed trial results
├── summary.json           # Aggregated statistics
└── dashboard.html         # Visual dashboard
```

### Trial Result Example
```json
{
  "model_label": "gpt5",
  "language": "en",
  "vocab_size": 1000,
  "trial_index": 0,
  "words": ["the", "child", "found", "a", "stone", "by", "the", "river"],
  "validation": {
    "only_vocab": true,
    "all_targets_present": true,
    "vocabulary_coverage": 0.85,
    "total_words": 150
  },
  "execution_time": 2.3,
  "success": true
}
```

## Metrics

- **Pass Rate**: Percentage of trials meeting all vocabulary constraints
- **Vocabulary Coverage**: How much of the allowed vocabulary is utilized
- **Execution Time**: Average time per trial
- **Success Rate**: Percentage of trials completing without errors
- **Word Count**: Average number of words generated

## CLI Options

### Fetch Command
```bash
python -m vocab_story_bench fetch --langs LANGS [--top N]
```
- `--langs`: Language codes or 'all' for all 12 languages
- `--top`: Number of top frequency words to fetch (default: 2000)

### Run Command
```bash
python -m vocab_story_bench run [OPTIONS]
```
- `--interactive, -i`: Run in interactive mode
- `--languages, -l`: Languages to test or 'all' for all languages
- `--vocab-sizes, -v`: Vocabulary sizes (e.g., 1000 2000 3000)
- `--models-config, -m`: Path to models configuration file
- `--num-targets`: Number of target words per language (default: 5)
- `--trials`: Number of trials per model (default: 3)
- `--story-length`: Approximate story length in words (default: 150)
- `--parallel`: Max parallel model executions (default: 8)
- `--output-dir, -o`: Output directory (default: outputs)

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys
echo "OPENAI_API_KEY=sk-..." > .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 3. Fetch vocabulary data for all languages
python -m vocab_story_bench fetch --langs all --top 2000

# 4. Run interactive benchmark
python -m vocab_story_bench --interactive

# 5. Or run with specific parameters
python -m vocab_story_bench run \
  --languages en es \
  --vocab-sizes 1000 \
  --models-config configs/models.yaml \
  --num-targets 5 \
  --trials 3

# 6. Run for all languages
python -m vocab_story_bench run \
  --languages all \
  --vocab-sizes 1000 \
  --models-config configs/models.yaml

# 7. View results
cat outputs/parallel_*/summary.json | jq
```

## Architecture

```
User Input → Beautiful CLI → Parallel Runner
                                    ↓
                          Thread Pool Executor
                          ↙        ↓        ↘
                   Language 1   Language 2   Language 3
                       ↓            ↓            ↓
                   [Models]     [Models]     [Models]
                       ↓            ↓            ↓
                  Structured    Structured   Structured
                   Outputs      Outputs      Outputs
                       ↓            ↓            ↓
                  Validation    Validation   Validation
                       ↘           ↓           ↙
                         Results Aggregation
                                ↓
                        Beautiful Results Table
```

## Adding New Providers

To add support for new LLM providers:

1. Create a provider in `vocab_story_bench/providers/`:
```python
class StructuredYourProvider:
    def generate_structured(self, model, system, user, response_format, ...):
        # Implementation
        return json_string
```

2. Register in `parallel_runner.py`:
```python
self.providers["your_provider"] = StructuredYourProvider()
```

## Performance

- **Parallel Execution**: Run hundreds of trials in minutes
- **Structured Outputs**: Eliminate parsing ambiguities
- **Efficient Validation**: Direct word-by-word checking
- **Thread Pool**: Configurable parallelization levels

## License

MIT License - see LICENSE file for details