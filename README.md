# Articulating Rules

Prompt evaluation framework with multi-turn articulation support.

## Setup

First, install dependencies (with [uv](https://docs.astral.sh/uv/#installation)) and add API keys:

```bash
uv sync
echo "OPENAI_API_KEY=your-key" >> .env
echo "ANTHROPIC_API_KEY=your-key" >> .env
```

This codebase has datasets for both the resume and paraphrase pipelines, but if you want to regenerate them, run the following commands:

```bash
# For resume (hiring_bias) pipeline, either run:
uv run scripts/generate_resumes.py configs/generate_resumes.yaml  # generates synthetic resumes and both base/easy example files
# or
uv run scripts/extract_data_from_resumes_csv.py  # requires resumes.csv from https://github.com/adamkarvonen/llm_bias

# For paraphrase_detection pipeline:
uv run scripts/json_dump.py  # contains samples from ~70 essays, dumps to an editable JSON file
uv run scripts/generate_paraphrases.py configs/generate_paraphrases.yaml  # generates paraphrases for every sample
uv run scripts/make_examples_paraphrases.py  # generates examples for system prompt (format depends on pipeline type)
```

Run the pipeline with:

```bash
# For resume (hiring_bias) pipeline
uv run main.py configs/hiring_bias.yaml

# For paraphrase_detection pipeline
uv run main.py configs/paraphrase_detection.yaml
```

You can vary the parameters in the yamls to try out different models or settings.

## Core Pipeline

**Stages:**
1. **Inference** - Query models with prompts
2. **Evaluation** - Frequency analysis or classifier-based evaluation
3. **Articulation Inference** - Multi-turn follow-up queries (optional)
4. **Articulation Evaluation** - Score or binary judgment of articulations (optional)
5. **Analysis** - Plot results and generate summary

**Input Types:**
- `hiring_bias` - Resume testing with demographic name variations only
- `hiring_bias_easy` - Resume testing with explicit demographic info in LinkedIn profile picture
- `paraphrase_detection` - Two-sample comparison (Sample 1 vs Sample 2)
- `paraphrase_detection_hard` - Single-sample classification (yes/no for original)

Results saved to timestamped `runs/` directory with logs, plots, and summaries.

## Directory Structure

```
configs/           # Experiment configurations
prompts/
  system_prompts/  # System prompts
  user_prompts/    # User prompts and templates
  examples/        # Few-shot examples and generated datasets
  classifiers/     # Judge prompts for evaluation
data/              # Input datasets
runs/              # Timestamped experiment outputs
scripts/           # Dataset generation utilities
```

## Scripts

**Fine-tuning:**

```bash
# Fine-tune OpenAI models
uv run finetune.py configs/finetune.yaml
```

## Config Examples

**Resume bias detection:**
```yaml
inputs:
  - type: "hiring_bias"  # or "hiring_bias_easy" for explicit demographic info
    dataset: "prompts/examples/resumes.json"
    template: "prompts/user_prompts/resume_template.txt"  # or resume_template_easy.txt
    num_samples: 100
```

**Paraphrase detection:**
```yaml
inputs:
  - type: "paraphrase_detection"  # or "paraphrase_detection_hard" for yes/no format
    dataset: "prompts/examples/paraphrases.json"
    template: "prompts/user_prompts/paraphrase_template.txt"
    num_samples: 50
```

**Multi-turn articulation:**
```yaml
evaluation:
  articulation:
    query_prompt: "prompts/user_prompts/articulation_prompt.txt"
    judge: "prompts/classifiers/articulation_judge.txt"
    judge_type: "score"  # or "binary"
```

Use `{{RESPONSE}}` in articulation prompts to split into multiple turns.
