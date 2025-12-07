# AGMI Creative Engine 🚀

Creative content generation using a divergence-convergence pipeline. Includes both a baseline few-shot approach and an advanced creative engine for comparison.

## Overview

- **Baseline** (`agmi-video-generator/`): Traditional few-shot learning → See [agmi-video-generator/README.md](agmi/agmi-video-generator/README.md)
- **Creative Engine** (`creative_engine/`): Advanced divergence-convergence pipeline with multi-phase generation and quality filtering

## Quick Start

### Installation

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"  # or ANTHROPIC_API_KEY, GOOGLE_API_KEY
```

### Basic Usage

```bash
# Generate a creative video script
python run_creative_engine.py --url https://www.cubic.dev --model gpt-4o

# With creativity evaluation
python run_creative_engine.py --url https://www.cubic.dev --model gpt-4o --evaluate

# Use reference style (varun/austin/mixed)
python run_creative_engine.py --url https://www.cubic.dev --model gpt-4o --reference-style varun

# Maximum creativity
python run_creative_engine.py --url https://www.cubic.dev --model gpt-4o --creativity 1.0
```

### Run Comparison Benchmark

```bash
python comparison_benchmark.py
```

Generates baseline vs creative scripts and produces comparison reports in `outputs/`.

## Supported Models

**OpenAI:** `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`  
**Anthropic:** `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`  
**Gemini:** `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash`

## Programmatic Usage

```python
from creative_engine import CreativeEngine, CreativityConfig, ContentType, extract_product_context
from creative_engine.generation import get_reference_examples

# Initialize
engine = CreativeEngine(
    model="gpt-4o",
    config=CreativityConfig(creativity_level=0.8, quality_threshold=0.75)
)

# Extract context and generate
product_context = extract_product_context("https://www.cubic.dev", engine.provider)
product_context["platform"] = "tiktok"

result = engine.generate(
    product_context=product_context,
    content_type=ContentType.VIDEO_SCRIPT,
    reference_examples=get_reference_examples("varun"),
    evaluate_creativity=True
)

# Access results
print(f"Concept: {result.selected_concept.title}")
print(f"Quality: {result.quality_score:.2f}")
if result.creativity_assessment:
    print(f"Creativity: {result.creativity_assessment.aggregate.overall.mean:.2f}")

# Save
gen_path, eval_path = engine.save_artifacts(result)
```

## Project Structure

```
agmi/
├── requirements.txt           # Unified dependencies
├── run_creative_engine.py     # CLI tool
├── comparison_benchmark.py    # Baseline vs Creative comparison
│
├── agmi-video-generator/      # Baseline (few-shot)
│   ├── README.md              # ← See for baseline details
│   └── src/
│
└── creative_engine/           # Creative engine library
    ├── core/                  # Config, LLM providers, utils
    ├── generation/            # Ideation, judging, drafting
    └── evaluation/            # Creativity assessment
```

## How It Works

**3-Phase Pipeline:**
1. **Ideation** → Generate diverse concepts (temperature sweeps)
2. **Judging** → Score concepts on originality, clarity, marketing viability
3. **Evaluation** → LLM-as-Judge (6 criteria: hook originality, visual creativity, narrative originality, entertainment value, brand integration, platform fit)

**Output:** Structured artifacts in `artifacts/` with generation and evaluation results.

## Documentation

- **Baseline System:** [agmi-video-generator/README.md](agmi/agmi-video-generator/README.md)
- **Challenge:** [agmi-video-generator/CHALLENGE.md](agmi-video-generator/CHALLENGE.md)

