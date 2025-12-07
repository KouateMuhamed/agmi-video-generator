# AGMI Creative Engine - Usage & API Guide

This guide details how to use the **Creative Engine** as a reusable library for generating and evaluating creative content. It serves as the technical documentation for the system described in the main [Research README](README.md).

## 🏗️ Project Structure

The codebase is organized into two distinct parts:

1.  **`creative_engine/`**: The core reusable library.
    *   **Content Agnostic**: Designed to generate any content type (video scripts, LinkedIn posts, etc.).
    *   **Divergence-Convergence**: Implements the Tree of Thoughts pipeline.
    *   **Creativity Scoring**: Includes the LLM-as-a-Judge evaluation module.

2.  **`agmi/agmi-video-generator/`**: The baseline "few-shot" system (legacy).
    *   Used as a control group to measure the Creative Engine's improvement.

---

## 🚀 Quick Start (CLI)

You can run the engine directly from the command line for quick testing.

### Prerequisites
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"  # or ANTHROPIC_API_KEY, GOOGLE_API_KEY
```

### Common Commands

**Generate a Creative Video Script:**
```bash
python run_creative_engine.py --url https://www.cubic.dev --model gpt-4o
```

**Evaluate Creativity (LLM-as-a-Judge):**
```bash
python run_creative_engine.py --url https://www.cubic.dev --model gpt-4o --evaluate
```

**Control Creativity Level:**
```bash
# Level 0.0 (Safe/Conventional) -> Level 1.0 (Maximum Novelty/Risk)
python run_creative_engine.py --url https://www.cubic.dev --creativity 0.9
```

**Run Benchmark (Baseline vs. Creative Engine):**
```bash
python comparison_benchmark.py
```

---

## 💻 Python API Usage

The core of this solution is the `CreativeEngine` class, designed to be embedded into any Python application.

### 1. Initialization

Configure the engine with a model and a creativity profile.

```python
from creative_engine import CreativeEngine, CreativityConfig

# Configure creativity parameters
config = CreativityConfig(
    creativity_level=0.8,    # 0.0 to 1.0 (Controls Temp, Top-P, Branching Factor)
    quality_threshold=0.75   # Minimum score (0-1) for a concept to be considered
)

# Initialize the engine
engine = CreativeEngine(
    model="gpt-4o",  # Auto-detects provider (OpenAI, Anthropic, Gemini)
    config=config
)
```

### 2. Generating Content

The `generate` method is the main entry point. It accepts product context and a content type.

```python
from creative_engine import ContentType

# Define product context (can be scraped or manual)
product_context = {
    "name": "Cubic",
    "target_audience": "DevOps Engineers",
    "pain_point": "Managing disparate tools",
    "key_benefit": "Unified observable platform",
    "offer": "Sign up for beta",
    "platform": "tiktok"
}

# Generate content
result = engine.generate(
    product_context=product_context,
    content_type=ContentType.VIDEO_SCRIPT,  # Extensible enum
    reference_examples=["..."],             # Optional: Examples to transcend
    evaluate_creativity=True                # Optional: Run LLM-as-Judge
)
```

### 3. Accessing Results

The `EngineResult` object contains the generated content, the selected concept, and evaluation scores.

```python
# The final generated artifact (e.g., Script object)
print(f"Final Script: {result.content}")

# The winning concept from the "Tree of Thoughts"
print(f"Selected Concept: {result.selected_concept.title}")
print(f"Concept Quality Score: {result.quality_score}")  # Internal judge score

# Creativity Assessment (if evaluate_creativity=True)
if result.creativity_assessment:
    score = result.creativity_assessment.aggregate.overall.mean
    print(f"Creativity Score: {score}/3.0")
    
    # Detailed breakdown
    for criterion, stats in result.creativity_assessment.aggregate.criteria.items():
        print(f" - {criterion}: {stats.mean}")
```

---

## 🛠️ Extending the Engine

The system is built to support any content format, as requested in the challenge.

### Adding New Content Types

To add support for a new format (e.g., LinkedIn Posts):

1.  **Update `ContentType` Enum** (`creative_engine/core/enums.py`):
    ```python
    class ContentType(Enum):
        VIDEO_SCRIPT = "video_script"
        LINKEDIN_POST = "linkedin_post"  # <--- Add this
    ```

2.  **Add Prompt Template** (`creative_engine/generation/models.py`):
    *   Create a data model for the output (e.g., `LinkedInPost` Pydantic model).
    *   Register the prompt template in the generator registry.

### Adding New LLM Providers

The engine uses a provider pattern. To add a new LLM (e.g., Llama 3 via Groq):

1.  Implement the `LLMProvider` protocol (`creative_engine/core/llm.py`).
2.  Register it in `create_provider_from_model`.

---

## 🎯 Meeting Challenge Requirements

This implementation directly addresses the core requirements of the AGMI Research Challenge:

| Requirement | Implementation |
| :--- | :--- |
| **Reusable Library** | `CreativeEngine` class is decoupled from CLI and specific scripts. |
| **Measurable Creativity** | `CreativityEvaluator` implements CAT (Consensual Assessment Technique) with 8 personas. |
| **Content Agnostic** | `ContentType` enum allows easy extension to Ads, Emails, etc. |
| **Creativity vs. Quality** | `creativity_level` knob balances Divergence (Temp/Branches) with Convergence (Quality Threshold). |
| **Integration** | `comparison_benchmark.py` proves integration by running side-by-side with the baseline. |

