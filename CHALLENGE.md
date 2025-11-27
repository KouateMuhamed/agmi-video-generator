# AGMI Research Challenge: Creative Engine

## Overview

Your video script generator works—but it's bounded by its few shot examples. It can only remix what it's seen.

We want you to build something way harder: **an engine that generates content more creative than its examples.**

If you solve this, the same engine powers everything we build: video scripts, ads, Reddit posts, LinkedIn content, emails.

---

## The Research Question

**How do you make LLM outputs more creative—measurably—without sacrificing quality?**

Not "more random." Not "higher temperature." Actually creative: novel, surprising, yet effective.

---

## Your Task

Build a **Creative Engine** that:

1. Plugs into your existing video script pipeline
2. Produces outputs that are **measurably more creative** than your baseline few-shot approach
3. Is architected as a **reusable library** we can drop into any content workflow

---

## Requirements

### 1. The Engine (Library/Class)

Build a content-agnostic module:

```python
from creative_engine import CreativeEngine, ContentType, CreativityConfig

engine = CreativeEngine(
    model="gemini-2.0-flash",  # or any LLM
    config=CreativityConfig(
        creativity_level=0.7,  # 0.0 = conventional, 1.0 = maximum divergence
        quality_threshold=0.8,  # reject outputs below this coherence
    )
)

# Works for ANY content type
result = engine.generate(
    product_context={"name": "...", "description": "...", "pain_points": [...]},
    content_type=ContentType.VIDEO_SCRIPT,  # or REDDIT_POST, AD_COPY, etc.
    reference_examples=["example 1", "example 2"],  # few-shot examples to transcend
)

print(result.content)
print(result.creativity_score)
print(result.quality_score)
```

**Requirements:**
- Clean API 
- Works with at least 2 different LLMs (to prove it's not model-specific)

### 2. Creativity Measurement

You must define what "creative" means and measure it. Build a scorer:

```python
from creative_engine import CreativityScorer

scorer = CreativityScorer()
score = scorer.score(
    content="the generated content",
    reference_examples=["example 1", "example 2"],
    product_context={...},
)

print(score....)     
```

### 3. Integration Demo

Show it working in your Veo3 pipeline:

- 5 scripts from your original few-shot system (baseline)
- 5 scripts from the Creative Engine (improved)
- Creativity scores for all 10
- Brief analysis: what changed? what's better? what's worse?

---

## Research Write-Up (1 page)

Address these questions:

1. **How do you define creativity for text content?**
   - What dimensions matter? How do you measure them?

2. **What's your approach to generating more creative outputs?**
   - What techniques did you try? What worked? What didn't?

3. **How do you balance creativity vs quality?**
   - More creative often means more risky. How do you control this?

4. **What are the limitations?**
   - Where does your approach fail? What would you try next?

5. **How would this evolve?**
   - With fine-tuning? With feedback from real performance data?

---

## Deliverables

| Deliverable | Description |
|-------------|-------------|
| `creative_engine/` | The reusable library with clean API |
| `outputs/` | The 10 comparison outputs (5 baseline, 5 creative) |
| `RESEARCH.md` | Your 2-3 page write-up |

---

## Evaluation Criteria

| Criteria | What We're Looking For |
|----------|------------------------|
| **Research thinking** | Novel approach to creativity, clear methodology, honest about limitations |
| **Results** | Measurable improvement with clear evaluation |
| **Integration** | Plugs cleanly into existing pipeline without hacks |

---

## Timeline

**3/4 days.**

**Use AI tools and think out of the box**

---