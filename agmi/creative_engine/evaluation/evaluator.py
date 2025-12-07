"""
Creativity evaluation module using LLMs as judges.

Implements Phase 3 of the creative engine: comprehensive creativity assessment
through temperature sweep and persona-based evaluation.
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any

from creative_engine.generation.models import VideoScript
from creative_engine.evaluation.models import (
    GenericJudgeOutput,
    PersonaJudgeOutput,
    TemperatureBlock,
    PersonaBlock,
    AggregateResults,
    CreativityAssessmentResult,
    EvaluationStats,
    CriterionStats,
)
from creative_engine.core.llm import LLMProvider

logger = logging.getLogger(__name__)


# Inline prompt templates (Phase 3)
GENERIC_SYSTEM_PROMPT = """You are an expert creativity assessor for short-form video ad scripts (TikTok, Reels, Shorts).

You will evaluate the CREATIVITY of ONE video ad script using SIX criteria, each scored from 1 to 3.
You must strictly follow the definitions and scoring rubrics below.

--------------------
CRITERIA DEFINITIONS
--------------------

1) Hook Originality & Stopping Power
- What it measures:
  - How surprising, attention-grabbing, and scroll-stopping the first 1–3 seconds are.
  - Novelty of the hook and presence of a clear pattern interrupt.
- Scoring rubric (1–3):
  - 1 = Weak:
    - Generic or predictable opening.
    - No clear pattern interrupt.
    - Slow or unclear start.
  - 2 = Moderate:
    - Somewhat interesting or curious hook.
    - Includes at least one engaging element (visual, line, sound).
  - 3 = Strong:
    - Highly original and surprising hook.
    - Strong pattern interruption.
    - Immediately commands attention in a TikTok-like feed.

2) Visual Creativity & Scene Dynamism
- What it measures:
  - Creativity of visuals, camera moves, and transitions.
  - How dynamic and varied the scenes feel, in line with short-form video grammar.
- Scoring rubric (1–3):
  - 1 = Weak:
    - Static, generic visuals.
    - Minimal camera movement.
    - No interesting transitions.
  - 2 = Moderate:
    - Some interesting visuals or transitions.
    - Basic dynamism and moderate variety between scenes.
  - 3 = Strong:
    - Highly dynamic, visually creative script.
    - Strong, TikTok-native visual ideas (POV shots, punch-ins, quick cuts, etc.).
    - Consistently engaging visual flow.

3) Narrative Originality & Idea Novelty
- What it measures:
  - The creativity and novelty of the underlying story or concept.
  - Twists, metaphors, unusual structures, or unique angles.
- Scoring rubric (1–3):
  - 1 = Weak:
    - Predictable storyline or cliché ad trope.
    - Repeats common marketing narratives.
  - 2 = Moderate:
    - Contains at least one interesting angle or twist.
    - Partially fresh but still somewhat familiar.
  - 3 = Strong:
    - Clearly distinctive and memorable idea.
    - Fresh creative angle or clever twist suitable for TikTok.

4) Entertainment Value & Emotional Impact
- What it measures:
  - How engaging, funny, relatable, emotional, or surprising the script is.
  - Entertainment value independent of the product.
- Scoring rubric (1–3):
  - 1 = Weak:
    - Emotionally flat or unengaging.
    - Feels like a plain informational ad.
  - 2 = Moderate:
    - Has some moments of humor, relatability, or emotional spark.
    - Mildly entertaining overall.
  - 3 = Strong:
    - Highly entertaining, emotionally punchy, or genuinely funny.
    - Strong engagement and high replay value for a TikTok user.

5) Creative Brand & Message Integration
- What it measures:
  - How creatively and naturally the product, benefit, or offer is integrated into the script.
  - Whether the message feels embedded in the story instead of bolted on.
- Scoring rubric (1–3):
  - 1 = Weak:
    - Message delivered in a forced, boring, or generic way.
    - Breaks immersion or entertainment flow.
  - 2 = Moderate:
    - Product integration is functional and makes sense.
    - Not highly creative, but not disruptive.
  - 3 = Strong:
    - Product is integrated in a clever, seamless, story-driven way.
    - The message enhances the entertainment instead of interrupting it.

6) Platform Fit & Trend Awareness
- What it measures:
  - How well the script fits TikTok-style content: pacing, UGC feel, trendfluency, meme grammar, and authenticity.
- Scoring rubric (1–3):
  - 1 = Weak:
    - Feels like a traditional TV/corporate ad.
    - Slow pacing, non-native tone.
    - No awareness of TikTok culture or trends.
  - 2 = Moderate:
    - Reasonably adapted to TikTok.
    - Some UGC elements or platform-appropriate tone.
  - 3 = Strong:
    - Strongly TikTok-native feel (UGC tone, casual voice, fast cuts).
    - Good alignment with trends, meme formats, POV styles, etc.

--------------------
SCORING RULES
--------------------

- For EACH of the six criteria above:
  - Assign an integer score: 1, 2, or 3.
  - Provide a short, concrete reason based on the script.
- Then compute an OVERALL creativity score:
  - overall_creativity.score = arithmetic mean of the six criterion scores.
  - This may be a decimal between 1.0 and 3.0 (e.g., 2.5).

--------------------
OUTPUT FORMAT (STRICT)
--------------------

You MUST output ONLY a valid JSON object with this exact structure:

{
  "hook_originality": {
    "score": <1-3 integer>,
    "reason": "<short explanation>"
  },
  "visual_creativity": {
    "score": <1-3 integer>,
    "reason": "<short explanation>"
  },
  "narrative_originality": {
    "score": <1-3 integer>,
    "reason": "<short explanation>"
  },
  "entertainment_value": {
    "score": <1-3 integer>,
    "reason": "<short explanation>"
  },
  "brand_integration": {
    "score": <1-3 integer>,
    "reason": "<short explanation>"
  },
  "platform_fit": {
    "score": <1-3 integer>,
    "reason": "<short explanation>"
  },
  "overall_creativity": {
    "score": <number between 1.0 and 3.0>,
    "reason": "<short global justification>"
  }
}

Do NOT include any text before or after the JSON.
Do NOT add or remove keys.
Do NOT use markdown."""

GENERIC_USER_PROMPT_TEMPLATE = """You will now evaluate the creativity of a generated TikTok video ad script.

PRODUCT CONTEXT:
- Product name: {product_name}
- Target audience: {target_audience}
- Main pain point: {pain_point}
- Key benefit: {key_benefit}
- Platform: {platform}

SCRIPT (JSON FORMAT):
{script_json_string}

Evaluate this script STRICTLY according to the six creativity criteria and the 1–3 scoring rubric defined in the system prompt.
Base your reasoning ONLY on the content of this script and the product context.
Return ONLY the JSON object as specified."""

PERSONA_SYSTEM_PROMPT_TEMPLATE = """You are evaluating the creativity of TikTok-style video ad scripts in the role of a specific expert persona.

Your persona for this evaluation is:

- Persona name: {PERSONA_NAME}
- Persona description: {PERSONA_DESCRIPTION}

You must THINK and JUDGE like this persona would:
- Focus on what this persona cares about the most.
- Keep the same six creativity criteria and the same 1–3 scoring scale.
- However, your explanations and emphasis should reflect this persona's priorities and biases.

--------------------
CREATIVITY CRITERIA (ALWAYS THE SAME)
--------------------

1) Hook Originality & Stopping Power
- What it measures:
  - How surprising, attention-grabbing, and scroll-stopping the first 1–3 seconds are.
- Scoring (1–3):
  - 1 = Weak (generic opening, no pattern interrupt, slow start)
  - 2 = Moderate (somewhat interesting or curious)
  - 3 = Strong (highly original and instantly attention-grabbing)

2) Visual Creativity & Scene Dynamism
- What it measures:
  - Creativity of visuals, camera moves, and transitions.
- Scoring (1–3):
  - 1 = Weak (static, generic, low dynamism)
  - 2 = Moderate (some interesting visuals or transitions)
  - 3 = Strong (highly dynamic, visually rich, TikTok-native)

3) Narrative Originality & Idea Novelty
- What it measures:
  - How original and conceptually fresh the underlying story or idea is.
- Scoring (1–3):
  - 1 = Weak (cliché, predictable story)
  - 2 = Moderate (partially fresh, at least one interesting angle)
  - 3 = Strong (distinctive, memorable, clearly innovative idea)

4) Entertainment Value & Emotional Impact
- What it measures:
  - How engaging, funny, emotional, or surprising the script is.
- Scoring (1–3):
  - 1 = Weak (flat, not entertaining)
  - 2 = Moderate (some engaging or emotional moments)
  - 3 = Strong (highly entertaining, strong emotional or comedic impact)

5) Creative Brand & Message Integration
- What it measures:
  - How creatively and naturally the product/offer is integrated into the story.
- Scoring (1–3):
  - 1 = Weak (forced, boring, disrupts the flow)
  - 2 = Moderate (functional, makes sense but not very creative)
  - 3 = Strong (clever, seamless, enhances the story)

6) Platform Fit & Trend Awareness
- What it measures:
  - How well the script matches TikTok culture, pacing, and style.
- Scoring (1–3):
  - 1 = Weak (feels like TV/corporate, non-native)
  - 2 = Moderate (reasonably adapted, some UGC or TikTok-like elements)
  - 3 = Strong (highly native, trend-aware, likely to feel organic on TikTok)

--------------------
SCORING RULES
--------------------

- For EACH of the six criteria:
  - Assign an integer score: 1, 2, or 3.
  - Provide a short explanation that reflects how {PERSONA_NAME} would think.
- Then compute an OVERALL creativity score as:
  - Mean of the six criterion scores (may be decimal, between 1.0 and 3.0).

--------------------
OUTPUT FORMAT (STRICT)
--------------------

You MUST output ONLY a valid JSON object with this exact structure. The "persona" field MUST be included and must match the persona name exactly:

{{
  "persona": "{PERSONA_NAME}",
  "hook_originality": {{
    "score": <1-3 integer>,
    "reason": "<short explanation from this persona's viewpoint>"
  }},
  "visual_creativity": {{
    "score": <1-3 integer>,
    "reason": "<short explanation from this persona's viewpoint>"
  }},
  "narrative_originality": {{
    "score": <1-3 integer>,
    "reason": "<short explanation from this persona's viewpoint>"
  }},
  "entertainment_value": {{
    "score": <1-3 integer>,
    "reason": "<short explanation from this persona's viewpoint>"
  }},
  "brand_integration": {{
    "score": <1-3 integer>,
    "reason": "<short explanation from this persona's viewpoint>"
  }},
  "platform_fit": {{
    "score": <1-3 integer>,
    "reason": "<short explanation from this persona's viewpoint>"
  }},
  "overall_creativity": {{
    "score": <number between 1.0 and 3.0>,
    "reason": "<short global justification from this persona's viewpoint>"
  }}
}}

CRITICAL: The "persona" field is REQUIRED and must be the first field in the JSON object.
Do NOT include any text before or after the JSON.
Do NOT add or remove keys.
Do NOT use markdown."""

PERSONA_USER_PROMPT_TEMPLATE = """You will now evaluate the creativity of a generated TikTok video ad script,
acting in the role of the persona defined in the system prompt.

PRODUCT CONTEXT:
- Product name: {product_name}
- Target audience: {target_audience}
- Main pain point: {pain_point}
- Key benefit: {key_benefit}
- Platform: {platform}

SCRIPT (JSON FORMAT):
{script_json_string}

Evaluate this script strictly according to the SIX creativity criteria and the 1–3 scoring rubric provided in the system prompt.
Your explanations must reflect the perspective and priorities of the persona.
Return ONLY the JSON object as specified. Remember to include the "persona" field as the first field."""


# Personas as constants
ALL_PERSONAS = [
    {
        "name": "Senior Creative Director",
        "description": "You are a top-tier agency Creative Director with 15+ years of experience in crafting high-impact advertising concepts. You evaluate ideas based on originality, conceptual strength, emotional resonance, and creative risk-taking. You value big ideas, freshness, memorable hooks, and storytelling clarity. You naturally penalize clichés, predictable structures, and anything that feels \"safe\" or uninspired."
    },
    {
        "name": "TikTok Native UGC Creator",
        "description": "You are a full-time TikTok creator specialized in UGC ads. You judge scripts based on authenticity, humor, relatability, pacing, and platform-native behavior. You value casual tone, real-person energy, low-friction storytelling, and trends that feel culturally alive. You penalize anything that feels like a corporate ad, overly polished, or \"trying too hard.\""
    },
    {
        "name": "Performance Marketer",
        "description": "You are a performance-driven marketer focused on conversions, retention, and messaging clarity. You evaluate scripts based on clear articulation of the value proposition, problem–solution logic, emotional triggering, and CTA effectiveness. You value clarity, benefit focus, product relevance, and hooks that immediately communicate value. You penalize scripts that are too abstract, confusing, slow, or weak on the selling point."
    },
    {
        "name": "Meme Culture Editor",
        "description": "You are a humor-first meme editor living inside TikTok culture. You judge scripts based on meme fluency, comedic timing, chaotic energy, trend remixability, and \"shareability.\" You value absurdity, humor sharpness, unexpected punchlines, and meme-native pacing. You penalize cringe humor, forced jokes, and anything that misuses or misunderstands meme logic."
    },
    {
        "name": "Cinematographer / Visual Director",
        "description": "You are a visual storyteller obsessed with framing, camera motion, transitions, and creative scene construction. You evaluate scripts on visual richness, dynamic pacing, shot variety, and cinematic expressiveness adapted to TikTok. You value POV shots, creative transitions, rhythm, kinetic visual energy, and clarity of visual storytelling. You penalize static visuals, generic framing, and scripts that lack dynamic visual imagination."
    },
    {
        "name": "Storytelling Coach",
        "description": "You are a professional storytelling instructor specializing in short-form narrative design. You judge scripts based on narrative arc, pacing, clarity of intention, emotional movement, and structural coherence. You value well-formed setups, satisfying payoffs, character voice, and narrative originality. You penalize chaotic structure, unclear motivations, weak payoffs, and stories without a strong through-line."
    },
    {
        "name": "Brand Strategist",
        "description": "You are a senior brand strategist focused on positioning, message clarity, differentiation, and audience fit. You evaluate scripts based on how well the product's value, benefit, and emotional promise are integrated into the creative idea. You value message coherence, brand consistency, persuasive storytelling, and meaningful differentiation. You penalize forced product mentions, weak benefit articulation, or scripts where the brand disappears behind creativity."
    },
    {
        "name": "Trend Analyst / Cultural Strategist",
        "description": "You are a cultural trend forecaster specializing in TikTok microcultures, aesthetics, and emerging content patterns. You judge scripts based on platform fit, trend alignment, cultural resonance, and relevance to audience behavior. You value trend fluency, meme alignment, authenticity, and formats that match current cultural waves. You penalize outdated styles, tone-deaf content, non-native pacing, or anything that misunderstands TikTok culture."
    }
]


def build_generic_user_prompt(product_context: Dict[str, Any], script_json: str) -> str:
    """
    Build user prompt for generic judge evaluation.
    
    Args:
        product_context: Product context dictionary
        script_json: JSON string of the VideoScript
        
    Returns:
        Formatted user prompt
    """
    return GENERIC_USER_PROMPT_TEMPLATE.format(
        product_name=product_context.get("name", ""),
        target_audience=product_context.get("target_audience", ""),
        pain_point=product_context.get("pain_point", ""),
        key_benefit=product_context.get("key_benefit", ""),
        platform=product_context.get("platform", "tiktok"),
        script_json_string=script_json,
    )


def build_persona_system_prompt(persona_name: str, persona_description: str) -> str:
    """
    Build system prompt for persona-based judge evaluation.
    
    Args:
        persona_name: Name of the persona
        persona_description: Description of the persona
        
    Returns:
        Formatted system prompt
    """
    return PERSONA_SYSTEM_PROMPT_TEMPLATE.format(
        PERSONA_NAME=persona_name,
        PERSONA_DESCRIPTION=persona_description,
    )


def build_persona_user_prompt(product_context: Dict[str, Any], script_json: str) -> str:
    """
    Build user prompt for persona-based judge evaluation.
    
    Args:
        product_context: Product context dictionary
        script_json: JSON string of the VideoScript
        
    Returns:
        Formatted user prompt
    """
    return PERSONA_USER_PROMPT_TEMPLATE.format(
        product_name=product_context.get("name", ""),
        target_audience=product_context.get("target_audience", ""),
        pain_point=product_context.get("pain_point", ""),
        key_benefit=product_context.get("key_benefit", ""),
        platform=product_context.get("platform", "tiktok"),
        script_json_string=script_json,
    )


class CreativityEvaluator:
    """
    LLM-based creativity evaluator for video scripts.
    
    Implements two evaluation blocks:
    - Block A: Temperature sweep (8 temperatures)
    - Block B: Persona sweep (8 personas)
    - Block C: Aggregation of results
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        temperature_grid: Optional[List[float]] = None,
        persona_temperature: float = 0.3,
    ):
        """
        Initialize the creativity evaluator.
        
        Args:
            provider: LLM provider to use for evaluation
            temperature_grid: List of temperatures for Block A (default: [0.1, ..., 0.8])
            persona_temperature: Fixed temperature for persona evaluations (default: 0.3)
        """
        self.provider = provider
        self.temperature_grid = temperature_grid or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.persona_temperature = persona_temperature
        self.personas = ALL_PERSONAS
        
        if len(self.personas) != 8:
            logger.warning(
                f"Expected 8 personas, found {len(self.personas)}. "
                "Some evaluations may be skipped."
            )
    
    def score_script(
        self,
        script: VideoScript,
        product_context: Dict[str, Any],
    ) -> CreativityAssessmentResult:
        """
        Evaluate a video script for creativity.
        
        Args:
            script: VideoScript to evaluate
            product_context: Product context dictionary
            
        Returns:
            CreativityAssessmentResult with all evaluation blocks
        """
        logger.info("Starting creativity evaluation for script")
        
        # Serialize script to JSON
        script_json = json.dumps(script.model_dump(), indent=2)
        
        # Block A: Temperature sweep
        logger.info("Block A: Running temperature sweep evaluation")
        temperature_block = self._temperature_sweep(script_json, product_context)
        
        # Block B: Persona sweep
        logger.info("Block B: Running persona sweep evaluation")
        persona_block = self._persona_sweep(script_json, product_context)
        
        # Block C: Aggregate results
        logger.info("Block C: Aggregating results")
        aggregate = self._aggregate_results(temperature_block, persona_block)
        
        # Build final result
        result = CreativityAssessmentResult(
            temperature_block=temperature_block,
            persona_block=persona_block,
            aggregate=aggregate,
        )
        
        logger.info(
            f"Evaluation complete. Overall creativity score: {aggregate.overall.mean:.2f} ± {aggregate.overall.std:.2f}"
        )
        
        return result
    
    def save_assessment(
        self,
        assessment: CreativityAssessmentResult,
        uuid: str,
        concept_title: str,
        directory: str = "artifacts",
    ) -> str:
        """
        Save the creativity assessment to a JSON file.
        
        Args:
            assessment: CreativityAssessmentResult to save
            uuid: UUID for linking with generation artifact
            concept_title: Title of the concept (for filename)
            directory: Directory to save the file in
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with UUID and concept title
        sanitized_title = "".join(
            c for c in concept_title 
            if c.isalnum() or c in (' ', '-', '_')
        ).strip().replace(' ', '_').lower()[:50]  # Limit length
        
        filename = f"{uuid}_evaluation_{sanitized_title}.json"
        filepath = Path(directory) / filename
        
        # Save evaluation artifact
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(assessment.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation artifact saved to: {filepath}")
        return str(filepath)
    
    def _temperature_sweep(
        self,
        script_json: str,
        product_context: Dict[str, Any],
    ) -> Optional[TemperatureBlock]:
        """
        Block A: Evaluate script across different temperatures.
        
        Args:
            script_json: JSON string of the script
            product_context: Product context
            
        Returns:
            TemperatureBlock with statistics
        """
        user_prompt = build_generic_user_prompt(product_context, script_json)
        outputs = []
        successful_evaluations = []
        
        for i, temp in enumerate(self.temperature_grid, 1):
            try:
                logger.info(f"  Temperature {i}/{len(self.temperature_grid)}: {temp}")
                output = self.provider.generate(
                    system_prompt=GENERIC_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    output_schema=GenericJudgeOutput,
                    temperature=temp,
                    top_p=0.9,
                )
                successful_evaluations.append(output)
                outputs.append({
                    "temperature": temp,
                    "overall_creativity": output.overall_creativity.score,
                    "judge_output": output.model_dump(),
                })
            except Exception as e:
                logger.warning(f"  Failed evaluation at temperature {temp}: {e}")
                continue
        
        if len(successful_evaluations) < 4:
            logger.error(
                f"Temperature sweep failed: only {len(successful_evaluations)}/8 successful evaluations"
            )
            return None
        
        # Compute statistics
        criterion_names = [
            "hook_originality",
            "visual_creativity",
            "narrative_originality",
            "entertainment_value",
            "brand_integration",
            "platform_fit",
        ]
        
        # Overall creativity stats
        overall_scores = [eval.overall_creativity.score for eval in successful_evaluations]
        overall_mean = statistics.mean(overall_scores)
        overall_std = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
        
        # Per-criterion stats
        criteria_stats = {}
        for criterion_name in criterion_names:
            scores = [
                getattr(eval, criterion_name).score
                for eval in successful_evaluations
            ]
            criteria_stats[criterion_name] = CriterionStats(
                mean=statistics.mean(scores),
                std=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            )
        
        return TemperatureBlock(
            overall=EvaluationStats(mean=overall_mean, std=overall_std),
            criteria=criteria_stats,
            by_temperature=outputs,
        )
    
    def _persona_sweep(
        self,
        script_json: str,
        product_context: Dict[str, Any],
    ) -> Optional[PersonaBlock]:
        """
        Block B: Evaluate script with different personas.
        
        Args:
            script_json: JSON string of the script
            product_context: Product context
            
        Returns:
            PersonaBlock with statistics
        """
        outputs = []
        successful_evaluations = []
        
        for i, persona in enumerate(self.personas, 1):
            try:
                persona_name = persona["name"]
                persona_desc = persona["description"]
                logger.info(f"  Persona {i}/{len(self.personas)}: {persona_name}")
                
                system_prompt = build_persona_system_prompt(persona_name, persona_desc)
                user_prompt = build_persona_user_prompt(product_context, script_json)
                
                output = self.provider.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=PersonaJudgeOutput,
                    temperature=self.persona_temperature,
                    top_p=0.9,
                )
                successful_evaluations.append(output)
                outputs.append({
                    "persona": persona_name,
                    "overall_creativity": output.overall_creativity.score,
                    "judge_output": output.model_dump(),
                })
            except Exception as e:
                logger.warning(f"  Failed evaluation for persona {persona.get('name', 'unknown')}: {e}")
                continue
        
        if len(successful_evaluations) < 4:
            logger.error(
                f"Persona sweep failed: only {len(successful_evaluations)}/8 successful evaluations"
            )
            return None
        
        # Compute statistics
        criterion_names = [
            "hook_originality",
            "visual_creativity",
            "narrative_originality",
            "entertainment_value",
            "brand_integration",
            "platform_fit",
        ]
        
        # Overall creativity stats
        overall_scores = [eval.overall_creativity.score for eval in successful_evaluations]
        overall_mean = statistics.mean(overall_scores)
        overall_std = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
        
        # Per-criterion stats
        criteria_stats = {}
        for criterion_name in criterion_names:
            scores = [
                getattr(eval, criterion_name).score
                for eval in successful_evaluations
            ]
            criteria_stats[criterion_name] = CriterionStats(
                mean=statistics.mean(scores),
                std=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            )
        
        return PersonaBlock(
            overall=EvaluationStats(mean=overall_mean, std=overall_std),
            criteria=criteria_stats,
            by_persona=outputs,
        )
    
    def _aggregate_results(
        self,
        temperature_block: Optional[TemperatureBlock],
        persona_block: Optional[PersonaBlock],
    ) -> AggregateResults:
        """
        Block C: Aggregate results from temperature and persona blocks.
        
        Args:
            temperature_block: Results from temperature sweep
            persona_block: Results from persona sweep
            
        Returns:
            AggregateResults with mean of means
        """
        criterion_names = [
            "hook_originality",
            "visual_creativity",
            "narrative_originality",
            "entertainment_value",
            "brand_integration",
            "platform_fit",
        ]
        
        # Handle cases where one block might be None
        if temperature_block and persona_block:
            # Mean of means for overall
            overall_mean = (temperature_block.overall.mean + persona_block.overall.mean) / 2
            overall_std = max(temperature_block.overall.std, persona_block.overall.std)
            
            # Mean of means for each criterion
            criteria_stats = {}
            for criterion_name in criterion_names:
                temp_mean = temperature_block.criteria[criterion_name].mean
                pers_mean = persona_block.criteria[criterion_name].mean
                aggregate_mean = (temp_mean + pers_mean) / 2
                aggregate_std = max(
                    temperature_block.criteria[criterion_name].std,
                    persona_block.criteria[criterion_name].std,
                )
                criteria_stats[criterion_name] = CriterionStats(
                    mean=aggregate_mean,
                    std=aggregate_std,
                )
        elif temperature_block:
            # Only temperature block available
            overall_mean = temperature_block.overall.mean
            overall_std = temperature_block.overall.std
            criteria_stats = temperature_block.criteria
        elif persona_block:
            # Only persona block available
            overall_mean = persona_block.overall.mean
            overall_std = persona_block.overall.std
            criteria_stats = persona_block.criteria
        else:
            # Both blocks failed - return default
            logger.error("Both evaluation blocks failed. Cannot aggregate results.")
            overall_mean = 0.0
            overall_std = 0.0
            criteria_stats = {
                name: CriterionStats(mean=0.0, std=0.0)
                for name in criterion_names
            }
        
        return AggregateResults(
            overall=EvaluationStats(mean=overall_mean, std=overall_std),
            criteria=criteria_stats,
        )
