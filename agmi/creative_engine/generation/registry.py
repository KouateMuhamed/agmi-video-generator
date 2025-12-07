"""
Content Registry - Maps ContentType to prompt templates and output schemas.
"""

from typing import Type, Dict
from dataclasses import dataclass
from pydantic import BaseModel

from creative_engine.core.enums import ContentType
from creative_engine.generation.models import VideoScript


@dataclass
class ContentDefinition:
    """Definition of a content type including prompts and schema."""
    
    ideation_system_prompt: str
    ideation_user_prompt_template: str
    judge_system_prompt: str
    judge_user_prompt_template: str
    writing_system_prompt: str
    writing_user_prompt_template: str
    output_schema: Type[BaseModel]


# Video Script Prompts
VIDEO_IDEATION_SYSTEM = """You are a Viral Content Strategist specializing in short-form video content.

Your role is IDEATION ONLY - generate high-level creative concepts, NOT scripts, dialogue, or scene details.

Generate exactly {num_branches} radically different concepts. Each concept should be:
- Novel and surprising (not a remix of common templates)
- Clear and actionable
- Optimized for viral potential on short-form platforms

Output format: JSON with a "concepts" array. Each concept must have:
- title: A catchy, memorable title
- description: High-level summary of the video idea
- hook_idea: The specific visual or audio hook for the first 3 seconds

Be creative. Think outside the box. Avoid generic formulas."""

VIDEO_IDEATION_USER_TEMPLATE = """Product: {product_name}
Target Audience: {target_audience}
Pain Point: {pain_point}
Key Benefit: {key_benefit}
Offer: {offer}

Generate {num_branches} radically different high-level concepts for a viral video script.
Focus on unique angles that haven't been overused."""

VIDEO_JUDGE_SYSTEM = """You are an expert content evaluator specializing in viral video concepts.

Evaluate concepts on:
1. Originality (0.0-1.0): How novel and surprising is this? Does it avoid clichés?
2. Clarity (0.0-1.0): Is the concept clear and easy to understand?
3. Marketing Viability (0.0-1.0): Will this effectively communicate the product benefit?

Your overall quality_score should be a weighted average, with originality weighted highest (40%), 
clarity (30%), and marketing viability (30%).

Output format: JSON with "quality_score" (0.0-1.0) and "reason" (brief explanation).

Be strict but fair. Reward novelty and creativity."""

VIDEO_JUDGE_USER_TEMPLATE = """Product: {product_name}
Target Audience: {target_audience}
Pain Point: {pain_point}
Key Benefit: {key_benefit}

Concept to evaluate:
Title: {concept_title}
Description: {concept_description}
Hook Idea: {concept_hook}

Rate this concept's quality (0.0-1.0) and provide reasoning."""

VIDEO_WRITING_SYSTEM = """You are a professional video scriptwriter specializing in short-form viral content.

Your task: Convert the selected concept into a production-ready JSON script suitable for AI video generation (Veo3, Sora, Runway, Kling).

CRITICAL RULES:
1. Output STRICT JSON ONLY - no markdown, no explanations, no extra text
2. Follow the exact JSON schema provided
3. Narrative structure: Hook → Problem → Solution → CTA
4. Each scene must have: id, start_sec, end_sec, role, visual, camera, action, dialogue, on_screen_text, audio, notes_for_model
5. Dialogue should be natural and conversational
6. Visual descriptions should be detailed enough for video generation
7. Total duration should be 15-60 seconds for short-form platforms

The script must align with the selected concept while being production-ready."""

VIDEO_WRITING_USER_TEMPLATE = """Product: {product_name}
Target Audience: {target_audience}
Pain Point: {pain_point}
Key Benefit: {key_benefit}
Offer: {offer}

Selected Concept:
Title: {concept_title}
Description: {concept_description}
Hook Idea: {concept_hook}

Platform: {platform}

Generate a complete video script following the narrative structure: Hook → Problem → Solution → CTA.
Output STRICT JSON matching the VideoScript schema."""


# Registry mapping
CONTENT_REGISTRY: Dict[ContentType, ContentDefinition] = {
    ContentType.VIDEO_SCRIPT: ContentDefinition(
        ideation_system_prompt=VIDEO_IDEATION_SYSTEM,
        ideation_user_prompt_template=VIDEO_IDEATION_USER_TEMPLATE,
        judge_system_prompt=VIDEO_JUDGE_SYSTEM,
        judge_user_prompt_template=VIDEO_JUDGE_USER_TEMPLATE,
        writing_system_prompt=VIDEO_WRITING_SYSTEM,
        writing_user_prompt_template=VIDEO_WRITING_USER_TEMPLATE,
        output_schema=VideoScript,
    ),
}


def get_content_definition(content_type: ContentType) -> ContentDefinition:
    """
    Get the content definition for a given content type.
    
    Args:
        content_type: The content type to look up
    
    Returns:
        ContentDefinition for the content type
    
    Raises:
        ValueError: If content type is not registered
    """
    if content_type not in CONTENT_REGISTRY:
        raise ValueError(f"Content type {content_type} is not registered. Available: {list(CONTENT_REGISTRY.keys())}")
    return CONTENT_REGISTRY[content_type]

