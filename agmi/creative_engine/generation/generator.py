"""
Content Generator - Handles Phase 1-2 (Ideation + Judging + Drafting).

Phase 1: Ideation - Generate multiple creative concepts
Phase 1b: Judge - Score each concept for quality
Phase 2: Draft - Convert selected concept to structured content
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from creative_engine.core.config import CreativityConfig, EngineParameters, map_creativity
from creative_engine.core.enums import ContentType
from creative_engine.generation.models import (
    Concept,
    ConceptScore,
    ScoredConcept,
    IdeationOutput,
)
from creative_engine.core.llm import LLMProvider
from creative_engine.generation.registry import get_content_definition

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a generation call (Phase 1-2 only)."""
    
    content: Any  # The generated content (VideoScript, LinkedInPost, etc.)
    selected_concept: Concept
    concept_score: float
    content_type: ContentType
    concepts: List[Concept]  # All generated concepts
    scored_concepts: List[ScoredConcept]  # All concepts with their scores
    product_context: Dict[str, Any]  # Product context used for generation
    reference_examples: Optional[List[str]] = None  # Reference examples used for ideation
    generation_uuid: str = None  # UUID for linking generation and evaluation artifacts
    
    def __post_init__(self):
        """Generate UUID if not provided."""
        if self.generation_uuid is None:
            self.generation_uuid = str(int(time.time())) + "_" + str(uuid.uuid4().hex[:8])
    
    @property
    def quality_score(self) -> float:
        """Quality score of the selected concept (alias for concept_score)."""
        return self.concept_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result = {
            "timestamp": time.time(),
            "iso_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "generation_uuid": self.generation_uuid,
            "content_type": self.content_type.value,
            "product_context": self.product_context,
            "reference_examples": self.reference_examples,
            "selected_concept": {
                "title": self.selected_concept.title,
                "description": self.selected_concept.description,
                "hook_idea": self.selected_concept.hook_idea,
            },
            "concept_score": self.concept_score,
            "generated_content": self.content.model_dump(),
            "all_concepts": [
                {
                    "concept": {
                        "title": sc.concept.title,
                        "description": sc.concept.description,
                        "hook_idea": sc.concept.hook_idea,
                    },
                    "score": sc.score.quality_score,
                    "reason": sc.score.reason
                }
                for sc in self.scored_concepts
            ],
            "total_concepts_generated": len(self.concepts),
            "total_concepts_scored": len(self.scored_concepts),
            "score_distribution": {
                "min": min(sc.quality_score for sc in self.scored_concepts) if self.scored_concepts else 0.0,
                "max": max(sc.quality_score for sc in self.scored_concepts) if self.scored_concepts else 0.0,
                "avg": sum(sc.quality_score for sc in self.scored_concepts) / len(self.scored_concepts) if self.scored_concepts else 0.0,
            }
        }
        return result
    
    def save_artifact(self, directory: str = "artifacts") -> str:
        """
        Save the generation result to a JSON file.
        
        Args:
            directory: Directory to save the file in
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with UUID and concept title
        sanitized_title = "".join(
            c for c in self.selected_concept.title 
            if c.isalnum() or c in (' ', '-', '_')
        ).strip().replace(' ', '_').lower()[:50]  # Limit length
        
        filename = f"{self.generation_uuid}_generation_{sanitized_title}.json"
        filepath = Path(directory) / filename
        
        # Save generation artifact
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generation artifact saved to: {filepath}")
        return str(filepath)


class ContentGenerator:
    """
    Content Generator implementing Phase 1-2 pipeline.
    
    Pipeline:
    1. Ideation: Generate N diverse concepts (divergence)
    2. Judge: Score each concept (quality filtering)
    3. Select: Choose best concept above threshold
    4. Draft: Convert concept to structured output (convergence)
    """
    
    def __init__(
        self,
        config: CreativityConfig,
        provider: LLMProvider,
    ):
        """
        Initialize the Content Generator.
        
        Args:
            config: Creativity configuration
            provider: LLM provider implementation
        """
        self.provider = provider
        self.config = config
        self._params: Optional[EngineParameters] = None
    
    def _get_params(self) -> EngineParameters:
        """Get or compute engine parameters."""
        if self._params is None:
            self._params = map_creativity(
                self.config.creativity_level,
                self.config.quality_threshold
            )
        return self._params
    
    def _ideate(
        self,
        product_context: Dict[str, Any],
        content_type: ContentType,
        reference_examples: Optional[List[str]] = None,
    ) -> List[Concept]:
        """
        Phase 1: Generate multiple creative concepts.
        
        Args:
            product_context: Product information (name, audience, pain_point, etc.)
            content_type: Type of content to generate
            reference_examples: Optional reference examples to transcend
        
        Returns:
            List of generated concepts
        """
        params = self._get_params()
        definition = get_content_definition(content_type)
        
        # Build prompts
        system_prompt = definition.ideation_system_prompt.format(
            num_branches=params.num_branches
        )
        
        user_prompt = definition.ideation_user_prompt_template.format(
            num_branches=params.num_branches,
            product_name=product_context.get("name", ""),
            target_audience=product_context.get("target_audience", ""),
            pain_point=product_context.get("pain_point", ""),
            key_benefit=product_context.get("key_benefit", ""),
            offer=product_context.get("offer", ""),
        )
        
        # Add reference examples if provided
        if reference_examples:
            user_prompt += "\n\nReference Examples (transcend these, don't remix):\n"
            for i, example in enumerate(reference_examples, 1):
                user_prompt += f"{i}. {example}\n"
        
        logger.info(
            "Phase 1 - Ideation: content_type=%s, num_branches=%d, temperature=%.2f, top_p=%.2f",
            content_type.value,
            params.num_branches,
            params.temperature,
            params.top_p,
        )
        
        # Generate concepts using structured output
        ideation_output = self.provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=IdeationOutput,
            temperature=params.temperature,
            top_p=params.top_p,
        )
        
        logger.info("Phase 1 - Ideation: generated %d concepts", len(ideation_output.concepts))
        for i, concept in enumerate(ideation_output.concepts, 1):
            logger.debug("  Concept %d: %s", i, concept.title)
        
        return ideation_output.concepts
    
    def _judge_concept(
        self,
        concept: Concept,
        product_context: Dict[str, Any],
        content_type: ContentType,
    ) -> ConceptScore:
        """
        Judge a single concept for quality.
        
        Args:
            concept: Concept to evaluate
            product_context: Product context
            content_type: Content type
        
        Returns:
            ConceptScore with quality score and reasoning
        """
        definition = get_content_definition(content_type)
        
        system_prompt = definition.judge_system_prompt
        user_prompt = definition.judge_user_prompt_template.format(
            product_name=product_context.get("name", ""),
            target_audience=product_context.get("target_audience", ""),
            pain_point=product_context.get("pain_point", ""),
            key_benefit=product_context.get("key_benefit", ""),
            concept_title=concept.title,
            concept_description=concept.description,
            concept_hook=concept.hook_idea,
        )
        
        logger.debug("Judging concept: %s", concept.title)
        
        score = self.provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ConceptScore,
            temperature=0.3,  # Lower temperature for consistent judging
            top_p=0.9,
        )
        
        logger.debug("  Score: %.3f - %s", score.quality_score, score.reason)
        return score
    
    def _judge_all_concepts(
        self,
        concepts: List[Concept],
        product_context: Dict[str, Any],
        content_type: ContentType,
    ) -> List[ScoredConcept]:
        """
        Judge all concepts in parallel.
        
        Args:
            concepts: List of concepts to evaluate
            product_context: Product context
            content_type: Content type
        
        Returns:
            List of ScoredConcepts
        """
        logger.info("Phase 1b - Judge: evaluating %d concepts", len(concepts))
        
        # For now, sequential execution
        # In production, use asyncio.gather for parallel execution
        scored = []
        for i, concept in enumerate(concepts, 1):
            logger.info("  Judging concept %d/%d: %s", i, len(concepts), concept.title)
            score = self._judge_concept(concept, product_context, content_type)
            scored.append(ScoredConcept(concept=concept, score=score))
            logger.info(
                "    Result: score=%.3f, reason=%s",
                score.quality_score,
                score.reason[:100] + "..." if len(score.reason) > 100 else score.reason,
            )
        return scored
    
    def _select_best_concept(
        self,
        scored_concepts: List[ScoredConcept],
    ) -> ScoredConcept:
        """
        Select the best concept above quality threshold.
        
        Args:
            scored_concepts: List of scored concepts
        
        Returns:
            Best ScoredConcept (highest score above threshold, or highest overall if none pass)
        """
        params = self._get_params()
        
        # Filter concepts above threshold
        valid_concepts = [
            sc for sc in scored_concepts
            if sc.quality_score >= params.quality_threshold
        ]
        
        logger.info(
            "Phase 1b - Selection: %d/%d concepts above threshold (%.2f)",
            len(valid_concepts),
            len(scored_concepts),
            params.quality_threshold,
        )
        
        if not valid_concepts:
            # Fallback: choose highest scoring even if below threshold
            logger.warning(
                "Phase 1b - Selection: no concepts above quality_threshold=%.2f, falling back to best overall",
                params.quality_threshold,
            )
            valid_concepts = scored_concepts
        
        # Safety check
        if not valid_concepts:
            raise ValueError("No scored concepts available for selection.")
        
        # Select highest scoring
        best = max(valid_concepts, key=lambda sc: sc.quality_score)
        logger.info(
            "Phase 1b - Selection: selected concept '%s' with score=%.3f (threshold=%.2f)",
            best.concept.title,
            best.quality_score,
            params.quality_threshold,
        )
        return best
    
    def _draft_content(
        self,
        selected_concept: ScoredConcept,
        product_context: Dict[str, Any],
        content_type: ContentType,
    ) -> Any:
        """
        Phase 2: Convert selected concept into structured content.
        
        Args:
            selected_concept: The selected concept with score
            product_context: Product context
            content_type: Content type
        
        Returns:
            Generated content matching the content type's schema
        """
        params = self._get_params()
        definition = get_content_definition(content_type)
        
        system_prompt = definition.writing_system_prompt
        user_prompt = definition.writing_user_prompt_template.format(
            product_name=product_context.get("name", ""),
            target_audience=product_context.get("target_audience", ""),
            pain_point=product_context.get("pain_point", ""),
            key_benefit=product_context.get("key_benefit", ""),
            offer=product_context.get("offer", ""),
            concept_title=selected_concept.concept.title,
            concept_description=selected_concept.concept.description,
            concept_hook=selected_concept.concept.hook_idea,
            platform=product_context.get("platform", "tiktok"),
        )
        
        logger.info(
            "Phase 2 - Writing: converting concept '%s' to %s",
            selected_concept.concept.title,
            content_type.value,
        )
        
        content = self.provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=definition.output_schema,
            temperature=params.temperature,
            top_p=params.top_p,
        )
        
        logger.info("Phase 2 - Writing: content generated successfully")
        return content
    
    def generate(
        self,
        product_context: Dict[str, Any],
        content_type: ContentType,
        reference_examples: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate creative content using the divergence-convergence pipeline.
        
        Args:
            product_context: Product information dict with keys:
                - name: Product name
                - target_audience: Target audience description
                - pain_point: Customer pain point
                - key_benefit: Main benefit
                - offer: Offer/CTA
                - platform: (optional) Platform for video scripts
            content_type: Type of content to generate
            reference_examples: Optional list of reference examples to transcend
        
        Returns:
            GenerationResult with content, selected concept, and score
        """
        # Phase 1: Ideation
        concepts = self._ideate(product_context, content_type, reference_examples)
        
        # Validate concepts
        if not concepts:
            raise ValueError(
                "Ideation phase returned no concepts. "
                "Please try again or adjust creativity parameters."
            )
        
        # Phase 1b: Judge all concepts
        scored_concepts = self._judge_all_concepts(concepts, product_context, content_type)
        
        # Select best concept
        best_concept = self._select_best_concept(scored_concepts)
        
        # Phase 2: Draft content
        content = self._draft_content(best_concept, product_context, content_type)
        
        logger.info(
            "Generation complete: selected '%s' (score=%.3f), generated %s",
            best_concept.concept.title,
            best_concept.quality_score,
            content_type.value,
        )
        
        return GenerationResult(
            content=content,
            selected_concept=best_concept.concept,
            concept_score=best_concept.quality_score,
            content_type=content_type,
            concepts=concepts,
            scored_concepts=scored_concepts,
            product_context=product_context,
            reference_examples=reference_examples,
        )
