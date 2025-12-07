"""
Creative Engine - Main orchestrator for Phases 1-3.

Orchestrates the complete pipeline:
- Phase 1-2: Content Generation (via ContentGenerator)
- Phase 3: Creativity Evaluation (via CreativityEvaluator)
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from creative_engine.core.config import CreativityConfig
from creative_engine.core.enums import ContentType
from creative_engine.core.llm import LLMProvider, create_provider_from_model
from creative_engine.generation.generator import ContentGenerator, GenerationResult
from creative_engine.evaluation.evaluator import CreativityEvaluator
from creative_engine.evaluation.models import CreativityAssessmentResult

logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """Result of a complete engine run (Generation + optional Evaluation)."""
    
    generation: GenerationResult
    evaluation: Optional[CreativityAssessmentResult] = None
    uuid: str = None
    
    def __post_init__(self):
        """Set UUID from generation if not provided."""
        if self.uuid is None:
            self.uuid = self.generation.generation_uuid
    
    @property
    def content(self) -> Any:
        """Convenience property to access generated content."""
        return self.generation.content
    
    @property
    def selected_concept(self):
        """Convenience property to access selected concept."""
        return self.generation.selected_concept
    
    @property
    def quality_score(self) -> float:
        """Convenience property to access quality score."""
        return self.generation.quality_score
    
    @property
    def creativity_assessment(self) -> Optional[CreativityAssessmentResult]:
        """Convenience property to access creativity assessment."""
        return self.evaluation


class CreativeEngine:
    """
    Creative Engine orchestrating Phases 1-3 pipeline.
    
    Pipeline:
    1. Ideation: Generate N diverse concepts (divergence)
    2. Judge: Score each concept (quality filtering)
    3. Select: Choose best concept above threshold
    4. Draft: Convert concept to structured output (convergence)
    5. Evaluate: (Optional) Assess creativity using LLM-as-Judge
    """
    
    def __init__(
        self,
        config: CreativityConfig,
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Creative Engine.
        
        Args:
            model: Model name string (e.g., "gpt-4o", "gemini-2.0-flash"). 
                   Provider will be auto-detected. Mutually exclusive with provider.
            provider: LLM provider implementation (OpenAIProvider, AnthropicProvider, etc.).
                      Mutually exclusive with model.
            config: Creativity configuration
            api_key: Optional API key (used if model is provided)
        
        Raises:
            ValueError: If both model and provider are provided, or neither is provided
        """
        if model and provider:
            raise ValueError("Cannot specify both 'model' and 'provider'. Use one or the other.")
        
        if not model and not provider:
            raise ValueError("Must specify either 'model' or 'provider'.")
        
        if model:
            self.provider = create_provider_from_model(model, api_key)
        else:
            self.provider = provider
        
        self.config = config
        
        # Initialize generator and evaluator
        self.generator = ContentGenerator(config=config, provider=self.provider)
        self.evaluator = CreativityEvaluator(provider=self.provider)
    
    def generate(
        self,
        product_context: Dict[str, Any],
        content_type: ContentType,
        reference_examples: Optional[List[str]] = None,
        evaluate_creativity: bool = False,
    ) -> EngineResult:
        """
        Generate creative content and optionally evaluate creativity.
        
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
            evaluate_creativity: If True, run Phase 3 creativity evaluation (LLM-as-Judge)
        
        Returns:
            EngineResult with generation and optional evaluation results
        """
        # Phase 1-2: Generate content
        gen_result = self.generator.generate(
            product_context=product_context,
            content_type=content_type,
            reference_examples=reference_examples,
        )
        
        # Phase 3: Evaluate creativity (optional)
        assessment = None
        if evaluate_creativity:
            # Only evaluate video scripts for now
            if content_type == ContentType.VIDEO_SCRIPT:
                try:
                    logger.info("Phase 3 - Creativity Evaluation: Starting LLM-as-Judge assessment")
                    assessment = self.evaluator.score_script(
                        gen_result.content,
                        product_context
                    )
                    logger.info(
                        "Phase 3 - Creativity Evaluation: Complete. "
                        f"Overall score: {assessment.aggregate.overall.mean:.2f}"
                    )
                except Exception as e:
                    logger.error(f"Phase 3 - Creativity Evaluation failed: {e}", exc_info=True)
                    # Continue without evaluation rather than failing entire generation
            else:
                logger.warning(
                    f"Creativity evaluation only supported for VIDEO_SCRIPT, "
                    f"got {content_type.value}. Skipping evaluation."
                )
        
        return EngineResult(
            generation=gen_result,
            evaluation=assessment,
            uuid=gen_result.generation_uuid,
        )
    
    def save_artifacts(
        self,
        result: EngineResult,
        directory: str = "artifacts",
    ) -> tuple[str, Optional[str]]:
        """
        Save generation and evaluation artifacts to separate files.
        
        Args:
            result: EngineResult containing generation and optional evaluation
            directory: Directory to save files in
            
        Returns:
            Tuple of (generation_filepath, evaluation_filepath)
            evaluation_filepath will be None if no evaluation was performed
        """
        # Save generation artifact
        gen_path = result.generation.save_artifact(directory)
        
        # Save evaluation artifact (if exists)
        eval_path = None
        if result.evaluation:
            eval_path = self.evaluator.save_assessment(
                assessment=result.evaluation,
                uuid=result.uuid,
                concept_title=result.generation.selected_concept.title,
                directory=directory,
            )
        
        return gen_path, eval_path
