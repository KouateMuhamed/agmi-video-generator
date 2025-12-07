"""
Creative Engine - A content-agnostic library for generating creative content
using divergence-convergence pipeline with quality filtering.
"""

from creative_engine.engine import CreativeEngine, EngineResult
from creative_engine.core import (
    ContentType,
    CreativityConfig,
    EngineParameters,
    map_creativity,
    create_provider_from_model,
    extract_product_context,
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
)
from creative_engine.generation import (
    ContentGenerator,
    GenerationResult,
    Concept,
    VideoScript,
)
from creative_engine.evaluation import (
    CreativityEvaluator,
    CreativityAssessmentResult,
)

# Backward compatibility aliases
GenerateResult = EngineResult  # For backward compatibility

__all__ = [
    # Main classes
    "CreativeEngine",
    "EngineResult",
    "GenerateResult",  # Backward compatibility
    # Core
    "ContentType",
    "CreativityConfig",
    "EngineParameters",
    "map_creativity",
    "create_provider_from_model",
    "extract_product_context",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    # Generation
    "ContentGenerator",
    "GenerationResult",
    "Concept",
    "VideoScript",
    # Evaluation
    "CreativityEvaluator",
    "CreativityAssessmentResult",
]
