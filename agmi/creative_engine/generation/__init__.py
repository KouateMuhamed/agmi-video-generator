"""
Content Generation module (Phase 1-2).

Handles ideation, judging, and drafting of creative content.
"""

from creative_engine.generation.generator import ContentGenerator, GenerationResult
from creative_engine.generation.models import (
    Concept,
    ConceptScore,
    ScoredConcept,
    IdeationOutput,
    VideoMeta,
    Audio,
    Scene,
    VideoScript,
    ContentArtifact,
)
from creative_engine.generation.registry import (
    ContentDefinition,
    get_content_definition,
    CONTENT_REGISTRY,
)
from creative_engine.generation.reference_examples import (
    VARUN_STYLE_EXAMPLES,
    AUSTIN_STYLE_EXAMPLES,
    get_reference_examples,
)

__all__ = [
    "ContentGenerator",
    "GenerationResult",
    "Concept",
    "ConceptScore",
    "ScoredConcept",
    "IdeationOutput",
    "VideoMeta",
    "Audio",
    "Scene",
    "VideoScript",
    "ContentArtifact",
    "ContentDefinition",
    "get_content_definition",
    "CONTENT_REGISTRY",
    "VARUN_STYLE_EXAMPLES",
    "AUSTIN_STYLE_EXAMPLES",
    "get_reference_examples",
]
