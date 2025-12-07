"""
Data models for the Generation pipeline (Phase 1-2).

Phase 1: Ideation - Generate multiple creative concepts
Phase 2: Drafting - Convert selected concept to structured content
"""

from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field


class Concept(BaseModel):
    """A high-level creative concept generated during ideation."""
    
    title: str = Field(..., description="A short, catchy title for the concept")
    description: str = Field(..., description="High-level summary of the idea")
    hook_idea: str = Field(..., description="The specific visual or audio hook to grab attention")


class IdeationOutput(BaseModel):
    """Output from the ideation phase containing multiple concepts."""
    
    concepts: List[Concept] = Field(..., description="List of generated concepts")


class ConceptScore(BaseModel):
    """Quality score and reasoning for a concept."""
    
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    reason: str = Field(..., description="Short justification for the score")


class ScoredConcept(BaseModel):
    """A concept paired with its quality score."""
    
    concept: Concept
    score: ConceptScore
    
    @property
    def quality_score(self) -> float:
        """Convenience property to access the score value."""
        return self.score.quality_score


# --- Phase 2: Writing Models (Video Script Schema) ---

class VideoMeta(BaseModel):
    """Metadata for a video script."""
    
    duration_seconds: int = Field(..., description="Total estimated duration in seconds")
    platform: Literal["tiktok", "instagram", "youtube_shorts", "linkedin"] = Field(
        ..., description="Target platform for the video"
    )


class Audio(BaseModel):
    """Audio specifications for a scene."""
    
    music: Optional[str] = Field(None, description="Mood or genre of background music")
    sfx: Optional[str] = Field(None, description="Specific sound effects (e.g., 'whoosh', 'notification sound')")


class Scene(BaseModel):
    """A single scene in a video script."""
    
    id: int = Field(..., description="Scene identifier")
    start_sec: float = Field(..., ge=0.0, description="Start time in seconds")
    end_sec: float = Field(..., ge=0.0, description="End time in seconds")
    role: Literal["hook", "problem", "solution", "cta", "other"] = Field(
        ..., description="Narrative role of the scene"
    )
    visual: str = Field(..., description="Detailed visual description for video generation")
    camera: str = Field(..., description="Camera movement or angle (e.g., 'Zoom in', 'Static shot')")
    action: str = Field(..., description="What happens in the scene")
    dialogue: str = Field(..., description="Spoken words by the actor or voiceover. Empty string if no dialogue.")
    on_screen_text: Optional[str] = Field(None, description="Text overlays")
    audio: Audio = Field(..., description="Audio specifications")
    notes_for_model: Optional[str] = Field(None, description="Technical notes for the video generation model")


class VideoScript(BaseModel):
    """Complete video script in structured JSON format."""
    
    video_meta: VideoMeta
    scenes: List[Scene]


# --- Generic Content Artifact (for future extensibility) ---

class ContentArtifact(BaseModel):
    """Generic content artifact that can hold any structured output."""
    
    content_type: str
    content: Any  # Can be VideoScript, LinkedInPost, etc.
    selected_concept: Concept
    concept_score: float
