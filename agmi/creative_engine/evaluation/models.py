"""
Data models for the Evaluation pipeline (Phase 3).

Phase 3: Creativity Assessment - LLM-as-Judge evaluation of generated content
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    """Score and reason for a single creativity criterion."""
    
    score: float = Field(..., ge=1.0, le=3.0, description="Score from 1.0 to 3.0 (accepts decimals)")
    reason: str = Field(..., description="Short explanation for the score")


class GenericJudgeOutput(BaseModel):
    """Output from generic judge evaluation (temperature sweep)."""
    
    hook_originality: CriterionScore
    visual_creativity: CriterionScore
    narrative_originality: CriterionScore
    entertainment_value: CriterionScore
    brand_integration: CriterionScore
    platform_fit: CriterionScore
    overall_creativity: CriterionScore


class PersonaJudgeOutput(BaseModel):
    """Output from persona-based judge evaluation."""
    
    persona: str = Field(..., description="Name of the persona used for evaluation")
    hook_originality: CriterionScore
    visual_creativity: CriterionScore
    narrative_originality: CriterionScore
    entertainment_value: CriterionScore
    brand_integration: CriterionScore
    platform_fit: CriterionScore
    overall_creativity: CriterionScore


class EvaluationStats(BaseModel):
    """Mean and standard deviation statistics for a criterion."""
    
    mean: float = Field(..., description="Mean score")
    std: float = Field(..., ge=0.0, description="Standard deviation")


class CriterionStats(BaseModel):
    """Statistics for a single criterion."""
    
    mean: float = Field(..., description="Mean score")
    std: float = Field(..., ge=0.0, description="Standard deviation")


class TemperatureBlock(BaseModel):
    """Results from temperature sweep evaluation (Block A)."""
    
    overall: EvaluationStats
    criteria: Dict[str, CriterionStats] = Field(
        ...,
        description="Statistics for each criterion: hook_originality, visual_creativity, etc."
    )
    by_temperature: List[Dict[str, Any]] = Field(
        ...,
        description="Individual judge outputs by temperature"
    )


class PersonaBlock(BaseModel):
    """Results from persona sweep evaluation (Block B)."""
    
    overall: EvaluationStats
    criteria: Dict[str, CriterionStats] = Field(
        ...,
        description="Statistics for each criterion: hook_originality, visual_creativity, etc."
    )
    by_persona: List[Dict[str, Any]] = Field(
        ...,
        description="Individual judge outputs by persona"
    )


class AggregateResults(BaseModel):
    """Aggregated results combining temperature and persona blocks."""
    
    overall: EvaluationStats
    criteria: Dict[str, CriterionStats] = Field(
        ...,
        description="Aggregated statistics for each criterion"
    )


class CreativityAssessmentResult(BaseModel):
    """Final creativity assessment result with all evaluation blocks."""
    
    temperature_block: Optional[TemperatureBlock] = None
    persona_block: Optional[PersonaBlock] = None
    aggregate: AggregateResults
    raw_judge_outputs: Optional[Dict[str, List[Any]]] = Field(
        None,
        description="Raw judge outputs for debugging/analysis"
    )
