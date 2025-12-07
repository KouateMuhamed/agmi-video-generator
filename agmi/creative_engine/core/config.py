"""
Configuration and creativity mapping logic for the Creative Engine.
"""

from dataclasses import dataclass


@dataclass
class CreativityConfig:
    """User-facing configuration for creativity control."""
    
    creativity_level: float  # 0.0 = conventional, 1.0 = maximum divergence
    quality_threshold: float  # 0.0-1.0, reject outputs below this score
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.creativity_level <= 1.0:
            raise ValueError("creativity_level must be between 0.0 and 1.0")
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0.0 and 1.0")


@dataclass
class EngineParameters:
    """Internal engine parameters derived from creativity_level."""
    
    temperature: float
    top_p: float
    num_branches: int
    quality_threshold: float
    
    def __post_init__(self):
        """Validate parameter values."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature should be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.num_branches < 1:
            raise ValueError("num_branches must be at least 1")


def map_creativity(creativity_level: float, quality_threshold: float = 0.7) -> EngineParameters:
    """
    Maps a user-facing creativity_level (0.0 to 1.0) to internal LLM parameters.
    
    Logic:
    - Low creativity (0.0): Conservative, fewer branches (cheaper), lower temp.
    - High creativity (1.0): High variance, max branches (more exploration), higher temp.
    
    Args:
        creativity_level: User-facing creativity control (0.0-1.0)
        quality_threshold: Quality threshold for filtering concepts (0.0-1.0)
    
    Returns:
        EngineParameters with mapped values
    """
    # Clamp level between 0 and 1
    level = max(0.0, min(1.0, creativity_level))
    
    return EngineParameters(
        # Temperature: 0.4 (safe) -> 1.2 (creative/random)
        # Cap at 1.2 to prevent total incoherence
        temperature=round(0.4 + (0.8 * level), 2),
        
        # Top_p: 0.6 (focused) -> 1.0 (broad vocabulary)
        top_p=round(0.6 + (0.4 * level), 2),
        
        # Branches: 2 (minimum) -> 8 (maximum exploration)
        # Using max to ensure at least 2 branches
        num_branches=max(2, int(2 + 6 * level)),
        
        quality_threshold=quality_threshold
    )

