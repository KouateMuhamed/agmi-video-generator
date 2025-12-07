"""
Reference examples for creative concept generation.

These examples are extracted from successful video styles (Varun Rana and Austin Nasso)
and are used to guide the ideation phase to generate concepts that transcend these patterns
rather than remix them.
"""

# ==========================================
# VARUN RANA STYLE CONCEPTS
# ==========================================

VARUN_STYLE_EXAMPLES = [
    "Deadpan fourth-wall break: exhausted developer stares directly into camera after discovering absurd tech situation",
    "Voiceover character skit: visible person talks to unseen roommate/colleague, creating dialogue-driven narrative",
    "Tech slang humor: uses 'cooked', 'locked in', 'wifey', 'bro' to create Gen-Z/tech worker authenticity",
    "Relatable exhaustion: low-energy delivery showing developer fatigue with corporate/tech life",
    "Multi-character single scene: one actor plays multiple roles through voiceover and visual context",
    "Real-world tech scenarios: code reviews, AI tools, dating apps - situations developers actually face",
    "Mood lighting contrast: switches between dark RGB monitor glow and bright natural apartment lighting",
    "Handheld camera intimacy: 'The Office' style direct-to-camera moments create personal connection",
    "Parental pressure humor: jokes about marriage expectations and cultural family dynamics",
    "Product discovery through frustration: tool introduced as solution to relatable developer pain point",
]

# ==========================================
# AUSTIN NASSO STYLE CONCEPTS
# ==========================================

AUSTIN_STYLE_EXAMPLES = [
    "Costume-based character switching: distinct outfits (beanie, polo, turtleneck) differentiate characters in rapid cuts",
    "Corporate hierarchy satire: 10x engineer vs junior dev, CEO demands, showing workplace power dynamics",
    "Fast-paced jump cuts: rapid editing between characters creates comedic rhythm and energy",
    "Dark twist ending: reveals uncomfortable truth (fired employee, hired VA) that subverts initial premise",
    "Character archetypes: Tech Bro (chaotic but praised), Senior Dev (biased), CEO (demanding), Junior Dev (desperate)",
    "Absurd approval logic: shows how corporate systems reward chaos while blocking simple fixes",
    "Visual character differentiation: relies on costume changes (grey beanie, blue polo, black turtleneck) for multi-character skits",
    "High energy vs low energy contrast: confident/arrogant characters vs desperate/panicked characters",
    "Impossible feature demands: CEO asks for unrealistic features, creating comedic tension",
    "Product as last resort: tool suggested when all other options fail, creating natural integration",
]

# ==========================================
# EXPORT
# ==========================================

# Convenience function to get examples by style
def get_reference_examples(style: str = "mixed") -> list:
    """
    Get reference examples by style.
    
    Args:
        style: One of "varun", "austin", or "mixed" (default)
        
    Returns:
        List of reference example strings
        
    Raises:
        ValueError: If style is not recognized
    """
    if style.lower() == "varun":
        return VARUN_STYLE_EXAMPLES
    elif style.lower() == "austin":
        return AUSTIN_STYLE_EXAMPLES
    elif style.lower() == "mixed":
        return VARUN_STYLE_EXAMPLES + AUSTIN_STYLE_EXAMPLES
    else:
        raise ValueError(f"Unknown style: {style}. Use 'varun', 'austin', or 'mixed'")

