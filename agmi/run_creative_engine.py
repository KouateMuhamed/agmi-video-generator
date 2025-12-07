"""
Standard entry point for running the Creative Engine.

Usage:
    python run_creative_engine.py --url https://www.cubic.dev --model gpt-4o
    python run_creative_engine.py --url https://www.cubic.dev --model gpt-4o --evaluate
    python run_creative_engine.py --url https://www.cubic.dev --model gpt-4o --reference-style varun
"""
import argparse
import logging
import sys
from typing import Optional

from creative_engine import (
    CreativeEngine,
    CreativityConfig,
    ContentType,
    extract_product_context
)
from creative_engine.generation import get_reference_examples

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("creative_engine_cli")

def run(
    url: str,
    model: str = "gpt-4o",
    creativity: float = 0.7,
    content_type: str = "video_script",
    platform: str = "tiktok",
    api_key: Optional[str] = None,
    evaluate_creativity: bool = False,
    reference_style: Optional[str] = None,
):
    """
    Run the creative engine pipeline.
    """
    logger.info(f"Initializing engine with model: {model}, creativity: {creativity}")
    
    try:
        # 1. Initialize Engine
        engine = CreativeEngine(
            model=model,
            config=CreativityConfig(
                creativity_level=creativity,
                quality_threshold=0.75
            ),
            api_key=api_key
        )
        
        # 2. Extract Context
        logger.info(f"Extracting product context from: {url}")
        product_context = extract_product_context(url, engine.provider)
        
        # Add platform context
        product_context["platform"] = platform
        logger.info(f"Context extracted: {product_context.get('name')} - {product_context.get('key_benefit')}")
        
        # Get reference examples if style is specified
        reference_examples = None
        if reference_style:
            logger.info(f"Using reference style: {reference_style}")
            reference_examples = get_reference_examples(reference_style)
            logger.info(f"Loaded {len(reference_examples)} reference examples")
        
        # 3. Generate
        logger.info("Starting generation pipeline (Ideation -> Judging -> Drafting)...")
        result = engine.generate(
            product_context=product_context,
            content_type=ContentType(content_type),
            reference_examples=reference_examples,
            evaluate_creativity=evaluate_creativity,
        )
        
        # 4. Result
        logger.info("Generation complete successfully.")
        logger.info(f"Selected Concept: {result.selected_concept.title}")
        logger.info(f"Quality Score: {result.quality_score:.2f}")
        
        # 5. Display creativity assessment if available
        if result.creativity_assessment:
            assessment = result.creativity_assessment
            logger.info("=" * 60)
            logger.info("CREATIVITY ASSESSMENT RESULTS")
            logger.info("=" * 60)
            
            # 1. Temperature Block (Block A)
            if assessment.temperature_block:
                temp_stats = assessment.temperature_block.overall
                logger.info(f"1) Temperature Sweep Score: {temp_stats.mean:.2f} ± {temp_stats.std:.2f}")
            
            # 2. Persona Block (Block B)
            if assessment.persona_block:
                pers_stats = assessment.persona_block.overall
                logger.info(f"2) Persona Sweep Score:     {pers_stats.mean:.2f} ± {pers_stats.std:.2f}")
            
            # 3. Aggregate (Block C)
            logger.info("-" * 60)
            logger.info(f"3) FINAL CREATIVITY SCORE:  {assessment.aggregate.overall.mean:.2f} ± {assessment.aggregate.overall.std:.2f}")
            
            logger.info("\nCriteria Breakdown (Aggregate):")
            for criterion, stats in assessment.aggregate.criteria.items():
                logger.info(f"  {criterion.replace('_', ' ').title()}: {stats.mean:.2f} ± {stats.std:.2f}")
            logger.info("=" * 60)
        
        # 6. Save artifacts
        gen_path, eval_path = engine.save_artifacts(result)
        logger.info(f"Generation artifact saved to: {gen_path}")
        if eval_path:
            logger.info(f"Evaluation artifact saved to: {eval_path}")
        
        return result

    except Exception as e:
        logger.error(f"Engine failed: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the Creative Engine")
    parser.add_argument("--url", required=True, help="Product landing page URL")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--creativity", type=float, default=0.7, help="Creativity level (0.0-1.0)")
    parser.add_argument("--type", default="video_script", help="Content type (video_script)")
    parser.add_argument("--platform", default="tiktok", help="Target platform")
    parser.add_argument("--evaluate", action="store_true", help="Enable creativity evaluation (Phase 3)")
    parser.add_argument("--reference-style", choices=["varun", "austin", "mixed"], help="Reference style examples to guide ideation (varun, austin, or mixed)")
    
    args = parser.parse_args()
    
    run(
        url=args.url,
        model=args.model,
        creativity=args.creativity,
        content_type=args.type,
        platform=args.platform,
        evaluate_creativity=args.evaluate,
        reference_style=args.reference_style,
    )

if __name__ == "__main__":
    main()

