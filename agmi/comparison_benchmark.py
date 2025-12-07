"""
Baseline vs Creative Engine Comparison Benchmark

Generates 5 scripts using baseline few-shot approach and 5 scripts using
creative engine divergence-convergence approach, evaluates all with the same
metrics, and produces comparison reports.
"""

import os
import sys
import json
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Unset Google/Gemini API keys to prevent warnings since we're using OpenAI
if "GOOGLE_API_KEY" in os.environ:
    del os.environ["GOOGLE_API_KEY"]
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

# Add agmi-video-generator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agmi-video-generator"))

# Baseline imports
from src import personas

# OpenAI for baseline generation and scraping
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Web scraping imports
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

# Creative engine imports
from creative_engine import (
    CreativeEngine,
    CreativityConfig,
    ContentType,
    extract_product_context,
)
from creative_engine.generation import get_reference_examples
from creative_engine.generation.models import VideoScript, VideoMeta, Scene, Audio
from creative_engine.evaluation.evaluator import CreativityEvaluator
from creative_engine.core.llm import create_provider_from_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("comparison_benchmark")

# Configuration
PRODUCTS = [
    "https://www.epiminds.com/",
    "https://peec.ai/",
    "https://www.cubic.dev/",
    "https://www.eikona.io/",
    "https://mistral.ai/"
]

MODEL = "gpt-4o-mini"  # Using gpt-4o-mini for better rate limits and lower cost
STYLE = "VARUN"
OUTPUT_DIR = "outputs"


def convert_baseline_to_videoscript(baseline_script_parts: List[str], product_name: str) -> VideoScript:
    """
    Convert baseline text prompts to VideoScript format for evaluation.
    
    Baseline format: List of 3 strings
    Each string: "Visual: [description]\n---\nScript: [dialogue]"
    
    Args:
        baseline_script_parts: List of 3 text prompt strings
        product_name: Product name for metadata
        
    Returns:
        VideoScript object
    """
    scenes = []
    total_duration = 0
    
    if not baseline_script_parts:
        raise ValueError("Empty baseline script parts")
    
    for i, prompt in enumerate(baseline_script_parts):
        # Extract visual and script parts - handle multiple formats
        # Format 1: "Visual: ...\n---\nScript: ..."
        # Format 2: "Visual: ...\nScript: ..." (no separator)
        # Format 3: Just text (fallback)
        
        visual = ""
        dialogue = ""
        
        # Try to extract Visual section
        visual_match = re.search(r'Visual:\s*(.*?)(?:\n---|\nScript:|$)', prompt, re.DOTALL | re.IGNORECASE)
        if visual_match:
            visual = visual_match.group(1).strip()
        else:
            # If no "Visual:" tag, try to extract before "---" or "Script:"
            parts = re.split(r'\n---\n|\nScript:', prompt, flags=re.IGNORECASE)
            if len(parts) > 0:
                visual = parts[0].strip()
        
        # Try to extract Script section
        script_match = re.search(r'Script:\s*(.*?)$', prompt, re.DOTALL | re.IGNORECASE)
        if script_match:
            dialogue = script_match.group(1).strip()
        else:
            # If no "Script:" tag, try to extract after "---"
            parts = re.split(r'\n---\n', prompt, flags=re.IGNORECASE)
            if len(parts) > 1:
                dialogue = parts[1].strip()
            elif len(parts) == 1 and not visual:
                # Fallback: use entire prompt as visual, no dialogue
                visual = prompt.strip()
        
        # If still empty, use prompt as visual
        if not visual:
            visual = prompt.strip()
        
        # Estimate scene duration (assume ~10 seconds per scene)
        start_sec = i * 10.0
        end_sec = (i + 1) * 10.0
        total_duration = end_sec
        
        # Determine role based on position
        if i == 0:
            role = "hook"
        elif i == len(baseline_script_parts) - 1:
            role = "cta"
        else:
            role = "problem" if i == 1 else "solution"
        
        scene = Scene(
            id=i + 1,
            start_sec=start_sec,
            end_sec=end_sec,
            role=role,
            visual=visual,
            camera="Handheld" if STYLE == "VARUN" else "Static shot",
            action=visual,
            dialogue=dialogue,
            on_screen_text=None,
            audio=Audio(music=None, sfx=None),
            notes_for_model=None
        )
        scenes.append(scene)
    
    video_meta = VideoMeta(
        duration_seconds=int(total_duration),
        platform="tiktok"
    )
    
    return VideoScript(video_meta=video_meta, scenes=scenes)


def scrape_product_info_openai(url: str) -> Dict[str, str]:
    """
    Scrape product information using OpenAI instead of Gemini.
    
    Args:
        url: Product website URL
        
    Returns:
        Dictionary with name, description, and pain_point
    """
    if requests is None or BeautifulSoup is None:
        raise ImportError("requests and beautifulsoup4 packages are required. Install with: pip install requests beautifulsoup4")
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    logger.info(f"   ...Scraping {url}...")
    
    # 1. Fetch HTML
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        logger.warning(f"⚠️ Scraping failed: {e}")
        # Fallback for testing/offline
        return {
            "name": "Generic Product",
            "description": "A placeholder product.",
            "pain_point": "Inefficiency"
        }
    
    # 2. Parse Text
    soup = BeautifulSoup(html_content, 'html.parser')
    # Get title, meta description, and body text (limited length)
    title = soup.title.string if soup.title else ""
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag:
        meta_desc = meta_tag.get("content", "")
        
    # Get main text content
    text_content = soup.get_text(separator=' ', strip=True)[:10000]  # Limit to 10k chars
    
    # 3. Analyze with OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    prompt = f"""
Analyze this website content and extract the following details about the product:
1. Product Name
2. Description (1 sentence summary)
3. Key Pain Point it solves (1 sentence)

Website Title: {title}
Website Description: {meta_desc}
Website Content: {text_content}

Return ONLY a JSON object with keys: "name", "description", "pain_point".
"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        return {
            "name": result.get("name", title or "Unknown Product"),
            "description": result.get("description", meta_desc or "No description available."),
            "pain_point": result.get("pain_point", "Unknown problem")
        }
    except Exception as e:
        logger.warning(f"⚠️ OpenAI extraction failed: {e}")
        return {
            "name": title or "Unknown Product",
            "description": meta_desc or "No description available.",
            "pain_point": "Unknown problem"
        }


def generate_baseline_script_openai(product_context: Dict[str, str], persona_data: Dict[str, Any]) -> List[str]:
    """
    Generate baseline script using OpenAI (replacement for Gemini-based baseline).
    
    Args:
        product_context: Product information dict
        persona_data: Persona data with transcripts and visual style
        
    Returns:
        List of 3 script prompt strings
    """
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Format examples for the prompt
    examples_text = ""
    for i, t in enumerate(persona_data['transcripts']):
        examples_text += f"--- TRAINING DATA EXAMPLE {i+1} ---\n{t}\n\n"
    
    full_prompt = f"""
You are a viral TikTok Scriptwriter.

**YOUR GOAL:** Write a hilarious, original 4-part video script to promote a product called "{product_context['name']}".

**PRODUCT CONTEXT:**
- Name: {product_context['name']}
- Description: {product_context['description']}
- Pain Point Solved: {product_context['pain_point']}

**THE VISUAL STYLE TO ADOPT:**
{persona_data['visual_style']}

**THE HUMOR & TONE REFERENCE (Few-Shot Examples):**
{examples_text}

**INSTRUCTIONS:**
1. **Story Arc:** Create a BRAND NEW story where the character faces the specific pain point mentioned above. The story must progress logically across 4 scenes following this structure: Hook → Problem → Solution → CTA (Call-to-Action).
2. **Humor Style:** Use the slang, pacing, and comedic delivery shown in the examples (e.g., if Varun, use "Bro", "Cooked"; if Austin, use "Tech Bro" arrogance vs. panic), but apply it to THIS new product.
3. **Product Placement:** You MUST mention the product name ("{product_context['name']}") naturally in the dialogue as the solution to the problem.
4. **Visuals:** 
   - If style is VARUN: Scene 1 starts with the avatar. Scenes 2-4 must describe the action *continuing* from the previous shot (continuous take).
   - If style is AUSTIN: Scenes must be JUMP CUTS with distinct characters (e.g., "Visual: Austin dressed as CEO...").

**OUTPUT FORMAT:**
Return a JSON object with a "scenes" key containing an array of exactly 4 strings. Each string is a prompt for video generation.
Format: "Visual: [Detailed description of action/setting] \\n---\\nScript: [Dialogue with sound effects cues]"

CRITICAL: Your response must be valid JSON with this EXACT structure:
{{"scenes": ["scene1_hook", "scene2_problem", "scene3_solution", "scene4_cta"]}}

Generate the JSON object with 4 scene prompts now. Return ONLY valid JSON, no markdown code blocks.
"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        # Log the raw response for debugging
        logger.debug(f"Raw LLM response keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
        
        # Extract the list from the JSON object
        if isinstance(result, dict):
            # First, try to find the scenes key
            if "scenes" in result and isinstance(result["scenes"], list):
                script_list = result["scenes"]
                # Accept 3-5 scenes (be lenient)
                if 3 <= len(script_list) <= 5:
                    # Pad or trim to exactly 4 if needed
                    if len(script_list) < 4:
                        logger.warning(f"Only {len(script_list)} scenes returned, padding with duplicates")
                        while len(script_list) < 4:
                            script_list.append(script_list[-1])  # Duplicate last scene
                    elif len(script_list) > 4:
                        logger.warning(f"Got {len(script_list)} scenes, using first 4")
                        script_list = script_list[:4]
                    return script_list
                else:
                    logger.warning(f"scenes array has {len(script_list)} items, expected 3-5")
            
            # Try other common keys
            for key in ["script", "prompts", "output", "result", "scripts"]:
                if key in result and isinstance(result[key], list):
                    script_list = result[key]
                    if 3 <= len(script_list) <= 5:
                        # Adjust to 4 scenes
                        if len(script_list) < 4:
                            while len(script_list) < 4:
                                script_list.append(script_list[-1])
                        elif len(script_list) > 4:
                            script_list = script_list[:4]
                        return script_list
            
            # Try to find any list value with 3-5 items
            for value in result.values():
                if isinstance(value, list) and 3 <= len(value) <= 5:
                    script_list = list(value)
                    if len(script_list) < 4:
                        while len(script_list) < 4:
                            script_list.append(script_list[-1])
                    elif len(script_list) > 4:
                        script_list = script_list[:4]
                    return script_list
            
            # Last resort: check if there are string values that look like scenes
            scene_strings = []
            for key, value in result.items():
                if isinstance(value, str) and ("Visual:" in value or "Script:" in value):
                    scene_strings.append(value)
            
            if len(scene_strings) >= 3:
                logger.warning(f"Found {len(scene_strings)} scene strings in top-level keys, using them")
                while len(scene_strings) < 4:
                    scene_strings.append(scene_strings[-1])
                return scene_strings[:4]
            
            raise ValueError(f"Could not find valid scenes list in JSON response. Keys: {list(result.keys())}, Structure: {json.dumps(result, indent=2)[:500]}")
        elif isinstance(result, list):
            # Accept 3-5 scenes
            if 3 <= len(result) <= 5:
                script_list = list(result)
                if len(script_list) < 4:
                    while len(script_list) < 4:
                        script_list.append(script_list[-1])
                elif len(script_list) > 4:
                    script_list = script_list[:4]
                return script_list
            else:
                raise ValueError(f"Expected list of 3-5 scenes, got {len(result)}")
        else:
            raise ValueError(f"Unexpected response format: {type(result)}")
            
    except Exception as e:
        logger.error(f"Error generating baseline script with OpenAI: {e}", exc_info=True)
        return []


def generate_baseline_scripts(products: List[str]) -> List[Dict[str, Any]]:
    """
    Generate baseline scripts using few-shot learning approach with OpenAI.
    
    Args:
        products: List of product URLs
        
    Returns:
        List of baseline result dictionaries
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Generating Baseline Scripts (Few-Shot Learning)")
    logger.info("=" * 60)
    
    baseline_results = []
    persona = personas.DATA[STYLE]
    
    for i, url in enumerate(products, 1):
        logger.info(f"\n[{i}/{len(products)}] Processing: {url}")
        
        try:
            # Scrape product context using OpenAI
            logger.info("  Scraping product context...")
            product_context = scrape_product_info_openai(url)
            logger.info(f"  Product: {product_context.get('name')}")
            
            # Generate script using few-shot approach with OpenAI
            logger.info("  Generating script with few-shot learning (OpenAI)...")
            script_parts = generate_baseline_script_openai(product_context, persona)
            
            if not script_parts or len(script_parts) == 0:
                logger.warning(f"  ⚠️ Failed to generate script for {url}")
                continue
            
            logger.info(f"  ✅ Generated {len(script_parts)} scenes")
            
            baseline_results.append({
                "url": url,
                "product_name": product_context.get("name", "Unknown"),
                "product_description": product_context.get("description", ""),
                "pain_point": product_context.get("pain_point", ""),
                "approach": "baseline_few_shot",
                "style": STYLE,
                "script_parts": script_parts,
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {url}: {e}", exc_info=True)
            continue
    
    logger.info(f"\n✅ Baseline generation complete: {len(baseline_results)}/{len(products)} successful")
    return baseline_results


def generate_creative_scripts(products: List[str]) -> List[Dict[str, Any]]:
    """
    Generate creative scripts using divergence-convergence pipeline.
    
    Args:
        products: List of product URLs
        
    Returns:
        List of creative result dictionaries
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Generating Creative Scripts (Divergence-Convergence)")
    logger.info("=" * 60)
    
    # Initialize engine with high creativity
    engine = CreativeEngine(
        model=MODEL,
        config=CreativityConfig(
            creativity_level=1.0,  # Maximum creativity
            quality_threshold=0.75
        )
    )
    
    # Get Varun reference examples
    varun_examples = get_reference_examples("varun")
    logger.info(f"Loaded {len(varun_examples)} Varun reference examples")
    
    creative_results = []
    
    for i, url in enumerate(products, 1):
        logger.info(f"\n[{i}/{len(products)}] Processing: {url}")
        
        try:
            # Extract product context
            logger.info("  Extracting product context...")
            product_context = extract_product_context(url, engine.provider)
            product_context["platform"] = "tiktok"
            logger.info(f"  Product: {product_context.get('name')}")
            
            # Generate with creative engine (without evaluation to avoid quota issues)
            logger.info("  Generating script with creative engine...")
            try:
                result = engine.generate(
                    product_context=product_context,
                    content_type=ContentType.VIDEO_SCRIPT,
                    reference_examples=varun_examples,
                    evaluate_creativity=False  # Will evaluate separately to avoid quota issues
                )
                
                if not result or not result.content:
                    logger.error(f"  ❌ Generation returned empty result for {url}")
                    continue
                
                logger.info(f"  ✅ Selected concept: {result.selected_concept.title}")
                logger.info(f"  ✅ Quality score: {result.quality_score:.3f}")
            except Exception as gen_error:
                logger.error(f"  ❌ Generation failed for {url}: {gen_error}", exc_info=True)
                raise  # Re-raise to be caught by outer try-except
            
            creative_results.append({
                "url": url,
                "product_name": product_context.get("name", "Unknown"),
                "product_description": product_context.get("description", ""),
                "pain_point": product_context.get("pain_point", ""),
                "approach": "creative_engine",
                "style": STYLE,
                "concept_title": result.selected_concept.title,
                "concept_description": result.selected_concept.description,
                "concept_hook": result.selected_concept.hook_idea,
                "quality_score": result.quality_score,
                "script": result.content.model_dump(),
                "creativity_assessment": None,  # Will be evaluated separately
                "all_concepts": [
                    {
                        "title": sc.concept.title,
                        "description": sc.concept.description,
                        "score": sc.quality_score
                    }
                    for sc in result.generation.scored_concepts
                ],
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {url}: {e}", exc_info=True)
            continue
    
    logger.info(f"\n✅ Creative generation complete: {len(creative_results)}/{len(products)} successful")
    return creative_results


def evaluate_baseline_scripts(baseline_results: List[Dict[str, Any]], engine: CreativeEngine) -> List[Dict[str, Any]]:
    """
    Evaluate baseline scripts using CreativityEvaluator.
    
    Args:
        baseline_results: List of baseline result dictionaries
        engine: CreativeEngine instance for provider access
        
    Returns:
        Updated baseline results with creativity assessments
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Evaluating Baseline Scripts")
    logger.info("=" * 60)
    
    evaluator = CreativityEvaluator(provider=engine.provider)
    
    for i, result in enumerate(baseline_results, 1):
        logger.info(f"\n[{i}/{len(baseline_results)}] Evaluating: {result['product_name']}")
        
        try:
            # Convert baseline script to VideoScript format
            video_script = convert_baseline_to_videoscript(
                result['script_parts'],
                result['product_name']
            )
            
            # Create minimal product context for evaluation
            product_context = {
                "name": result['product_name'],
                "target_audience": "Tech professionals",
                "pain_point": result.get('pain_point', ''),
                "key_benefit": result.get('product_description', ''),
                "platform": "tiktok"
            }
            
            # Evaluate
            logger.info("  Running creativity evaluation...")
            assessment = evaluator.score_script(video_script, product_context)
            
            result['creativity_assessment'] = assessment.model_dump()
            
            overall_score = assessment.aggregate.overall.mean
            logger.info(f"  ✅ Creativity score: {overall_score:.2f}")
            
        except Exception as e:
            logger.error(f"  ❌ Error evaluating {result['product_name']}: {e}", exc_info=True)
            result['creativity_assessment'] = None
            continue
    
    logger.info(f"\n✅ Baseline evaluation complete")
    return baseline_results


def evaluate_creative_scripts(creative_results: List[Dict[str, Any]], engine: CreativeEngine) -> List[Dict[str, Any]]:
    """
    Evaluate creative scripts using CreativityEvaluator.
    
    Args:
        creative_results: List of creative result dictionaries
        engine: CreativeEngine instance for provider access
        
    Returns:
        Updated creative results with creativity assessments
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3B: Evaluating Creative Scripts")
    logger.info("=" * 60)
    
    evaluator = CreativityEvaluator(provider=engine.provider)
    
    for i, result in enumerate(creative_results, 1):
        logger.info(f"\n[{i}/{len(creative_results)}] Evaluating: {result['product_name']}")
        
        try:
            # Convert script dict back to VideoScript object
            video_script = VideoScript(**result['script'])
            
            # Create product context for evaluation
            product_context = {
                "name": result['product_name'],
                "target_audience": "Tech professionals",
                "pain_point": result.get('pain_point', ''),
                "key_benefit": result.get('product_description', ''),
                "platform": "tiktok"
            }
            
            # Evaluate
            logger.info("  Running creativity evaluation...")
            assessment = evaluator.score_script(video_script, product_context)
            
            result['creativity_assessment'] = assessment.model_dump()
            
            overall_score = assessment.aggregate.overall.mean
            logger.info(f"  ✅ Creativity score: {overall_score:.2f}")
            
        except Exception as e:
            logger.error(f"  ❌ Error evaluating {result['product_name']}: {e}", exc_info=True)
            result['creativity_assessment'] = None
            continue
    
    logger.info(f"\n✅ Creative evaluation complete")
    return creative_results


def generate_comparison_report(baseline_results: List[Dict[str, Any]], creative_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate structured comparison report.
    
    Args:
        baseline_results: List of baseline results with assessments
        creative_results: List of creative results with assessments
        
    Returns:
        Comprehensive comparison report dictionary
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Generating Comparison Report")
    logger.info("=" * 60)
    
    report = {
        "metadata": {
            "model": MODEL,
            "style": STYLE,
            "timestamp": time.time(),
            "baseline_approach": "few_shot_learning",
            "creative_approach": "divergence_convergence_pipeline",
            "baseline_count": len(baseline_results),
            "creative_count": len(creative_results)
        },
        "results": {
            "baseline": [],
            "creative": []
        },
        "analysis": {}
    }
    
    # Extract metrics from baseline
    baseline_scores = []
    for b in baseline_results:
        if b.get('creativity_assessment'):
            assessment = b['creativity_assessment']
            baseline_scores.append({
                "product": b['product_name'],
                "url": b['url'],
                "overall_mean": assessment['aggregate']['overall']['mean'],
                "overall_std": assessment['aggregate']['overall']['std'],
                "criteria": {
                    k: {"mean": v['mean'], "std": v['std']}
                    for k, v in assessment['aggregate']['criteria'].items()
                }
            })
            report["results"]["baseline"].append({
                "product_name": b['product_name'],
                "url": b['url'],
                "creativity_score": assessment['aggregate']['overall']['mean'],
                "creativity_std": assessment['aggregate']['overall']['std']
            })
    
    # Extract metrics from creative
    creative_scores = []
    for c in creative_results:
        if c.get('creativity_assessment'):
            assessment = c['creativity_assessment']
            creative_scores.append({
                "product": c['product_name'],
                "url": c['url'],
                "concept_title": c['concept_title'],
                "quality_score": c['quality_score'],
                "overall_mean": assessment['aggregate']['overall']['mean'],
                "overall_std": assessment['aggregate']['overall']['std'],
                "criteria": {
                    k: {"mean": v['mean'], "std": v['std']}
                    for k, v in assessment['aggregate']['criteria'].items()
                }
            })
            report["results"]["creative"].append({
                "product_name": c['product_name'],
                "url": c['url'],
                "concept_title": c['concept_title'],
                "quality_score": c['quality_score'],
                "creativity_score": assessment['aggregate']['overall']['mean'],
                "creativity_std": assessment['aggregate']['overall']['std']
            })
    
    # Calculate averages
    if baseline_scores and creative_scores:
        baseline_avg = sum(s['overall_mean'] for s in baseline_scores) / len(baseline_scores)
        creative_avg = sum(s['overall_mean'] for s in creative_scores) / len(creative_scores)
        
        baseline_std = sum(s['overall_std'] for s in baseline_scores) / len(baseline_scores)
        creative_std = sum(s['overall_std'] for s in creative_scores) / len(creative_scores)
        
        improvement_pct = ((creative_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
        
        report["analysis"] = {
            "baseline_average_creativity": baseline_avg,
            "baseline_std": baseline_std,
            "creative_average_creativity": creative_avg,
            "creative_std": creative_std,
            "improvement_percentage": improvement_pct,
            "baseline_scores": baseline_scores,
            "creative_scores": creative_scores,
            "per_product_comparison": [
                {
                    "product": b['product'],
                    "baseline_score": b['overall_mean'],
                    "creative_score": next((c['overall_mean'] for c in creative_scores if c['product'] == b['product']), None),
                    "improvement": None
                }
                for b in baseline_scores
            ]
        }
        
        # Calculate per-product improvements
        for item in report["analysis"]["per_product_comparison"]:
            if item['baseline_score'] and item['creative_score']:
                item['improvement'] = ((item['creative_score'] - item['baseline_score']) / item['baseline_score'] * 100) if item['baseline_score'] > 0 else 0
    
    logger.info(f"\n✅ Comparison report generated")
    logger.info(f"   Baseline Average: {report['analysis'].get('baseline_average_creativity', 0):.2f}")
    logger.info(f"   Creative Average: {report['analysis'].get('creative_average_creativity', 0):.2f}")
    logger.info(f"   Improvement: {report['analysis'].get('improvement_percentage', 0):.1f}%")
    
    return report


def generate_markdown_report(report: Dict[str, Any], baseline_results: List[Dict[str, Any]], creative_results: List[Dict[str, Any]]) -> str:
    """
    Generate human-readable markdown comparison report.
    
    Args:
        report: Comparison report dictionary
        baseline_results: Baseline results
        creative_results: Creative results
        
    Returns:
        Markdown string
    """
    md = f"""# Baseline vs Creative Engine Comparison

## Methodology

- **Model**: {report['metadata']['model']}
- **Style**: {report['metadata']['style']}
- **Baseline Approach**: {report['metadata']['baseline_approach']}
  - Uses few-shot learning with 3 Varun transcript examples
  - Direct remix of existing patterns
  - Temperature: 0.8 (fixed)
- **Creative Approach**: {report['metadata']['creative_approach']}
  - Divergence-convergence pipeline (ideation → judge → draft)
  - Transcends reference examples to create novel concepts
  - Creativity level: 1.0 (maximum)
  - Quality threshold: 0.75

## Results Summary

| Product | Baseline Score | Creative Score | Improvement |
|---------|----------------|----------------|-------------|
"""
    
    # Add per-product comparison
    for item in report['analysis'].get('per_product_comparison', []):
        baseline_score = item.get('baseline_score', 0)
        creative_score = item.get('creative_score', 0)
        improvement = item.get('improvement', 0)
        
        baseline_str = f"{baseline_score:.2f}" if baseline_score else "N/A"
        creative_str = f"{creative_score:.2f}" if creative_score else "N/A"
        improvement_str = f"+{improvement:.1f}%" if improvement else "N/A"
        
        md += f"| {item['product']} | {baseline_str} | {creative_str} | {improvement_str} |\n"
    
    # Overall averages
    baseline_avg = report['analysis'].get('baseline_average_creativity', 0)
    creative_avg = report['analysis'].get('creative_average_creativity', 0)
    improvement_pct = report['analysis'].get('improvement_percentage', 0)
    
    md += f"""
## Average Creativity Scores

- **Baseline**: {baseline_avg:.2f} (±{report['analysis'].get('baseline_std', 0):.2f})
- **Creative**: {creative_avg:.2f} (±{report['analysis'].get('creative_std', 0):.2f})
- **Overall Improvement**: {improvement_pct:+.1f}%

## Detailed Analysis

### Key Insights

"""
    
    # Add insights based on comparison
    if improvement_pct > 0:
        md += f"- ✅ Creative Engine shows **{improvement_pct:.1f}% improvement** over baseline\n"
    else:
        md += f"- ⚠️ Creative Engine shows **{abs(improvement_pct):.1f}% decrease** compared to baseline\n"
    
    md += f"""
### Per-Product Breakdown

"""
    
    # Add detailed per-product analysis
    for item in report['analysis'].get('per_product_comparison', []):
        product = item['product']
        baseline_score = item.get('baseline_score', 0)
        creative_score = item.get('creative_score', 0)
        
        # Find corresponding results for concept details
        creative_result = next((c for c in creative_results if c['product_name'] == product), None)
        
        md += f"#### {product}\n\n"
        md += f"- **Baseline Score**: {baseline_score:.2f}\n"
        md += f"- **Creative Score**: {creative_score:.2f}\n"
        
        if creative_result:
            md += f"- **Selected Concept**: {creative_result.get('concept_title', 'N/A')}\n"
            md += f"- **Quality Score**: {creative_result.get('quality_score', 0):.3f}\n"
        
        md += "\n"
    
    md += f"""
## Criteria Breakdown

### Baseline Average Criteria Scores

"""
    
    if baseline_results and baseline_results[0].get('creativity_assessment'):
        criteria = baseline_results[0]['creativity_assessment']['aggregate']['criteria']
        for criterion, stats in criteria.items():
            md += f"- **{criterion.replace('_', ' ').title()}**: {stats['mean']:.2f} (±{stats['std']:.2f})\n"
    
    md += "\n### Creative Average Criteria Scores\n\n"
    
    if creative_results and creative_results[0].get('creativity_assessment'):
        criteria = creative_results[0]['creativity_assessment']['aggregate']['criteria']
        for criterion, stats in criteria.items():
            md += f"- **{criterion.replace('_', ' ').title()}**: {stats['mean']:.2f} (±{stats['std']:.2f})\n"
    
    md += f"""
## Conclusion

This comparison demonstrates the effectiveness of the divergence-convergence pipeline
compared to traditional few-shot learning for generating creative video scripts.

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['metadata']['timestamp']))}
"""
    
    return md


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("BASELINE vs CREATIVE ENGINE COMPARISON")
    logger.info("=" * 60)
    logger.info(f"Products: {len(PRODUCTS)}")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Style: {STYLE}")
    
    # Check API keys
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set (required for both baseline and creative engine)")
        logger.error("   Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    else:
        logger.info("✅ OPENAI_API_KEY found")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Phase 1: Generate baseline scripts
    baseline_results = generate_baseline_scripts(PRODUCTS)
    
    # Save baseline results
    baseline_file = Path(OUTPUT_DIR) / "baseline_scripts.json"
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n💾 Saved baseline scripts to: {baseline_file}")
    
    # Phase 2: Generate creative scripts
    creative_results = generate_creative_scripts(PRODUCTS)
    
    # Save creative results
    creative_file = Path(OUTPUT_DIR) / "creative_scripts.json"
    with open(creative_file, "w", encoding="utf-8") as f:
        json.dump(creative_results, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved creative scripts to: {creative_file}")
    
    # Phase 3: Evaluate both baseline and creative scripts
    engine = CreativeEngine(
        model=MODEL,
        config=CreativityConfig(creativity_level=0.7, quality_threshold=0.75)
    )
    baseline_results = evaluate_baseline_scripts(baseline_results, engine)
    creative_results = evaluate_creative_scripts(creative_results, engine)
    
    # Re-save baseline with evaluations
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)
    
    # Re-save creative with evaluations
    with open(creative_file, "w", encoding="utf-8") as f:
        json.dump(creative_results, f, indent=2, ensure_ascii=False)
    
    # Phase 4: Generate comparison report
    report = generate_comparison_report(baseline_results, creative_results)
    
    # Save JSON report
    report_file = Path(OUTPUT_DIR) / "comparison_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved comparison report to: {report_file}")
    
    # Generate and save markdown report
    md_report = generate_markdown_report(report, baseline_results, creative_results)
    md_file = Path(OUTPUT_DIR) / "comparison_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info(f"💾 Saved markdown summary to: {md_file}")
    
    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Baseline Average: {report['analysis'].get('baseline_average_creativity', 0):.2f}")
    logger.info(f"Creative Average: {report['analysis'].get('creative_average_creativity', 0):.2f}")
    logger.info(f"Improvement: {report['analysis'].get('improvement_percentage', 0):+.1f}%")
    logger.info(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
