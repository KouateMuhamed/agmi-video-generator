import os
import json
import re
from google import genai
from google.genai import types

def generate_styled_script(product_context, persona_data):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    # 1. format the examples for the prompt
    examples_text = ""
    for i, t in enumerate(persona_data['transcripts']):
        examples_text += f"--- TRAINING DATA EXAMPLE {i+1} ---\n{t}\n\n"

    # 2. System Instruction (The "Style Transfer")
    
    full_prompt = f"""
    You are a viral TikTok Scriptwriter.
    
    **YOUR GOAL:** Write a hilarious, original 3-part video script to promote a product called "{product_context['name']}".
    
    **PRODUCT CONTEXT:**
    - Name: {product_context['name']}
    - Description: {product_context['description']}
    - Pain Point Solved: {product_context['pain_point']}

    **THE VISUAL STYLE TO ADOPT:**
    {persona_data['visual_style']}
    
    **THE HUMOR & TONE REFERENCE (Few-Shot Examples):**
    {examples_text}
    
    **INSTRUCTIONS:**
    1. **Story Arc:** Create a BRAND NEW story where the character faces the specific pain point mentioned above. The story must progress logically across 3 scenes.
    2. **Humor Style:** Use the slang, pacing, and comedic delivery shown in the examples (e.g., if Varun, use "Bro", "Cooked"; if Austin, use "Tech Bro" arrogance vs. panic), but apply it to THIS new product.
    3. **Product Placement:** You MUST mention the product name ("{product_context['name']}") naturally in the dialogue as the solution to the problem.
    4. **Visuals:** 
       - If style is VARUN: Scene 1 starts with the avatar. Scene 2 and 3 must describe the action *continuing* from the previous shot (continuous take).
       - If style is AUSTIN: Scenes must be JUMP CUTS with distinct characters (e.g., "Visual: Austin dressed as CEO...").
    
    **OUTPUT FORMAT:**
    Return a raw JSON list of 3 strings. Each string is a prompt for Google Veo.
    Format: "Visual: [Detailed description of action/setting] \\n---\\nScript: [Dialogue with sound effects cues]"

    Generate the JSON list of 3 scene prompts now.
    """

    # 3. Call Gemini
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8 # Increased creativity for humor
            ),
            contents=full_prompt
        )
        
        # 4. Clean and Parse
        return json.loads(response.text)
    except Exception as e:
        print(f"⚠️ Error parsing Script JSON or calling API: {e}")
        return []
