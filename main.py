import sys
import os
import json
import hashlib
from src import personas, script_engine, video_generator, post_process, scraper

def run_ad_generator(url, selected_style):
    
    # 1. Validation & Setup
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not set.")
        return

    selected_style = selected_style.upper()
    if selected_style not in personas.DATA:
        print(f"❌ Error: Style '{selected_style}' not found. Available: {list(personas.DATA.keys())}")
        return

    # Ensure output dir exists
    os.makedirs("output", exist_ok=True)
    
    print(f"\n🚀 --- STARTING AD GEN: {selected_style} MODE ---")
    print(f"Target: {url}")
    
    # Create a unique identifier for this URL (using hash)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    
    # Define and create a subfolder for this specific run ID
    output_dir = f"output/{url_hash}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths for saving context and script within the subfolder
    context_file = f"{output_dir}/context_{selected_style.lower()}.json"
    script_file = f"{output_dir}/script_{selected_style.lower()}.json"
    
    # 2. Scrape Context (or load if exists and URL matches)
    should_regenerate_context = True
    if os.path.exists(context_file):
        print(f"📂 Found saved context at {context_file}...")
        with open(context_file, 'r') as f:
            saved_context = json.load(f)
        # Check if the saved context is for the same URL (extra safety check)
        if saved_context.get('url') == url:
            print(f"✅ Loading saved context (matches URL)...")
            product_context = saved_context
            should_regenerate_context = False
            print(f"✅ Loaded Context: {product_context.get('name')} - {product_context.get('pain_point')}")
        else:
            print(f"⚠️  Saved context is for different URL, regenerating...")
    
    if should_regenerate_context:
        product_context = scraper.get_product_info(url)
        # Add URL to context for future verification
        product_context['url'] = url
        print(f"✅ Context: {product_context.get('name')} - {product_context.get('pain_point')}")
        # Save context
        with open(context_file, 'w') as f:
            json.dump(product_context, f, indent=2)
        print(f"💾 Saved context to {context_file}")
    
    # 3. Generate Styled Script (or load if exists and context matches)
    persona = personas.DATA[selected_style]  # Needed for both script generation and video generation
    
    should_regenerate_script = True
    if os.path.exists(script_file) and not should_regenerate_context:
        # Only load script if we're using the saved context (same URL)
        print(f"📂 Loading saved script from {script_file}...")
        with open(script_file, 'r') as f:
            script_parts = json.load(f)
        print(f"✅ Loaded {len(script_parts)} script parts")
        should_regenerate_script = False
    else:
        if should_regenerate_context:
            print("\n✍️  Writing Script (Few-Shot Learning)...")
        else:
            print("\n✍️  Regenerating Script (context changed)...")
        
        script_parts = script_engine.generate_styled_script(product_context, persona)
        
        # Debug: Print the first prompt to verify style
        if script_parts and len(script_parts) > 0:
            print(f"   > Scene 1 Prompt Preview: {str(script_parts[0])[:100]}...")
            # Save script
            with open(script_file, 'w') as f:
                json.dump(script_parts, f, indent=2)
            print(f"💾 Saved script to {script_file}")
        else:
            print("❌ Script generation failed. Exiting.")
            return

    # 4. Generate Video
    # Note: We pass the 'selected_style' as the 'mode' to trigger Logic Split
    # We pass 'output_dir' so videos are saved in the subfolder
    video_segments = video_generator.generate_sequence(
        prompts=script_parts, 
        avatar_path=persona['avatar_path'],
        mode=selected_style,
        output_dir=output_dir,
        url_hash=url_hash
    )
    
    # 5. Stitch or Select Final Output
    if video_segments:
        if selected_style == "VARUN":
            # In Varun mode, the last segment IS the full extended video.
            # We don't stitch; we just rename/use the last generated file.
            print("\n✂️  Varun Mode: Selecting final extended clip (no stitching required)...")
            last_segment = video_segments[-1]
            final_output_name = f"{output_dir}/final_{selected_style.lower()}_{url_hash}.mp4"
            
            # Simple rename/copy logic (using OS rename if possible, or just pointing to it)
            # Here we'll try to rename it for clarity, or just copy it.
            # Let's just use moviepy to write it to the final name to be consistent/safe with formats
            # OR just a file copy for speed. Let's do file copy/rename.
            import shutil
            try:
                shutil.copy(last_segment, final_output_name)
                print(f"\n🎉 DONE! Video saved: {final_output_name}")
            except Exception as e:
                print(f"❌ Error copying final file: {e}")
                
        else:
            # Austin Mode: Stitch all independent segments
            print("\n✂️  Stitching Final Cut...")
            final_file = post_process.stitch_segments(video_segments, output_name=f"{output_dir}/final_{selected_style.lower()}_{url_hash}.mp4")
            if final_file:
                print(f"\n🎉 DONE! Video saved: {final_file}")
    else:
        print("\n❌ Generation failed, no videos to stitch.")

if __name__ == "__main__":
    # Usage: python main.py [URL] [STYLE]
    # Default: Cubic.dev / VARUN
    
    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://cubic.dev"
    style = sys.argv[2] if len(sys.argv) > 2 else "VARUN"
    
    run_ad_generator(target_url, style)

