import time
import os
import requests
from google import genai
from google.genai import types

def generate_sequence(prompts, avatar_path, mode="VARUN", output_dir="output", url_hash=""):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # Load Avatar
    try:
        with open(avatar_path, "rb") as f:
            avatar_bytes = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Avatar file not found at {avatar_path}")
        return []

    last_video_response = None
    saved_files = []
    
    print(f"--- 🎥 Starting Video Generation (Mode: {mode}) ---")
    
    for i, prompt_text in enumerate(prompts):
        print(f"\n   ...Processing Scene {i+1}/{len(prompts)}")
        
        # Define Config (Vertical Video)
        config = types.GenerateVideosConfig(
            aspect_ratio="9:16",
            duration_seconds=8 
        )
        
        try:
            # --- LOGIC SPLIT ---
            # Using "veo-3.1-generate-preview" as the model for video generation
            model_name = "veo-3.1-generate-preview"

            # MODE A: VARUN (Continuous Shot)
            # We pass 'video=last_video_response' to extend the previous clip.
            if mode == "VARUN":
                if i == 0:
                    # First shot uses Image
                    print("   ...Starting sequence with Avatar Image...")
                    operation = client.models.generate_videos(
                        model=model_name,
                        prompt=prompt_text,
                        image=types.Image(image_bytes=avatar_bytes, mime_type="image/jpeg"),
                        config=config
                    )
                else:
                    # Subsequent shots use Previous Video (Daisy Chain)
                    # Note: Passing the previous video object for extension
                    print("   ...Extending previous video (Daisy Chain)...")
                    operation = client.models.generate_videos(
                        model=model_name,
                        prompt=prompt_text,
                        video=last_video_response, 
                        config=config
                    )

            # MODE B: AUSTIN (Jump Cuts / Skit)
            # We ALWAYS use 'image=avatar_bytes' to reset the shot.
            elif mode == "AUSTIN":
                operation = client.models.generate_videos(
                    model=model_name,
                    prompt=prompt_text,
                    image=types.Image(image_bytes=avatar_bytes, mime_type="image/jpeg"),
                    config=config
                )
            
            # --- POLLING ---
            while not operation.done:
                time.sleep(5)
                print(".", end="", flush=True)
                operation = client.operations.get(operation)
            
            print() # Newline

            # --- SAVE ---
            if operation.response.generated_videos:
                generated_video = operation.response.generated_videos[0]
                video_obj = generated_video.video
                
                # Download the video file
                filename = f"{output_dir}/segment_{i}_{url_hash}.mp4"
                try:
                    # Try to download using the client's download method
                    downloaded_file = client.files.download(file=video_obj)
                    
                    # Check various possible attributes for video bytes
                    video_bytes = None
                    if hasattr(downloaded_file, 'video_bytes'):
                        video_bytes = downloaded_file.video_bytes
                    elif hasattr(downloaded_file, 'bytes'):
                        video_bytes = downloaded_file.bytes
                    elif hasattr(video_obj, 'video_bytes'):
                        video_bytes = video_obj.video_bytes
                    elif hasattr(video_obj, 'bytes'):
                        video_bytes = video_obj.bytes
                    elif hasattr(video_obj, 'uri'):
                        # If we have a URI, download it directly with requests
                        video_uri = video_obj.uri
                        headers = {"x-goog-api-key": os.environ.get("GEMINI_API_KEY")}
                        response = requests.get(video_uri, headers=headers, stream=True)
                        response.raise_for_status()
                        with open(filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        video_bytes = True  # Mark as downloaded
                    
                    # Write video bytes if we got them
                    if video_bytes:
                        if video_bytes is not True:  # Not already written via URI download
                            with open(filename, 'wb') as f:
                                f.write(video_bytes)
                        saved_files.append(filename)
                        print(f" ✅ Saved: {filename}")
                    else:
                        # Fallback: try save method
                        video_obj.save(filename)
                        saved_files.append(filename)
                        print(f" ✅ Saved (fallback method): {filename}")
                    
                    # For VARUN mode, we need to keep the video object for extension
                    # The video object itself is what we pass for extension
                    if mode == "VARUN":
                        last_video_response = video_obj
                except Exception as download_error:
                    print(f" ❌ Error downloading video: {download_error}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            print(f" ❌ Error generating scene {i+1}: {e}")

    return saved_files

