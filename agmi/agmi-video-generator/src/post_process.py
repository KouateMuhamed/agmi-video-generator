import os
from moviepy import VideoFileClip, concatenate_videoclips

def stitch_segments(file_paths, output_name="final_ad.mp4"):
    clips = []
    
    # 1. Load Clips
    for f in file_paths:
        if os.path.exists(f):
            try:
                clip = VideoFileClip(f)
                clips.append(clip)
            except Exception as e:
                print(f"⚠️ Could not load {f}: {e}")
    
    if not clips:
        print("❌ No valid clips to stitch.")
        return None

    # 2. Concatenate
    print(f"   ...Merging {len(clips)} clips...")
    try:
        # method="compose" ensures they snap together even if formats differ slightly
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # 3. Export
        final_clip.write_videofile(
            output_name, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac",
            logger=None # Silence FFMPEG logs
        )
        return output_name
    except Exception as e:
        print(f"❌ Error during stitching: {e}")
        return None

