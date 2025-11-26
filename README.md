# Agmi Video Ad Generator 🎥

A "Persona-Aware" AI video generator that creates viral-style marketing videos for products using Google Gemini and Veo. It scrapes a product URL, writes a funny script based on a specific creator persona (Varun or Austin), and generates a video sequence.

## Features

-   **Persona-Aware Generation**:
    -   **Varun Mode**: Generates a continuous, "daisy-chained" video (one-take rant style).
    -   **Austin Mode**: Generates independent clips stitched together (skit style with jump cuts).
-   **Smart Scripting**: Uses Few-Shot Learning to mimic specific humor and visual styles.
-   **Automated Workflow**: Scrapes URL -> Writes Script -> Generates Video -> Stitches Output.
-   **Efficient Management**: Saves context, scripts, and video segments in hashed subfolders to avoid re-work.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Add Assets**:
    Place avatar images in the `assets/` folder:
    -   `assets/varun_avatar.jpg`
    -   `assets/austin_avatar.jpg`
    
    > **Note**: These avatars serve as the **starting frame** for the first video clip.
    > - In **Varun Mode**, the avatar initializes the first shot, and subsequent clips extend from that motion (preserving the character).
    > - In **Austin Mode**, the avatar resets the scene for each clip to allow for costume changes or jump cuts.

3.  **Set API Key**:
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```

## Usage

Run the generator with a target URL and a persona style:

```bash
# Syntax: python main.py [URL] [STYLE]

# Mode 1: Varun (Continuous Shot)
python main.py https://cubic.dev VARUN

# Mode 2: Austin (Jump Cut Skit)
python main.py https://cubic.dev AUSTIN
```

## Output

All files are saved in `output/[URL_HASH]/`:
-   `context_*.json`: Scraped product info.
-   `script_*.json`: Generated script prompts.
-   `segment_*.mp4`: Individual video clips.
-   `final_*.mp4`: The final ready-to-use video.

