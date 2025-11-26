import os
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import json

def get_product_info(url):
    print(f"   ...Scraping {url}...")
    
    # 1. Fetch HTML
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        print(f"⚠️ Scraping failed: {e}")
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
    text_content = soup.get_text(separator=' ', strip=True)[:10000] # Limit to 10k chars

    # 3. Analyze with Gemini
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
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
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
            contents=prompt
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"⚠️ Gemini extraction failed: {e}")
        return {
            "name": title or "Unknown Product",
            "description": meta_desc or "No description available.",
            "pain_point": "Unknown problem"
        }

