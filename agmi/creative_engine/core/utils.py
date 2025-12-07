"""
Utility functions for the Creative Engine.
"""

import logging
from typing import Dict, Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import requests

from creative_engine.core.llm import LLMProvider
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProductContext(BaseModel):
    """Structured product context extracted from a landing page."""
    
    name: str = Field(..., description="Product or company name")
    target_audience: str = Field(..., description="Target audience description")
    pain_point: str = Field(..., description="Main customer pain point addressed")
    key_benefit: str = Field(..., description="Primary benefit or value proposition")


def extract_text_from_html(html_content: str, max_words: int = 2000) -> str:
    """
    Extract and clean text content from HTML.
    
    Args:
        html_content: Raw HTML content
        max_words: Maximum number of words to extract
        
    Returns:
        Cleaned text content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    
    # Extract text from body
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    # Truncate to max_words
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
        logger.info(f"Truncated HTML content from {len(words)} to {max_words} words")
    
    return text


def extract_product_context(url: str, provider: LLMProvider) -> Dict[str, Any]:
    """
    Extract product context from a landing page URL.
    
    This function:
    1. Fetches the HTML content from the URL
    2. Extracts and cleans the text content
    3. Uses an LLM to analyze and extract structured product information
    
    Args:
        url: URL of the product landing page
        provider: LLM provider to use for extraction
        
    Returns:
        Dictionary containing: name, target_audience, pain_point, key_benefit
        
    Raises:
        ValueError: If URL is invalid or extraction fails
        requests.RequestException: If HTTP request fails
    """
    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")
    
    logger.info(f"Extracting product context from URL: {url}")
    
    # Fetch HTML content
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
        logger.info(f"Successfully fetched HTML content ({len(html_content)} chars)")
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch URL {url}: {e}")
    
    # Extract text content
    text_content = extract_text_from_html(html_content, max_words=2000)
    logger.info(f"Extracted {len(text_content.split())} words from HTML")
    
    # Use LLM to extract structured information
    system_prompt = """You are an expert marketing analyst specializing in extracting key product information from landing pages.

Your task is to analyze the provided landing page text and extract the following information:
1. Product/Company Name: The main product or company name
2. Target Audience: Who is this product for? Be specific about demographics, roles, or user types.
3. Pain Point: What problem or pain point does this product solve? What are customers struggling with?
4. Key Benefit: What is the primary value proposition or benefit? What makes this product valuable?

Be precise and concise. Extract only information that is clearly stated or strongly implied in the text.
If information is not available, make reasonable inferences based on the content, but be clear about what is inferred."""

    user_prompt = f"""Analyze the following landing page content and extract the product marketing information:

{text_content}

Extract the product name, target audience, pain point, and key benefit. Output as JSON matching the ProductContext schema."""

    try:
        logger.info("Calling LLM to extract product context...")
        product_context = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ProductContext,
            temperature=0.3,  # Lower temperature for more consistent extraction
            top_p=0.9,
        )
        
        logger.info(f"Successfully extracted product context: {product_context.name}")
        
        # Convert to dictionary
        return {
            "name": product_context.name,
            "target_audience": product_context.target_audience,
            "pain_point": product_context.pain_point,
            "key_benefit": product_context.key_benefit,
        }
    except Exception as e:
        raise ValueError(f"Failed to extract product context from URL: {e}")

