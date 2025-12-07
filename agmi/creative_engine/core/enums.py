"""
Content type enumerations for the Creative Engine.
"""

from enum import Enum


class ContentType(Enum):
    """Supported content types for generation."""
    
    VIDEO_SCRIPT = "video_script"
    LINKEDIN_POST = "linkedin_post"
    TWITTER_THREAD = "twitter_thread"
    EMAIL_COPY = "email_copy"
    REDDIT_POST = "reddit_post"
    AD_COPY = "ad_copy"

