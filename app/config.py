"""
Configuration settings for Sugar-AI.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Dict, List, Any, Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Dev mode (THIS MUST EXIST)
    DEV_MODE: bool = os.getenv("DEV_MODE", "0") == "1"
    DEV_MODEL_NAME: str | None = None
    PROD_MODEL_NAME: str | None = None
    DEFAULT_MODEL: str | None = None
    
    API_KEYS: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    MODEL_CHANGE_PASSWORD: str = ""
    DOC_PATHS: List[str] = Field(default_factory=list)
    MAX_DAILY_REQUESTS: int = 100

    # OAuth
    github_client_id: Optional[str] = None
    github_client_secret: Optional[str] = None
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    oauth_redirect_uri: Optional[str] = None
    session_secret_key: Optional[str] = None
    
    port: Optional[str] = None
    
    # application settings
    TEMPLATES_DIR: str = "templates"
    
    class Config:
        env_file = ".env"
        extra = "allow"  # this allows extra attribute if we have any

settings = Settings()