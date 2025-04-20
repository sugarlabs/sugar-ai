"""
Configuration settings for Sugar-AI.
"""
import os
import json
from pydantic_settings import BaseSettings
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    API_KEYS: Dict[str, Dict[str, Any]] = json.loads(os.getenv("API_KEYS", "{}"))
    MODEL_CHANGE_PASSWORD: str = os.getenv("MODEL_CHANGE_PASSWORD", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "Qwen/Qwen2-1.5B-Instruct")
    DOC_PATHS: List[str] = json.loads(os.getenv("DOC_PATHS", '["./docs/Pygame Documentation.pdf", "./docs/Python GTK+3 Documentation.pdf", "./docs/Sugar Toolkit Documentation.pdf"]'))
    MAX_DAILY_REQUESTS: int = int(os.getenv("MAX_DAILY_REQUESTS", 100))
    
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
