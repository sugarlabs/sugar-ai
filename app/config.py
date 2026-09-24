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

    # Provider selection
    AI_PROVIDER: str = 'huggingface'
    AI_MODEL: str | None = None
    OLLAMA_BASE_URL: str = 'http://localhost:11434'

    # OpenAI-compatible provider (Groq, Cerebras, OpenRouter, OpenAI, Mistral, ...)
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str = 'https://api.openai.com/v1'

    # Google Gemini provider
    GEMINI_API_KEY: str | None = None
    GEMINI_BASE_URL: str = 'https://generativelanguage.googleapis.com/v1beta'

    API_KEYS: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    MODEL_CHANGE_PASSWORD: str = ""
    DOC_PATHS: List[str] = Field(default_factory=list)
    MAX_DAILY_REQUESTS: int = 100

    # Reasoning shares the output token budget with the answer. When think is on,
    # add this many tokens on top of the caller's max_length so reasoning cannot
    # starve the answer to empty. Tune per model (larger models need less).
    THINKING_HEADROOM: int = Field(
        2048,
        ge=0,
        le=8192,
        description="Extra output tokens reserved for reasoning",
    )

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
