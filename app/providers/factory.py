import os
from app.providers.base import BaseProvider

def get_provider() -> BaseProvider:
    provider = os.getenv("AI_PROVIDER", "huggingface_local")

    if provider == "ollama":
        from app.providers.ollama_provider import OllamaProvider
        return OllamaProvider()
    elif provider == "huggingface_local":
        from app.providers.huggingface_provider import HuggingFaceLocalProvider
        return HuggingFaceLocalProvider()
    else:
        raise ValueError(f"Unknown AI_PROVIDER: {provider}")