"""app/backends/config.py - Configuration system for sugar-ai backends.

Loads backend configuration from:
  1. sugar_ai.yaml (primary config file)
  2. Environment variables (override yaml values)
  3. .env file (for secrets)

Example sugar_ai.yaml:

  primary_backend:
    type: huggingface
    model_name: Qwen/Qwen2-1.5B-Instruct
    device: cpu
    torch_dtype: float32
    max_context: 2048
    system_prompt: "You are a helpful assistant for children."

  fallback_backends:
    - type: openai_compat
      provider: groq
      api_key: ${GROQ_API_KEY}
      model_name: llama-3.1-8b-instant

    - type: openai_compat
      provider: gemini
      api_key: ${GEMINI_API_KEY}
      model_name: gemini-2.0-flash

  max_history_tokens: 1500
  token_budget_per_request: 512
  profanity_check: true

Environment variable overrides:
  SUGAR_AI_BACKEND_TYPE     — override primary backend type
  SUGAR_AI_MODEL_NAME       — override primary model name
  SUGAR_AI_API_KEY          — override primary API key
  SUGAR_AI_BASE_URL         — override primary base URL
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("sugar_ai.config")

# Default config if no yaml file is found
_DEFAULT_CONFIG = {
    "primary_backend": {
        "type": "huggingface",
        "model_name": "Qwen/Qwen2-1.5B-Instruct",
        "device": "cpu",
        "torch_dtype": "float32",
        "max_context": 2048,
    },
    "fallback_backends": [],
    "max_history_tokens": 1500,
    "token_budget_per_request": 512,
    "profanity_check": True,
}


@dataclass
class SugarAIConfig:
    """Parsed and validated sugar-ai backend configuration."""

    primary_backend: dict = field(default_factory=dict)
    fallback_backends: list[dict] = field(default_factory=list)
    max_history_tokens: int = 1500
    token_budget_per_request: int = 512
    profanity_check: bool = True

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "SugarAIConfig":
        """Load config from yaml file, then apply env var overrides.

        Parameters
        ----------
        config_path : str | Path | None
            Path to sugar_ai.yaml. If None, searches standard locations.
        """
        raw = cls._load_yaml(config_path)
        raw = cls._apply_env_overrides(raw)
        raw = cls._interpolate_env_vars(raw)
        return cls._parse(raw)

    @classmethod
    def _load_yaml(cls, config_path: str | Path | None) -> dict:
        """Load yaml config file."""
        search_paths = [
            config_path,
            Path("sugar_ai.yaml"),
            Path("config/sugar_ai.yaml"),
            Path(os.environ.get("SUGAR_AI_CONFIG", "sugar_ai.yaml")),
        ]

        for path in search_paths:
            if path and Path(path).exists():
                try:
                    import yaml
                    with open(path) as f:
                        raw = yaml.safe_load(f)
                    logger.info("Loaded config from: %s", path)
                    return raw or {}
                except ImportError:
                    logger.warning(
                        "PyYAML not installed. Using default config. "
                        "Install with: pip install pyyaml"
                    )
                except Exception as e:
                    logger.warning("Failed to load %s: %s", path, e)

        logger.info("No config file found, using defaults")
        return dict(_DEFAULT_CONFIG)

    @classmethod
    def _apply_env_overrides(cls, raw: dict) -> dict:
        """Apply top-level environment variable overrides."""
        primary = raw.get("primary_backend", {})

        if os.environ.get("SUGAR_AI_BACKEND_TYPE"):
            primary["type"] = os.environ["SUGAR_AI_BACKEND_TYPE"]
        if os.environ.get("SUGAR_AI_MODEL_NAME"):
            primary["model_name"] = os.environ["SUGAR_AI_MODEL_NAME"]
        if os.environ.get("SUGAR_AI_API_KEY"):
            primary["api_key"] = os.environ["SUGAR_AI_API_KEY"]
        if os.environ.get("SUGAR_AI_BASE_URL"):
            primary["base_url"] = os.environ["SUGAR_AI_BASE_URL"]

        raw["primary_backend"] = primary
        return raw

    @classmethod
    def _interpolate_env_vars(cls, obj: Any) -> Any:
        """Replace ${VAR_NAME} placeholders with environment variable values."""
        if isinstance(obj, str):
            pattern = re.compile(r"\$\{([^}]+)\}")
            def replacer(match):
                var_name = match.group(1)
                value = os.environ.get(var_name, "")
                if not value:
                    logger.warning("Environment variable %s is not set", var_name)
                return value
            return pattern.sub(replacer, obj)
        elif isinstance(obj, dict):
            return {k: cls._interpolate_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._interpolate_env_vars(item) for item in obj]
        return obj

    @classmethod
    def _parse(cls, raw: dict) -> "SugarAIConfig":
        primary = raw.get("primary_backend", _DEFAULT_CONFIG["primary_backend"])
        if not primary.get("type"):
            primary["type"] = "huggingface"

        fallbacks = raw.get("fallback_backends", [])
        if not isinstance(fallbacks, list):
            fallbacks = []

        return cls(
            primary_backend=primary,
            fallback_backends=fallbacks,
            max_history_tokens=int(raw.get("max_history_tokens", 1500)),
            token_budget_per_request=int(raw.get("token_budget_per_request", 512)),
            profanity_check=bool(raw.get("profanity_check", True)),
        )

    def summary(self) -> dict:
        """Return a safe (no secrets) summary for logging."""
        def _redact(d: dict) -> dict:
            return {
                k: "***" if "key" in k.lower() or "token" in k.lower() or "secret" in k.lower() else v
                for k, v in d.items()
            }

        return {
            "primary_backend": _redact(self.primary_backend),
            "fallback_backends": [_redact(fb) for fb in self.fallback_backends],
            "max_history_tokens": self.max_history_tokens,
            "token_budget_per_request": self.token_budget_per_request,
        }
