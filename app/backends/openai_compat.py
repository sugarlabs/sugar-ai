"""app/backends/openai_compat.py - OpenAI-compatible REST API backend.

Covers any provider that implements the OpenAI chat completions API:
  - OpenAI          (gpt-4o, gpt-4o-mini, etc.)
  - Groq            (llama-3.1, mixtral, gemma — very fast inference)
  - Together AI     (open models via API)
  - Anthropic       (claude-3-haiku, claude-3-sonnet — via compat layer)
  - Google Gemini   (gemini-1.5-flash, gemini-2.0-flash — free tier)
  - Ollama          (local models via /v1/chat/completions)
  - Any OpenAI-compatible endpoint

Config keys:
  api_key      : API key (required for cloud providers)
  base_url     : API base URL (default: https://api.openai.com/v1)
  model_name   : model identifier (required)
  timeout      : request timeout seconds (default: 30)
  max_retries  : number of retries on transient errors (default: 2)
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from app.backends.base import (
    BackendCapabilities,
    BackendError,
    BackendQuotaError,
    BackendResponse,
    BackendTimeoutError,
    BackendUnavailableError,
    GenerationConfig,
    Message,
    ModelBackend,
)

logger = logging.getLogger("sugar_ai.backends.openai_compat")

_openai = None


def _import_deps():
    global _openai
    if _openai is None:
        try:
            import openai
            _openai = openai
        except ImportError as e:
            raise BackendUnavailableError(
                f"OpenAI-compatible backend requires 'openai': {e}",
                backend="openai_compat",
            )
    return _openai


PROVIDER_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3-8b-chat-hf",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-3-haiku-20240307",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3",
        "api_key": "ollama",  
    },
}


class OpenAICompatBackend(ModelBackend):
    """OpenAI-compatible REST API backend.

    One class handles all cloud providers and local Ollama,
    configured via base_url and api_key.
    """

    name = "openai_compat"

    def __init__(self, config: dict):
        super().__init__(config)

        # Support provider shorthand: provider: "groq"
        provider = config.get("provider", "")
        preset = PROVIDER_PRESETS.get(provider, {})

        self._api_key = config.get("api_key") or preset.get("api_key", "")
        self._base_url = config.get("base_url") or preset.get("base_url", "https://api.openai.com/v1")
        self._model_name = config.get("model_name") or preset.get("default_model", "gpt-4o-mini")
        self._timeout = int(config.get("timeout", 30))
        self._max_retries = int(config.get("max_retries", 2))
        self._provider = provider

        self._client = None

    def is_available(self) -> bool:
        try:
            _import_deps()
            # Ollama doesn't need a real key
            if self._provider == "ollama":
                return True
            return bool(self._api_key)
        except BackendUnavailableError:
            return False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            chat_history=True,
            system_prompt=True,
            token_counting=False,
            model_switching=True,
            local=(self._provider == "ollama"),
        )

    def ask(
        self,
        question: str,
        history: list[Message] | None = None,
        config: GenerationConfig | None = None,
    ) -> BackendResponse:
        config = config or GenerationConfig()
        openai = _import_deps()
        client = self._get_client(openai)
        messages = self._build_messages(question, history)

        for attempt in range(self._max_retries + 1):
            try:
                t0 = time.monotonic()
                response = client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    timeout=self._timeout,
                )
                latency_ms = (time.monotonic() - t0) * 1000

                content = response.choices[0].message.content or ""
                usage = response.usage

                return BackendResponse(
                    content=content.strip(),
                    model=response.model,
                    backend=self.name,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency_ms,
                    finish_reason=response.choices[0].finish_reason or "stop",
                )

            except openai.RateLimitError as e:
                raise BackendQuotaError(str(e), backend=self.name)
            except openai.APITimeoutError as e:
                if attempt < self._max_retries:
                    logger.warning("Timeout on attempt %d, retrying...", attempt + 1)
                    time.sleep(2 ** attempt)
                    continue
                raise BackendTimeoutError(str(e), backend=self.name)
            except openai.APIError as e:
                raise BackendError(
                    f"API error ({self._provider}): {e}", backend=self.name
                ) from e

    def stream(
        self,
        question: str,
        history: list[Message] | None = None,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        config = config or GenerationConfig()
        openai = _import_deps()
        client = self._get_client(openai)
        messages = self._build_messages(question, history)

        try:
            with client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                stream=True,
                timeout=self._timeout,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except openai.APIError as e:
            raise BackendError(str(e), backend=self.name) from e

    def _get_client(self, openai):
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self._api_key or "no-key",
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=0,  # we handle retries ourselves
            )
        return self._client
