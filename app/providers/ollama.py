# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""Ollama REST API provider for Sugar-AI."""
import httpx
import logging
from typing import Optional

from app.providers.base import BaseProvider, GenerationParams

logger = logging.getLogger("sugar-ai")

# Ollama can be slow on first request (cold model load).
# 5 minutes allows for pulling + loading a model on first use.
_DEFAULT_TIMEOUT = 300.0


class OllamaProvider(BaseProvider):
    """Provider that connects to an Ollama server via HTTP.

    Works with any Ollama instance: local (localhost:11434),
    LAN (school server), or remote (Sugar Labs AWS).
    The only difference is the base_url.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=_DEFAULT_TIMEOUT)

        logger.info(
            "OllamaProvider initialized: model=%s, server=%s",
            model_name,
            self.base_url,
        )

    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """Generate text from a plain string prompt."""
        if params is None:
            params = GenerationParams()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self._params_to_options(params),
        }
        # think is a top-level Ollama field, not part of options.
        if params.think is not None:
            payload["think"] = params.think

        data = self._post_with_think_fallback("/api/generate", payload)
        return data.get("response", "").strip()

    def chat(self, messages: list[dict], params: Optional[GenerationParams] = None) -> str:
        """Generate response from chat messages."""
        if params is None:
            params = GenerationParams()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": self._params_to_options(params),
        }
        # think is a top-level Ollama field, not part of options.
        if params.think is not None:
            payload["think"] = params.think

        data = self._post_with_think_fallback("/api/chat", payload)
        message = data.get("message", {})
        return message.get("content", "").strip()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def health_check(self) -> bool:
        """Check if the Ollama server is reachable and the model is available."""
        try:
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
            return response.status_code == 200
        except Exception:
            return False

    def _post_with_think_fallback(self, path: str, payload: dict) -> dict:
        """POST to Ollama, retrying once without the think flag if it is rejected.

        Models that do not support reasoning reject a request carrying the
        think field, so drop it and retry to return a normal answer instead
        of failing.
        """
        response = self._client.post(f"{self.base_url}{path}", json=payload)

        if "think" in payload and self._is_thinking_unsupported(response):
            payload.pop("think")
            logger.warning(
                "Model %s does not support thinking; retrying without it.",
                self.model_name,
            )
            response = self._client.post(f"{self.base_url}{path}", json=payload)

        response.raise_for_status()
        return response.json()

    @staticmethod
    def _is_thinking_unsupported(response: httpx.Response) -> bool:
        """Return whether Ollama rejected the request because think is unsupported."""
        if response.status_code != httpx.codes.BAD_REQUEST:
            return False

        try:
            error = response.json().get("error", "")
        except (ValueError, AttributeError):
            return False

        return isinstance(error, str) and "does not support thinking" in error.lower()

    def _params_to_options(self, params: GenerationParams) -> dict:
        """Convert GenerationParams to Ollama's options format."""
        return {
            "num_predict": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "repeat_penalty": params.repetition_penalty,
        }
