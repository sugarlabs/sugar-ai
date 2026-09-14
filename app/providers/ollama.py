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
        self._client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

        logger.info(
            "OllamaProvider initialized: model=%s, server=%s",
            model_name,
            self.base_url,
        )

    async def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """Generate text from a plain string prompt."""
        if params is None:
            params = GenerationParams()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self._params_to_options(params),
        }

        response = await self._client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    async def chat(self, messages: list[dict], params: Optional[GenerationParams] = None) -> str:
        """Generate response from chat messages."""
        if params is None:
            params = GenerationParams()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": self._params_to_options(params),
        }

        response = await self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        return message.get("content", "").strip()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if the Ollama server is reachable and the model is available."""
        try:
            response = await self._client.post(
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

    def _params_to_options(self, params: GenerationParams) -> dict:
        """Convert GenerationParams to Ollama's options format."""
        return {
            "num_predict": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "repeat_penalty": params.repetition_penalty,
        }
