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


"""Google Gemini provider for Sugar-AI."""
import httpx
import logging
from typing import Optional

from app.providers.base import BaseProvider, GenerationParams

logger = logging.getLogger("sugar-ai")

_DEFAULT_TIMEOUT = 120.0
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(BaseProvider):
    """Provider that connects to Google's Gemini generateContent API.

    The model name is placed in the URL path per-request, so one client
    can serve any Gemini model. Only base_url and api_key differ per setup.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
    ):
        if not api_key:
            raise ValueError(
                "GeminiProvider requires an api_key. "
                "Set GEMINI_API_KEY in your environment."
            )
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "GeminiProvider initialized: model=%s, server=%s",
            model_name,
            self.base_url,
        )

    async def chat(self, messages: list[dict], params: Optional[GenerationParams] = None) -> str:
        """Generate a response from chat messages."""
        if params is None:
            params = GenerationParams()

        contents, system_instruction = self._to_gemini_contents(messages)

        payload = {
            "contents": contents,
            "generationConfig": self._params_to_config(params),
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        response = await self._client.post(
            f"{self.base_url}/models/{self.model_name}:generateContent",
            json=payload,
        )
        response.raise_for_status()

        return self._extract_text(response.json())

    async def health_check(self) -> bool:
        """Check if the endpoint is reachable and the key/model are valid."""
        try:
            response = await self._client.post(
                f"{self.base_url}/models/{self.model_name}:generateContent",
                json={
                    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                    "generationConfig": {"maxOutputTokens": 1},
                },
            )
            return response.status_code == 200
        except Exception:
            return False

    def _to_gemini_contents(self, messages: list[dict]) -> tuple[list[dict], str]:
        """Translate role/content messages into (contents, system_instruction)."""
        contents = []
        system_parts = []
        for message in messages:
            role = message.get("role", "user")
            text = message.get("content", "")
            if role == "system":
                if text:
                    system_parts.append(text)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
        return contents, "\n\n".join(system_parts)

    def _extract_text(self, data: dict) -> str:
        """Pull the response text out of a generateContent payload."""
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()

    def _params_to_config(self, params: GenerationParams) -> dict:
        """Convert GenerationParams to Gemini's generationConfig format."""
        return {
            "maxOutputTokens": params.max_new_tokens,
            "temperature": params.temperature,
            "topP": params.top_p,
            "topK": params.top_k,
        }
