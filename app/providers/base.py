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


"""Base provider interface for Sugar-AI."""
import httpx
import logging
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger("sugar-ai")

# Every provider handles text. Anything beyond it is declared per provider,
# and per model, because support varies within a single API.
TEXT_ONLY = frozenset({"text"})

# OpenAI names an audio clip's format, not its mime type.
_AUDIO_FORMATS = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
}

# Cloud APIs are usually fast, but allow headroom for cold routes / rate-limit
# retries handled upstream. 120s is generous without hanging forever.
_DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class GenerationParams:
    """Parameters controlling text generation behavior."""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    truncation: bool = True
    do_sample: bool = True

    def __post_init__(self):
        object.__setattr__(self, "do_sample", self.temperature > 0)


class BaseProvider:
    """OpenAI-compatible provider: speaks /v1/chat/completions over HTTP."""

    # Whether an OpenAI-compatible endpoint accepts images or audio depends
    # on the model behind it, so the default stays text and a deployment
    # widens it through AI_SUPPORTED_MODALITIES.
    default_modalities = TEXT_ONLY

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        supported_modalities: Optional[Iterable[str]] = None,
    ):
        if not api_key:
            raise ValueError(
                f"{type(self).__name__} requires an api_key. "
                "Set OPENAI_API_KEY in your environment."
            )
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.set_supported_modalities(supported_modalities)
        self._client = httpx.Client(
            timeout=_DEFAULT_TIMEOUT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "%s initialized: model=%s, server=%s",
            type(self).__name__,
            model_name,
            self.base_url,
        )

    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """Generate text from a plain prompt by wrapping it as a user message."""
        return self.chat([{"role": "user", "content": prompt}], params)

    def chat(self, messages: list[dict], params: Optional[GenerationParams] = None) -> str:
        """Generate a response from chat messages via /chat/completions."""
        if params is None:
            params = GenerationParams()

        payload = {
            "model": self.model_name,
            "messages": [self._to_openai_message(message) for message in messages],
            "stream": False,
            **self._params_to_options(params),
        }

        response = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return (message.get("content") or "").strip()

    def _to_openai_message(self, message: dict) -> dict:
        """Render one message in the OpenAI content format."""
        content = message.get("content", "")
        if isinstance(content, str):
            return message
        return {
            **message,
            "content": [self._to_openai_part(part) for part in content],
        }

    def _to_openai_part(self, part: dict) -> dict:
        """Render one content part as an OpenAI content block."""
        kind = part.get("type")

        if kind == "text":
            return {"type": "text", "text": part["text"]}

        if kind == "image":
            data_url = f"data:{part['mime_type']};base64,{part['data']}"
            return {"type": "image_url", "image_url": {"url": data_url}}

        if kind == "audio":
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": part["data"],
                    "format": _AUDIO_FORMATS[part["mime_type"]],
                },
            }

        raise ValueError(f"Unsupported content part: {kind}")

    def set_supported_modalities(
        self, modalities: Optional[Iterable[str]] = None
    ) -> None:
        """Declare what this provider accepts, defaulting to its own class."""
        if modalities is None:
            self.supported_modalities = frozenset(self.default_modalities)
        else:
            self.supported_modalities = frozenset(modalities) | TEXT_ONLY

    def get_model_name(self) -> str:
        return self.model_name

    def health_check(self) -> bool:
        """Verify the endpoint is reachable and the key/model are valid."""
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            return response.status_code == 200
        except Exception:
            return False

    def _params_to_options(self, params: GenerationParams) -> dict:
        """Map GenerationParams to OpenAI chat-completions fields.

        Only OpenAI-standard fields are sent. top_k and repetition_penalty
        are not part of the spec and are intentionally omitted."""
        
        return {
            "max_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }

    def get_eos_token(self) -> Optional[str]:
        """Return the provider's EOS token string if one is known."""
        return None

    def close(self) -> None:
        """Release provider resources."""
        pass
