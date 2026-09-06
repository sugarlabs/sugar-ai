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
import os
from dataclasses import dataclass, replace
from typing import Optional

from app.context import ContextBudget, estimate_tokens, fit_messages, fit_text

logger = logging.getLogger("sugar-ai")

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

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
    ):
        if not api_key:
            raise ValueError(
                f"{type(self).__name__} requires an api_key. "
                "Set OPENAI_API_KEY in your environment."
            )
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
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

    def get_context_window(self) -> int:
        """Return the model context window, with a portable safe default."""
        return max(1, int(os.getenv("AI_CONTEXT_WINDOW", "4096")))

    def get_model_metadata(self) -> dict:
        """Expose model limits to the runtime and frontend."""
        max_output = min(1024, max(1, self.get_context_window() - 1))
        return {
            "model": self.get_model_name(),
            "provider": type(self).__name__,
            "context_window": self.get_context_window(),
            "max_output_tokens": max_output,
            "safe_input_tokens": self.get_context_window() - max_output,
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens for budgeting; providers may use an exact tokenizer."""
        return estimate_tokens(text)

    def prepare_messages(self, messages: list[dict], params: GenerationParams) -> list[dict]:
        """Compress older turns so input plus output fits the model window."""
        budget = ContextBudget(
            context_window=self.get_context_window(),
            output_tokens=min(params.max_new_tokens, self.get_context_window() - 1),
        )
        return fit_messages(messages, budget, counter=self.count_tokens)

    def bound_params(self, params: GenerationParams) -> GenerationParams:
        """Clamp output tokens so input and output cannot exceed the window."""
        return replace(
            params,
            max_new_tokens=min(params.max_new_tokens, self.get_context_window() - 1),
        )

    def prepare_prompt(self, prompt: str, params: GenerationParams) -> str:
        """Trim a plain prompt so an output reservation is always preserved."""
        budget = ContextBudget(
            context_window=self.get_context_window(),
            output_tokens=min(params.max_new_tokens, self.get_context_window() - 1),
        )
        return fit_text(prompt, budget, counter=self.count_tokens)

    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """Generate text from a plain prompt by wrapping it as a user message."""
        params = params or GenerationParams()
        prompt = self.prepare_prompt(prompt, params)
        return self.chat([{"role": "user", "content": prompt}], params)

    def chat(self, messages: list[dict], params: Optional[GenerationParams] = None) -> str:
        """Generate a response from chat messages via /chat/completions."""
        if params is None:
            params = GenerationParams()
        params = self.bound_params(params)
        messages = self.prepare_messages(messages, params)

        payload = {
            "model": self.model_name,
            "messages": messages,
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
