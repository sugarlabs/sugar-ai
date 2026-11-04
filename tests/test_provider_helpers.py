# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Tests for provider helper methods."""

from unittest.mock import AsyncMock, Mock

import httpx

from app.providers.base import BaseProvider, GenerationParams
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider


def test_openai_compatible_options_use_standard_fields():
    provider = object.__new__(BaseProvider)
    params = GenerationParams(
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.8,
        top_k=10,
        repetition_penalty=1.3,
    )

    options = provider._params_to_options(params)

    assert options == {
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.8,
    }


def test_ollama_options_use_ollama_field_names():
    provider = object.__new__(OllamaProvider)
    params = GenerationParams(
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.8,
        top_k=10,
        repetition_penalty=1.3,
    )

    options = provider._params_to_options(params)

    assert options == {
        "num_predict": 256,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 10,
        "repeat_penalty": 1.3,
    }


async def test_ollama_generate_forwards_think_to_async_fallback():
    provider = object.__new__(OllamaProvider)
    provider.model_name = "test-model"
    provider._post_with_think_fallback = AsyncMock(
        return_value={"response": " reasoned answer "}
    )

    params = GenerationParams(think=True)

    answer = await provider.generate("Explain gravity.", params)

    assert answer == "reasoned answer"
    provider._post_with_think_fallback.assert_awaited_once_with(
        "/api/generate",
        {
            "model": "test-model",
            "prompt": "Explain gravity.",
            "stream": False,
            "options": {
                "num_predict": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.1,
            },
            "think": True,
        },
    )


async def test_ollama_retries_without_think_when_model_rejects_it():
    provider = object.__new__(OllamaProvider)
    provider.model_name = "test-model"
    provider.base_url = "http://ollama.example:11434"

    unsupported = Mock(status_code=httpx.codes.BAD_REQUEST)
    unsupported.json.return_value = {
        "error": "model does not support thinking",
    }
    successful = Mock(status_code=httpx.codes.OK)
    successful.json.return_value = {"response": "fallback answer"}

    captured_payloads = []

    async def post(url, json):
        captured_payloads.append((url, json.copy()))
        return unsupported if len(captured_payloads) == 1 else successful

    client = Mock()
    client.post = AsyncMock(side_effect=post)
    provider._client = client
    payload = {
        "model": "test-model",
        "prompt": "Explain gravity.",
        "think": True,
    }

    data = await provider._post_with_think_fallback("/api/generate", payload)

    assert data == {"response": "fallback answer"}
    assert captured_payloads == [
        (
            "http://ollama.example:11434/api/generate",
            {
                "model": "test-model",
                "prompt": "Explain gravity.",
                "think": True,
            },
        ),
        (
            "http://ollama.example:11434/api/generate",
            {
                "model": "test-model",
                "prompt": "Explain gravity.",
            },
        ),
    ]
    successful.raise_for_status.assert_called_once_with()


def test_gemini_config_uses_gemini_field_names():
    provider = object.__new__(GeminiProvider)
    params = GenerationParams(
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.8,
        top_k=10,
    )

    config = provider._params_to_config(params)

    assert config == {
        "maxOutputTokens": 256,
        "temperature": 0.2,
        "topP": 0.8,
        "topK": 10,
    }


def test_gemini_converts_chat_messages_to_gemini_format():
    provider = object.__new__(GeminiProvider)

    contents, system_instruction = provider._to_gemini_contents(
        [
            {"role": "system", "content": "You are a patient tutor."},
            {"role": "user", "content": "What is a loop?"},
            {"role": "assistant", "content": "A loop repeats instructions."},
        ]
    )

    assert system_instruction == "You are a patient tutor."
    assert contents == [
        {
            "role": "user",
            "parts": [{"text": "What is a loop?"}],
        },
        {
            "role": "model",
            "parts": [{"text": "A loop repeats instructions."}],
        },
    ]


def test_gemini_extracts_text_from_all_response_parts():
    provider = object.__new__(GeminiProvider)

    text = provider._extract_text(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Hello"},
                            {"text": ", learner!"},
                        ]
                    }
                }
            ]
        }
    )

    assert text == "Hello, learner!"


def test_gemini_returns_empty_text_when_response_has_no_candidates():
    provider = object.__new__(GeminiProvider)

    assert provider._extract_text({}) == ""


def test_base_provider_returns_its_configured_model_name():
    provider = object.__new__(BaseProvider)
    provider.model_name = "test-model"

    assert provider.get_model_name() == "test-model"


def test_base_provider_has_no_known_eos_token():
    provider = object.__new__(BaseProvider)

    assert provider.get_eos_token() is None
