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


    