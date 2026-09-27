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

"""Tests for provider construction and configuration forwarding."""

from unittest.mock import patch

import pytest

from app.providers import create_provider
from app.providers.base import BaseProvider
from app.providers.gemini import GeminiProvider


def test_factory_creates_huggingface_provider_with_requested_options():
    with patch("app.providers.HuggingFaceProvider") as provider_class:
        expected_provider = provider_class.return_value

        provider = create_provider(
            "huggingface",
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            quantize=False,
            dev_mode=True,
        )

    assert provider is expected_provider
    provider_class.assert_called_once_with(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        quantize=False,
        dev_mode=True,
    )


def test_factory_creates_ollama_provider_with_requested_base_url():
    with patch("app.providers.OllamaProvider") as provider_class:
        expected_provider = provider_class.return_value

        provider = create_provider(
            "ollama",
            "qwen3.5:0.8b",
            base_url="http://ollama.example:11434",
        )

    assert provider is expected_provider
    provider_class.assert_called_once_with(
        model_name="qwen3.5:0.8b",
        base_url="http://ollama.example:11434",
    )


@pytest.mark.parametrize(
    "provider_name",
    ["openai", "openai-compatible", "openai_compatible"],
)
def test_factory_creates_openai_compatible_provider_for_each_alias(provider_name):
    with patch("app.providers.BaseProvider") as provider_class:
        expected_provider = provider_class.return_value

        provider = create_provider(
            provider_name,
            "test-model",
            api_key="test-api-key",
            openai_base_url="https://provider.example/v1",
        )

    assert provider is expected_provider
    provider_class.assert_called_once_with(
        model_name="test-model",
        api_key="test-api-key",
        base_url="https://provider.example/v1",
    )


def test_factory_creates_gemini_provider_with_requested_base_url():
    with patch("app.providers.GeminiProvider") as provider_class:
        expected_provider = provider_class.return_value

        provider = create_provider(
            "gemini",
            "gemini-2.5-flash",
            gemini_api_key="test-api-key",
            gemini_base_url="https://gemini.example/v1beta",
        )

    assert provider is expected_provider
    provider_class.assert_called_once_with(
        model_name="gemini-2.5-flash",
        api_key="test-api-key",
        base_url="https://gemini.example/v1beta",
    )


def test_factory_rejects_unknown_provider_name():
    with pytest.raises(ValueError, match="Unknown provider: 'unknown'"):
        create_provider("unknown", "test-model")


@pytest.mark.parametrize(
    ("provider_class", "expected_message"),
    [
        (BaseProvider, "BaseProvider requires an api_key"),
        (GeminiProvider, "GeminiProvider requires an api_key"),
    ],
)
def test_cloud_providers_require_an_api_key(provider_class, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        provider_class(model_name="test-model", api_key="")
