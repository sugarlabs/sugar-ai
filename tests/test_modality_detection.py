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

"""A backend that reports per-model capabilities is believed over the default."""

from unittest.mock import patch

import httpx
import pytest

from app.providers import create_provider
from app.providers.base import BaseProvider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider


def ollama_with_show(capabilities=None, status=200, fail=False):
    """An OllamaProvider whose /api/show answer is scripted."""
    asked = {}

    def handler(request: httpx.Request) -> httpx.Response:
        asked["url"] = str(request.url)
        asked["body"] = request.read()
        if fail:
            raise httpx.ConnectError("refused")
        return httpx.Response(status, json={"capabilities": capabilities or []})

    provider = OllamaProvider("some-model")
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    provider.asked = asked
    return provider


# --- Ollama ----------------------------------------------------------------


def test_vision_capability_becomes_image():
    provider = ollama_with_show(["completion", "vision", "tools"])
    provider.detect_modalities()
    assert provider.supported_modalities == {"text", "image"}


def test_a_text_model_loses_the_image_default():
    provider = ollama_with_show(["completion", "tools", "thinking"])
    provider.detect_modalities()
    assert provider.supported_modalities == {"text"}


def test_the_model_is_named_in_the_query():
    provider = ollama_with_show(["completion"])
    provider.detect_modalities()
    assert provider.asked["url"].endswith("/api/show")
    assert b"some-model" in provider.asked["body"]


def test_unknown_capabilities_are_ignored():
    provider = ollama_with_show(["completion", "embedding", "future-thing"])
    provider.detect_modalities()
    assert provider.supported_modalities == {"text"}


def test_unreachable_server_keeps_the_default():
    provider = ollama_with_show(fail=True)
    provider.detect_modalities()
    assert provider.supported_modalities == OllamaProvider.default_modalities


def test_unknown_model_keeps_the_default():
    provider = ollama_with_show(status=404)
    provider.detect_modalities()
    assert provider.supported_modalities == OllamaProvider.default_modalities


def test_construction_alone_does_not_query():
    """Detection is a separate step, so building a provider is offline."""
    calls = []
    with patch.object(httpx.Client, "post", side_effect=lambda *a, **k: calls.append(1)):
        OllamaProvider("some-model")
    assert calls == []


# --- Other providers -------------------------------------------------------


def test_gemini_has_nothing_to_ask_and_keeps_its_default():
    provider = GeminiProvider("gemini-2.5-flash", api_key="k")
    provider.detect_modalities()
    assert provider.supported_modalities == GeminiProvider.default_modalities


def test_openai_compatible_keeps_its_default():
    provider = BaseProvider("gpt-4o", api_key="k")
    provider.detect_modalities()
    assert provider.supported_modalities == {"text"}


# --- Factory wiring --------------------------------------------------------


def test_factory_detects_when_nothing_is_configured():
    with patch.object(OllamaProvider, "detect_modalities") as detect:
        create_provider("ollama", "some-model")
    detect.assert_called_once()


def test_factory_skips_detection_when_configured():
    """AI_SUPPORTED_MODALITIES is the operator's word and is not second-guessed."""
    with patch.object(OllamaProvider, "detect_modalities") as detect:
        provider = create_provider(
            "ollama", "some-model", supported_modalities=["image", "audio"]
        )
    detect.assert_not_called()
    assert provider.supported_modalities == {"text", "image", "audio"}


@pytest.mark.parametrize("name", ["gemini", "openai"])
def test_factory_calls_the_hook_on_every_provider(name):
    with patch.object(BaseProvider, "detect_modalities") as detect:
        create_provider(name, "m", api_key="k", gemini_api_key="k")
    detect.assert_called_once()
