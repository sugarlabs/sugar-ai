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

"""Requests are refused when the active model cannot accept their media."""

import base64

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.config import Settings, settings
from app.providers.base import BaseProvider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.routes import api

PIXEL = base64.b64encode(b"\x89PNG fake").decode()
CLIP = base64.b64encode(b"RIFF fake").decode()
HEADERS = {"X-API-Key": "test-key"}


class FakeProvider:
    def __init__(self, modalities):
        self.supported_modalities = frozenset(modalities)

    def get_model_name(self):
        return "fake-model"

    def health_check(self):
        return True


class FakeAgent:
    def __init__(self, modalities):
        self.provider = FakeProvider(modalities)

    def run_chat_completion(self, messages, params=None):
        return "chat answer"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setitem(settings.API_KEYS, "test-key", {"name": "tester"})


def use_provider_with(monkeypatch, modalities):
    monkeypatch.setattr(api, "agent", FakeAgent(modalities))


def chat(client, part):
    return client.post(
        "/ask-llm-prompted",
        json={
            "chat": True,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "what is this?"}, part],
                }
            ],
        },
        headers=HEADERS,
    )


IMAGE_PART = {"type": "image", "mime_type": "image/png", "data": PIXEL}
AUDIO_PART = {"type": "audio", "mime_type": "audio/wav", "data": CLIP}


# --- Refusal ---------------------------------------------------------------


@pytest.mark.parametrize("part,name", [(IMAGE_PART, "image"), (AUDIO_PART, "audio")])
def test_text_only_model_refuses_media(client, monkeypatch, part, name):
    use_provider_with(monkeypatch, {"text"})
    response = chat(client, part)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "modality_not_supported"
    assert name in error["message"]
    assert "fake-model" in error["message"]


def test_image_model_still_refuses_audio(client, monkeypatch):
    use_provider_with(monkeypatch, {"text", "image"})
    response = chat(client, AUDIO_PART)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "modality_not_supported"


# --- Acceptance ------------------------------------------------------------


@pytest.mark.parametrize("part", [IMAGE_PART, AUDIO_PART])
def test_supported_media_passes_the_gate(client, monkeypatch, part):
    use_provider_with(monkeypatch, {"text", "image", "audio"})
    response = chat(client, part)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "chat answer"


def test_text_is_always_allowed(client, monkeypatch):
    use_provider_with(monkeypatch, {"text"})
    response = client.post(
        "/ask-llm-prompted",
        json={"chat": True, "messages": [{"role": "user", "content": "hello"}]},
        headers=HEADERS,
    )
    assert response.status_code == 200


def test_health_reports_the_accepted_modalities(client, monkeypatch):
    use_provider_with(monkeypatch, {"text", "image"})
    body = client.get("/health").json()
    assert body["modalities"] == ["image", "text"]


# --- Provider declarations -------------------------------------------------


def test_provider_defaults():
    assert OllamaProvider("llava").supported_modalities == {"text", "image"}
    assert GeminiProvider("gemini-2.0-flash", api_key="k").supported_modalities == {
        "text", "image", "audio",
    }
    # Support on an OpenAI-compatible endpoint depends on the model.
    assert BaseProvider("gpt-4o", api_key="k").supported_modalities == {"text"}


def test_configured_modalities_override_the_default():
    provider = BaseProvider("gpt-4o", api_key="k", supported_modalities=["image"])
    assert provider.supported_modalities == {"text", "image"}


def test_text_cannot_be_configured_away():
    provider = OllamaProvider("llama3", supported_modalities=[])
    assert provider.supported_modalities == {"text"}


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("text,image", ["text", "image"]),
        (" TEXT , Audio ", ["text", "audio"]),
    ],
)
def test_setting_is_parsed_into_a_list(value, expected):
    assert Settings(AI_SUPPORTED_MODALITIES=value).supported_modalities() == expected
