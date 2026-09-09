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

"""An image travels the whole stack: HTTP request to provider payload.

Every layer here is the real one - route, schema, RAGAgent and provider.
Only the network is replaced, so what these tests assert is the request
that would have gone to the model.
"""

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.ai import RAGAgent
from app.config import settings
from app.providers.gemini import GeminiProvider
from app.routes import api

PIXEL = base64.b64encode(b"\x89PNG fake").decode()
CLIP = base64.b64encode(b"RIFF fake").decode()
HEADERS = {"X-API-Key": "test-key"}
IMAGE_PART = {"type": "image", "mime_type": "image/png", "data": PIXEL}
AUDIO_PART = {"type": "audio", "mime_type": "audio/wav", "data": CLIP}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture
def sent(monkeypatch):
    """Wire a real agent and provider into the API, capturing the payload."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "a red square"}]}}]},
        )

    provider = GeminiProvider("gemini-2.5-flash", api_key="k")
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(api, "agent", RAGAgent(provider=provider))
    monkeypatch.setitem(settings.API_KEYS, "test-key", {"name": "tester"})
    return captured


def test_an_attachment_reaches_the_model_as_inline_data(client, sent):
    response = client.post(
        "/ask-llm",
        json={"question": "What colour is this?", "attachments": [IMAGE_PART]},
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "a red square"
    assert sent["payload"]["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "What colour is this?"},
                {"inline_data": {"mime_type": "image/png", "data": PIXEL}},
            ],
        }
    ]


def test_audio_reaches_the_model(client, sent):
    response = client.post(
        "/ask-llm",
        json={"question": "What is this sound?", "attachments": [AUDIO_PART]},
        headers=HEADERS,
    )

    assert response.status_code == 200
    parts = sent["payload"]["contents"][0]["parts"]
    assert parts[1] == {"inline_data": {"mime_type": "audio/wav", "data": CLIP}}


def test_chat_mode_carries_parts_through(client, sent):
    response = client.post(
        "/ask-llm-prompted",
        json={
            "chat": True,
            "messages": [
                {"role": "system", "content": "Answer simply."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is this?"}, IMAGE_PART],
                },
            ],
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = sent["payload"]
    assert payload["systemInstruction"] == {"parts": [{"text": "Answer simply."}]}
    assert payload["contents"][0]["parts"][1]["inline_data"]["data"] == PIXEL


def test_prompted_mode_sends_the_prompt_as_a_system_instruction(client, sent):
    response = client.post(
        "/ask-llm-prompted",
        json={
            "chat": False,
            "question": "What is this?",
            "custom_prompt": "Answer simply.",
            "attachments": [IMAGE_PART],
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = sent["payload"]
    assert payload["systemInstruction"] == {"parts": [{"text": "Answer simply."}]}
    assert payload["contents"][0]["parts"][1]["inline_data"]["data"] == PIXEL


def test_generation_parameters_reach_the_model(client, sent):
    client.post(
        "/ask-llm-prompted",
        json={
            "chat": True,
            "messages": [{"role": "user", "content": [IMAGE_PART]}],
            "temperature": 0.2,
            "max_length": 64,
        },
        headers=HEADERS,
    )

    config = sent["payload"]["generationConfig"]
    assert config["temperature"] == 0.2
    assert config["maxOutputTokens"] == 64


def test_a_text_only_request_is_unchanged(client, sent):
    """The plain path must not have acquired a parts wrapper."""
    response = client.post(
        "/ask-llm-prompted",
        json={"chat": True, "messages": [{"role": "user", "content": "hello"}]},
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert sent["payload"]["contents"] == [
        {"role": "user", "parts": [{"text": "hello"}]}
    ]
