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

"""Content parts reach Gemini's generateContent API in its own format."""

import base64
import json

import httpx
import pytest

from app.providers.gemini import GeminiProvider

PIXEL = base64.b64encode(b"\x89PNG fake").decode()
CLIP = base64.b64encode(b"RIFF fake").decode()


@pytest.fixture
def provider():
    """A provider whose HTTP calls are captured instead of sent."""
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "an answer"}]}}]},
        )

    provider = GeminiProvider("gemini-2.0-flash", api_key="k")
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    provider.sent = sent
    return provider


def payload(provider):
    return provider.sent["payload"]


# --- Text is untouched -----------------------------------------------------


def test_string_content_still_becomes_a_text_part(provider):
    assert provider.chat([{"role": "user", "content": "hello"}]) == "an answer"
    assert payload(provider)["contents"] == [
        {"role": "user", "parts": [{"text": "hello"}]}
    ]


def test_system_message_still_becomes_a_system_instruction(provider):
    provider.chat([
        {"role": "system", "content": "be kind"},
        {"role": "user", "content": "hello"},
    ])
    assert payload(provider)["systemInstruction"] == {"parts": [{"text": "be kind"}]}


def test_assistant_maps_to_the_model_role(provider):
    provider.chat([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert [c["role"] for c in payload(provider)["contents"]] == ["user", "model"]


# --- Media translation -----------------------------------------------------


def test_image_becomes_inline_data(provider):
    provider.chat([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "mime_type": "image/png", "data": PIXEL},
            ],
        }
    ])

    assert payload(provider)["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "what is this?"},
                {"inline_data": {"mime_type": "image/png", "data": PIXEL}},
            ],
        }
    ]


def test_audio_uses_the_same_inline_data_shape(provider):
    provider.chat([
        {
            "role": "user",
            "content": [{"type": "audio", "mime_type": "audio/mpeg", "data": CLIP}],
        }
    ])

    assert payload(provider)["contents"][0]["parts"] == [
        {"inline_data": {"mime_type": "audio/mpeg", "data": CLIP}}
    ]


def test_media_in_a_system_message_is_dropped_but_its_text_survives(provider):
    """A system instruction takes text only; media belongs in a turn."""
    provider.chat([
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "be kind"},
                {"type": "image", "mime_type": "image/png", "data": PIXEL},
            ],
        },
        {"role": "user", "content": "hi"},
    ])
    assert payload(provider)["systemInstruction"] == {"parts": [{"text": "be kind"}]}


def test_unknown_part_is_refused(provider):
    with pytest.raises(ValueError, match="Unsupported content part"):
        provider.chat([{"role": "user", "content": [{"type": "video", "data": PIXEL}]}])
