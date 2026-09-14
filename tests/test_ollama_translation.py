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

"""Image parts reach an Ollama server in its own message format."""

import base64
import json

import httpx
import pytest

from app.providers.ollama import OllamaProvider

PIXEL = base64.b64encode(b"\x89PNG fake").decode()
OTHER = base64.b64encode(b"\x89PNG second").decode()
CLIP = base64.b64encode(b"RIFF fake").decode()


@pytest.fixture
def provider():
    """A provider whose HTTP calls are captured instead of sent."""
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "an answer"}})

    provider = OllamaProvider("llava")
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    provider.sent = sent
    return provider


def messages_sent(provider):
    return provider.sent["payload"]["messages"]


# --- Text is untouched -----------------------------------------------------


def test_string_content_is_sent_unchanged(provider):
    assert provider.chat([{"role": "user", "content": "hello"}]) == "an answer"
    assert messages_sent(provider) == [{"role": "user", "content": "hello"}]


def test_a_message_without_images_gets_no_images_field(provider):
    provider.chat([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
    assert messages_sent(provider) == [{"role": "user", "content": "hello"}]


# --- Image translation -----------------------------------------------------


def test_image_travels_beside_the_text(provider):
    provider.chat([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "mime_type": "image/png", "data": PIXEL},
            ],
        }
    ])

    assert messages_sent(provider) == [
        {"role": "user", "content": "what is this?", "images": [PIXEL]}
    ]


def test_several_images_are_collected(provider):
    provider.chat([
        {
            "role": "user",
            "content": [
                {"type": "image", "mime_type": "image/png", "data": PIXEL},
                {"type": "image", "mime_type": "image/jpeg", "data": OTHER},
            ],
        }
    ])
    assert messages_sent(provider)[0]["images"] == [PIXEL, OTHER]


def test_text_parts_are_joined(provider):
    provider.chat([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is"},
                {"type": "image", "mime_type": "image/png", "data": PIXEL},
                {"type": "text", "text": "this?"},
            ],
        }
    ])
    assert messages_sent(provider)[0]["content"] == "what is this?"


def test_the_role_survives(provider):
    provider.chat([
        {"role": "system", "content": "be kind"},
        {
            "role": "user",
            "content": [{"type": "image", "mime_type": "image/png", "data": PIXEL}],
        },
    ])
    sent = messages_sent(provider)
    assert sent[0] == {"role": "system", "content": "be kind"}
    assert sent[1]["role"] == "user"


# --- Refusals --------------------------------------------------------------


def test_audio_is_refused():
    """The gate blocks audio first; this guards a misconfigured provider."""
    assert "audio" not in OllamaProvider("llava").supported_modalities

    provider = OllamaProvider("llava")
    with pytest.raises(ValueError, match="does not accept audio"):
        provider.chat([
            {
                "role": "user",
                "content": [{"type": "audio", "mime_type": "audio/wav", "data": CLIP}],
            }
        ])


def test_unknown_part_is_refused(provider):
    with pytest.raises(ValueError, match="Unsupported content part"):
        provider.chat([{"role": "user", "content": [{"type": "video", "data": PIXEL}]}])
