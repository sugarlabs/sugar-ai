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

"""Content parts reach an OpenAI-compatible endpoint in its own format."""

import base64

import httpx
import pytest

from app.providers.base import BaseProvider

PIXEL = base64.b64encode(b"\x89PNG fake").decode()
CLIP = base64.b64encode(b"RIFF fake").decode()


@pytest.fixture
def provider():
    """A provider whose HTTP calls are captured instead of sent."""
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["payload"] = __import__("json").loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "an answer"}}]}
        )

    provider = BaseProvider("gpt-4o", api_key="k")
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    provider.sent = sent
    return provider


def messages_sent(provider):
    return provider.sent["payload"]["messages"]


# --- Text is untouched -----------------------------------------------------


def test_string_content_is_sent_unchanged(provider):
    assert provider.chat([{"role": "user", "content": "hello"}]) == "an answer"
    assert messages_sent(provider) == [{"role": "user", "content": "hello"}]


def test_generate_still_wraps_a_plain_prompt(provider):
    provider.generate("hello")
    assert messages_sent(provider) == [{"role": "user", "content": "hello"}]


# --- Media translation -----------------------------------------------------


def test_image_becomes_a_data_url(provider):
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
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{PIXEL}"},
                },
            ],
        }
    ]


def test_audio_becomes_an_input_audio_block(provider):
    provider.chat([
        {
            "role": "user",
            "content": [{"type": "audio", "mime_type": "audio/wav", "data": CLIP}],
        }
    ])

    assert messages_sent(provider)[0]["content"] == [
        {"type": "input_audio", "input_audio": {"data": CLIP, "format": "wav"}}
    ]


@pytest.mark.parametrize(
    "mime,expected", [("audio/wav", "wav"), ("audio/mpeg", "mp3"), ("audio/ogg", "ogg")]
)
def test_audio_format_is_named_not_mime_typed(provider, mime, expected):
    provider.chat([
        {"role": "user", "content": [{"type": "audio", "mime_type": mime, "data": CLIP}]}
    ])
    audio = messages_sent(provider)[0]["content"][0]["input_audio"]
    assert audio["format"] == expected


def test_roles_and_other_fields_survive(provider):
    provider.chat([
        {"role": "system", "content": "be kind"},
        {
            "role": "user",
            "content": [{"type": "image", "mime_type": "image/jpeg", "data": PIXEL}],
        },
    ])
    sent = messages_sent(provider)
    assert sent[0] == {"role": "system", "content": "be kind"}
    assert sent[1]["role"] == "user"


def test_unknown_part_is_refused(provider):
    with pytest.raises(ValueError, match="Unsupported content part"):
        provider.chat([{"role": "user", "content": [{"type": "video", "data": PIXEL}]}])
