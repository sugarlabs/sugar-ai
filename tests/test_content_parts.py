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

"""Typed content parts: dispatch, validation and provider normalization."""

import base64

import pytest
from pydantic import ValidationError

from app.schemas.content import (
    MAX_AUDIO_BYTES,
    MAX_IMAGE_BYTES,
    AudioPart,
    ImagePart,
    TextPart,
    content_to_provider,
    messages_to_provider,
    modalities_of,
)
from app.schemas.requests import ChatMessage

PIXEL = base64.b64encode(b"\x89PNG\r\n\x1a\n fake pixel").decode()
CLIP = base64.b64encode(b"RIFF fake wav").decode()


def message(content):
    return ChatMessage(role="user", content=content)


# --- Discriminated dispatch ------------------------------------------------


def test_parts_dispatch_on_type():
    msg = message(
        [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "mime_type": "image/png", "data": PIXEL},
            {"type": "audio", "mime_type": "audio/wav", "data": CLIP},
        ]
    )
    assert [type(part) for part in msg.content] == [TextPart, ImagePart, AudioPart]


def test_unknown_part_type_is_rejected():
    with pytest.raises(ValidationError):
        message([{"type": "video", "data": PIXEL}])


# --- Plain strings stay valid ----------------------------------------------


def test_string_content_is_still_accepted():
    msg = message("plain question")
    assert msg.content == "plain question"
    assert msg.modalities() == {"text"}
    assert content_to_provider(msg.content) == "plain question"


# --- Validation ------------------------------------------------------------


@pytest.mark.parametrize("part_type,mime", [("image", "image/png"), ("audio", "audio/wav")])
def test_malformed_base64_is_rejected(part_type, mime):
    with pytest.raises(ValidationError, match="valid base64"):
        message([{"type": part_type, "mime_type": mime, "data": "not!base64!"}])


@pytest.mark.parametrize("part_type,mime", [("image", "image/png"), ("audio", "audio/wav")])
def test_empty_data_is_rejected(part_type, mime):
    with pytest.raises(ValidationError, match="must not be empty"):
        message([{"type": part_type, "mime_type": mime, "data": ""}])


def test_oversized_image_is_rejected():
    too_big = base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode()
    with pytest.raises(ValidationError, match="the limit is"):
        message([{"type": "image", "mime_type": "image/png", "data": too_big}])


def test_oversized_audio_is_rejected():
    too_big = base64.b64encode(b"x" * (MAX_AUDIO_BYTES + 1)).decode()
    with pytest.raises(ValidationError, match="the limit is"):
        message([{"type": "audio", "mime_type": "audio/wav", "data": too_big}])


def test_audio_may_be_larger_than_the_image_limit():
    clip = base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode()
    msg = message([{"type": "audio", "mime_type": "audio/wav", "data": clip}])
    assert msg.modalities() == {"audio"}


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image", "mime_type": "image/gif", "data": PIXEL},
        {"type": "audio", "mime_type": "audio/flac", "data": CLIP},
        {"type": "image", "mime_type": "audio/wav", "data": PIXEL},
    ],
)
def test_unsupported_mime_type_is_rejected(part):
    with pytest.raises(ValidationError):
        message([part])


def test_empty_text_part_is_rejected():
    with pytest.raises(ValidationError):
        message([{"type": "text", "text": ""}])


# --- Modalities and normalization ------------------------------------------


def test_modalities_reflect_every_part():
    msg = message(
        [
            {"type": "text", "text": "listen"},
            {"type": "audio", "mime_type": "audio/mpeg", "data": CLIP},
        ]
    )
    assert msg.modalities() == {"text", "audio"}
    assert modalities_of("just text") == {"text"}


def test_text_helper_ignores_non_text_parts():
    msg = message(
        [
            {"type": "text", "text": "what is"},
            {"type": "image", "mime_type": "image/png", "data": PIXEL},
            {"type": "text", "text": "this?"},
        ]
    )
    assert msg.text() == "what is this?"


def test_messages_are_normalized_to_plain_dicts():
    messages = [
        message("hello"),
        message([{"type": "image", "mime_type": "image/png", "data": PIXEL}]),
    ]
    assert messages_to_provider(messages) == [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [{"type": "image", "mime_type": "image/png", "data": PIXEL}],
        },
    ]
