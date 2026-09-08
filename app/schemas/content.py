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

"""Typed message content for Sugar-AI.

A message carries either a plain string, as it always has, or a list of
parts. Each part declares its own modality, so an image or a recording
travels through the same request as the text that describes it.
"""
import base64
import binascii
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field, field_validator

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 20 * 1024 * 1024

IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/webp")
AUDIO_MIME_TYPES = ("audio/wav", "audio/mpeg", "audio/ogg")


def _decoded_size(data: str) -> int:
    """Return the byte length of base64 data, rejecting anything malformed."""
    try:
        return len(base64.b64decode(data, validate=True))
    except (binascii.Error, ValueError):
        raise ValueError("data must be valid base64")


class TextPart(BaseModel):
    """A run of text inside a message."""
    type: Literal["text"]
    text: str = Field(..., min_length=1, max_length=32_000)


class ImagePart(BaseModel):
    """A base64-encoded image."""
    type: Literal["image"]
    mime_type: Literal[IMAGE_MIME_TYPES]
    data: str = Field(..., description="Base64-encoded image bytes, without a data: prefix")

    @field_validator("data")
    @classmethod
    def check_data(cls, value: str) -> str:
        size = _decoded_size(value)
        if size == 0:
            raise ValueError("data must not be empty")
        if size > MAX_IMAGE_BYTES:
            raise ValueError(
                f"image is {size} bytes; the limit is {MAX_IMAGE_BYTES} bytes"
            )
        return value


class AudioPart(BaseModel):
    """A base64-encoded audio clip.

    Audio is input only: the model answers in text. Spoken replies are
    not part of this contract.
    """
    type: Literal["audio"]
    mime_type: Literal[AUDIO_MIME_TYPES]
    data: str = Field(..., description="Base64-encoded audio bytes, without a data: prefix")

    @field_validator("data")
    @classmethod
    def check_data(cls, value: str) -> str:
        size = _decoded_size(value)
        if size == 0:
            raise ValueError("data must not be empty")
        if size > MAX_AUDIO_BYTES:
            raise ValueError(
                f"audio is {size} bytes; the limit is {MAX_AUDIO_BYTES} bytes"
            )
        return value


ContentPart = Annotated[
    Union[TextPart, ImagePart, AudioPart],
    Field(discriminator="type"),
]

# What a part contributes to a request's modality set. Kept beside the
# part types so a new modality is one entry, not a scattered edit.
MODALITY_OF_PART = {"text": "text", "image": "image", "audio": "audio"}


def modalities_of(content: Union[str, List[ContentPart]]) -> set:
    """Return the modalities one message's content uses."""
    if isinstance(content, str):
        return {"text"}
    return {MODALITY_OF_PART[part.type] for part in content}


def part_to_dict(part) -> dict:
    """Render a part as the plain dict providers translate from."""
    return part.model_dump()


def content_to_provider(content: Union[str, List[ContentPart]]):
    """Normalize content for providers: a string stays a string."""
    if isinstance(content, str):
        return content
    return [part_to_dict(part) for part in content]


def messages_to_provider(messages) -> List[dict]:
    """Render messages as the plain dicts providers translate from."""
    return [
        {"role": message.role, "content": content_to_provider(message.content)}
        for message in messages
    ]
