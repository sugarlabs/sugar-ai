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

"""Request/response models for the reflection endpoint.

Field bounds are the server's own ceiling, not the client's: the Sugar
client already clips what it sends, but one quota unit must never buy
an arbitrarily large model call from a client that doesn't.

The conversation travels as the engine's own trace records, passed
through opaquely: this server decodes them with the engine's
from_record and never interprets or re-shapes them itself, so the
wire format has exactly one definition (the engine spec, section 1).
"""
import json
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# A record is one engine or child turn of the engine's trace encoding;
# 64 covers the engine's longest session with room, without letting a
# hostile client ship a book.
_MAX_RECORDS_WIRE = 64
# One record: a turn's text plus its typed envelope. The engine caps
# its own turns far below this; the ceiling exists for what a client
# other than ours might send.
_MAX_RECORD_CHARS = 6000
# The work context carries base64 images (the Journal's preview is a
# 720x540 PNG, moment snaps 960x600 JPEGs); 6 MB of JSON covers a
# full set with margin, and one quota unit buys nothing bigger.
_MAX_CONTEXT_CHARS = 6 * 1024 * 1024


class ReflectChatRequest(BaseModel):
    """Payload for /reflect/chat.

    Deliberately narrow: no full Journal metadata dict, no reflections
    blob, no preview image. The category the engine needs is derived
    server-side from activity_id.
    """
    title: str = Field(max_length=512, description="Title of the Journal entry")
    description: str = Field(
        default="",
        max_length=8192,
        description="The child's own description of their project, in their words",
    )
    activity_id: str = Field(
        max_length=128,
        description="Sugar activity bundle ID, e.g. 'org.laptop.TurtleArtActivity'",
        examples=["org.laptop.TurtleArtActivity"],
    )
    records: List[dict] = Field(
        default_factory=list,
        max_length=_MAX_RECORDS_WIRE,
        description="Conversation so far, as engine trace records "
                    "(engine spec, section 1). Empty starts a new session.",
    )
    previous_next_steps: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="What the child said they wanted to try next, from a "
                    "previous reflection on this activity type",
    )
    work_context: Optional[dict] = Field(
        default=None,
        description="Richer work context in the engine's own session_start "
                    "encoding: preview, moments, spent_seconds. Passed to "
                    "the engine's decoder untouched; this server never "
                    "interprets it.",
    )

    @field_validator("records")
    @classmethod
    def _each_record_bounded(cls, records: List[dict]) -> List[dict]:
        for record in records:
            if len(json.dumps(record)) > _MAX_RECORD_CHARS:
                raise ValueError("a history record exceeds the size ceiling")
        return records

    @field_validator("work_context")
    @classmethod
    def _context_bounded(cls, context: Optional[dict]) -> Optional[dict]:
        # One ceiling for the whole blob (base64 images included);
        # everything finer-grained is the engine decoder's job.
        if context is not None and len(json.dumps(context)) > _MAX_CONTEXT_CHARS:
            raise ValueError("work context exceeds the size ceiling")
        return context


class ReflectChatResponse(BaseModel):
    record: dict = Field(
        description="One engine trace record: an engine_turn for the "
                    "client to render, or a session_end.",
    )
    user: str
    quota: dict
