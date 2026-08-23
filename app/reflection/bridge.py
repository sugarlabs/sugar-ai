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

"""Adapts a sugar-ai BaseProvider to the engine's Provider seam.

reflection_engine's next_turn takes any object shaped
complete(*, system, user, schema) -> dict, with the model's reply
already decoded against the given JSON schema. Sugar-AI's providers
are shaped chat(messages, params) -> str, know nothing about JSON
schemas, and model selection, quotas, and health checks all already
live on that interface. This class is the whole translation: it asks
for the schema in the prompt, decodes the text reply, and raises when
the reply is not the JSON it asked for -- the engine turns a raise
into a floored turn, which is better than guessing.

The host owns provider policy here: generation is capped well above
the engine's own output bound, so a long reasoning preamble truncates
into a guard event instead of silently eating the answer, and an
empty reply raises rather than reaching a child as a blank companion.
"""
import base64
import json

from app.providers.base import BaseProvider, GenerationParams

_PARAMS = GenerationParams(max_new_tokens=2048)


class ProviderBridge:
    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    def complete(
        self, *, system: str, user: str, schema: dict, images: tuple = ()
    ) -> dict:
        # images attach only when the configured provider declares it
        # can carry them; otherwise they drop and the text flows on -
        # the engine's contract for a provider without vision. Each
        # image is (label, mime, bytes) and rides as an OpenAI-style
        # content part next to its label.
        content = user
        if images and getattr(self._provider, "supports_images", False):
            parts = [{"type": "text", "text": user}]
            for label, mime, data in images:
                encoded = base64.b64encode(data).decode("ascii")
                parts.append({"type": "text", "text": f"({label}:)"})
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                })
            content = parts
        messages = [
            {
                "role": "system",
                "content": (
                    f"{system}\n\nReply with a single JSON object "
                    f"matching this schema, and nothing else:\n"
                    f"{json.dumps(schema)}"
                ),
            },
            {"role": "user", "content": content},
        ]
        text = (self._provider.chat(messages, _PARAMS) or "").strip()
        if not text:
            raise RuntimeError("provider returned an empty reply")
        return _decode(text)


def _decode(text: str) -> dict:
    """The model's text as one JSON object, or ValueError. A fenced
    code block is unwrapped first: models without a JSON mode wrap
    their answer often enough that refusing the fence would floor
    turns for punctuation.
    """
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError("model reply is not a JSON object")
    return decoded
