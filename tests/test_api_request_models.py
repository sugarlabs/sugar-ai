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
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Tests for API request validation models."""

import pytest
from pydantic import ValidationError

from app.routes.api import ChatMessage, PromptedLLMRequest


def test_prompted_request_accepts_maximum_valid_length():
    request = PromptedLLMRequest(max_length=8192)

    assert request.max_length == 8192


def test_prompted_request_rejects_length_above_limit():
    with pytest.raises(ValidationError):
        PromptedLLMRequest(max_length=8193)


def test_prompted_request_has_expected_defaults():
    request = PromptedLLMRequest()

    assert request.chat is False
    assert request.question is None
    assert request.custom_prompt is None
    assert request.messages is None
    assert request.max_length == 1024
    assert request.truncation is True
    assert request.repetition_penalty == 1.1
    assert request.temperature == 0.7
    assert request.top_p == 0.9
    assert request.top_k == 50


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("max_length", 0),
        ("repetition_penalty", 0.0),
        ("repetition_penalty", 2.1),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", 0.0),
        ("top_p", 1.1),
        ("top_k", -1),
    ],
)
def test_prompted_request_rejects_invalid_generation_values(
    field_name,
    invalid_value,
):
    with pytest.raises(ValidationError) as error:
        PromptedLLMRequest(**{field_name: invalid_value})

    assert field_name in str(error.value)


def test_prompted_request_converts_message_data_to_chat_messages():
    request = PromptedLLMRequest(
        chat=True,
        messages=[
            {"role": "user", "content": "What is a loop?"},
        ],
    )

    assert len(request.messages) == 1
    assert isinstance(request.messages[0], ChatMessage)
    assert request.messages[0].role == "user"
    assert request.messages[0].content == "What is a loop?"
