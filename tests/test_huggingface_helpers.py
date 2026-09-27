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

"""Tests for Hugging Face provider helper methods."""

from app.providers.huggingface import HuggingFaceProvider


def test_huggingface_removes_echoed_prompt_and_eos_token():
    provider = object.__new__(HuggingFaceProvider)

    answer = provider._extract_after_prompt(
        full_text="Explain a loop. A loop repeats instructions.</s>",
        prompt="Explain a loop.",
        eos_token="</s>",
    )

    assert answer == "A loop repeats instructions."


def test_huggingface_keeps_response_when_prompt_is_not_echoed():
    provider = object.__new__(HuggingFaceProvider)

    answer = provider._extract_after_prompt(
        full_text="A loop repeats instructions.",
        prompt="Explain a loop.",
    )

    assert answer == "A loop repeats instructions."


def test_huggingface_combines_system_message_with_first_user_message():
    provider = object.__new__(HuggingFaceProvider)
    provider.model_name = "test-model"

    messages = provider._normalize_chat_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Explain loops."},
        ]
    )

    assert messages == [
        {
            "role": "user",
            "content": "Be concise.\n\nExplain loops.",
        }
    ]


def test_huggingface_maps_assistant_role_to_model_for_gemma():
    provider = object.__new__(HuggingFaceProvider)
    provider.model_name = "google/gemma-3-1b"

    messages = provider._normalize_chat_messages(
        [
            {"role": "user", "content": "What is a loop?"},
            {"role": "assistant", "content": "It repeats instructions."},
        ]
    )

    assert messages == [
        {"role": "user", "content": "What is a loop?"},
        {"role": "model", "content": "It repeats instructions."},
    ]


def test_huggingface_prepends_system_message_when_assistant_starts():
    provider = object.__new__(HuggingFaceProvider)
    provider.model_name = "test-model"

    messages = provider._normalize_chat_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "assistant", "content": "Here is an answer."},
        ]
    )

    assert messages == [
        {"role": "user", "content": "Be concise."},
        {"role": "assistant", "content": "Here is an answer."},
    ]
