# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Tests for RAG pipeline chat message structuring and prompt templates."""

from unittest.mock import MagicMock
import pytest

import app.prompts as prompts
from app.ai import RAGAgent
from app.providers.base import BaseProvider, GenerationParams


def test_prompt_template_contains_only_system_instructions_and_context():
    """PROMPT_TEMPLATE should only contain system instructions and {context}."""
    assert "{context}" in prompts.PROMPT_TEMPLATE
    assert "{question}" not in prompts.PROMPT_TEMPLATE
    assert "Answer:" not in prompts.PROMPT_TEMPLATE
    assert "Sugar Learning Platform" in prompts.PROMPT_TEMPLATE


def test_child_friendly_prompt_contains_only_system_instructions():
    """CHILD_FRIENDLY_PROMPT should only contain system instructions."""
    assert "{original_answer}" not in prompts.CHILD_FRIENDLY_PROMPT
    assert "Child-friendly answer:" not in prompts.CHILD_FRIENDLY_PROMPT
    assert "paraphrase" in prompts.CHILD_FRIENDLY_PROMPT


def test_rag_agent_run_builds_structured_messages_with_retrieved_context():
    """RAGAgent.run() should pass structured messages to provider.chat() in both stages."""
    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.get_model_name.return_value = "mock-model"
    mock_provider.chat.side_effect = [
        "A sprite in Pygame is a 2D image object inheriting from pygame.sprite.Sprite.",
        "Think of a sprite like a digital sticker or game character that you can move around!",
    ]

    agent = RAGAgent(mock_provider)

    mock_doc = MagicMock()
    mock_doc.page_content = "pygame.sprite.Sprite is the base class for visible game objects."
    agent.get_relevant_document = MagicMock(return_value=(mock_doc, 0.85))

    params = GenerationParams(think=True, max_new_tokens=512)
    result = agent.run("How do I make a sprite?", params=params)

    assert result == "Think of a sprite like a digital sticker or game character that you can move around!"

    # Verify provider.generate was never called
    mock_provider.generate.assert_not_called()

    # Verify provider.chat was called exactly twice
    assert mock_provider.chat.call_count == 2

    # Verify Stage 1 call
    first_call_args, first_call_kwargs = mock_provider.chat.call_args_list[0]
    first_messages = first_call_args[0]
    first_params = first_call_args[1] if len(first_call_args) > 1 else first_call_kwargs.get("params")

    assert first_params is params
    assert len(first_messages) == 2
    assert first_messages[0]["role"] == "system"
    assert "pygame.sprite.Sprite is the base class" in first_messages[0]["content"]
    assert "How do I make a sprite?" not in first_messages[0]["content"]
    assert first_messages[1] == {"role": "user", "content": "How do I make a sprite?"}

    # Verify Stage 2 call
    second_call_args, second_call_kwargs = mock_provider.chat.call_args_list[1]
    second_messages = second_call_args[0]
    second_params = second_call_args[1] if len(second_call_args) > 1 else second_call_kwargs.get("params")

    # Stage 2 runs with default no-think params (None passed)
    assert second_params is None
    assert len(second_messages) == 2
    assert second_messages[0]["role"] == "system"
    assert "paraphrase" in second_messages[0]["content"]
    assert second_messages[1] == {
        "role": "user",
        "content": "A sprite in Pygame is a 2D image object inheriting from pygame.sprite.Sprite.",
    }


def test_rag_agent_run_builds_structured_messages_without_retrieved_context():
    """RAGAgent.run() should handle missing context gracefully using fallback string."""
    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.get_model_name.return_value = "mock-model"
    mock_provider.chat.side_effect = [
        "Raw technical answer without doc context.",
        "Simple answer for kids.",
    ]

    agent = RAGAgent(mock_provider)
    agent.get_relevant_document = MagicMock(return_value=(None, 0.0))

    result = agent.run("What is Python?")

    assert result == "Simple answer for kids."
    mock_provider.generate.assert_not_called()
    assert mock_provider.chat.call_count == 2

    first_messages = mock_provider.chat.call_args_list[0][0][0]
    assert first_messages[0]["role"] == "system"
    assert "No relevant documentation found." in first_messages[0]["content"]
    assert first_messages[1] == {"role": "user", "content": "What is Python?"}
