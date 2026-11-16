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

"""Tests for RAGAgent."""

from unittest.mock import Mock

import pytest

from app.ai import RAGAgent


def _make_agent():
    return RAGAgent(Mock())


def test_get_relevant_document_returns_none_when_no_retriever():
    agent = _make_agent()
    agent.retriever = None

    doc, score = agent.get_relevant_document("anything")

    assert doc is None
    assert score == 0.0


def test_get_relevant_document_returns_match_above_threshold():
    agent = _make_agent()
    fake_doc = Mock()
    agent.retriever = Mock()
    agent.retriever.search_kwargs = {}
    agent.retriever.vectorstore.similarity_search_with_relevance_scores.return_value = [
        (fake_doc, 0.8),
    ]

    doc, score = agent.get_relevant_document("how do I make a window?", threshold=0.5)

    assert doc is fake_doc
    assert score == 0.8


def test_get_relevant_document_rejects_match_below_threshold():
    agent = _make_agent()
    fake_doc = Mock()
    agent.retriever = Mock()
    agent.retriever.search_kwargs = {}
    agent.retriever.vectorstore.similarity_search_with_relevance_scores.return_value = [
        (fake_doc, 0.2),
    ]

    doc, score = agent.get_relevant_document("unrelated question", threshold=0.5)

    assert doc is None
    assert score == 0.0


def test_truncate_at_eos_cuts_at_provider_eos_token():
    agent = _make_agent()
    agent.provider = Mock()
    agent.provider.get_eos_token.return_value = "<|end|>"

    result = agent._truncate_at_eos("The answer is 42.<|end|>garbage after")

    assert result == "The answer is 42."


def test_truncate_at_eos_cuts_at_fallback_eos_token_when_provider_has_none():
    agent = _make_agent()
    agent.provider = Mock()
    agent.provider.get_eos_token.return_value = None

    result = agent._truncate_at_eos("The answer is 42.</s>garbage after")

    assert result == "The answer is 42."


def test_truncate_at_eos_returns_full_text_when_no_eos_present():
    agent = _make_agent()
    agent.provider = Mock()
    agent.provider.get_eos_token.return_value = None

    result = agent._truncate_at_eos("  The answer is 42.  ")

    assert result == "The answer is 42."


def test_run_uses_retrieved_context_when_relevant_doc_found():
    agent = _make_agent()
    fake_doc = Mock()
    fake_doc.page_content = "Pygame windows are created with pygame.display.set_mode()."
    agent.get_relevant_document = Mock(return_value=(fake_doc, 0.8))

    agent.provider = Mock()
    agent.provider.generate.side_effect = [
        "Answer: Use pygame.display.set_mode().",
        "Child-friendly answer: You make a window with one line of code!",
    ]

    result = agent.run("How do I create a Pygame window?")

    assert result == "You make a window with one line of code!"
    first_call_prompt = agent.provider.generate.call_args_list[0].args[0]
    assert "Pygame windows are created with pygame.display.set_mode()." in first_call_prompt


def test_run_falls_back_when_no_relevant_doc_found():
    agent = _make_agent()
    agent.get_relevant_document = Mock(return_value=(None, 0.0))

    agent.provider = Mock()
    agent.provider.generate.side_effect = [
        "Answer: I don't have specific documentation on that.",
        "Child-friendly answer: I'm not sure about that one!",
    ]

    result = agent.run("What is the capital of France?")

    assert result == "I'm not sure about that one!"
    first_call_prompt = agent.provider.generate.call_args_list[0].args[0]
    assert "No relevant documentation found." in first_call_prompt


def test_run_with_custom_prompt_strips_answer_prefix_and_truncates_at_eos():
    agent = _make_agent()
    agent.provider = Mock()
    agent.provider.generate.return_value = "Answer: 42 is the answer.<|end|>ignored"
    agent.provider.get_eos_token.return_value = "<|end|>"

    result = agent.run_with_custom_prompt("What is the answer?", "You are helpful.")

    assert result == "42 is the answer."


def test_run_with_custom_prompt_wraps_provider_errors():
    agent = _make_agent()
    agent.provider = Mock()
    agent.provider.generate.side_effect = RuntimeError("model crashed")

    with pytest.raises(Exception, match="model crashed"):
        agent.run_with_custom_prompt("question", "prompt")


def test_run_chat_completion_delegates_to_provider_chat():
    agent = _make_agent()
    agent.provider = Mock()
    agent.provider.chat.return_value = "A loop repeats instructions."

    messages = [{"role": "user", "content": "What is a loop?"}]
    result = agent.run_chat_completion(messages)

    assert result == "A loop repeats instructions."
    agent.provider.chat.assert_called_once()
    called_messages = agent.provider.chat.call_args.args[0]
    assert called_messages == messages


def test_debug_without_context_runs_two_stage_generation():
    agent = _make_agent()
    agent.provider = Mock()
    agent.provider.generate.side_effect = [
        "raw debug output",
        "kid-friendly debug output",
    ]

    result = agent.debug("print(hello)", context=False)

    assert result == "kid-friendly debug output"
    assert agent.provider.generate.call_count == 2


def test_debug_with_context_runs_two_stage_generation():
    agent = _make_agent()
    agent.provider = Mock()
    agent.provider.generate.side_effect = [
        "raw context output",
        "kid-friendly context output",
    ]

    result = agent.debug("print(hello)", context=True)

    assert result == "kid-friendly context output"
    assert agent.provider.generate.call_count == 2


def test_set_model_closes_old_provider():
    agent = _make_agent()
    old_provider = Mock()
    new_provider = Mock()
    new_provider.get_model_name.return_value = "new-model"
    agent.provider = old_provider
    agent.model_name = "old-model"

    agent.set_model(new_provider)

    assert agent.provider is new_provider
    assert agent.model_name == "new-model"
    old_provider.close.assert_called_once()


def test_set_model_does_not_crash_if_old_provider_close_fails():
    agent = _make_agent()
    old_provider = Mock()
    old_provider.close.side_effect = RuntimeError("cleanup failed")
    new_provider = Mock()
    new_provider.get_model_name.return_value = "new-model"
    agent.provider = old_provider
    agent.model_name = "old-model"

    agent.set_model(new_provider)

    assert agent.provider is new_provider