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

"""Tests for Context-Window Overflow & Conversation Budgeting (Issue #120)."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.ai import (
    estimate_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
    budget_conversation_history,
    RAGAgent,
)
from app.providers.base import BaseProvider, GenerationParams
from app.config import settings
from app.routes import api
from app import create_app


# ---------------------------------------------------------------------------
# Unit Tests: Token Estimation
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    """Unit tests for token count estimation functions."""

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_estimate_tokens_short_text(self):
        text = "Hello world"  # 11 chars -> ceil(11/4) = 3 tokens
        tokens = estimate_tokens(text)
        assert tokens == 3

    def test_estimate_tokens_long_text(self):
        text = "a" * 400  # 400 chars -> ceil(400/4) = 100 tokens
        tokens = estimate_tokens(text)
        assert tokens == 100

    def test_estimate_message_tokens(self):
        assert estimate_message_tokens({}) == 0
        assert estimate_message_tokens(None) == 0

        msg = {"role": "user", "content": "Hello!"}
        # role: "user" (4 chars -> 1 token)
        # content: "Hello!" (6 chars -> 2 tokens)
        # overhead: 4 tokens
        # total: 1 + 2 + 4 = 7 tokens
        tokens = estimate_message_tokens(msg)
        assert tokens == 7

    def test_estimate_messages_tokens(self):
        assert estimate_messages_tokens([]) == 0

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        tokens = estimate_messages_tokens(messages)
        assert tokens > 0
        # Should equal sum of message tokens + 2 priming tokens
        expected = sum(estimate_message_tokens(m) for m in messages) + 2
        assert tokens == expected


# ---------------------------------------------------------------------------
# Unit Tests: Sliding-Window Conversation Budgeting
# ---------------------------------------------------------------------------

class TestConversationBudgeting:
    """Unit tests for budget_conversation_history."""

    def test_empty_messages(self):
        assert budget_conversation_history([]) == []

    def test_small_conversation_passes_untouched(self):
        messages = [
            {"role": "system", "content": "You are a friendly Sugar-AI tutor."},
            {"role": "user", "content": "How do I print in Python?"},
            {"role": "assistant", "content": "Use the print() function!"},
            {"role": "user", "content": "Can you give an example?"},
        ]
        budgeted = budget_conversation_history(messages, max_tokens=1024, reserve_for_response=100)
        assert len(budgeted) == len(messages)
        assert budgeted == messages

    def test_long_conversation_trims_oldest_turns_and_keeps_system(self):
        system_msg = {"role": "system", "content": "System instruction: Be concise."}
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"Question number {i}: " + "details " * 10})
            history.append({"role": "assistant", "content": f"Answer number {i}: " + "explanation " * 10})

        messages = [system_msg] + history

        # Apply a tight budget
        budgeted = budget_conversation_history(messages, max_tokens=300, reserve_for_response=50)

        # 1. System message must be preserved at index 0
        assert budgeted[0]["role"] == "system"
        assert budgeted[0]["content"] == system_msg["content"]

        # 2. Result length must be less than original
        assert len(budgeted) < len(messages)

        # 3. Latest messages should be preserved
        assert budgeted[-1]["role"] == "assistant"
        assert "Answer number 19" in budgeted[-1]["content"]

        # 4. Oldest question 0 should have been dropped
        all_contents = " ".join(m["content"] for m in budgeted)
        assert "Question number 0:" not in all_contents

        # 5. Chronological order must be maintained
        turn_indices = []
        for m in budgeted:
            if "number" in m["content"]:
                num = int(m["content"].split("number ")[1].split(":")[0])
                turn_indices.append(num)
        assert turn_indices == sorted(turn_indices)

    def test_conversation_without_system_prompt(self):
        messages = []
        for i in range(15):
            messages.append({"role": "user", "content": f"User query {i}: " + "data " * 12})
            messages.append({"role": "assistant", "content": f"Assistant reply {i}: " + "text " * 12})

        budgeted = budget_conversation_history(messages, max_tokens=250, reserve_for_response=50)
        assert len(budgeted) < len(messages)
        # Latest query and reply should be retained
        assert "User query 14" in budgeted[-2]["content"] or "Assistant reply 14" in budgeted[-1]["content"]

    def test_oversized_single_message_is_truncated(self):
        system_msg = {"role": "system", "content": "You are a tutor."}
        huge_user_msg = {"role": "user", "content": "A" * 5000}
        messages = [system_msg, huge_user_msg]

        budgeted = budget_conversation_history(messages, max_tokens=300, reserve_for_response=50)
        assert len(budgeted) == 2
        assert budgeted[0]["role"] == "system"
        assert budgeted[1]["role"] == "user"
        assert len(budgeted[1]["content"]) < 5000
        assert len(budgeted[1]["content"]) > 0

    def test_oversized_system_prompt_is_truncated_safely(self):
        huge_sys_msg = {"role": "system", "content": "SYS " * 1000}
        user_msg = {"role": "user", "content": "Short question"}
        messages = [huge_sys_msg, user_msg]

        budgeted = budget_conversation_history(messages, max_tokens=200, reserve_for_response=50)
        assert len(budgeted) >= 1
        assert budgeted[0]["role"] == "system"
        assert len(budgeted[0]["content"]) < len(huge_sys_msg["content"])

    def test_reserve_for_response_reduces_effective_budget(self):
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Message A: " + "a" * 80},
            {"role": "assistant", "content": "Message B: " + "b" * 80},
            {"role": "user", "content": "Message C: " + "c" * 80},
        ]
        # With 0 reserved tokens, all messages fit
        budgeted_large = budget_conversation_history(messages, max_tokens=300, reserve_for_response=0)
        # With high reserved tokens, oldest conversation turn is trimmed
        budgeted_small = budget_conversation_history(messages, max_tokens=300, reserve_for_response=150)
        assert len(budgeted_small) <= len(budgeted_large)


# ---------------------------------------------------------------------------
# Integration Tests: RAGAgent Integration
# ---------------------------------------------------------------------------

class TestRAGAgentBudgeting:
    """Integration tests for RAGAgent methods with context budgeting."""

    def test_run_chat_completion_budgets_messages_before_provider(self):
        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.get_model_name.return_value = "mock-model"
        mock_provider.get_eos_token.return_value = None
        mock_provider.chat.return_value = "Mocked chat response"

        agent = RAGAgent(mock_provider)

        # Build long conversation
        messages = [{"role": "system", "content": "Always remember this system instruction."}]
        for i in range(25):
            messages.append({"role": "user", "content": f"Long turn {i}: " + "word " * 15})
            messages.append({"role": "assistant", "content": f"Long response {i}: " + "info " * 15})

        params = GenerationParams(max_new_tokens=100)
        result = agent.run_chat_completion(messages, params=params, max_context_tokens=350)

        assert result == "Mocked chat response"
        mock_provider.chat.assert_called_once()
        passed_messages = mock_provider.chat.call_args[0][0]

        # Verify passed messages were pruned
        assert len(passed_messages) < len(messages)
        # Verify system prompt was retained
        assert passed_messages[0]["role"] == "system"
        assert passed_messages[0]["content"] == "Always remember this system instruction."
        # Verify latest response turn was retained
        assert any("Long turn 24" in m["content"] or "Long response 24" in m["content"] for m in passed_messages)

    def test_run_with_custom_prompt_budgets_oversized_prompt(self):
        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.get_model_name.return_value = "mock-model"
        mock_provider.get_eos_token.return_value = None
        mock_provider.generate.return_value = "Answer: Mocked custom answer"

        agent = RAGAgent(mock_provider)
        huge_prompt = "Custom system instruction: " + "rule " * 1000
        question = "What is 2 + 2?"

        params = GenerationParams(max_new_tokens=100)
        result = agent.run_with_custom_prompt(
            question=question,
            custom_prompt=huge_prompt,
            params=params,
            max_context_tokens=300
        )

        assert result == "Mocked custom answer"
        mock_provider.generate.assert_called_once()
        sent_prompt = mock_provider.generate.call_args[0][0]
        assert "What is 2 + 2?" in sent_prompt
        assert len(sent_prompt) < len(huge_prompt)


# ---------------------------------------------------------------------------
# API Endpoint Tests: /ask-llm-prompted
# ---------------------------------------------------------------------------

class TestAPIBudgetingEndpoint:
    """Tests for API routes handling context budgeting."""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        self.mock_provider = MagicMock(spec=BaseProvider)
        self.mock_provider.get_model_name.return_value = "mock-model"
        self.mock_provider.get_eos_token.return_value = None
        self.mock_provider.chat.return_value = "Assistant response from mock"
        self.mock_provider.generate.return_value = "Generated answer"

        self.agent = RAGAgent(self.mock_provider)
        api.agent = self.agent

        self.valid_key = "test_key_budget"
        settings.API_KEYS = {
            self.valid_key: {"name": "Test User", "can_change_model": False}
        }
        api.user_quotas.clear()

        self.app = create_app()
        self.client = TestClient(self.app)

    def test_ask_llm_prompted_chat_with_budgeting(self):
        # Construct long chat history
        messages = [{"role": "system", "content": "You are Sugar AI."}]
        for i in range(20):
            messages.append({"role": "user", "content": f"Message {i}: " + "query " * 10})
            messages.append({"role": "assistant", "content": f"Reply {i}: " + "text " * 10})

        payload = {
            "chat": True,
            "messages": messages,
            "max_context_tokens": 400,
            "max_length": 100,
        }

        response = self.client.post(
            "/ask-llm-prompted",
            headers={"X-API-Key": self.valid_key},
            json=payload,
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"] == "Assistant response from mock"
        assert data["generation_params"]["max_context_tokens"] == 400

        # Provider must have received a budgeted payload
        self.mock_provider.chat.assert_called_once()
        sent_messages = self.mock_provider.chat.call_args[0][0]
        assert len(sent_messages) < len(messages)
        assert sent_messages[0]["role"] == "system"
