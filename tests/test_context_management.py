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

from app.context import ContextBudget, estimate_tokens, fit_messages, fit_text
from app.providers.base import BaseProvider, GenerationParams
from app.providers.huggingface import HuggingFaceProvider


class DummyProvider(BaseProvider):
    def __init__(self):
        # Avoid network setup: these tests exercise the provider seam only.
        self.model_name = "dummy"
        self.base_url = "http://example.test"

    def get_context_window(self):
        return 64


def test_estimator_is_stable_and_nonzero_for_nonempty_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 5) == 2


def test_fit_messages_preserves_system_and_latest_turns():
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "old " * 30},
        {"role": "assistant", "content": "old answer " * 20},
        {"role": "user", "content": "latest question"},
    ]
    result = fit_messages(messages, ContextBudget(64, 16))
    assert result[0]["role"] == "system"
    assert result[-1]["content"] == "latest question"
    assert any("compressed" in item["content"] for item in result if item["role"] == "system")
    assert sum(estimate_tokens(item["content"]) + 4 for item in result) <= 48


def test_fit_text_reserves_requested_output_budget():
    result = fit_text("word " * 100, ContextBudget(32, 8))
    assert estimate_tokens(result) <= 24
    assert "truncated" in result


def test_fit_text_uses_custom_counter_without_character_ratio_assumption():
    # Deliberately use a non-4-character tokenizer to catch accidental coupling
    # between character slicing and token accounting.
    counter = lambda text: len(text.split())
    result = fit_text("one two three four five six", ContextBudget(5, 2), counter)
    assert counter(result) <= 3
    assert result.endswith("[…truncated…]")


def test_provider_metadata_and_output_are_bounded(monkeypatch):
    provider = DummyProvider()
    monkeypatch.setenv("AI_CONTEXT_WINDOW", "999")
    # DummyProvider intentionally advertises 64, proving provider-specific limits win.
    metadata = provider.get_model_metadata()
    assert metadata == {
        "model": "dummy",
        "provider": "DummyProvider",
        "context_window": 64,
        "max_output_tokens": 63,
        "safe_input_tokens": 1,
    }
    params = provider.bound_params(GenerationParams(max_new_tokens=1000))
    assert params.max_new_tokens == 63


def test_prepare_messages_never_uses_more_than_input_budget():
    provider = DummyProvider()
    messages = [{"role": "user", "content": "long " * 100}]
    prepared = provider.prepare_messages(messages, GenerationParams(max_new_tokens=16))
    assert sum(estimate_tokens(item["content"]) + 4 for item in prepared) <= 48


def test_huggingface_count_tokens_uses_tokenizer_and_falls_back_safely():
    provider = object.__new__(HuggingFaceProvider)

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    provider._pipeline = type("Pipeline", (), {"tokenizer": Tokenizer()})()
    assert provider.count_tokens("any text") == 3

    class BrokenTokenizer:
        def __call__(self, text, **kwargs):
            raise RuntimeError("tokenizer unavailable")

    provider._pipeline.tokenizer = BrokenTokenizer()
    assert provider.count_tokens("abcd") == estimate_tokens("abcd")


def test_huggingface_normalization_preserves_compressed_system_history():
    provider = object.__new__(HuggingFaceProvider)
    provider.model_name = "test-model"
    normalized = provider._normalize_chat_messages([
        {"role": "system", "content": "Be helpful."},
        {"role": "system", "content": "Earlier conversation (compressed): old turn"},
        {"role": "user", "content": "latest question"},
    ])
    assert normalized[0]["content"] == (
        "Be helpful.\n\nEarlier conversation (compressed): old turn\n\nlatest question"
    )
