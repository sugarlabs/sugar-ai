# Copyright (C) 2024 Sugar Labs, Inc.
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

"""
Tests for utility functions in Sugar-AI that do not require
model loading or GPU access.
"""

from app.ai import extract_answer_from_output, format_docs, combine_messages, RAGAgent


class TestExtractAnswerFromOutput:
    """Tests for extract_answer_from_output()"""

    def test_none_input_returns_empty(self):
        assert extract_answer_from_output(None) == ""

    def test_empty_list_returns_empty(self):
        assert extract_answer_from_output([]) == ""

    def test_empty_dict_returns_empty(self):
        assert extract_answer_from_output([{}]) == ""

    def test_none_generated_text_returns_empty(self):
        assert extract_answer_from_output([{"generated_text": None}]) == ""

    def test_non_string_generated_text_returns_empty(self):
        assert extract_answer_from_output([{"generated_text": 123}]) == ""

    def test_extracts_after_answer_marker(self):
        outputs = [{"generated_text": "Some preamble\nAnswer: Use pygame.init()"}]
        assert extract_answer_from_output(outputs) == "Use pygame.init()"

    def test_extracts_after_child_friendly_marker(self):
        outputs = [{"generated_text": "Prompt text\nChild-friendly answer: Try this!"}]
        assert extract_answer_from_output(outputs) == "Try this!"

    def test_child_friendly_marker_takes_precedence_over_answer(self):
        outputs = [{"generated_text": "Answer: raw\nChild-friendly answer: nice"}]
        assert extract_answer_from_output(outputs) == "nice"

    def test_fallback_returns_full_text_stripped(self):
        outputs = [{"generated_text": "  Just a plain response  "}]
        assert extract_answer_from_output(outputs) == "Just a plain response"

    def test_multiple_answer_markers_uses_last(self):
        outputs = [{"generated_text": "Answer: first\nAnswer: second"}]
        assert extract_answer_from_output(outputs) == "second"


class TestFormatDocs:
    """Tests for format_docs()"""

    class FakeDoc:
        def __init__(self, content):
            self.page_content = content

    def test_single_doc(self):
        docs = [self.FakeDoc("Hello world")]
        assert format_docs(docs) == "Hello world"

    def test_multiple_docs_joined_with_double_newline(self):
        docs = [self.FakeDoc("Doc 1"), self.FakeDoc("Doc 2"), self.FakeDoc("Doc 3")]
        assert format_docs(docs) == "Doc 1\n\nDoc 2\n\nDoc 3"

    def test_empty_list_returns_empty(self):
        assert format_docs([]) == ""


class TestCombineMessages:
    """Tests for combine_messages()"""

    def test_string_passthrough(self):
        assert combine_messages("hello") == "hello"

    def test_non_string_converted_via_str(self):
        assert combine_messages(42) == "42"


class TestExtractAfterPrompt:
    """Tests for RAGAgent._extract_after_prompt()

    This is an instance method but only uses its arguments (no model needed).
    We call the underlying function directly, passing a dummy self.
    """

    @staticmethod
    def _call(full_text, prompt, eos_token=None):
        """Helper to call _extract_after_prompt without a real RAGAgent."""
        return RAGAgent._extract_after_prompt(None, full_text, prompt, eos_token)

    def test_removes_prompt_prefix(self):
        assert self._call("PROMPT hello world", "PROMPT") == "hello world"

    def test_no_prompt_match_returns_stripped_text(self):
        assert self._call("  different text  ", "PROMPT") == "different text"

    def test_trims_at_eos_token(self):
        result = self._call("PROMPT answer here<eos>junk after", "PROMPT", "<eos>")
        assert result == "answer here"

    def test_eos_none_does_not_trim(self):
        assert self._call("PROMPT just answer", "PROMPT", None) == "just answer"

    def test_double_newline_truncates_long_text(self):
        # Candidate before \n\n is "this is a long answer text" (26 chars > 10)
        result = self._call("this is a long answer text\n\nextra stuff", "")
        assert result == "this is a long answer text"

    def test_double_newline_keeps_short_text(self):
        # Candidate before \n\n is "short" (5 chars <= 10), so no truncation
        result = self._call("short\n\nmore text here", "")
        assert result == "short\n\nmore text here"
