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

"""Tests for RAGAgent document retrieval and relevance scoring."""
import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.ai import RAGAgent


class TestGetRelevantDocument(unittest.TestCase):
    """Verify relevance scoring and threshold behavior."""

    def setUp(self):
        self.provider = MagicMock()
        self.provider.get_model_name.return_value = "test-model"
        self.agent = RAGAgent(provider=self.provider)

    def test_vector_store_returns_document_when_score_meets_threshold(self):
        doc = Document(page_content="Sugar activity docs", metadata={"source": "a.txt"})
        vector_store = MagicMock()
        vector_store.similarity_search_with_relevance_scores.return_value = [
            (doc, 0.82)
        ]
        self.agent.vector_store = vector_store

        result_doc, score = self.agent.get_relevant_document("What is Sugar?", 0.5)

        self.assertIs(result_doc, doc)
        self.assertEqual(score, 0.82)
        vector_store.similarity_search_with_relevance_scores.assert_called_once_with(
            "What is Sugar?", k=1
        )

    def test_vector_store_rejects_document_below_threshold(self):
        doc = Document(page_content="Unrelated content")
        vector_store = MagicMock()
        vector_store.similarity_search_with_relevance_scores.return_value = [
            (doc, 0.31)
        ]
        self.agent.vector_store = vector_store

        result_doc, score = self.agent.get_relevant_document("What is Sugar?", 0.5)

        self.assertIsNone(result_doc)
        self.assertEqual(score, 0.0)

    def test_vector_store_returns_fallback_when_no_results(self):
        vector_store = MagicMock()
        vector_store.similarity_search_with_relevance_scores.return_value = []
        self.agent.vector_store = vector_store

        result_doc, score = self.agent.get_relevant_document("What is Sugar?", 0.5)

        self.assertIsNone(result_doc)
        self.assertEqual(score, 0.0)

    def test_retriever_uses_metadata_score_when_vector_store_unavailable(self):
        doc = Document(page_content="Pygame guide", metadata={"score": 0.75})
        retriever = MagicMock()
        retriever.invoke.return_value = [doc]
        self.agent.retriever = retriever

        result_doc, score = self.agent.get_relevant_document("Pygame sprites", 0.5)

        self.assertIs(result_doc, doc)
        self.assertEqual(score, 0.75)
        retriever.invoke.assert_called_once_with("Pygame sprites")

    def test_retriever_rejects_when_metadata_score_below_threshold(self):
        doc = Document(page_content="Pygame guide", metadata={"score": 0.2})
        retriever = MagicMock()
        retriever.invoke.return_value = [doc]
        self.agent.retriever = retriever

        result_doc, score = self.agent.get_relevant_document("Pygame sprites", 0.5)

        self.assertIsNone(result_doc)
        self.assertEqual(score, 0.0)

    def test_retriever_without_score_trusts_ranking(self):
        doc = Document(page_content="GTK tutorial", metadata={"source": "gtk.txt"})
        retriever = MagicMock()
        retriever.invoke.return_value = [doc]
        self.agent.retriever = retriever

        result_doc, score = self.agent.get_relevant_document("GTK buttons", 0.5)

        self.assertIs(result_doc, doc)
        self.assertEqual(score, 1.0)

    def test_run_uses_fallback_context_when_no_relevant_document(self):
        vector_store = MagicMock()
        vector_store.similarity_search_with_relevance_scores.return_value = []
        self.agent.vector_store = vector_store
        self.provider.generate.side_effect = [
            "Answer: Generic answer.",
            "Child-friendly answer: A friendly generic answer.",
        ]

        response = self.agent.run("Unknown topic")

        self.assertEqual(response, "A friendly generic answer.")
        first_prompt = self.provider.generate.call_args_list[0].args[0]
        self.assertIn("No relevant documentation found.", first_prompt)

    def test_run_uses_retrieved_context_when_document_is_relevant(self):
        doc = Document(page_content="Sugar is a learning platform.")
        vector_store = MagicMock()
        vector_store.similarity_search_with_relevance_scores.return_value = [
            (doc, 0.9)
        ]
        self.agent.vector_store = vector_store
        self.provider.generate.side_effect = [
            "Answer: Sugar helps kids learn.",
            "Child-friendly answer: Sugar is fun for learning!",
        ]

        response = self.agent.run("What is Sugar?")

        self.assertEqual(response, "Sugar is fun for learning!")
        first_prompt = self.provider.generate.call_args_list[0].args[0]
        self.assertIn("Sugar is a learning platform.", first_prompt)
        self.assertNotIn("No relevant documentation found.", first_prompt)


if __name__ == "__main__":
    unittest.main()
