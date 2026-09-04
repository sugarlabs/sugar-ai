"""Tests for RAG retrieval pipeline in RAGAgent.

Verifies that:
- get_relevant_document() returns documents above the score threshold
- get_relevant_document() rejects documents below the score threshold
- get_relevant_document() handles empty results and missing vector store
- run() includes retrieved doc context in the prompt when a match is found
- run() falls back to 'no documentation found' when nothing matches
"""
import unittest
from unittest.mock import MagicMock, patch, call
from langchain_core.documents import Document
from app.ai import RAGAgent


class TestGetRelevantDocument(unittest.TestCase):
    """Unit tests for RAGAgent.get_relevant_document()."""

    def setUp(self):
        # Create a mock provider
        self.mock_provider = MagicMock()
        self.mock_provider.get_model_name.return_value = "mock-model"
        self.mock_provider.get_eos_token.return_value = None

        # Initialize RAGAgent with mock provider
        self.agent = RAGAgent(provider=self.mock_provider)

        # Mock the FAISS vector store directly
        self.mock_vector_store = MagicMock()
        self.agent.vector_store = self.mock_vector_store

    def test_above_threshold_returns_document(self):
        """Documents with L2 distance close to 0 should score above 0.5."""
        doc = Document(
            page_content="Use pygame.display.set_mode() to create a window.",
            metadata={"source": "pygame_docs.pdf"},
        )
        # L2 distance of 0.5 → similarity = 1/(1+0.5) = 0.667
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (doc, 0.5)
        ]

        result_doc, score = self.agent.get_relevant_document(
            "How to create a Pygame window?"
        )

        self.assertIsNotNone(result_doc)
        self.assertEqual(
            result_doc.page_content,
            "Use pygame.display.set_mode() to create a window.",
        )
        self.assertAlmostEqual(score, 1.0 / (1.0 + 0.5))
        self.mock_vector_store.similarity_search_with_score.assert_called_once_with(
            "How to create a Pygame window?", k=1
        )

    def test_below_threshold_returns_none(self):
        """Documents with high L2 distance should score below threshold."""
        doc = Document(
            page_content="Irrelevant content.",
            metadata={"source": "random.txt"},
        )
        # L2 distance of 5.0 → similarity = 1/(1+5) = 0.167 < 0.5
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (doc, 5.0)
        ]

        result_doc, score = self.agent.get_relevant_document(
            "How to create a Pygame window?"
        )

        self.assertIsNone(result_doc)
        self.assertEqual(score, 0.0)

    def test_empty_results_returns_none(self):
        """When FAISS returns no results, should return (None, 0.0)."""
        self.mock_vector_store.similarity_search_with_score.return_value = []

        result_doc, score = self.agent.get_relevant_document(
            "What is Sugar-AI?"
        )

        self.assertIsNone(result_doc)
        self.assertEqual(score, 0.0)

    def test_no_vector_store_returns_none(self):
        """When vector store is not initialized, should return (None, 0.0)."""
        self.agent.vector_store = None

        result_doc, score = self.agent.get_relevant_document(
            "What is Sugar-AI?"
        )

        self.assertIsNone(result_doc)
        self.assertEqual(score, 0.0)

    def test_custom_threshold(self):
        """Should respect a custom threshold value."""
        doc = Document(
            page_content="GTK Button docs.",
            metadata={"source": "gtk_docs.pdf"},
        )
        # L2 distance of 0.3 → similarity = 1/(1+0.3) ≈ 0.769
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (doc, 0.3)
        ]

        # With high threshold of 0.8 → 0.769 < 0.8, should reject
        result_doc, score = self.agent.get_relevant_document(
            "How to make a GTK button?", threshold=0.8
        )
        self.assertIsNone(result_doc)

        # With lower threshold of 0.7 → 0.769 >= 0.7, should accept
        result_doc, score = self.agent.get_relevant_document(
            "How to make a GTK button?", threshold=0.7
        )
        self.assertIsNotNone(result_doc)


class TestRunRAGPipeline(unittest.TestCase):
    """Tests that run() correctly uses or skips RAG context."""

    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.get_model_name.return_value = "mock-model"
        self.mock_provider.get_eos_token.return_value = None
        self.mock_provider.generate.return_value = "Here is your answer."

        self.agent = RAGAgent(provider=self.mock_provider)
        self.mock_vector_store = MagicMock()
        self.agent.vector_store = self.mock_vector_store

    def test_run_uses_doc_context_when_found(self):
        """When a relevant doc is found, its content should appear in the prompt."""
        doc = Document(
            page_content="pygame.display.set_mode((800, 600)) creates a window.",
            metadata={"source": "pygame_docs.pdf"},
        )
        # Close match: L2 distance 0.3 → similarity ≈ 0.77
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (doc, 0.3)
        ]

        self.agent.run("How to create a Pygame window?")

        # Check that provider.generate was called and the prompt
        # contains the document content (not the fallback message)
        first_call_prompt = self.mock_provider.generate.call_args_list[0][0][0]
        self.assertIn(
            "pygame.display.set_mode((800, 600)) creates a window.",
            first_call_prompt,
        )
        self.assertNotIn("No relevant documentation found", first_call_prompt)

    def test_run_uses_fallback_when_no_doc(self):
        """When no relevant doc is found, prompt should contain fallback text."""
        # No match: empty results
        self.mock_vector_store.similarity_search_with_score.return_value = []

        self.agent.run("Tell me about quantum physics")

        first_call_prompt = self.mock_provider.generate.call_args_list[0][0][0]
        self.assertIn("No relevant documentation found", first_call_prompt)

    def test_run_uses_fallback_when_no_vector_store(self):
        """When vector store is None, should still work with fallback."""
        self.agent.vector_store = None

        self.agent.run("How to draw a circle?")

        first_call_prompt = self.mock_provider.generate.call_args_list[0][0][0]
        self.assertIn("No relevant documentation found", first_call_prompt)


if __name__ == "__main__":
    unittest.main()
