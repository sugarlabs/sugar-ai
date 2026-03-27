"""
AI functionality for Sugar-AI, including RAG orchestration and LLM access.
"""
from __future__ import annotations

import math
import os
import logging
from typing import Optional, List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate

import app.prompts as prompts
from app.llm import LLMProvider

logger = logging.getLogger("sugar-ai")


def format_docs(docs):
    """Return document content separated by newlines."""
    return "\n\n".join(doc.page_content for doc in docs)


def combine_messages(x):
    """Combine message content with newlines."""
    if hasattr(x, "to_messages"):
        return "\n".join(msg.content for msg in x.to_messages())
    return str(x)


class RAGAgent:
    """Retrieval-Augmented Generation agent for Sugar-AI."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.retriever: Optional[FAISS] = None
        self.prompt = ChatPromptTemplate.from_template(prompts.PROMPT_TEMPLATE)
        self.child_prompt = ChatPromptTemplate.from_template(prompts.CHILD_FRIENDLY_PROMPT)
        self.debug_prompt = ChatPromptTemplate.from_template(prompts.CODE_DEBUG_PROMPT)
        self.context_prompt = ChatPromptTemplate.from_template(prompts.CODE_CONTEXT_PROMPT)
        self.kids_debug_prompt = ChatPromptTemplate.from_template(prompts.KIDS_DEBUG_PROMPT)
        self.kids_context_prompt = ChatPromptTemplate.from_template(prompts.KIDS_CONTEXT_PROMPT)

    def set_provider(self, provider: LLMProvider) -> None:
        """Swap the backing provider used for generation."""
        self.provider = provider

    def setup_vectorstore(self, file_paths: List[str]) -> Optional[FAISS]:
        """Load documents and create a vector store for retrieval."""
        all_documents = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                if file_path.endswith(".pdf"):
                    loader = PyMuPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)

        if not all_documents:
            logger.warning("No documents found for vectorstore setup; RAG retrieval will run without local docs.")
            self.retriever = None
            return None

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(all_documents, embeddings)
        self.retriever = vector_store.as_retriever()
        return self.retriever

    def get_relevant_document(self, query: str, threshold: float = 0.5):
        """Get the most relevant document for a query."""
        if not self.retriever:
            return None, 0.0

        results = self.retriever.invoke(query)
        if results:
            top_result = results[0]
            score = top_result.metadata.get("score", 0.0)
            if score >= threshold:
                return top_result, score
        return None, 0.0

    def debug(self, code: str, context: bool) -> str:
        """Generate debugging or context help and rewrite it for kids."""
        if context:
            context_prompt = combine_messages(self.context_prompt.invoke({"code": code}))
            context_output = self._generate_prompt_text(
                context_prompt,
                temperature=0.2,
                top_p=0.9,
            )
            kids_prompt = combine_messages(
                self.kids_context_prompt.invoke({"context_output": context_output})
            )
            return self._generate_prompt_text(
                kids_prompt,
                temperature=0.4,
                top_p=0.9,
            )

        debug_prompt = combine_messages(self.debug_prompt.invoke({"code": code}))
        debug_output = self._generate_prompt_text(
            debug_prompt,
            temperature=0.2,
            top_p=0.9,
        )
        kids_prompt = combine_messages(
            self.kids_debug_prompt.invoke({"debug_output": debug_output})
        )
        return self._generate_prompt_text(
            kids_prompt,
            temperature=0.4,
            top_p=0.9,
        )

    def run(self, question: str) -> str:
        """Process a question through the RAG pipeline."""
        doc_result, _ = self.get_relevant_document(question)
        if doc_result:
            prompt_text = combine_messages(
                self.prompt.invoke(
                    {
                        "question": question,
                        "context": doc_result.page_content,
                    }
                )
            )
        else:
            prompt_text = combine_messages(
                self.prompt.invoke(
                    {
                        "question": question,
                        "context": "",
                    }
                )
            )

        first_response = self._generate_prompt_text(
            prompt_text,
            temperature=0.3,
            top_p=0.9,
        )
        second_prompt = combine_messages(
            self.child_prompt.invoke({"original_answer": first_response})
        )
        return self._generate_prompt_text(
            second_prompt,
            temperature=0.4,
            top_p=0.9,
        )

    def run_with_custom_prompt(
        self,
        question: str,
        custom_prompt: str,
        max_length: int = 1024,
        truncation: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Process a question with custom prompt and generation parameters (no RAG)."""
        messages = [
            {"role": "system", "content": custom_prompt},
            {"role": "user", "content": question},
        ]
        prepared_messages = self._prepare_messages_for_generation(
            messages,
            max_tokens=max_length,
            truncation=truncation,
        )
        return self.provider.generate(
            prepared_messages,
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
        )

    def run_chat_completion(
        self,
        messages: list,
        max_length: int = 1024,
        truncation: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Process chat messages with generation parameters."""
        normalized = self._normalize_chat_messages(messages)
        prepared_messages = self._prepare_messages_for_generation(
            normalized,
            max_tokens=max_length,
            truncation=truncation,
        )
        return self.provider.generate(
            prepared_messages,
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
        )

    def run_direct(self, question: str) -> str:
        """Generate a direct answer without retrieval."""
        prepared_messages = self._prepare_messages_for_generation(
            [{"role": "user", "content": question}],
            max_tokens=1024,
            truncation=True,
        )
        return self.provider.generate(
            prepared_messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )

    def _normalize_chat_messages(self, messages: list[dict]) -> list[dict]:
        """Ensure system prompt placement and roles stay OpenAI-compatible."""
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system" and msg.get("content"):
                system_content = msg["content"]
                break

        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        if not non_system_messages:
            return []

        normalized: list[dict[str, str]] = []
        if system_content:
            normalized.append({"role": "system", "content": system_content})

        for msg in non_system_messages:
            normalized.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
            )
        return normalized

    def _generate_prompt_text(
        self,
        prompt_text: str,
        *,
        max_tokens: int = 1024,
        truncation: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        prepared_messages = self._prepare_messages_for_generation(
            [{"role": "user", "content": prompt_text}],
            max_tokens=max_tokens,
            truncation=truncation,
        )
        return self.provider.generate(
            prepared_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _prepare_messages_for_generation(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        truncation: bool,
    ) -> list[dict[str, str]]:
        if not truncation:
            return messages

        model_limit = getattr(self.provider, "max_model_length", None)
        if not model_limit or model_limit <= 0:
            return messages

        input_budget = max(model_limit - max_tokens, 1)
        total_tokens = sum(self._approx_token_count(msg.get("content", "")) + 4 for msg in messages)
        if total_tokens <= input_budget:
            return messages

        system_message = None
        remaining_messages = messages
        if messages and messages[0].get("role") == "system":
            system_message = messages[0]
            remaining_messages = messages[1:]

        prepared_reversed: list[dict[str, str]] = []
        remaining_budget = input_budget

        if system_message:
            system_budget = input_budget if not remaining_messages else max(1, input_budget // 2)
            truncated_system = self._truncate_message(system_message, system_budget, keep_tail=False)
            if truncated_system:
                prepared_reversed.append(truncated_system)
                remaining_budget = max(
                    remaining_budget - self._message_token_cost(truncated_system),
                    0,
                )

        recent_messages: list[dict[str, str]] = []
        for message in reversed(remaining_messages):
            if remaining_budget <= 0:
                break
            truncated_message = self._truncate_message(message, remaining_budget, keep_tail=True)
            if not truncated_message:
                continue
            recent_messages.append(truncated_message)
            remaining_budget = max(
                remaining_budget - self._message_token_cost(truncated_message),
                0,
            )

        recent_messages.reverse()
        if system_message:
            return prepared_reversed + recent_messages
        return recent_messages

    def _truncate_message(
        self,
        message: dict[str, str],
        budget: int,
        *,
        keep_tail: bool,
    ) -> Optional[dict[str, str]]:
        if budget <= 4:
            return None

        content_budget = budget - 4
        content = message.get("content", "")
        if self._approx_token_count(content) <= content_budget:
            return {"role": message.get("role", "user"), "content": content}

        truncated_content = self._truncate_text(content, content_budget, keep_tail=keep_tail)
        if not truncated_content:
            return None
        return {"role": message.get("role", "user"), "content": truncated_content}

    def _message_token_cost(self, message: dict[str, str]) -> int:
        return self._approx_token_count(message.get("content", "")) + 4

    def _approx_token_count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    def _truncate_text(self, text: str, token_budget: int, *, keep_tail: bool) -> str:
        if token_budget <= 0:
            return ""

        char_budget = max(token_budget * 4, 1)
        if len(text) <= char_budget:
            return text
        if keep_tail:
            return text[-char_budget:]
        return text[:char_budget]
