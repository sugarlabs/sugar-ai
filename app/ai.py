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

"""RAG and LLM components for Sugar-AI."""
import asyncio
import os
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from typing import AsyncIterator, Optional, List
import app.prompts as prompts
from app.config import settings
from app.providers.base import BaseProvider, GenerationParams
from starlette.concurrency import run_in_threadpool
import logging

logger = logging.getLogger("sugar-ai")

EOS_TOKENS = (
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end|>",
    "</s>",
    "<eos>",
)


def format_docs(docs):
    """Return document content separated by newlines"""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGAgent:
    """Retrieval-Augmented Generation agent for Sugar-AI.

    This class handles document retrieval and prompt orchestration,
    delegating model loading and inference to a BaseProvider.
    """

    def __init__(self, provider: BaseProvider):
        """Initialize RAGAgent with a provider."""
        self.provider = provider
        self.model_name = provider.get_model_name()
        self._provider_condition = asyncio.Condition()
        self._active_provider_requests = 0
        self.retriever: Optional[FAISS] = None

        self.prompt_template = prompts.PROMPT_TEMPLATE
        self.child_prompt_template = prompts.CHILD_FRIENDLY_PROMPT
        self.debug_prompt_template = prompts.CODE_DEBUG_PROMPT
        self.context_prompt_template = prompts.CODE_CONTEXT_PROMPT
        self.kids_debug_prompt_template = prompts.KIDS_DEBUG_PROMPT
        self.kids_context_prompt_template = prompts.KIDS_CONTEXT_PROMPT

    @asynccontextmanager
    async def use_provider(self) -> AsyncIterator[BaseProvider]:
        """Keep one provider active for the duration of an AI operation."""
        async with self._provider_condition:
            provider = self.provider
            self._active_provider_requests += 1

        try:
            yield provider
        finally:
            async with self._provider_condition:
                self._active_provider_requests -= 1
                if self._active_provider_requests == 0:
                    self._provider_condition.notify_all()

    async def set_model(self, provider: BaseProvider) -> None:
        """Update the current provider."""
        old_provider = self.provider
        self.provider = provider
        self.model_name = provider.get_model_name()
        if old_provider is not provider:
            try:
                await old_provider.close()
            except Exception as e:
                logger.warning("Failed to close previous provider: %s", e)

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

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(all_documents, embeddings)
        self.retriever = vector_store.as_retriever()
        return self.retriever

    async def get_relevant_document(self, query: str, threshold: float = 0.5):
        """Get the most relevant document for a query."""
        results = await run_in_threadpool(self.retriever.invoke, query)
        if results:
            top_result = results[0]
            score = top_result.metadata.get("score", 0.0)
            if score >= threshold:
                return top_result, score
        return None, 0.0

    async def debug(self, code: str, context: bool) -> str:
        """Debug or explain python code using provider."""
        async with self.use_provider() as provider:
            if context:
                context_prompt = self.context_prompt_template.format(code=code)
                raw_context = await provider.generate(context_prompt)

                kids_prompt = self.kids_context_prompt_template.format(context_output=raw_context)
                kid_friendly = await provider.generate(kids_prompt)
                return kid_friendly

            debug_prompt = self.debug_prompt_template.format(code=code)
            raw_debug = await provider.generate(debug_prompt)

            kids_prompt = self.kids_debug_prompt_template.format(debug_output=raw_debug)
            kid_friendly = await provider.generate(kids_prompt)
            return kid_friendly

    async def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
    ) -> str:
        """Generate text while holding a lease on the current provider."""
        async with self.use_provider() as provider:
            return await provider.generate(prompt, params)

    async def run(self, question: str) -> str:
        """Process a question through the RAG pipeline."""
        async with self.use_provider() as provider:
            doc_result, _ = await self.get_relevant_document(question)
            if doc_result:
                prompt = self.prompt_template.format(
                    question=question,
                    context=doc_result.page_content
                )
            else:
                prompt = self.prompt_template.format(
                    question=question,
                    context="No relevant documentation found."
                )

            first_response = await provider.generate(prompt)

            if "Child-friendly answer:" in first_response:
                first_response = first_response.split("Child-friendly answer:")[-1].strip()
            elif "Answer:" in first_response:
                first_response = first_response.split("Answer:")[-1].strip()

            child_prompt = self.child_prompt_template.format(original_answer=first_response)
            final_response = await provider.generate(child_prompt)

            if "Child-friendly answer:" in final_response:
                final_response = final_response.split("Child-friendly answer:")[-1].strip()

            return final_response

    async def run_with_custom_prompt(self, question: str, custom_prompt: str,
                               params: Optional[GenerationParams] = None) -> str:
        """Process a question with custom prompt and parameters (no RAG)."""
        params = params or GenerationParams()
        full_prompt = f"{custom_prompt}\n\nQuestion: {question}\nAnswer:"

        try:
            async with self.use_provider() as provider:
                answer = await provider.generate(full_prompt, params)

                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()

                return self._truncate_at_eos(answer, provider)

        except Exception as e:
            raise Exception(f"Error generating response with custom prompt: {str(e)}")

    async def run_chat_completion(self, messages: list,
                            params: Optional[GenerationParams] = None) -> str:
        """Process chat messages using the provider's chat interface."""
        params = params or GenerationParams()

        try:
            async with self.use_provider() as provider:
                answer = await provider.chat(messages, params)
                return answer
        except Exception as e:
            raise Exception(f"Error generating chat completion: {str(e)}")

    def _truncate_at_eos(
        self,
        text: str,
        provider: Optional[BaseProvider] = None,
    ) -> str:
        """Trim model output at an explicit end-of-sequence token."""
        eos_tokens = []
        active_provider = provider or self.provider
        provider_eos = active_provider.get_eos_token()
        if provider_eos:
            eos_tokens.append(provider_eos)
        eos_tokens.extend(EOS_TOKENS)

        eos_positions = [
            text.find(token)
            for token in eos_tokens
            if token and text.find(token) != -1
        ]
        if eos_positions:
            return text[:min(eos_positions)].strip()
        return text.strip()
