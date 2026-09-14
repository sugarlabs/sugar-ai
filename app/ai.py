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
from typing import AsyncIterator, Callable, Dict, Optional, List
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
        self.provider: BaseProvider = provider
        self.model_name = provider.get_model_name()
        self._model_change_lock = asyncio.Lock()
        self._provider_users: Dict[int, int] = {}
        self.retriever: Optional[FAISS] = None

        self.prompt_template = prompts.PROMPT_TEMPLATE
        self.child_prompt_template = prompts.CHILD_FRIENDLY_PROMPT
        self.debug_prompt_template = prompts.CODE_DEBUG_PROMPT
        self.context_prompt_template = prompts.CODE_CONTEXT_PROMPT
        self.kids_debug_prompt_template = prompts.KIDS_DEBUG_PROMPT
        self.kids_context_prompt_template = prompts.KIDS_CONTEXT_PROMPT

    @asynccontextmanager
    async def use_provider(self) -> AsyncIterator[BaseProvider]:
        """Hold one provider for the duration of an AI operation.

        The provider is captured when the operation starts, so a model change
        never pulls it out from under a request already using it. A retired
        provider is closed by whichever request releases it last.
        """
        provider = self.provider
        key = id(provider)
        self._provider_users[key] = self._provider_users.get(key, 0) + 1

        try:
            yield provider
        finally:
            remaining = self._provider_users[key] - 1
            if remaining:
                self._provider_users[key] = remaining
            else:
                del self._provider_users[key]
                if provider is not self.provider:
                    await self._close_provider(provider)

    @asynccontextmanager
    async def _provider_operation(
        self,
        provider: Optional[BaseProvider] = None,
    ) -> AsyncIterator[BaseProvider]:
        """Use an admitted provider or acquire a new provider lease."""
        if provider is not None:
            yield provider
            return

        async with self.use_provider() as leased_provider:
            yield leased_provider

    async def _close_provider(self, provider: BaseProvider) -> None:
        """Close a retired provider without disturbing live requests."""
        try:
            await provider.close()
            logger.info("Closed retired provider for model %s", provider.get_model_name())
        except Exception as error:
            logger.warning("Failed to close retired provider: %s", error)

    async def replace_provider(
        self,
        provider_factory: Callable[[], BaseProvider],
    ) -> None:
        """Install a new provider without interrupting active requests.

        The replacement is built before anything is swapped, so a failed build
        leaves the current provider serving. Requests already running keep the
        provider they started with until they finish.
        """
        async with self._model_change_lock:
            old_provider = self.provider
            logger.info("Model change requested while serving %s", self.model_name)

            new_provider = await run_in_threadpool(provider_factory)
            try:
                new_model_name = new_provider.get_model_name()
            except BaseException:
                await self._close_provider(new_provider)
                raise

            # Nothing may await between here and the swap, so no request can
            # observe a half-changed agent.
            previous_model_name = self.model_name
            self.provider = new_provider
            self.model_name = new_model_name
            logger.info(
                "Model changed from %s to %s", previous_model_name, new_model_name
            )

            if old_provider is not new_provider:
                if id(old_provider) not in self._provider_users:
                    await self._close_provider(old_provider)
                else:
                    logger.info(
                        "Retired provider %s stays open for %d active request(s)",
                        previous_model_name,
                        self._provider_users[id(old_provider)],
                    )

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

    async def debug(
        self,
        code: str,
        context: bool,
        provider: Optional[BaseProvider] = None,
    ) -> str:
        """Debug or explain python code using provider."""
        async with self._provider_operation(provider) as active_provider:
            if context:
                context_prompt = self.context_prompt_template.format(code=code)
                raw_context = await active_provider.generate(context_prompt)

                kids_prompt = self.kids_context_prompt_template.format(context_output=raw_context)
                kid_friendly = await active_provider.generate(kids_prompt)
                return kid_friendly

            debug_prompt = self.debug_prompt_template.format(code=code)
            raw_debug = await active_provider.generate(debug_prompt)

            kids_prompt = self.kids_debug_prompt_template.format(debug_output=raw_debug)
            kid_friendly = await active_provider.generate(kids_prompt)
            return kid_friendly

    async def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        provider: Optional[BaseProvider] = None,
    ) -> str:
        """Generate text while holding a lease on the current provider."""
        async with self._provider_operation(provider) as active_provider:
            return await active_provider.generate(prompt, params)

    async def run(
        self,
        question: str,
        provider: Optional[BaseProvider] = None,
    ) -> str:
        """Process a question through the RAG pipeline."""
        async with self._provider_operation(provider) as active_provider:
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

            first_response = await active_provider.generate(prompt)

            if "Child-friendly answer:" in first_response:
                first_response = first_response.split("Child-friendly answer:")[-1].strip()
            elif "Answer:" in first_response:
                first_response = first_response.split("Answer:")[-1].strip()

            child_prompt = self.child_prompt_template.format(original_answer=first_response)
            final_response = await active_provider.generate(child_prompt)

            if "Child-friendly answer:" in final_response:
                final_response = final_response.split("Child-friendly answer:")[-1].strip()

            return final_response

    async def run_with_custom_prompt(self, question: str, custom_prompt: str,
                               params: Optional[GenerationParams] = None,
                               provider: Optional[BaseProvider] = None) -> str:
        """Process a question with custom prompt and parameters (no RAG)."""
        params = params or GenerationParams()
        full_prompt = f"{custom_prompt}\n\nQuestion: {question}\nAnswer:"

        try:
            async with self._provider_operation(provider) as active_provider:
                answer = await active_provider.generate(full_prompt, params)

                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()

                return self._truncate_at_eos(answer, active_provider)

        except Exception as e:
            raise Exception(f"Error generating response with custom prompt: {str(e)}")

    async def run_chat_completion(self, messages: list,
                            params: Optional[GenerationParams] = None,
                            provider: Optional[BaseProvider] = None) -> str:
        """Process chat messages using the provider's chat interface."""
        params = params or GenerationParams()

        try:
            async with self._provider_operation(provider) as active_provider:
                answer = await active_provider.chat(messages, params)
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
