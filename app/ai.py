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
import os
import math
import logging
from typing import Optional, List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
import app.prompts as prompts
from app.config import settings
from app.providers.base import BaseProvider, GenerationParams

logger = logging.getLogger("sugar-ai")

EOS_TOKENS = (
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end|>",
    "</s>",
    "<eos>",
)


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.
    
    Uses standard character and word heuristics (~4 chars/token).
    Returns 0 for empty or None text.
    """
    if not text:
        return 0
    # 1 token is roughly ~4 characters for standard English / code
    return max(1, math.ceil(len(text) / 4.0))


def estimate_message_tokens(message: dict) -> int:
    """Estimate token count for a single message dictionary.
    
    Includes role and framing delimiters overhead (~4 tokens).
    """
    if not message:
        return 0
    role = str(message.get("role", ""))
    content = str(message.get("content", ""))
    # Role + content + 4 tokens overhead per message
    return estimate_tokens(role) + estimate_tokens(content) + 4


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total token count for a list of chat message dictionaries."""
    if not messages:
        return 0
    # Sum of message tokens + 2 tokens for assistant prefix priming
    return sum(estimate_message_tokens(m) for m in messages) + 2


def budget_conversation_history(
    messages: list[dict],
    max_tokens: int = 4096,
    reserve_for_response: int = 0,
) -> list[dict]:
    """Prune conversation history to fit within a token budget using a sliding window.
    
    Strategy:
    1. System messages are prioritized and preserved at the front of history.
    2. The latest conversation turns (from newest to oldest) are preserved.
    3. Oldest conversation turns exceeding the remaining budget are dropped.
    4. If the latest message alone exceeds the remaining budget, its content
       is truncated to fit within the available budget.
       
    Args:
        messages: List of message dicts with 'role' and 'content'.
        max_tokens: Maximum allowable total tokens.
        reserve_for_response: Tokens reserved for model generation output.
        
    Returns:
        Budgeted list of message dicts preserving chronological order.
    """
    if not messages:
        return []

    # Calculate effective budget available for input messages
    effective_budget = max(128, max_tokens - max(0, reserve_for_response))

    # Fast path: check if total message tokens already fit
    total_tokens = estimate_messages_tokens(messages)
    if total_tokens <= effective_budget:
        return [dict(m) for m in messages]

    logger.info(
        "Conversation token count (%d) exceeds budget (%d, reserved=%d). Trimming context...",
        total_tokens,
        effective_budget,
        reserve_for_response,
    )

    # Separate system messages and non-system messages
    system_messages: list[dict] = []
    conversation_messages: list[dict] = []

    for msg in messages:
        if msg.get("role") == "system":
            system_messages.append(dict(msg))
        else:
            conversation_messages.append(dict(msg))

    # Calculate tokens used by system messages
    sys_tokens = sum(estimate_message_tokens(m) for m in system_messages) + 2

    # If system messages themselves exceed or nearly exhaust the budget, truncate system prompt
    if sys_tokens >= effective_budget:
        logger.warning(
            "System prompt tokens (%d) exceed or match budget (%d). Truncating system prompt.",
            sys_tokens,
            effective_budget,
        )
        avail_sys_tokens = max(64, effective_budget // 2)
        budgeted_sys: list[dict] = []
        for sm in system_messages:
            content = sm.get("content", "")
            char_limit = avail_sys_tokens * 4
            truncated_content = content[:char_limit]
            budgeted_sys.append({"role": sm.get("role", "system"), "content": truncated_content})
        system_messages = budgeted_sys
        sys_tokens = sum(estimate_message_tokens(m) for m in system_messages) + 2

    remaining_budget = max(64, effective_budget - sys_tokens)

    # Sliding window on conversation messages: newest to oldest
    selected_conv_messages: list[dict] = []
    accumulated_tokens = 0

    for msg in reversed(conversation_messages):
        msg_tokens = estimate_message_tokens(msg)
        if accumulated_tokens + msg_tokens <= remaining_budget:
            selected_conv_messages.append(dict(msg))
            accumulated_tokens += msg_tokens
        else:
            # If we couldn't even fit the single newest message, truncate its content
            if not selected_conv_messages:
                content = msg.get("content", "")
                avail_content_tokens = max(16, remaining_budget - 8)
                char_limit = avail_content_tokens * 4
                truncated_content = content[-char_limit:] if len(content) > char_limit else content
                selected_conv_messages.append({
                    "role": msg.get("role", "user"),
                    "content": truncated_content
                })
                accumulated_tokens += estimate_message_tokens(selected_conv_messages[-1])
            break

    # Restore chronological order
    selected_conv_messages.reverse()

    result = system_messages + selected_conv_messages
    logger.info(
        "Budgeted conversation from %d to %d messages (estimated tokens: %d / %d).",
        len(messages),
        len(result),
        estimate_messages_tokens(result),
        effective_budget,
    )
    return result


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
        self.retriever: Optional[FAISS] = None

        self.prompt_template = prompts.PROMPT_TEMPLATE
        self.child_prompt_template = prompts.CHILD_FRIENDLY_PROMPT
        self.debug_prompt_template = prompts.CODE_DEBUG_PROMPT
        self.context_prompt_template = prompts.CODE_CONTEXT_PROMPT
        self.kids_debug_prompt_template = prompts.KIDS_DEBUG_PROMPT
        self.kids_context_prompt_template = prompts.KIDS_CONTEXT_PROMPT

    def set_model(self, provider: BaseProvider) -> None:
        """Update the current provider."""
        old_provider = self.provider
        self.provider = provider
        self.model_name = provider.get_model_name()
        if old_provider is not provider:
            try:
                old_provider.close()
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

    def get_relevant_document(self, query: str, threshold: float = 0.5):
        """Get the most relevant document for a query."""
        if not self.retriever:
            return None, 0.0
        results = self.retriever.vectorstore.similarity_search_with_relevance_scores(
            query,
            **self.retriever.search_kwargs
        )
        if results:
            top_result, score = results[0]
            if score >= threshold:
                return top_result, score
        return None, 0.0

    def debug(self, code: str, context: bool) -> str:
        """Debug or explain python code using provider."""
        if context:
            context_prompt = self.context_prompt_template.format(code=code)
            raw_context = self.provider.generate(context_prompt)

            kids_prompt = self.kids_context_prompt_template.format(context_output=raw_context)
            kid_friendly = self.provider.generate(kids_prompt)
            return kid_friendly
        else:
            debug_prompt = self.debug_prompt_template.format(code=code)
            raw_debug = self.provider.generate(debug_prompt)

            kids_prompt = self.kids_debug_prompt_template.format(debug_output=raw_debug)
            kid_friendly = self.provider.generate(kids_prompt)
            return kid_friendly

    def run(self, question: str, max_context_tokens: Optional[int] = None) -> str:
        """Process a question through the RAG pipeline with context budgeting."""
        budget = max_context_tokens or getattr(settings, "MAX_CONTEXT_TOKENS", 4096)
        effective_budget = max(256, budget - 1024)

        doc_result, _ = self.get_relevant_document(question)
        if doc_result:
            context_text = doc_result.page_content
            # Ensure retrieved document context fits within half the effective budget
            if estimate_tokens(context_text) > (effective_budget // 2):
                char_limit = (effective_budget // 2) * 4
                context_text = context_text[:char_limit] + "..."
            prompt = self.prompt_template.format(
                question=question,
                context=context_text
            )
        else:
            prompt = self.prompt_template.format(
                question=question,
                context="No relevant documentation found."
            )

        first_response = self.provider.generate(prompt)

        if "Child-friendly answer:" in first_response:
            first_response = first_response.split("Child-friendly answer:")[-1].strip()
        elif "Answer:" in first_response:
            first_response = first_response.split("Answer:")[-1].strip()

        child_prompt = self.child_prompt_template.format(original_answer=first_response)
        final_response = self.provider.generate(child_prompt)

        if "Child-friendly answer:" in final_response:
            final_response = final_response.split("Child-friendly answer:")[-1].strip()

        return final_response

    def run_with_custom_prompt(self, question: str, custom_prompt: str,
                               params: Optional[GenerationParams] = None,
                               max_context_tokens: Optional[int] = None) -> str:
        """Process a question with custom prompt and parameters (no RAG) with context budgeting."""
        params = params or GenerationParams()
        budget = max_context_tokens or getattr(settings, "MAX_CONTEXT_TOKENS", 4096)
        reserve = params.max_new_tokens if params else 1024
        effective_budget = max(128, budget - reserve)

        full_prompt = f"{custom_prompt}\n\nQuestion: {question}\nAnswer:"
        if estimate_tokens(full_prompt) > effective_budget:
            q_tokens = estimate_tokens(question)
            avail_prompt_tokens = max(64, effective_budget - q_tokens - 32)
            char_limit = avail_prompt_tokens * 4
            custom_prompt_truncated = custom_prompt[:char_limit] + "..."
            full_prompt = f"{custom_prompt_truncated}\n\nQuestion: {question}\nAnswer:"

        try:
            answer = self.provider.generate(full_prompt, params)

            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

            return self._truncate_at_eos(answer)

        except Exception as e:
            raise Exception(f"Error generating response with custom prompt: {str(e)}")

    def run_chat_completion(self, messages: list,
                            params: Optional[GenerationParams] = None,
                            max_context_tokens: Optional[int] = None) -> str:
        """Process chat messages using the provider's chat interface with context budgeting."""
        params = params or GenerationParams()
        budget = max_context_tokens or getattr(settings, "MAX_CONTEXT_TOKENS", 4096)
        reserve = params.max_new_tokens if params else 1024

        budgeted_messages = budget_conversation_history(
            messages=messages,
            max_tokens=budget,
            reserve_for_response=reserve,
        )

        try:
            answer = self.provider.chat(budgeted_messages, params)
            return answer
        except Exception as e:
            raise Exception(f"Error generating chat completion: {str(e)}")

    def _truncate_at_eos(self, text: str) -> str:
        """Trim model output at an explicit end-of-sequence token."""
        eos_tokens = []
        provider_eos = self.provider.get_eos_token()
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
