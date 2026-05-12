"""
AI functionality for Sugar-AI, including RAG and LLM components.

Backend selection is now driven by sugar_ai.yaml (or environment variables).
The RAG pipeline (FAISS, LangChain, prompts) is unchanged.
Only the inference calls route through BackendRouter instead of a hardcoded
HuggingFace pipeline.
"""

import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List
import app.prompts as prompts
from app.config import settings
import logging

# Backend abstraction layer
from app.backends import BackendRouter, BackendUnavailableError, GenerationConfig, Message, Role, SugarAIConfig

logger = logging.getLogger("sugar-ai")


def format_docs(docs):
    """Return document content separated by newlines"""
    return "\n\n".join(doc.page_content for doc in docs)


def combine_messages(x):
    """Combine message content with newlines"""
    if hasattr(x, "to_messages"):
        return "\n".join(msg.content for msg in x.to_messages())
    return str(x)


def extract_answer_from_output(outputs):
    """Extract the answer text from model output safely."""
    if not outputs:
        return ""

    first = outputs[0] or {}
    generated_text = first.get("generated_text")
    if not isinstance(generated_text, str):
        return ""

    if "Child-friendly answer:" in generated_text:
        return generated_text.split("Child-friendly answer:")[-1].strip()

    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()

    return generated_text.strip()


# ---------------------------------------------------------------------------
# Backend-aware callable wrapper
# ---------------------------------------------------------------------------

class _RouterCallable:
    """Wraps BackendRouter so it can be called like a HuggingFace pipeline.

    This lets the existing LangChain chains (| self.model | ...) work
    without modification. The router receives the prompt string and returns
    output in the same format as the HuggingFace pipeline.
    """

    def __init__(self, router: BackendRouter, gen_config: GenerationConfig):
        self._router = router
        self._gen_config = gen_config
        # Expose tokenizer attribute so existing code that accesses
        # self.model.tokenizer doesn't crash (returns None gracefully)
        self.tokenizer = None

    def __call__(self, prompt_or_messages, **kwargs) -> list[dict]:
        """Handle both string prompts and message lists."""
        gen_config = GenerationConfig(
            max_tokens=kwargs.get("max_new_tokens", kwargs.get("max_length", self._gen_config.max_tokens)),
            temperature=kwargs.get("temperature", self._gen_config.temperature),
            top_p=kwargs.get("top_p", self._gen_config.top_p),
            top_k=kwargs.get("top_k", self._gen_config.top_k),
            repeat_penalty=kwargs.get("repetition_penalty", self._gen_config.repeat_penalty),
        )

        if isinstance(prompt_or_messages, list):
            # Chat messages format from run_chat_completion
            history = []
            question = ""
            for msg in prompt_or_messages:
                role_str = msg.get("role", "user")
                content = msg.get("content", "")
                if role_str == "user":
                    question = content
                elif role_str in ("assistant", "model"):
                    history.append(Message(role=Role.ASSISTANT, content=content))
            response = self._router.ask(question, history=history or None, config=gen_config)
        else:
            # Plain string prompt
            response = self._router.ask(str(prompt_or_messages), config=gen_config)

        # Return in HuggingFace pipeline format so existing chains work
        return [{"generated_text": response.content}]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Stub: return concatenated message content as a plain string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------

class RAGAgent:
    """Retrieval-Augmented Generation agent for Sugar-AI.

    Model selection is now driven by sugar_ai.yaml.
    The RAG pipeline (FAISS, LangChain, prompts) is unchanged.
    """

    def __init__(self, model: Optional[str] = None, quantize: bool = True):
        # Load backend config
        ai_config = SugarAIConfig.load()

        # Allow explicit model override to propagate into the primary backend
        if model:
            ai_config.primary_backend["model_name"] = model
            logger.info("Model override: %s", model)

        # Build the router
        try:
            self._router = BackendRouter.from_config(ai_config)
            logger.info(
                "BackendRouter initialised: primary=%s",
                self._router.primary.name,
            )
        except BackendUnavailableError as e:
            logger.error("Failed to initialise backend: %s", e)
            raise

        self.model_name = ai_config.primary_backend.get("model_name", "unknown")

        # Generation config defaults
        self._gen_config = GenerationConfig(
            max_tokens=ai_config.token_budget_per_request,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repeat_penalty=1.1,
        )

        # Wrap router as a callable so LangChain chains work unchanged
        self.model = _RouterCallable(self._router, self._gen_config)
        self.simplify_model = self.model

        # Prompts (unchanged)
        self.retriever: Optional[FAISS] = None
        self.prompt = ChatPromptTemplate.from_template(prompts.PROMPT_TEMPLATE)
        self.child_prompt = ChatPromptTemplate.from_template(prompts.CHILD_FRIENDLY_PROMPT)
        self.debug_prompt = ChatPromptTemplate.from_template(prompts.CODE_DEBUG_PROMPT)
        self.context_prompt = ChatPromptTemplate.from_template(prompts.CODE_CONTEXT_PROMPT)
        self.kids_debug_prompt = ChatPromptTemplate.from_template(prompts.KIDS_DEBUG_PROMPT)
        self.kids_context_prompt = ChatPromptTemplate.from_template(prompts.KIDS_CONTEXT_PROMPT)

    def set_model(self, model: str) -> None:
        """Hot-swap the primary model.

        If the primary backend supports model switching (HuggingFace does),
        swaps in place. Otherwise re-initialises the router.
        """
        primary = self._router.primary
        if hasattr(primary, "load_model"):
            primary.load_model(model)
            self.model_name = model
            logger.info("Switched model to: %s", model)
        else:
            # Re-initialise router with new model name
            config = SugarAIConfig.load()
            config.primary_backend["model_name"] = model
            self._router = BackendRouter.from_config(config)
            self.model = _RouterCallable(self._router, self._gen_config)
            self.simplify_model = self.model
            self.model_name = model
            logger.info("Re-initialised router with model: %s", model)

    def backend_health(self) -> dict:
        """Return health status of all configured backends."""
        return self._router.health()

    # ------------------------------------------------------------------
    # RAG pipeline (unchanged from original)
    # ------------------------------------------------------------------

    def setup_vectorstore(self, file_paths: List[str]) -> Optional[FAISS]:
        """Load documents and create a vector store for retrieval"""
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
        """Get the most relevant document for a query"""
        results = self.retriever.invoke(query)
        if results:
            top_result = results[0]
            score = top_result.metadata.get("score", 0.0)
            if score >= threshold:
                return top_result, score
        return None, 0.0

    def debug(self, code: str, context: bool) -> str:
        """Debugging chain (unchanged)"""
        debug_chain = (
            self.debug_prompt
            | combine_messages
            | self.model
            | extract_answer_from_output
            | self.kids_debug_prompt
            | combine_messages
            | self.model
            | extract_answer_from_output
        )

        context_chain = (
            self.context_prompt
            | combine_messages
            | self.model
            | extract_answer_from_output
            | self.kids_context_prompt
            | combine_messages
            | self.model
            | extract_answer_from_output
        )

        if context:
            return context_chain.invoke({"code": code})
        return debug_chain.invoke({"code": code})

    def run(self, question: str) -> str:
        """Process a question through the RAG pipeline (unchanged)"""
        chain_input = {
            "context": self.retriever | format_docs,
            "question": RunnablePassthrough()
        }

        first_chain = (
            chain_input
            | self.prompt
            | combine_messages
            | self.model
            | extract_answer_from_output
        )

        doc_result, _ = self.get_relevant_document(question)
        if doc_result:
            first_response = first_chain.invoke({
                "query": question,
                "context": doc_result.page_content
            })
        else:
            first_response = first_chain.invoke(question)

        second_chain = (
            {"original_answer": lambda x: x}
            | self.child_prompt
            | combine_messages
            | self.simplify_model
            | extract_answer_from_output
        )

        return second_chain.invoke(first_response)

    def run_with_custom_prompt(
        self,
        question: str,
        custom_prompt: str,
        max_length: int = 1024,
        truncation: bool = True,
        repetition_penalty: float = 1.1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Process a question with custom prompt and generation parameters (no RAG)"""
        full_prompt = f"{custom_prompt}\n\nQuestion: {question}\nAnswer:"

        try:
            gen_config = GenerationConfig(
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
            )
            response = self._router.ask(full_prompt, config=gen_config)
            answer = response.content

            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

            if "\n\n" in answer:
                candidate = answer.split("\n\n", 1)[0].strip()
                if len(candidate) > 10:
                    answer = candidate

            return answer

        except Exception as e:
            raise Exception(f"Error generating response with custom prompt: {str(e)}")

    def _normalize_chat_messages(self, messages: list[dict]) -> list[dict]:
        """Normalize messages to roles expected by chat template (unchanged)"""
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system" and msg.get("content"):
                system_content = msg["content"]
                break

        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

        if not non_system_messages:
            return []

        normalized = []
        first_role = non_system_messages[0].get("role")

        if first_role == "assistant" and system_content:
            normalized.append({"role": "user", "content": system_content})

        for i, msg in enumerate(non_system_messages):
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "assistant" and "gemma" in str(self.model_name).lower():
                role = "model"

            if role == "user" and i == 0 and first_role == "user" and system_content:
                content = f"{system_content}\n\n{content}"

            normalized.append({"role": role, "content": content})

        return normalized

    def _extract_after_prompt(self, full_text: str, prompt: str, eos_token: str = None) -> str:
        """Extract model output after the prompt (unchanged)"""
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            answer = full_text.strip()

        if eos_token and eos_token in answer:
            answer = answer.split(eos_token)[0].strip()

        if "\n\n" in answer:
            candidate = answer.split("\n\n", 1)[0].strip()
            if len(candidate) > 10:
                answer = candidate

        return answer

    def run_chat_completion(
        self,
        messages: list,
        max_length: int = 1024,
        truncation: bool = True,
        repetition_penalty: float = 1.1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Process chat messages with generation parameters."""
        history = []
        question = ""

        normalized = self._normalize_chat_messages(messages)
        for msg in normalized:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                question = content
            elif role in ("assistant", "model"):
                history.append(Message(role=Role.ASSISTANT, content=content))

        try:
            gen_config = GenerationConfig(
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
            )
            response = self._router.ask(
                question,
                history=history or None,
                config=gen_config,
            )
            return response.content

        except Exception as e:
            raise Exception(f"Error generating chat completion: {str(e)}")