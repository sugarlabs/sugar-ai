"""
AI functionality for Sugar-AI, including RAG and LLM components.
"""
import os
import torch
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, List
import app.prompts as prompts
from app.config import settings
import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger("sugar-ai")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def combine_messages(x):
    if hasattr(x, "to_messages"):
        return "\n".join(msg.content for msg in x.to_messages())
    return str(x)


def extract_answer_from_output(outputs):
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


class RAGAgent:
    """Retrieval-Augmented Generation agent for Sugar-AI"""

    def __init__(self, model: Optional[str] = None, quantize: bool = True, use_reranker: bool = True):

        if model:
            self.model_name = model
            logger.info("Using explicit model argument: %s", self.model_name)
        else:
            if getattr(settings, "DEV_MODE", False):
                self.model_name = getattr(settings, "DEV_MODEL_NAME", settings.DEFAULT_MODEL)
            else:
                self.model_name = getattr(settings, "PROD_MODEL_NAME", settings.DEFAULT_MODEL)

        self.use_quant = quantize and torch.cuda.is_available() and not getattr(settings, "DEV_MODE", False)
        device = 0 if torch.cuda.is_available() and not getattr(settings, "DEV_MODE", False) else -1
        dtype = torch.float16 if device == 0 else torch.float32

        if self.use_quant:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model_obj = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            self.model = pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                truncation=True,
            )

            self.simplify_model = self.model

        else:
            self.model = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=1024,
                truncation=True,
                torch_dtype=dtype,
                device=device,
            )

            self.simplify_model = self.model

    
        self.use_reranker = use_reranker
        if self.use_reranker:
            logger.info("Loading reranker model")
            self.reranker = CrossEncoder("BAAI/bge-reranker-base")

        self.retriever: Optional[FAISS] = None

        self.prompt = ChatPromptTemplate.from_template(prompts.PROMPT_TEMPLATE)
        self.child_prompt = ChatPromptTemplate.from_template(prompts.CHILD_FRIENDLY_PROMPT)
        self.debug_prompt = ChatPromptTemplate.from_template(prompts.CODE_DEBUG_PROMPT)
        self.context_prompt = ChatPromptTemplate.from_template(prompts.CODE_CONTEXT_PROMPT)
        self.kids_debug_prompt = ChatPromptTemplate.from_template(prompts.KIDS_DEBUG_PROMPT)
        self.kids_context_prompt = ChatPromptTemplate.from_template(prompts.KIDS_CONTEXT_PROMPT)

    def set_model(self, model: str) -> None:
        self.model_name = model
        self.model = pipeline(
            "text-generation",
            model=self.model_name,
            max_length=1024,
            truncation=True,
            torch_dtype=torch.float16
        )

        self.simplify_model = self.model

    def setup_vectorstore(self, file_paths: List[str]) -> Optional[FAISS]:
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
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        return self.retriever

    def get_relevant_document(self, query: str):
        results = self.retriever.invoke(query)
        return results if results else []

    def rerank_documents(self, query: str, docs, top_k: int = 2):
        if not self.use_reranker or not docs:
            return docs[:top_k]

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc for doc, _ in ranked[:top_k]]

    def debug(self, code: str, context: bool) -> str:

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
        if not self.retriever:
            raise ValueError("Vector store not initialized.")

        docs = self.get_relevant_document(question)
        top_docs = self.rerank_documents(question, docs, top_k=1)

        context = format_docs(top_docs)

        chain_input = {
            "context": lambda _: context,
            "question": RunnablePassthrough()
        }

        first_chain = (
            chain_input
            | self.prompt
            | combine_messages
            | self.model
            | extract_answer_from_output
        )

        first_response = first_chain.invoke({
            "context": context,
            "question": question
        })

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

        full_prompt = f"{custom_prompt}\n\nQuestion: {question}\nAnswer:"

        try:
            response = self.model(
                full_prompt,
                max_length=max_length,
                truncation=truncation,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.model.tokenizer.eos_token_id,
            )

            generated_text = response[0]["generated_text"]

            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.replace(full_prompt, "").strip()

            if "\n\n" in answer:
                answer = answer.split("\n\n")[0].strip()

            return answer

        except Exception as e:
            raise Exception(f"Error generating response with custom prompt: {str(e)}")

    def _normalize_chat_messages(self, messages: list[dict]) -> list[dict]:

        system_content = ""

        for msg in messages:
            if msg.get("role") == "system" and msg.get("content"):
                system_content = msg["content"]
                break

        non_system_messages = [
            msg for msg in messages if msg.get("role") != "system"
        ]

        if not non_system_messages:
            return []

        normalized = []
        first_role = non_system_messages[0].get("role")

        if first_role == "assistant" and system_content:
            normalized.append({"role": "user", "content": system_content})

        for i, msg in enumerate(non_system_messages):

            role = msg.get("role")
            content = msg.get("content", "")

            # Convert assistant to model only for Gemma-style chat templates
            if role == "assistant" and "gemma" in str(self.model_name).lower():
                role = "model"

            if role == "user" and i == 0 and first_role == "user" and system_content:
                content = f"{system_content}\n\n{content}"

            normalized.append({"role": role, "content": content})

        return normalized

    def _extract_after_prompt(self, full_text: str, prompt: str, eos_token: str = None):

        if full_text.startswith(prompt):
            answer = full_text[len(prompt) :].strip()
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

        chat = self._normalize_chat_messages(messages)

        full_prompt = self.model.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            response = self.model(
                full_prompt,
                max_length=max_length,
                truncation=truncation,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.model.tokenizer.eos_token_id,
            )

            generated_text = response[0]["generated_text"]

            answer = self._extract_after_prompt(
                generated_text,
                full_prompt,
                getattr(self.model.tokenizer, "eos_token", None),
            )

            return answer

        except Exception as e:
            raise Exception(f"Error generating chat completion: {str(e)}")
