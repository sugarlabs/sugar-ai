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
from app.token_tracker import TokenTracker

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

    # Fallback: return the full generated text trimmed.
    return generated_text.strip()


class RAGAgent:
    """Retrieval-Augmented Generation agent for Sugar-AI"""

    def __init__(self, model: Optional[str] = None, quantize: bool = True):
        # 1) Determine model name with clear precedence:
        #    explicit argument > DEV_MODEL_NAME (if DEV_MODE) > PROD_MODEL_NAME > DEFAULT_MODEL
        if model:
            self.model_name = model
            logger.info("Using explicit model argument: %s", self.model_name)
        else:
            if getattr(settings, "DEV_MODE", False):
                # prefer DEV_MODEL_NAME, then fallback to DEFAULT_MODEL
                self.model_name = getattr(
                    settings, "DEV_MODEL_NAME", settings.DEFAULT_MODEL
                )
                logger.info(
                    "DEV_MODE active: using lightweight model %s", self.model_name
                )
            else:
                # production: prefer PROD_MODEL_NAME, else DEFAULT_MODEL
                self.model_name = getattr(
                    settings, "PROD_MODEL_NAME", settings.DEFAULT_MODEL
                )
                logger.info("Using production model %s", self.model_name)

        # 2) Compute quantization/device choices. Keep quantization off in DEV_MODE by default.
        self.use_quant = (
            quantize
            and torch.cuda.is_available()
            and not getattr(settings, "DEV_MODE", False)
        )
        device = (
            0
            if torch.cuda.is_available() and not getattr(settings, "DEV_MODE", False)
            else -1
        )
        dtype = torch.float16 if device == 0 else torch.float32

        if self.use_quant:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model_obj = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                truncation=True,
            )

            self.simplify_model = pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                truncation=True,
            )
        else:
            self.model = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=1024,
                truncation=True,
                torch_dtype=dtype,  # Use the dynamic dtype
                device=device,  # Use the dynamic device
            )

            self.simplify_model = self.model

        self.retriever: Optional[FAISS] = None
        self.prompt = ChatPromptTemplate.from_template(prompts.PROMPT_TEMPLATE)
        self.child_prompt = ChatPromptTemplate.from_template(
            prompts.CHILD_FRIENDLY_PROMPT
        )
        self.debug_prompt = ChatPromptTemplate.from_template(prompts.CODE_DEBUG_PROMPT)
        self.context_prompt = ChatPromptTemplate.from_template(
            prompts.CODE_CONTEXT_PROMPT
        )
        self.kids_debug_prompt = ChatPromptTemplate.from_template(
            prompts.KIDS_DEBUG_PROMPT
        )
        self.kids_context_prompt = ChatPromptTemplate.from_template(
            prompts.KIDS_CONTEXT_PROMPT
        )
        self.token_tracker = TokenTracker(self.model_name)

    def set_model(self, model: str) -> None:
        """Update the model used by the agent"""
        self.model_name = model
        self.model = pipeline(
            "text-generation",
            model=self.model_name,
            max_length=1024,
            truncation=True,
            torch_dtype=torch.float16,
        )

        self.simplify_model = self.model

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
        # self.token_tracker = TokenTracker(self.model_name)
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
        """
        Debugging chain (dual-chain):
        Chain 1 - Debugging suggestion: code → debug prompt → combine → model → extract answer
        Chain 2 - Kid friendly formatting: answer → kids_debug prompt → combine → model → extract answer
        """
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

        """
        Contextualization chain (dual-chain):
        Chain 1 - Context generation: code → context prompt → combine → model → extract answer
        Chain 2 - Kid friendly formatting: answer → kids_context prompt → combine → model → extract answer
        """
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
            context_response = context_chain.invoke({"code": code})
            return context_response

        debug_response = debug_chain.invoke({"code": code})
        return debug_response

    def run(self, question: str, api_key: str = None, user_name: str = None) -> tuple:
        """
        Process question through RAG pipeline

        Returns:
            Tuple of (answer, token_stats)
        """
        import time

        start_time = time.time()

        # Get documents
        # docs = self.get_relevant_documents_advanced(question, top_k=3)
        if not self.retriever:
            raise ValueError(
                "Vector store not initialized. Call setup_vectorstore() first."
            )

        docs = self.retriever.invoke(question)

        if not docs:
            docs = self.get_relevant_documents_simple(question, k=3)

        context = format_docs(docs)

        # Hard guard against hallucination
        if not context.strip() or len(context) < 50:
            no_info_response = (
                "I don't have information about that in the Sugar/Pygame documentation."
            )

            # Track tokens even for "no info" responses
            if api_key and user_name:
                token_stats = self.token_tracker.track_usage(
                    api_key=api_key,
                    user_name=user_name,
                    endpoint="/ask",
                    question=question,
                    prompt=question,
                    response=no_info_response,
                    model_name=self.model_name,
                    response_time=time.time() - start_time,
                )
            else:
                token_stats = {}

            return no_info_response, token_stats

        # Build prompt
        prompt_text = f"""You are a Python assistant for kids. Answer ONLY using the context below.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        # Generate response
        try:
            response = self.model(
                prompt_text,
                max_new_tokens=300,
                temperature=0.2,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )

            text = response[0]["generated_text"]

            # Extract answer
            if "ANSWER:" in text:
                answer = text.split("ANSWER:")[-1].strip()
            else:
                answer = text.replace(prompt_text, "").strip()

            # Clean output
            # answer = clean_output(answer)

            # TRACK TOKEN USAGE
            token_stats = {}
            if api_key and user_name:
                token_stats = self.token_tracker.track_usage(
                    api_key=api_key,
                    user_name=user_name,
                    endpoint="/ask",
                    question=question,
                    prompt=prompt_text,
                    response=answer,
                    model_name=self.model_name,
                    response_time=time.time() - start_time,
                )

            return answer, token_stats

        except Exception as e:
            logger.exception("Inference error occurred")
            error_response = "Sorry, I encountered an error. Please try again."
            return error_response, {}

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

        # Combine custom prompt with question
        full_prompt = f"{custom_prompt}\n\nQuestion: {question}\nAnswer:"

        # Generate response with custom parameters
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

            # Extract the answer from the generated text
            generated_text = response[0]["generated_text"]

            # Remove the original prompt from the response
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                # Fallback: remove the input prompt
                answer = generated_text.replace(full_prompt, "").strip()

            # Stop at double newlines - this is our main stopping condition, else model continues with generating next user input, which we don't want
            if "\n\n" in answer:
                # Find the first occurrence of double newlines and cut there
                double_newline_pos = answer.find("\n\n")
                answer = answer[:double_newline_pos].strip()

            return answer

        except Exception as e:
            raise Exception(f"Error generating response with custom prompt: {str(e)}")

    def _normalize_chat_messages(self, messages: list[dict]) -> list[dict]:
        """
        Normalize messages to roles expected by Gemma chat template.
        - Convert 'assistant' -> 'model'
        - Handle system message placement based on first non-system message:
        * If first message is 'user': merge system into first user message
        * If first message is 'assistant': create user message with system content, then assistant message
        """
        # Extract system content
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system" and msg.get("content"):
                system_content = msg["content"]
                break

        # Filter out system messages and find first non-system message
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

        if not non_system_messages:
            return []

        normalized = []
        first_role = non_system_messages[0].get("role")

        # If first message is assistant and we have system content, add system as user message
        if first_role == "assistant" and system_content:
            normalized.append({"role": "user", "content": system_content})

        # Process all non-system messages
        for i, msg in enumerate(non_system_messages):
            role = msg.get("role")
            content = msg.get("content", "")

            # Convert assistant to model only for Gemma-style chat templates
            if role == "assistant" and "gemma" in str(self.model_name).lower():
                role = "model"

            # Merge system into first user message (if first message is user)
            if role == "user" and i == 0 and first_role == "user" and system_content:
                content = f"{system_content}\n\n{content}"

            normalized.append({"role": role, "content": content})

        return normalized

    def _extract_after_prompt(
        self, full_text: str, prompt: str, eos_token: str = None
    ) -> str:
        """
        Return the model's generated output that follows the input prompt.
        Keeps logic minimal; optionally trims at eos token or first blank paragraph.
        """
        # Remove prompt prefix if present
        if full_text.startswith(prompt):
            answer = full_text[len(prompt) :].strip()
        else:
            answer = full_text.strip()

        # Trim on EOS token if available
        if eos_token and eos_token in answer:
            answer = answer.split(eos_token)[0].strip()

        # Conservative stop at first double newline if very long
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
        """
        Process chat messages with chat template format and generation parameters.
        """

        # Normalize messages and build prompt using tokenizer's chat template
        chat = self._normalize_chat_messages(messages)
        full_prompt = self.model.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate response with custom parameters
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

            # Extract the answer from the generated text
            generated_text = response[0]["generated_text"]

            # Extract only the new model response
            answer = self._extract_after_prompt(
                generated_text,
                full_prompt,
                getattr(self.model.tokenizer, "eos_token", None),
            )

            return answer

        except Exception as e:
            raise Exception(f"Error generating chat completion: {str(e)}")
