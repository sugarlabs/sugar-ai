"""
AI functionality for Sugar-AI, including RAG and LLM components.
"""
import os
import logging
import typing
from typing import Optional, List
import app.prompts as prompts
from app.config import settings

logger = logging.getLogger("sugar-ai")

# Lazy imports for heavy libraries
torch = None
transformers = None
FAISS = None
HuggingFaceEmbeddings = None
PyMuPDFLoader = None
TextLoader = None
RunnablePassthrough = None
ChatPromptTemplate = None

def _lazy_load_deps():
    """Lazily load heavy AI dependencies only when needed"""
    global torch, transformers, FAISS, HuggingFaceEmbeddings, PyMuPDFLoader, TextLoader, RunnablePassthrough, ChatPromptTemplate
    if torch is None:
        import torch as _torch
        torch = _torch
    if transformers is None:
        import transformers as _transformers
        transformers = _transformers
    if FAISS is None:
        from langchain_community.vectorstores import FAISS as _FAISS
        FAISS = _FAISS
    if HuggingFaceEmbeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
        HuggingFaceEmbeddings = _HuggingFaceEmbeddings
    if PyMuPDFLoader is None:
        from langchain_community.document_loaders import PyMuPDFLoader as _PyMuPDFLoader
        PyMuPDFLoader = _PyMuPDFLoader
    if TextLoader is None:
        from langchain_community.document_loaders import TextLoader as _TextLoader
        TextLoader = _TextLoader
    if RunnablePassthrough is None:
        from langchain_core.runnables import RunnablePassthrough as _RunnablePassthrough
        RunnablePassthrough = _RunnablePassthrough
    if ChatPromptTemplate is None:
        from langchain_core.prompts import ChatPromptTemplate as _ChatPromptTemplate
        ChatPromptTemplate = _ChatPromptTemplate

class ModelManager:
    """Manages model loading and caching to optimize memory usage."""
    _models = {}
    _tokenizers = {}

    @classmethod
    def get_model(cls, model_name: str, quantize: bool = True):
        """Load and return model and tokenizer (cached)"""
        if settings.DEV_MODE:
            logger.info("DEV_MODE active: skipping actual model load in ModelManager")
            return None, None
            
        _lazy_load_deps()
        
        if model_name in cls._models:
            return cls._models[model_name], cls._tokenizers[model_name]

        logger.info(f"Loading model: {model_name}")
        
        # Determine device and dtype
        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if device == 0 else torch.float32
        
        if quantize and device == 0:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model_obj = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            pipe = transformers.pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                truncation=True,
            )
        else:
            pipe = transformers.pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=1024,
                truncation=True,
                torch_dtype=dtype,
                device=device,
            )
            tokenizer = pipe.tokenizer

        cls._models[model_name] = pipe
        cls._tokenizers[model_name] = tokenizer
        return pipe, tokenizer

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
        self.model = None
        self.simplify_model = None
        self.retriever: Optional[typing.Any] = None
        self.quantize = quantize
        
        # Determine model name with clear precedence
        if model:
            self.model_name = model
        else:
            if settings.DEV_MODE:
                self.model_name = getattr(settings, "DEV_MODEL_NAME", settings.DEFAULT_MODEL)
                logger.info("DEV_MODE active: using placeholder for agent model")
            else:
                self.model_name = getattr(settings, "PROD_MODEL_NAME", settings.DEFAULT_MODEL)
        
        _lazy_load_deps()
        self.prompt = ChatPromptTemplate.from_template(prompts.PROMPT_TEMPLATE)
        self.child_prompt = ChatPromptTemplate.from_template(prompts.CHILD_FRIENDLY_PROMPT)
        self.debug_prompt = ChatPromptTemplate.from_template(prompts.CODE_DEBUG_PROMPT)
        self.context_prompt = ChatPromptTemplate.from_template(prompts.CODE_CONTEXT_PROMPT)
        self.kids_debug_prompt = ChatPromptTemplate.from_template(prompts.KIDS_DEBUG_PROMPT)
        self.kids_context_prompt = ChatPromptTemplate.from_template(prompts.KIDS_CONTEXT_PROMPT)

    def ensure_model_loaded(self):
        """Ensure the underlying models are loaded (skipped in DEV_MODE)"""
        if settings.DEV_MODE:
            logger.info("DEV_MODE active: skipping actual model loading in ensure_model_loaded")
            return

        if self.model is None:
            logger.info(f"Lazily loading model for RAGAgent: {self.model_name}")
            self.model, _ = ModelManager.get_model(self.model_name, quantize=self.quantize)
            self.simplify_model = self.model


    def set_model(self, model: str) -> None:
        """Update the model used by the agent"""
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

    def run(self, question: str) -> str:
        """Process a question through the RAG pipeline"""
        # build chain components
        chain_input = {
            "context": self.retriever | format_docs,
            "question": RunnablePassthrough()
        }
        
        # first chain: prompt -> combine messages -> model -> extract answer
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

        # second chain for making answer child-friendly
        second_chain = (
            {"original_answer": lambda x: x}
            | self.child_prompt
            | combine_messages
            | self.simplify_model
            | extract_answer_from_output
        )
        
        final_response = second_chain.invoke(first_response)
        return final_response

    def run_with_custom_prompt(self, question: str, custom_prompt: str, 
                             max_length: int = 1024, truncation: bool = True,
                             repetition_penalty: float = 1.1, temperature: float = 0.7,
                             top_p: float = 0.9, top_k: int = 50) -> str:
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
            generated_text = response[0]['generated_text']
            
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


    def _extract_after_prompt(self, full_text: str, prompt: str, eos_token: str = None) -> str:
        """
        Return the model's generated output that follows the input prompt.
        Keeps logic minimal; optionally trims at eos token or first blank paragraph.
        """
        # Remove prompt prefix if present
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
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


    def run_chat_completion(self, messages: list, 
                        max_length: int = 1024, truncation: bool = True,
                        repetition_penalty: float = 1.1, temperature: float = 0.7,
                        top_p: float = 0.9, top_k: int = 50) -> str:
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
            generated_text = response[0]['generated_text']
            
            # Extract only the new model response
            answer = self._extract_after_prompt(
                generated_text,
                full_prompt,
                getattr(self.model.tokenizer, "eos_token", None),
            )
            
            return answer
            
        except Exception as e:
            raise Exception(f"Error generating chat completion: {str(e)}")
