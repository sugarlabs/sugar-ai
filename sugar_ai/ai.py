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
import sugar_ai.prompts as prompts
from sugar_ai.config import settings
from sugar_ai.core.model_router import run_model
import logging
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
        # 1) Determine provider and model with clear precedence:
        self.provider = getattr(settings, "DEFAULT_PROVIDER", "local")
        if model:
            self.model_name = model
            logger.info("Using explicit model argument: %s", self.model_name)
        else:
            if getattr(settings, "DEV_MODE", False):
                self.model_name = getattr(settings, "DEV_MODEL_NAME", settings.DEFAULT_MODEL)
                logger.info("DEV_MODE active: using lightweight model %s", self.model_name)
            else:
                self.model_name = getattr(settings, "PROD_MODEL_NAME", settings.DEFAULT_MODEL)
                logger.info("Using production model %s", self.model_name)

        # 2) Compute quantization/device choices. Skip if using cloud provider.
        if self.provider in ["openai"]:
            logger.info("Using cloud provider: %s. Skipping local model load.", self.provider)
            self.model = None
            self.simplify_model = None
            self.use_quant = False
        else:
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
                    torch_dtype=dtype, 
                    device=device,
                )
                self.simplify_model = self.model

        self.retriever: Optional[FAISS] = None
        self.prompt = ChatPromptTemplate.from_template(prompts.PROMPT_TEMPLATE)
        self.child_prompt = ChatPromptTemplate.from_template(prompts.CHILD_FRIENDLY_PROMPT)
        self.debug_prompt = ChatPromptTemplate.from_template(prompts.CODE_DEBUG_PROMPT)
        self.context_prompt = ChatPromptTemplate.from_template(prompts.CODE_CONTEXT_PROMPT)
        self.kids_debug_prompt = ChatPromptTemplate.from_template(prompts.KIDS_DEBUG_PROMPT)
        self.kids_context_prompt = ChatPromptTemplate.from_template(prompts.KIDS_CONTEXT_PROMPT)

    def set_model(self, model: str, provider: str = "openai") -> None:
        """Update the model and provider used by the agent"""
        self.model_name = model
        self.provider = provider
        logger.info(f"Updated agent to model {model} using provider {provider}")
        
        # Optionally re-load local model if switching to local/hf
        if self.provider in ["local", "huggingface"]:
            self.model = pipeline(
                "text-generation",
                model=self.model_name,
                max_length=1024,
                truncation=True,
                torch_dtype=torch.float16
            )
            self.simplify_model = self.model
        else:
            self.model = None
            self.simplify_model = None

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
        Debugging logic using the modular router.
        """
        config = {"model": self.model_name}
        
        if context:
            # Contextualization
            prompt1 = prompts.CODE_CONTEXT_PROMPT.format(code=code)
            res1 = run_model(prompt1, provider=self.provider, config=config)
            context_output = res1.get("response") if isinstance(res1, dict) else str(res1)
            
            prompt2 = prompts.KIDS_CONTEXT_PROMPT.format(context_output=context_output)
            res2 = run_model(prompt2, provider=self.provider, config=config)
            return res2.get("response") if isinstance(res2, dict) else str(res2)
        
        # Debugging
        prompt1 = prompts.CODE_DEBUG_PROMPT.format(code=code)
        res1 = run_model(prompt1, provider=self.provider, config=config)
        debug_output = res1.get("response") if isinstance(res1, dict) else str(res1)
        
        prompt2 = prompts.KIDS_DEBUG_PROMPT.format(debug_output=debug_output)
        res2 = run_model(prompt2, provider=self.provider, config=config)
        return res2.get("response") if isinstance(res2, dict) else str(res2)

    def run(self, question: str) -> str:
        """Process a question through the RAG pipeline using the modular router"""
        # Retrieval
        doc_result, _ = self.get_relevant_document(question)
        context = doc_result.page_content if doc_result else "No context available."
        
        config = {"model": self.model_name}
        
        # Step 1: Initial Answer
        # Note: Since the original template didn't have {context}, we'll stick to what the router supports
        # or update the prompt here.
        full_question = f"Context: {context}\n\nQuestion: {question}"
        res1 = run_model(full_question, provider=self.provider, config=config)
        first_response = res1.get("response") if isinstance(res1, dict) else str(res1)
        
        # Step 2: Make child-friendly
        child_prompt = prompts.CHILD_FRIENDLY_PROMPT.format(original_answer=first_response)
        res2 = run_model(child_prompt, provider=self.provider, config=config)
        final_response = res2.get("response") if isinstance(res2, dict) else str(res2)
        
        return final_response

    def run_with_custom_prompt(self, question: str, custom_prompt: str, 
                             **kwargs) -> str:
        """Process a question with custom prompt using the router"""
        full_prompt = f"{custom_prompt}\n\nQuestion: {question}\nAnswer:"
        
        config = {"model": self.model_name, **kwargs}
        res = run_model(full_prompt, provider=self.provider, config=config)
        return res.get("response") if isinstance(res, dict) else str(res)

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


    def run_chat_completion(self, messages: list, **kwargs) -> str:
        """
        Process chat messages with chat format and generation parameters using the router.
        """
        if self.provider == "openai":
            last_msg = messages[-1]["content"] if messages else ""
            res = run_model(last_msg, provider=self.provider, config={"model": self.model_name, **kwargs})
            return res.get("response") if isinstance(res, dict) else str(res)
        else:
            # Fallback for local/hf: combine messages
            combined = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            res = run_model(combined, provider=self.provider, config={"model": self.model_name, **kwargs})
            return res.get("response") if isinstance(res, dict) else str(res)
