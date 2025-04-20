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
from langchain.prompts import ChatPromptTemplate
from typing import Optional, List

PROMPT_TEMPLATE = """
You are a highly intelligent Python coding assistant built for kids using the Sugar Learning Platform.
1. Focus on coding-related problems, errors, and explanations.
2. Use the knowledge from the provided Pygame, GTK, and Sugar Toolkit documentation.
3. Provide complete, clear and concise answers.
4. Your answer must be easy to understand for kids.
5. Always include Sugar-specific guidance when relevant to the question.
6. Always answer in English only.

Question: {question}
Answer:
"""

CHILD_FRIENDLY_PROMPT = """
Your task is to answer children's questions using simple language.
You will be given an answer, you will have to paraphrase it.
Explain any difficult words in a way a 5-12-years-old can understand.

Original answer: {original_answer}

Child-friendly answer:
"""

def format_docs(docs):
    """Return document content separated by newlines"""
    return "\n\n".join(doc.page_content for doc in docs)

def combine_messages(x):
    """Combine message content with newlines"""
    if hasattr(x, "to_messages"):
        return "\n".join(msg.content for msg in x.to_messages())
    return str(x)

def extract_answer_from_output(outputs):
    """Extract the answer text from model output"""
    generated_text = outputs[0]['generated_text']

    if "Child-friendly answer:" in generated_text:
        return generated_text.split("Child-friendly answer:")[-1].strip()
    
    return generated_text.split("Answer:")[-1].strip()


class RAGAgent:
    """Retrieval-Augmented Generation agent for Sugar-AI"""
    
    def __init__(self, model: str = "Qwen/Qwen2-1.5B-Instruct", quantize: bool = True):
        # disable quantization if CUDA is not available
        self.use_quant = quantize and torch.cuda.is_available()
        self.model_name = model
        
        if self.use_quant:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                max_length=1024,
                truncation=True,
            )
            
            self.simplify_model = pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                max_length=1024,
                truncation=True,
            )
        else:
            self.model = pipeline(
                "text-generation",
                model=model,
                max_length=1024,
                truncation=True,
                torch_dtype=torch.float16,
                device=0 if torch.cuda.is_available() else -1,
            )

            self.simplify_model = self.model

        self.retriever: Optional[FAISS] = None
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.child_prompt = ChatPromptTemplate.from_template(CHILD_FRIENDLY_PROMPT)

    def set_model(self, model: str) -> None:
        """Update the model used by the agent"""
        self.model_name = model
        self.model = pipeline(
            "text-generation",
            model=model,
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
