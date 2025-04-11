# Pippy's AI-Coding Assistant
# Uses a model from HuggingFace with optional 4-bit quantization

import os
import re
import argparse
import torch
from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate


PROMPT_TEMPLATE = """
You are a smart and helpful assistant designed to answer coding questions using context given.

1. The context above contains the relevant information needed to answer this specific question.
2. Use the information from this context to formulate your answer.
3. Prioritize and include any relevant details from the context.
4. Always answer in clear, complete and helpful way.

Context: {context}
Question: {question}

Answer:
"""

CHILD_FRIENDLY_PROMPT = """
You are a helpful assistant who rewrites answers so that children aged 3 to 10 can understand them.

Your task:
- ONLY rewrite the Original answer using simple words and short sentences.
- Do NOT explain what the answer means.
- Do NOT explain what you're doing.
- Do NOT say anything extra.
- Do NOT repeat ideas.
- Do NOT add tips or encouragement.

Just give the rewritten answer. Nothing else.

Original answer:
{original_answer}

Child-friendly answer:
"""

def format_docs(docs):
    """Return all document content separated by two newlines."""
    return "\n\n".join(doc.page_content for doc in docs)

def trim_incomplete_sentence(text):
    matches = list(re.finditer(r'\.\s', text))
    if matches:
        last_complete = matches[-1].end()
        return text[:last_complete].strip()
    else:
        return text.strip()
    
def combine_messages(x):
    """
    If 'x' has a method to_messages, combine message content with newline.
    Otherwise, return string representation.
    """
    if hasattr(x, "to_messages"):
        return "\n".join(msg.content for msg in x.to_messages())
    return str(x)


def extract_answer_from_output(outputs):
    """
    Extract the answer text from the model's output after the keyword 'Answer:'.
    """
    generated_text = outputs[0]['generated_text']

    if "Child-friendly answer:" in generated_text:
        return generated_text.split("Child-friendly answer:")[-1].strip()
    
    return generated_text.split("Answer:")[-1].strip()


class RAG_Agent:
    def __init__(self, model="Qwen/Qwen2-1.5B-Instruct",
                 quantize=True):
        # Disable quantization if CUDA is not available
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
                max_new_tokens=1024,
                temperature=0.3, 
                truncation=True
            )
            
            tokenizer2 = AutoTokenizer.from_pretrained(model)
            self.simplify_model = pipeline(
                "text-generation",
                model=model_obj,  
                tokenizer=tokenizer2,
                max_new_tokens=1024,
                temperature=0.3,  
                truncation=True
            )
        else:
            self.model = pipeline(
                "text-generation",
                model=model,
                max_new_tokens=1024, 
                temperature=0.3, 
                truncation=True,
                torch_dtype=torch.float16,
                device=0 if torch.cuda.is_available() else -1,
            )

            self.simplify_model = pipeline(
                "text-generation",
                model=model,
                max_new_tokens=1024, 
                temperature=0.3, 
                truncation=True,
                torch_dtype=torch.float16,
                device=0 if torch.cuda.is_available() else -1,
            )

        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.child_prompt = ChatPromptTemplate.from_template(CHILD_FRIENDLY_PROMPT)

    def set_model(self, model):
        # Update both models
        self.model_name = model
        self.model = pipeline(
            "text-generation",
            model=model,
            max_new_tokens=1024, 
            temperature=0.3, 
            truncation=True,
            torch_dtype=torch.float16
        )
        
        self.simplify_model = pipeline(
            "text-generation",
            model=model,
            max_new_tokens=1024,
            temperature=0.3,   
            truncation=True,
            torch_dtype=torch.float16
        )

    def get_model(self):
        return self.model

    def setup_vectorstore(self, file_paths):
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
        retriever = vector_store.as_retriever()
        return retriever

    def get_relevant_document(self, query):
        results = self.retriever.invoke(query)
            
        print(f"[DEBUG] Retrieved results: {results}")
            
        if results:
            return results[:2], 1.0
        return None, 0.0

    def run(self, question):
        """
        Build the QA chain and process the output from model generation.
        Apply double prompting to make answers child-friendly.
        """
        doc_result, _ = self.get_relevant_document(question)
        context_text = format_docs(doc_result) if doc_result else ""

        if not context_text.strip():
            return "I couldn't find an answer in the documents."

        first_chain = (
            {"context": lambda x: context_text, "question": lambda x: x}
            | self.prompt
            | combine_messages
            | self.model
            | extract_answer_from_output
        )
        first_response = first_chain.invoke(question)
        print(self.prompt.format(context=context_text, question=question))
        # The chain applies: prompt -> combine messages -> model ->
        # extract answer from output.
       
        second_chain = (
            {"original_answer": lambda x: x}
            | self.child_prompt
            | combine_messages
            | self.simplify_model  
            | extract_answer_from_output
        )

        final_response = second_chain.invoke(first_response)
        return trim_incomplete_sentence(final_response)


def main():
    parser = argparse.ArgumentParser(description="Pippy's AI-Coding Assistant")
    parser.add_argument(
        '--model',
        type=str,
        choices=[
            'bigscience/bloom-1b1',
            'facebook/opt-350m',
            'EleutherAI/gpt-neo-1.3B'
        ],
        default='bigscience/bloom-1b1',
        help='Model name to use for text generation'
    )
    parser.add_argument(
        '--docs',
        nargs='+',
        default=[
            './docs/Pygame Documentation.pdf',
            './docs/Python GTK+3 Documentation.pdf',
            './docs/Sugar Toolkit Documentation.pdf'
        ],
        help='List of document paths to load into the vector store'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Enable 4-bit quantization (only works with CUDA)'
    )
    args = parser.parse_args()

    try:
        agent = RAG_Agent(model=args.model, quantize=args.quantize)
        agent.retriever = agent.setup_vectorstore(args.docs)
        while True:
            question = input("Enter your question: ").strip()
            if not question:
                print("Please enter a valid question.")
                continue
            response = agent.run(question)
            print("Response:", response)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    