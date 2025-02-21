# Pippy's AI-Coding Assistant
# Uses a model from HuggingFace with optional 4-bit quantization

import os
import argparse
import torch
from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate


PROMPT_TEMPLATE = """
You are a highly intelligent Python coding assistant built for kids.
You are ONLY allowed to answer Python and GTK-based coding questions.
1. Focus on coding-related problems, errors, and explanations.
2. Use the knowledge from the provided Pygame and GTK documentation without
   explicitly mentioning the documents as the source.
3. Provide a clear and concise answer.
4. Your answer must be easy to understand for the kids.

Question: {question}
Answer:
"""


def format_docs(docs):
    """Return all document content separated by two newlines."""
    return "\n\n".join(doc.page_content for doc in docs)


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
    Extract the answer text from the model's output after the keyword
    'Answer:'
    """
    generated_text = outputs[0]['generated_text']
    return generated_text.split("Answer:")[-1].strip()


class RAG_Agent:
    def __init__(self, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                 quantize=True):
        # Use 4-bit quantization if enabled
        if quantize:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                load_in_4bit=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                max_length=300,
                truncation=True,
            )
        else:
            self.model = pipeline(
                "text-generation",
                model=model,
                max_length=300,
                truncation=True,
                torch_dtype=torch.float16,
                device=0 if torch.cuda.is_available() else -1,
            )

        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def set_model(self, model):
        self.model = pipeline(
            "text-generation",
            model=model,
            max_length=300,
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

    def get_relevant_document(self, query, threshold=0.5):
        results = self.retriever.invoke(query)
        if results:
            top_result = results[0]
            score = top_result.metadata.get("score", 0.0)
            if score >= threshold:
                return top_result, score
        return None, 0.0

    def run(self, question):
        """
        Build the QA chain and process the output from model generation.
        """
        # Build the chain components:
        chain_input = {
            "context": self.retriever | format_docs,
            "question": RunnablePassthrough()
        }
        # The chain applies: prompt -> combine messages -> model ->
        # extract answer from output.
        chain = (
            chain_input
            | self.prompt
            | combine_messages
            | self.model
            | extract_answer_from_output
        )
        doc_result, _ = self.get_relevant_document(question)
        if doc_result:
            response = chain.invoke({
                "query": question,
                "context": doc_result.page_content
            })
        else:
            response = chain.invoke(question)
        return response


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
        help='Enable 4-bit quantization'
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


if __name__ == "__main__":
    main()
