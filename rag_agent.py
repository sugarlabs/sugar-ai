# Pippy's AI-Coding Assistant
# Date: 2023-02-15
# Uses Llama3.1 model from Ollama

import os
import warnings, logging

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


# Suppress future warnings related to deprecations
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

document_paths = [
    './docs/Pygame Documentation.pdf'),
    './docs/Python GTK+3 Documentation.pdf'),
]

PROMPT = """
You are a highly intelligent Python coding assistant with
access to both general knowledge and specific Pygame documentation.
1. You only have to answer Python and GTK based coding queries.
2. Prioritize answers based on the documentation when the query is related to it.
   However make sure you are not biased towards documentation provided to you.
3. Make sure that you don't mention words like context or documentation
   stating what has been provided to you.
4. Provide step-by-step explanations wherever applicable.
5. If the documentation does not contain relevant information, use your
   general knowledge.
6. Always be clear, concise, and provide examples where necessary.
"""

TEMPLATE = f"""{PROMPT}
Question: {{question}}
Answer: Let's think step by step.
"""


class RAG_Agent():
    def __init__(self):
        self.model = None
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(TEMPLATE)

    def set_model(self, model):
        self.model = OllamaLLM(model=model)

    def get_model(self):
        return self.model

    def setup_vectorstore(self, file_paths):
        """
        Set up a vector store from the provided document files.
    
        Args:
            file_paths (list): List of paths to document files.
    
        Returns:
            retriever: A retriever object for document retrieval.
        """
        all_documents = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                if file_path.endswith(".pdf"):
                    loader = PyMuPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
    
                documents = loader.load()
                all_documents.extend(documents)

        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_documents(all_documents, embeddings)
        retriever = vector_store.as_retriever()

        return retriever

    def get_relevant_document(self, query, retriever, threshold=0.5):
        """
        Check if the query is related to the document by using the retriever.
    
        Args:
            query (str): The user query.
            retriever: The document retriever.
            threshold (float): The confidence threshold to decide
                               if a query is related to the document.
    
        Returns:
            result (dict): Retrieved document information and similarity score.
        """
        self.retriever = self.setup_vectorstore(document_paths)
        results = self.retriever.invoke(query)
        if len(results) > 0:
            # Check the confidence score of the first result
            # (assuming retriever returns sorted results)
            top_result = results[0]
            score = top_result.metadata.get("score", 0.0)
    
            if score >= threshold:
                return top_result, score
        return None, 0.0
    
    def run(self):
        if not self.retriever:
            return

        # This is deprecated according to langchain docs
        # please migrate it to the new API
        # https://python.langchain.com/v0.2/docs/versions/migrating_chains/retrieval_qa/
        rag_chain = RetrievalQA.from_chain_type(
            llm=self.get_model(),
            chain_type="stuff",
            retriever=self.retriever)
    
        while True:
            # We'll need to change this as time goes on as we won't
            # get user input this way.
            question = input("Enter your question (or type 'exit' to quit): ").strip()
            
            if question.lower() == 'exit':
                print("Exiting the application. Goodbye!")
                break
    
            doc_result, relevance_score = get_relevant_document(question, retriever)
    
            if doc_result:
                print(f"Document is relevant (Score: {relevance_score:.2f}). Using document-specific response...")
                response = rag_chain({"query": question})
                if 'result' in response:
                    print(f"Document Response: {response['result']}")
                else:
                    print("No result found in documents, trying general knowledge...")
                    response = model.invoke(question)
                    print(f"Response: {response}")
            else:
                print(f"Document is not relevant (Score: {relevance_score:.2f}). Using general knowledge response...")
                response = model.invoke(question)
                print(f"Response: {response}")
