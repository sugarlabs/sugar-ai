# Pippy's AI-Coding Assistant
# Uses Llama3.1 model from Ollama

import os
import warnings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# Document paths loaded in the RAG model
document_paths = [
    './docs/Pygame Documentation.pdf',
    './docs/Python GTK+3 Documentation.pdf',
    './docs/Sugar Toolkit Documentation.pdf'
]

# Revised Prompt Template to avoid mentioning the source
PROMPT_TEMPLATE = """
You are a highly intelligent Python coding assistant built for kids. 
You are ONLY allowed to answer Python and GTK-based coding questions. 
1. Focus on coding-related problems, errors, and explanations.
2. Use the knowledge from the provided Pygame and GTK documentation without explicitly mentioning the documents as the source.
3. Provide step-by-step explanations wherever applicable.
4. If the documentation does not contain relevant information, use your general knowledge.
5. Always be clear, concise, and provide examples where necessary.
6. Your answer must be easy to understand for the kids.

Question: {question}
Answer: Let's think step by step.
"""

class RAG_Agent:
    def __init__(self, model="llama3.1"):
        self.model = OllamaLLM(model=model)
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def set_model(self, model):
        self.model = OllamaLLM(model=model)

    def get_model(self):
        return self.model
    
    # Loading the docs for retrieval in Vector Database
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
            try:
                if os.path.exists(file_path):
                    print(f"Loading {file_path}...")
                    if file_path.endswith(".pdf"):
                        loader = PyMuPDFLoader(file_path)
                    else:
                        loader = TextLoader(file_path)

                    documents = loader.load()
                    all_documents.extend(documents)
                else:
                    warnings.warn(f"File {file_path} not found.")
            except Exception as e:
                warnings.warn(f"Error processing file {file_path}: {str(e)}")
            
        if all_documents:
            embeddings = HuggingFaceEmbeddings()
            vector_store = FAISS.from_documents(all_documents, embeddings)
            retriever = vector_store.as_retriever()
            print("Document loading complete.")
            return retriever
        else:
            raise ValueError("No valid documents found to load.")
    
    def get_relevant_document(self, query, threshold=0.5):
        """
        Check if the query is related to a document by using the retriever.
        
        Args:
            query (str): The user query.
            threshold (float): The confidence threshold to decide
                               if a query is related to the document.
        
        Returns:
            tuple: (top_result, score) if relevant document found, otherwise (None, 0.0)
        """
        try:
            results = self.retriever.invoke(query)
            if results:
                # Check the confidence score of the top result
                top_result = results[0]
                score = top_result.metadata.get("score", 0.0)

                if score >= threshold:
                    return top_result, score
            return None, 0.0
        except Exception as e:
            warnings.warn(f"Error retrieving relevant document: {str(e)}")
            return None, 0.0

    def run(self):
        """
        Main loop to interact with the user.
        """
        print("Ask your Python or GTK-based coding questions! Type 'exit' to quit.")
        
        while True:
            question = input().strip()

            if question.lower() == 'exit':
                print("Exiting the assistant...")
                break

            doc_result, relevance_score = self.get_relevant_document(question)
            if doc_result:
                response = qa_chain.invoke({"query": question, "context": doc_result.page_content})
            else:
                response = qa_chain.invoke(question)
            
            print(response)


if __name__ == "__main__":
    try:
        agent = RAG_Agent()
        agent.retriever = agent.setup_vectorstore(document_paths)
        agent.run()
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

