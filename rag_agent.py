# Pippy's AI-Coding Assistant
# Uses Llama3.1 model from Ollama

import os
import warnings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# Suppress future warnings related to deprecations
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Document paths
document_paths = [
    '/home/kshitij/Downloads/Sugarlabs/Pippy-Activity/Pygame Documentation.pdf',
    '/home/kshitij/Downloads/AI-model/Python GTK+3 Documentation.pdf',
]

# Revised Prompt Template to avoid mentioning the source
PROMPT_TEMPLATE = """
You are a highly intelligent Python coding assistant. 
You are ONLY allowed to answer Python and GTK-based coding questions. 
1. Focus on coding-related problems, errors, and explanations.
2. Use the knowledge from the provided Pygame and GTK documentation without explicitly mentioning the documents as the source.
3. Provide step-by-step explanations wherever applicable.
4. If the documentation does not contain relevant information, use your general knowledge.
5. Always be clear, concise, and provide examples where necessary.

Question: {question}
Answer: Let's think step by step.
"""

class RAG_Agent:
    def __init__(self):
        self.model = None
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def set_model(self, model="llama3.1"):
        """Set the Llama 3.1 model from Ollama."""
        self.model = OllamaLLM(model=model)

    def get_model(self):
        """Return the LLM model."""
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

        # Using HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_documents(all_documents, embeddings)
        retriever = vector_store.as_retriever()
        return retriever


    def is_coding_query(self, question):
        """
        Check if the question is coding-related by looking for specific keywords.
        
        Args:
            question (str): The user query.
        
        Returns:
            bool: True if the query is coding-related, False otherwise.
        """
        coding_keywords = [
            'code', 'Python', 'bug', 'error', 'exception', 'syntax', 'Pygame', 'debug', 
            'program', 'programming', 'variable', 'loop', 'if', 'function', 'print', 
            'input', 'string', 'number', 'list', 'tuple', 'dictionary', 'module', 
            'import', 'turtle', 'indentation', 'syntax', 'Pygame', 'math', 'shape', 
            'for', 'while', 'repeat', 'class', 'object', 'method', 'attribute', 
            'operator', 'condition', 'true', 'false', 'boolean', 'integer', 'float', 
            'type', 'index', 'range', 'len', 'append', 'pop', 'remove', 'sort', 
            'reverse', 'random', 'sleep', 'time', 'pygame', 'GTK', 'logic', 'numbers', 'correct', 'answer'
        ]
        return any(keyword.lower() in question.lower() for keyword in coding_keywords)


    def run(self):
        """Run the main logic of the RAG agent."""
        
        # Format documents for context
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create the LCEL chain
        qa_chain = (
            {
                "context": self.retriever | format_docs, 
                "question": RunnablePassthrough()  
            }
            | self.prompt  
            | self.model   
            | StrOutputParser()  
        )

        while True:
            question = input("Hey there, let me know your coding doubt").strip()

            if not self.is_coding_query(question):
                response="Sorry, I can only assist with coding-related questions. Please ask a programming question."
                return response

            
            response = qa_chain.invoke(question)
            return response

if __name__ == "__main__":
    agent = RAG_Agent()
    agent.set_model("llama3.1")  
    agent.retriever = agent.setup_vectorstore(document_paths)  
    agent.run()

