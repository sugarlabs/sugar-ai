from fastapi import APIRouter
from pydantic import BaseModel
import os
import warnings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define document paths
document_paths = [
    '/home/kshitij/Downloads/AI-model/Pygame Documentation.pdf',
    '/home/kshitij/Downloads/AI-model/AI-model(Streamlitfree)/Python GTK+3 Documentation.pdf',
]

# Define the Pydantic model for input
class Question(BaseModel):
    query: str

router = APIRouter()

# Helper function to set up the vector store
def setup_vectorstore(file_paths):
    try:
        all_documents = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"Loading document from: {file_path}")
                if file_path.endswith(".pdf"):
                    loader = PyMuPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)

                documents = loader.load()
                print(f"Loaded {len(documents)} documents from {file_path}.")
                all_documents.extend(documents)
            else:
                print(f"File not found: {file_path}")

        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_documents(all_documents, embeddings)
        return vector_store.as_retriever()

    except Exception as e:
        print(f"Failed to set up the retriever: {e}")
        return None

# System prompt definition
system_prompt = """
You are a highly intelligent Python coding assistant with access to both general knowledge and specific Pygame documentation.
1. You only have to answer Python and GTK based coding queries.
2. Prioritize answers based on the documentation when the query is related to it. However make sure you are not biased towards documentation provided to you.
3. Make sure that you don't mention words like context or documentation stating what has been provided to you.
4. Provide step-by-step explanations wherever applicable.
5. If the documentation does not contain relevant information, use your general knowledge.
6. Always be clear, concise, and provide examples where necessary.
"""

template = f"""{system_prompt}
Question: {{question}}
Answer: Let's think step by step.
"""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.1")

retriever = setup_vectorstore(document_paths)

if retriever:
    rag_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)
else:
    raise RuntimeError("Unable to initialize retriever. Check document paths.")

@router.post("/generate_answer")
def generate_answer(question: Question):
    try:
        # Retrieve relevant documents
        results = retriever.get_relevant_documents(question.query)
        if results:
            print("Relevant document found. Using document-specific response...")
            response = rag_chain({"query": question.query})
            return {
                "success": True,
                "response": response.get("result", "No result found.")
            }
        else:
            print("No relevant document found. Using general knowledge response...")
            response = model.invoke(question.query)
            return {
                "success": True,
                "response": response
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }