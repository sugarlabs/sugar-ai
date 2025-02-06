from fastapi import FastAPI, HTTPException, Request, APIRouter
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import os

# Document paths loaded in the RAG model
document_paths = [
    './docs/Pygame Documentation.pdf',
    './docs/Python GTK+3 Documentation.pdf',
    './docs/Sugar Toolkit Documentation.pdf'
]

PROMPT_TEMPLATE = """
You are a highly intelligent Python coding assistant built for kids. 
You are ONLY allowed to answer Python and GTK-based coding questions. 
1. Focus on coding-related problems, errors, and explanations.
2. Use the knowledge from the provided Pygame and GTK documentation without explicitly mentioning the documents as the source.
3. Provide step-by-step explanations wherever applicable.
4. If the documentation does not contain relevant information, use your general knowledge.
5. Always be clear, concise, and provide examples where necessary.
6. Your answer must be easy to understand for the kids.

Context: {context}
Question: {question}
Answer: Let's think step by step.
"""

router = APIRouter()

class Pippy_RAG_Agent:
    def __init__(self, model="llama3.1"):
        self.model = OllamaLLM(model=model)
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    def set_model(self, model):
        self.model = OllamaLLM(model=model)

    def get_model(self):
        return self.model

    def setup_vectorstore(self, file_paths):
        """
        Set up a vector store from the provided document files.
        """
        all_documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        for file_path in file_paths:
            if os.path.exists(file_path):
                if file_path.endswith(".pdf"):
                    loader = PyMuPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)

                documents = loader.load()
                for doc in documents:
                    chunks = text_splitter.split_text(doc.page_content)
                    for chunk in chunks:
                        all_documents.append(Document(page_content=chunk, metadata=doc.metadata))

        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_documents(all_documents, embeddings)
        self.retriever = vector_store.as_retriever()

    def get_relevant_document(self, query, threshold=0.5):
        """
        Check if the query is related to a document using the retriever.

        Args:
            query (str): The user query.
            threshold (float): The confidence threshold to decide
                               if a query is related to the document.
        
        Returns:
            tuple: (top_result, score) if relevant document found, otherwise (None, 0.0)
        """
        if not self.retriever:
            raise ValueError("Retriever is not set up.")

        results = self.retriever.invoke(query)
        
        if results:
            top_result = results[0]
            score = top_result.metadata.get("score", 0.0)

            if score >= threshold:
                return top_result, score
        return None, 0.0

    def format_docs(self, docs):
        """
        Format retrieved documents for input into the model prompt.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_qa_chain(self):
        """
        Create a dynamic QA chain using LangChain primitives- LCEL chain.
        """
        return (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )
    
    def format_prompt(self, question, context):
        """
        Formatting the input prompt dynamically.
        """
        return self.prompt.format(question=question, context=context)
     
    def qa_chain(self, question):
        """
        Process a question using retrieved documents and return a response.
        """
        if not self.retriever:
            raise ValueError("Retriever is not set up.")
        
        # Retrieve relevant documents
        results = self.retriever.invoke(question)
        context = self.format_docs(results) if results else "No relevant documents found."

        # Use the dynamic QA chain
        chain = self.create_qa_chain()
        response = chain.invoke({"context": context, "question": question})
        return response
    
    def run(self, question):
        """
        Process a question and return a response using relevant documents or general knowledge.
        """
        if not self.retriever:
            raise ValueError("Retriever is not set up.")
        
        relevant_doc, score = self.get_relevant_document(question)
        if relevant_doc and score > 0.5:
            response = self.qa_chain(question)
        else:
            response = self.model.invoke(question)
        return response

class Query(BaseModel):
    question: str

agent = Pippy_RAG_Agent()
agent.setup_vectorstore(document_paths)

# Define API routes
@router.get("/")
async def root():
    return {"message": "Welcome to Pippy AI Assistant"}

@router.post("/query")
async def handle_query(query: Query, request: Request):
    try:
        response = agent.run(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

