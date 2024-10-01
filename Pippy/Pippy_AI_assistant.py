# Pippy's AI-Coding Assistant
# Uses Llama3.1 model from Ollama

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import warnings
import os

# Suppress future warnings related to deprecations
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Paths to your PDF/text documents to be retrieved in the RAG model (you can add more paths to this list)
document_paths = [
    '/home/kshitij/Downloads/AI-model/Pygame Documentation.pdf',
    '/home/kshitij/Downloads/AI-model/AI-model(Streamlitfree)/Python GTK+3 Documentation.pdf',
]


def setup_vectorstore(file_paths):
    """
    Set up a vector store from the provided document files.

    Args:
        file_paths (list): List of paths to document files.

    Returns:
        retriever: A retriever object for document retrieval.
    """
    try:
        all_documents = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"Loading document from: {file_path}")
                # Load PDF or text documents based on file extension
                if file_path.endswith(".pdf"):
                    loader = PyMuPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)

                documents = loader.load()
                print(f"Loaded {len(documents)} documents from {file_path}.")
                all_documents.extend(documents)  # Combine documents from all files
            else:
                print(f"File not found: {file_path}")

        # Create embeddings using HuggingFace or OpenAI embeddings
        embeddings = HuggingFaceEmbeddings()  # You can switch to OpenAIEmbeddings if needed

        # Create a vector store with all combined documents
        vector_store = FAISS.from_documents(all_documents, embeddings)

        # Create a retriever
        retriever = vector_store.as_retriever()
        return retriever

    except Exception as e:
        print(f"Failed to set up the retriever: {e}")
        return None


# Defining a system prompt to prioritize coding-specific responses and guide the model
system_prompt = """
You are a highly intelligent Python coding assistant with access to both general knowledge and specific Pygame documentation.
1. You only have to answer Python and GTK based coding queries.
2. Prioritize answers based on the documentation when the query is related to it. However make sure you are not biased towards documentation provided to you.
3. Make sure that you don't mention words like context or documentation stating what has been provided to you.
4. Provide step-by-step explanations wherever applicable.
5. If the documentation does not contain relevant information, use your general knowledge.
6. Always be clear, concise, and provide examples where necessary.
"""


# Set up the RAG model with system prompt
template = f"""{system_prompt}
Question: {{question}}
Answer: Let's think step by step.
"""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.1")

# Initialize the vector store retriever using multiple documents
retriever = setup_vectorstore(document_paths)

# Used to calculate relevance of a query
def get_relevant_document(query, retriever, threshold=0.5):
    """
    Check if the query is related to the document by using the retriever.

    Args:
        query (str): The user query.
        retriever: The document retriever.
        threshold (float): The confidence threshold to decide if a query is related to the document.

    Returns:
        result (dict): Retrieved document information and similarity score.
    """
    results = retriever.get_relevant_documents(query)
    if results and len(results) > 0:
        # Check the confidence score of the first result (assuming retriever returns sorted results)
        top_result = results[0]
        score = top_result.metadata.get("score", 0.0)  # Default to 0.0 if no score is found

        # Return the result and its score if it meets the threshold
        if score >= threshold:
            return top_result, score
    return None, 0.0

if retriever:
    # Combine the retriever and model into a Retrieval-Augmented Generation (RAG) chain
    rag_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)

    print("RAG model setup completed successfully!")

    while True:
        # Capture user input from the terminal
        question = input("Enter your question (or type 'exit' to quit): ").strip()
        
        if question.lower() == 'exit':
            print("Exiting the application. Goodbye!")
            break

        # Check if the query is relevant to the document content
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

else:
    print("Unable to set up the retriever. Please check the documents and try again.")