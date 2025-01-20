# Pippy's AI-Coding Assistant
# Uses a model from HuggingFace

import os
import argparse
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline

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
    def __init__(self, model="bigscience/bloom-1b1"):
        self.model = pipeline("text-generation", model=model, max_length = 200)
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def set_model(self, model):
        self.model = pipeline("text-generation", model=model)

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
        results = self.retriever.invoke(query)

        if results:
            # Check the confidence score of the top result
            top_result = results[0]
            score = top_result.metadata.get("score", 0.0)

            if score >= threshold:
                return top_result, score
        return None, 0.0

    def run(self, question):
        # Format documents for context
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create the LCEL chain 
        # Ref. https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/
        qa_chain = (
            {
                "context": self.retriever | format_docs, 
                "question": RunnablePassthrough()  
            }
            | self.prompt  
            | (lambda x: str(x))  # Convert ChatPromptValue to string
            | self.model   
            | (lambda outputs: outputs[0]['generated_text'])  # Extract generated text
            | StrOutputParser()  
        )

        doc_result, relevance_score = self.get_relevant_document(question)
        # Classifying query based on its relevance with retrieved context
        if doc_result:
            response = qa_chain.invoke({"query": question, "context": doc_result.page_content})
        else:
            response = qa_chain.invoke(question)
        
        return response



def main():
    parser = argparse.ArgumentParser(description="Pippy's AI-Coding Assistant")
    parser.add_argument('--model', type=str, choices=[
        'bigscience/bloom-1b1',
        'facebook/opt-350m',
        'EleutherAI/gpt-neo-1.3B'
    ], default='bigscience/bloom-1b1', help='Model name to use for text generation')
    parser.add_argument('--docs', nargs='+', default=[
        './docs/Pygame Documentation.pdf',
        './docs/Python GTK+3 Documentation.pdf',
        './docs/Sugar Toolkit Documentation.pdf'
    ], help='List of document paths to load into the vector store')
    args = parser.parse_args()

    try:
        agent = RAG_Agent(model=args.model)
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