from rag_agent import RAG_Agent

def test_rag_agent():
    agent = RAG_Agent(model="bigscience/bloom-1b1")

    document_paths = [
        './docs/Pygame Documentation.pdf',
        './docs/Python GTK+3 Documentation.pdf',
        './docs/Sugar Toolkit Documentation.pdf'
    ]
    agent.retriever = agent.setup_vectorstore(document_paths)

    question = "How do I create a Pygame window?"
    response = agent.run(question)

    print("Question:", question)
    print("Response:", response)

if __name__ == "__main__":
    test_rag_agent()