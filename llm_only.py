"""
This script demonstrates a simple CLI that calls the LLM directly,
without any retrieval steps.
"""

from rag_agent import RAG_Agent

def main():
    agent = RAG_Agent()
    
    print("Enter your question for the LLM ('quit' to exit):")
    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "q", ""}:
            break
        
        response = agent.model(question)
        answer = response[0]['generated_text'].split("Answer:")[-1].strip()
        print("Answer:", answer)
        

if __name__ == "__main__":
    main()
    