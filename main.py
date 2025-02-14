# Copyright (C) 2024 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from fastapi import FastAPI
import uvicorn
from rag_agent import RAG_Agent

app = FastAPI()

# Initialize the RAG_Agent and its vector store (retriever) for the RAG endpoint.
agent = RAG_Agent()
agent.retriever = agent.setup_vectorstore([
    './docs/Pygame Documentation.pdf',
    './docs/Python GTK+3 Documentation.pdf',
    './docs/Sugar Toolkit Documentation.pdf'
])

@app.get("/")
def root():
    return {"message": "Welcome to Sugar-AI with FastAPI!"}

@app.post("/ask")
def ask_question(question: str):
    """
    Process a question using the full RAG pipeline (LLM + retrieval).
    """
    answer = agent.run(question)
    return {"answer": answer}

@app.post("/ask-llm")
def ask_llm(question: str):
    """
    Process a question by calling the LLM directly without retrieval.
    """
    response = agent.model(question)
    answer = response[0]['generated_text'].split("Answer:")[-1].strip()
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
