# Sugar AI: Child-Friendly AI Assistant & Python Coding Helper

A dual-purpose AI system combining child-friendly question answering with Python coding assistance using Retrieval-Augmented Generation (RAG).

## Features

### 🌟 Child-Friendly QA (main.py)
- Answers children's questions using simple language
- Automatically explains complex terms for 3-year-old understanding
- GPT-2 model with response limit (60 words)
- Safety-focused content generation

### 💻 Python Coding Assistant (rag_agent.py)
- Specialized in Pygame, GTK+, and Sugar Toolkit
- RAG system with Llama3.1 model
- Documentation sources:
  - Pygame Documentation
  - Python GTK+3 Documentation
  - Sugar Toolkit Documentation
- Step-by-step explanations
- Context-aware error troubleshooting

## Installation

### Prerequisites
- Python 3.10+
- Ollama (for Llama3.1 model)
- GPU recommended for better performance

```bash
# Clone repository
git clone https://github.com/sugarlabs/sugar-ai.git
cd sugar-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up Ollama
ollama pull llama3.1
```

## Usage

### Child-Friendly QA
```python
from main import AI_Test

assistant = AI_Test()
response = assistant.generate_bot_response("Why is the sky blue?")
print(response)
```

### Coding Assistant
Start the assistant:
```bash
python rag_agent.py
```

Example session:
```
> How to create a window in GTK+3?
Let's think step by step.
1. Import Gtk module
2. Create Window object
3. Set title and size
4. Connect delete event
5. Show all elements
...
```

## Project Structure
```
sugar-ai/
├── main.py               - Child QA with GPT-2
├── rag_agent.py          - Coding assistant with RAG
├── docs/                 - Documentation PDFs
└── requirements.txt      - Dependency list
```

## Configuration

### Document Paths (rag_agent.py)
```python
document_paths = [
    './docs/Pygame Documentation.pdf',
    './docs/Python GTK+3 Documentation.pdf',
    './docs/Sugar Toolkit Documentation.pdf'
]
```

### Prompt Templates
**Child QA:**
```python
prompt = '''
Your task is to answer children's questions using simple language.
Explain any difficult words in a way a 3-year-old can understand.
Keep responses under 60 words.
\n\nQuestion:
'''
```

**Coding Assistant:**
```python
PROMPT_TEMPLATE = """
You are a highly intelligent Python coding assistant built for kids.
You are ONLY allowed to answer Python and GTK-based coding questions.
1. Focus on coding-related problems...
"""
```

## Dependencies

### Core Technologies
- Transformers (Hugging Face)
- LangChain
- FAISS (Vector Store)
- Ollama
- PyMuPDF (PDF processing)

### Key Libraries
```txt
transformers==4.45.2
torch==2.4.1
langchain-ollama==0.3.3
faiss-cpu==1.9.0
sentence-transformers==3.1.1
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

GNU General Public License v3.0  
See [LICENSE](COPYING) for full text.

---

> **Note:** Ensure Ollama service is running before using the RAG agent  
> `ollama serve`