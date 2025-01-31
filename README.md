# Sugar 🇦🇮

Sugar AI is a **central hub for all AI-powered chatbots and assistants** integrated within Sugar. Designed to enhance interactive learning, it **brings the power of AI and LLMs** into various educational activities, making learning more engaging and accessible for children.
By seamlessly integrating AI into Sugar activities, Sugar AI fosters creativity, problem-solving, and interactive learning. Children can engage with AI chatbots for conversations, receive coding assistance through intelligent programming guides, and explore new concepts in an intuitive and friendly environment.

## Current AI Models🤖

### 1. Chatbot for the [Chat Activity](https://github.com/sugarlabs/chat)💬
An interactive AI that helps kids learn through engaging conversations.

### 2. AI Coding Assistant for the [Pippy Activity](https://github.com/sugarlabs/Pippy)👩‍💻
A Python coding assistant integrated with Pippy to support children in learning programming in more efficient way. Pippy's AI-assistant uses a Llama3.1 model from Ollama with a [RAG architecture](https://arxiv.org/pdf/2005.11401) for retrieving relevent examples from the [documentations](https://github.com/sugarlabs/sugar-ai/tree/main/docs). Specially enhanced to handle Python, GTK, and Pygame queries, making it an essential tool for beginner programmers.

## Project Structure🗂️
```
sugar-ai/
├── main.py               - Child QA with GPT-2
├── rag_agent.py          - Coding assistant with RAG
├── docs/                 - Documentation PDFs
└── requirements.txt      - Dependency list
```

## Installation🛠️

### Prerequisites
✔️ Python 3.10+

✔️ Ollama (for Llama3.1 model)

✔️ GPU recommended for better performance

```bash
# Clone repository
git clone https://github.com/sugarlabs/sugar-ai.git
cd sugar-ai
```
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```
or you can also use conda refer this [documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) to setup conda locally on your system 
```bash
# Create and activate a Conda environment
conda create --name <env-name> python=<python-version>
conda activate <env-name>
```
```bash
# Install dependencies
pip install -r requirements.txt
```

## License

GNU General Public License v3.0  
See [LICENSE](COPYING) for full text.
