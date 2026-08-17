# 🤖 Agentic RAG System

An advanced Retrieval-Augmented Generation (RAG) system that uses intelligent agents with hybrid search.

## 🚀 Features

- **Hybrid Search**: Combines semantic (dense) + keyword (sparse) search
- **Multi-Agent Workflow**: Built with LangGraph for intelligent reasoning
- **Parent-Child Chunking**: Optimal retrieval with context preservation
- **Local LLM**: Runs with Ollama (no API costs, full privacy)
- **Interactive UI**: Clean Streamlit interface

## 📁 Project Structure
agentic-rag-project/
├── app/ # Core application
├── data/ # Document storage
├── ui/ # Streamlit interface
├── requirements.txt
└── run.py # Entry point

## 🛠️ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd agentic-rag-project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama serve