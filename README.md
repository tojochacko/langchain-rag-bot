# LangChain RAG Bot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain that can process multiple document types and provide intelligent answers based on your document collection.

## Features

- **Multiple Document Support**: PDF, CSV, Text files, and websites
- **Flexible Input Sources**: Individual files, directories, or web URLs
- **Local LLM Support**: Uses Ollama for embeddings and chat (with OpenAI option)
- **Persistent Vector Storage**: FAISS vector database for efficient retrieval
- **Interactive Chat Interface**: Command-line chat interface for querying documents
- **Modular Design**: Separate document processing and chat functionality

## Prerequisites

### Required Software

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Ollama models: `nomic-embed-text` and `llama3.2`

### Install Ollama Models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Python Dependencies

```bash
pip install langchain langchain-community langchain-ollama langchain-openai
pip install faiss-cpu python-dotenv
pip install pypdf beautifulsoup4 lxml  # For PDF and web support
```

## Installation

1. **Clone or download the project files**
2. **Install dependencies** (see above)
3. **Create environment file** (optional, for OpenAI):

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Project Structure

```
langchain-rag-bot/
├── document_processor.py    # Document processing and vector store creation
├── chat_interface.py        # Interactive chat interface  
├── README.md               # This file
├── .env                    # Environment variables (optional)
├── documents/              # Your document directory (optional)
│   ├── manual.pdf
│   ├── data.csv
│   └── notes.txt
└── vectors/                # Generated vector store (auto-created)
    └── faiss_index/
```

## Quick Start

### 1. Process Your Documents

First, create the vector store from your documents:

```bash
# Process files in current directory
python document_processor.py

# Process specific files
python document_processor.py --files data.txt document.pdf report.csv

# Process entire directories
python document_processor.py --dirs ./documents ./research

# Load from websites
python document_processor.py --urls https://docs.python.org/3/

# Combine multiple sources
python document_processor.py --files manual.pdf --dirs ./docs --urls https://example.com
```

### 2. Start Chatting

Launch the interactive chat interface:

```bash
python chat_interface.py
```

#### Advanced Options

```bash
python document_processor.py \
  --files data.txt \
  --dirs ./documents \
  --urls https://example.com \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --index-path custom/vector/path
```

### Customization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chunk-size` | 1000 | Size of text chunks for processing |
| `--chunk-overlap` | 200 | Overlap between consecutive chunks |
| `--index-path` | `vectors/faiss_index` | Path to save vector store |

### Chat Interface

The chat interface provides an interactive way to query your documents:


### Ollama Configuration

- **Default base URL**: `http://localhost:11434`
- **Required models**: `nomic-embed-text`, `llama3.2`
- **To change models**: Edit the model names in the respective files

## Supported File Types

| Type | Extension | Loader Used |
|------|-----------|-------------|
| Text | `.txt` | TextLoader |
| PDF | `.pdf` | PyPDFLoader |
| CSV | `.csv` | CSVLoader |
| Web | URLs | WebBaseLoader |


## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this RAG bot.

## License

This project is open source. Please check individual package licenses for dependencies.