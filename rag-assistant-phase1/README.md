# RAG Assistant Phase 1

## 🎯 Project Overview
A modular Python-based Retrieval Augmented Generation (RAG) system for PDF document Q&A using open-source models and vector similarity search.

## 🚀 Features
- **PDF Text Extraction**: Extract and process text from PDF documents
- **Smart Text Chunking**: Split documents into semantically meaningful chunks
- **Vector Embeddings**: Generate embeddings using SentenceTransformers
- **Semantic Search**: Fast similarity search with FAISS vector store
- **Local LLM Integration**: Support for both Hugging Face Transformers and Ollama
- **Modular Architecture**: Clean, testable, and extensible codebase

## 🏗️ Architecture

### Project Structure
```
rag-assistant-phase1/
├── modules/
│   ├── __init__.py          # Module initialization
│   ├── pdf_parser.py        # PDF text extraction
│   ├── text_splitter.py     # Text chunking logic
│   ├── embedder.py          # Embedding generation
│   ├── vector_store.py      # FAISS vector storage
│   └── rag_pipeline.py      # Main RAG orchestration
├── tests/                   # Unit tests
├── sample_data/            # Sample PDFs and test data
├── main.py                 # CLI application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Data Flow
1. **PDF Upload** → Extract text using PyPDF2
2. **Text Processing** → Split into overlapping chunks
3. **Embedding Generation** → Convert chunks to vectors using SentenceTransformers
4. **Vector Storage** → Index vectors in FAISS for fast retrieval
5. **Query Processing** → Find relevant chunks via similarity search
6. **Answer Generation** → Use local LLM with retrieved context

## 🛠️ Installation

### Prerequisites
- Python 3.8+ (no specific version required)
- 8GB+ RAM recommended for local LLM inference
- Git for version control

### Setup Instructions

1. **Clone and navigate to project**:
```bash
git clone <repository-url>
cd rag-assistant-phase1
```

2. **Create virtual environment**:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python main.py --help
```

## 🎮 Usage

### Basic Usage
```bash
# Process a PDF and ask questions
python main.py --pdf "sample_data/sample_document.pdf" --query "What is the main topic?"

# Interactive mode
python main.py --interactive
```

### Python API Usage
```python
from modules.rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline()

# Process document
rag.load_document("path/to/document.pdf")

# Ask questions
answer = rag.query("What are the key findings?")
print(answer)
```

## 🔧 Configuration

### LLM Options
The system supports two local LLM backends:

**Option 1: Hugging Face Transformers**
```python
# config.py
LLM_BACKEND = "huggingface"
HF_MODEL_NAME = "microsoft/DialoGPT-medium"  # Lightweight option
# HF_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # More powerful
```

**Option 2: Ollama**
```python
# config.py
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "llama2:7b"
```

### Embedding Configuration
```python
# config.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=modules --cov-report=html
```

### Manual Testing
```bash
# Test individual modules
python -m modules.pdf_parser sample_data/sample_document.pdf
python -m modules.embedder --test
```

## 📊 Performance

### Benchmarks
- **PDF Processing**: ~2-5 seconds for typical documents (10-50 pages)
- **Embedding Generation**: ~1-3 seconds per 1000 chunks
- **Query Response**: ~2-8 seconds (depending on LLM size)
- **Memory Usage**: 2-8GB (varies by LLM model)

### Optimization Tips
- Use smaller LLM models for faster inference
- Increase `CHUNK_SIZE` for fewer, longer contexts
- Adjust `TOP_K_RETRIEVAL` based on document complexity

## 🐛 Troubleshooting

### Common Issues

**1. Memory Errors**
```bash
# Solution: Use smaller models or increase system RAM
# In config.py, try:
HF_MODEL_NAME = "distilgpt2"  # Lightweight alternative
```

**2. Slow Performance**
```bash
# Solution: Reduce model size or chunk count
CHUNK_SIZE = 800  # Larger chunks = fewer embeddings
TOP_K_RETRIEVAL = 3  # Fewer context chunks
```

**3. Poor Answer Quality**
```bash
# Solution: Adjust retrieval parameters
CHUNK_OVERLAP = 100  # More context overlap
TOP_K_RETRIEVAL = 7  # More relevant context
```

## 🔄 Development Workflow

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement in appropriate module
3. Add comprehensive tests in `tests/`
4. Update documentation
5. Submit pull request

### Module Development
- Follow existing code patterns
- Include docstrings with examples
- Add type hints where applicable
- Write unit tests for new functions

## 🎯 Phase 1 Completion Checklist
- [x] Project structure and environment
- [x] PDF text extraction (`pdf_parser.py`)
- [x] Text chunking system (`text_splitter.py`)
- [x] Embedding generation (`embedder.py`)
- [x] Vector storage with FAISS (`vector_store.py`)
- [x] LLM integration (both HF and Ollama)
- [x] End-to-end RAG pipeline (`rag_pipeline.py`)
- [x] CLI interface (`main.py`)
- [x] Error handling and logging
- [x] Documentation and examples

## 🚀 Next Steps (Future Phases)
- **Phase 2**: Multilingual support and speech integration
- **Phase 3**: Web interface with Streamlit
- **Phase 4**: Advanced features (memory, chat history)

## 🤝 Contributing
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request with clear description

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for efficient, local RAG implementations**
