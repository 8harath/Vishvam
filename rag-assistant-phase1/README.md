# 🧠 AI-Powered Voice-Driven Multilingual RAG Assistant

## 🔍 Overview

An intelligent assistant that allows users to upload PDFs (manuals, legal docs, research papers), ask questions via **text or speech**, and receive answers **in multiple languages** — as text, voice, or AI-generated avatar video.

This assistant leverages **open-source LLMs**, **document RAG pipelines**, and **voice/multilingual support** for accessibility across diverse audiences.

---

## 🎯 Key Features

- 📄 **PDF Upload** for custom knowledge base
- 🧠 **Open-source RAG pipeline** with DeepSeek / Mistral
- 🔊 **Speech I/O**: Ask and receive answers via voice
- 🌍 **Multilingual detection + translation**
- 🧏 **AI Avatar with lip-synced answers**
- 🌐 **Streamlit UI for interactive use**

---

## 🧪 Example Use Cases

- Legal assistants for multilingual clients  
- Voice-driven appliance manuals  
- Rural education tools in local languages  
- Elderly-friendly voice assistants

---

## 🔧 Tech Stack

| Component | Tech |
|----------|------|
| LLM       | DeepSeek / Mistral via Transformers |
| Embeddings | SentenceTransformers + FAISS |
| PDF Parsing | PyPDF2 / PyMuPDF |
| STT       | SpeechRecognition / Whisper |
| TTS       | gTTS |
| Language Detection | langdetect + translate |
| UI        | Streamlit + streamlit-audio-recorder |
| Avatar    | D-ID API |

---

## 🚀 Current Phase 1 Features

- **PDF Text Extraction**: Extract and process text from PDF documents
- **Smart Text Chunking**: Split documents into semantically meaningful chunks
- **Vector Embeddings**: Generate embeddings using SentenceTransformers
- **Semantic Search**: Fast similarity search with FAISS vector store
- **Local LLM Integration**: Support for both Hugging Face Transformers and Ollama
- **Modular Architecture**: Clean, testable, and extensible codebase

## 🧩 Phase-wise Development Roadmap

---

### ✅ **Phase 1: Core RAG Logic & Basic Text Interaction** (Current Phase)
**Status**: In Development  
**Target**: July 23

#### 📌 Completed Tasks:
1. ✅ Setup VS Code project + GitHub repo  
2. ✅ Implement PDF parsing (`PyPDF2`)  
3. ✅ Add text chunking logic  
4. ✅ Generate embeddings using `SentenceTransformers`  
5. ✅ Load and query open-source LLM (DeepSeek / Mistral)  
6. ✅ Build RAG core pipeline (parse → embed → retrieve → respond)  
7. ✅ Integrate everything in a simple CLI via `main.py`

> ✅ **Current Output**: Given a PDF + user question, return context-aware text response using RAG.

---

### 🔄 **Phase 2: Speech I/O and Multilingual Support**  
**Owner**: Suhas  
**Target**: July 25

#### 📌 Planned Tasks:
1. Add microphone input using `SpeechRecognition`  
2. Add voice output using `gTTS`  
3. Implement language detection (`langdetect`)  
4. Translate queries to English using `translate`  
5. Translate answers back to user language  
6. Test full speech → RAG → speech flow

> 🎯 **Expected Output**: Ask questions via mic, get spoken answers in your language.

---

### 🔄 **Phase 3: AI Avatar Integration**  
**Owner**: Sreeya  
**Target**: July 27

#### 📌 Planned Tasks:
1. Integrate D-ID API for avatar generation  
2. Convert audio/text to lip-synced video response  
3. Modularize avatar code for use in Streamlit later

> 🎯 **Expected Output**: AI avatar delivers spoken responses visually.

---

### 🔄 **Phase 4: Streamlit Interface - Text & File UI**  
**Owner**: Kiran  
**Target**: July 28

#### 📌 Planned Tasks:
1. UI for PDF upload  
2. Text input for chat interface  
3. Display text answers in chat  
4. UI for choosing output language  
5. Test full text-to-text loop in browser

> � **Expected Output**: Browser app for PDF upload + text chat.

---

### 🔄 **Phase 5: Streamlit Interface - Voice & Avatar UI + Deployment**  
**Owner**: Vipul  
**Target**: July 28

#### 📌 Planned Tasks:
1. Add microphone component in Streamlit  
2. Output TTS audio response via player  
3. Display D-ID avatar video inline  
4. Deploy app using Streamlit Cloud / HuggingFace Spaces

> 🎯 **Expected Output**: End-to-end web demo with full voice + avatar interaction.

---

## �🏗️ Project Architecture

### Full Project Structure (All Phases)
```
ai-rag-assistant/
├── main.py                  # CLI entry point (Phase 1)
├── requirements.txt         # Python dependencies
├── README.md               # This comprehensive guide
├── config.py               # Configuration settings
├── .gitignore              # Git ignore rules
├── modules/                # Core RAG modules
│   ├── __init__.py         # Module initialization
│   ├── pdf_parser.py       # PDF text extraction (Phase 1)
│   ├── text_splitter.py    # Text chunking logic (Phase 1)
│   ├── embedder.py         # Embedding generation (Phase 1)
│   ├── vector_store.py     # FAISS vector storage (Phase 1)
│   ├── rag_pipeline.py     # Main RAG orchestration (Phase 1)
│   ├── speech_module.py    # STT/TTS functionality (Phase 2)
│   ├── multilingual.py     # Translation & lang detection (Phase 2)
│   ├── avatar_generator.py # D-ID API integration (Phase 3)
├── frontend/               # Web UI components (Phases 4 & 5)
│   ├── streamlit_ui.py     # Main Streamlit app
│   ├── audio_components.py # Voice recording widgets
│   ├── avatar_components.py# Avatar display widgets
├── sample_data/            # Test documents and data
│   ├── sample_content.txt  # Generated sample content
│   └── sample_document.html# Sample HTML document
├── tests/                  # Unit tests for all modules
├── assets/                 # Static assets (images, icons)
└── docs/                   # Additional documentation
```

### Current Phase 1 Structure
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
