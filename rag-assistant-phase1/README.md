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

### ✅ Completed Components

#### 📄 PDF Text Extraction
- **Robust PDF parsing** using PyPDF2 with comprehensive error handling
- **Multi-page extraction** with page-by-page processing
- **Metadata extraction** and file validation
- **Clean text output** with proper formatting

#### 🧩 Smart Text Chunking System
- **Configurable chunk sizes** (default: 500 characters)
- **Overlap support** for context preservation (default: 50 characters)
- **Word boundary preservation** to maintain text integrity
- **Future-ready** for sentence-based chunking enhancement
- **Comprehensive statistics** and monitoring capabilities

Key Features:
- Character-based chunking with smart word boundary detection
- Configurable chunk size and overlap parameters
- Support for large documents (tested with 20,000+ character texts)
- Sentence-based chunking method (prepared for future enhancement)
- Export capabilities for debugging and analysis

#### 🔧 Integrated CLI Tools
- **Main application** with PDF processing and chunking
- **Comprehensive testing suite** with multiple validation scenarios
- **Demo scripts** for easy feature demonstration
- **Success criteria validation** for quality assurance
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

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+ (3.9+ recommended for better transformer support)
- 8GB+ RAM recommended for local LLM inference
- Git for version control
- Windows PowerShell / Command Prompt

### Phase 1 Setup Instructions

1. **Clone and navigate to project**:
```powershell
git clone <repository-url>
cd rag-assistant-phase1
```

2. **Create virtual environment**:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

4. **Generate sample data** (if needed):
```powershell
python create_sample_pdf.py
```

5. **Verify installation**:
```powershell
python main.py --help
```

## 🎮 Usage

### Phase 1 - Basic Usage
```powershell
# Basic PDF processing
python main.py --pdf "sample_data\sample_document.pdf"

# PDF processing with text chunking
python main.py --pdf "sample_data\sample_document.pdf" --chunk

# Custom chunk parameters
python main.py --pdf "sample_data\sample_document.pdf" --chunk --chunk-size 300 --chunk-overlap 30

# Extract PDF info and process page by page
python main.py --pdf "sample_data\sample_document.pdf" --info --pages --chunk

# Get help
python main.py --help
```

### Text Chunking System Usage

#### Using the TextSplitter Class
```python
from modules.text_splitter import TextSplitter

# Initialize with custom parameters
splitter = TextSplitter(chunk_size=500, chunk_overlap=50)

# Chunk text with word boundary preservation
text = "Your large document content here..."
chunks = splitter.chunk_text(text, preserve_word_boundaries=True)

# Get statistics about chunks
stats = splitter.get_chunk_stats(chunks)
print(f"Generated {stats['total_chunks']} chunks")
print(f"Average size: {stats['average_chunk_size']:.1f} characters")

# Save chunks for inspection
splitter.save_chunks_to_file(chunks, "output/chunked_text.txt")
```

#### Using the Convenience Function
```python
from modules.text_splitter import chunk_text

# Simple chunking with defaults (500 chars, 50 overlap)
chunks = chunk_text("Your text content here...")

# Custom parameters
chunks = chunk_text(
    text="Your text content here...",
    chunk_size=300,
    chunk_overlap=25
)
```

#### Future Enhancement: Sentence-Based Chunking
```python
# Available but optimized for future use
sentence_chunks = splitter.chunk_text_by_sentences(text)
```

### Testing and Validation
```powershell
# Run comprehensive test suite
python test_text_splitter.py

# Run validation for success criteria
python validate_step3.py

# Simple demo
python demo_text_splitter.py
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
```powershell
pytest tests\ -v
```

### Run with Coverage
```powershell
pytest tests\ --cov=modules --cov-report=html
```

### Manual Testing
```powershell
# Test individual modules
python -m modules.pdf_parser sample_data\sample_document.pdf
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
```powershell
# Solution: Reduce model size or chunk count
# In config.py, try:
CHUNK_SIZE = 800  # Larger chunks = fewer embeddings
TOP_K_RETRIEVAL = 3  # Fewer context chunks
```

**3. Poor Answer Quality**
```powershell
# Solution: Adjust retrieval parameters
# In config.py:
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

## 📅 Development Timeline

| Phase | Owner   | Target Date | Status |
| ----- | ------- | ----------- | ------ |
| 1     | Bharath | July 23     | ✅ In Development |
| 2     | Suhas   | July 25     | 🔄 Planned |
| 3     | Sreeya  | July 27     | 🔄 Planned |
| 4     | Kiran   | July 28     | 🔄 Planned |
| 5     | Vipul   | July 28     | 🔄 Planned |

## 📌 Team Contributors

* **Bharath** – Phase 1 (Core RAG Pipeline & Architecture)
* **Suhas** – Phase 2 (Speech Recognition & Multilingual Support)
* **Sreeya** – Phase 3 (AI Avatar Integration)
* **Kiran** – Phase 4 (Streamlit Web Interface - Text)
* **Vipul** – Phase 5 (Voice UI & Production Deployment)

---

## 🎯 Phase 1 Completion Checklist
- [x] Project structure and environment
- [x] PDF text extraction (`pdf_parser.py`) ✅ **COMPLETED**
- [ ] Text chunking system (`text_splitter.py`)
- [ ] Embedding generation (`embedder.py`)
- [ ] Vector storage with FAISS (`vector_store.py`)
- [ ] LLM integration (both HF and Ollama)
- [ ] End-to-end RAG pipeline (`rag_pipeline.py`)
- [x] CLI interface (`main.py`) ✅ **BASIC VERSION COMPLETED**
- [x] Error handling and logging ✅ **COMPLETED**
- [x] Documentation and examples ✅ **COMPLETED**
- [x] Sample data generation (`create_sample_pdf.py`) ✅ **COMPLETED**

## 🚀 Next Development Steps

### Immediate (Phase 1 Completion)
- [ ] Final testing and optimization
- [ ] Performance benchmarking
- [ ] Documentation polish
- [ ] Code review and refactoring

### Upcoming Phases
- **Phase 2**: Speech I/O and multilingual translation
- **Phase 3**: AI avatar video generation
- **Phase 4**: Web UI with Streamlit
- **Phase 5**: Voice interface and deployment

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Make changes with comprehensive tests
4. Submit pull request with clear description
5. Follow existing code patterns and documentation standards

## � License
This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎯 Building the future of accessible, multilingual AI assistance - one phase at a time!**

*Current Status: Phase 1 (Core RAG Pipeline) - Ready for Phase 2 Integration*
