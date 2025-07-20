# AI Assistant Instructions for VS Code Agent

## 🧠 Project Title
AI-Powered Voice-Driven Multilingual RAG Assistant

## 🎯 Goal
Build a system where users can upload PDFs, ask questions (via text or speech), and receive multilingual answers. Phase 1 focuses purely on the core RAG logic in Python.

## ✅ What You Should Help With
- Modularizing the code with clear separation of logic
- Writing reusable, testable components
- Fixing bugs in the embedding + retrieval pipeline
- Providing sample docstrings, and function tests
- Writing example usages for each module

## ⛔ What to Avoid
- Do not scaffold any speech, audio, Streamlit, or UI logic in this phase
- Do not write D-ID avatar integration logic
- Avoid non-Python suggestions unless explicitly asked

## 🔧 Libraries to Use
- PyPDF2 (for PDF text extraction)
- sentence-transformers (`all-MiniLM-L6-v2`)
- faiss-cpu (for vector similarity search)
- transformers (Hugging Face for local LLMs)
- torch (PyTorch backend)

## 📋 Current Implementation Details

### Phase 1 - RAG Pipeline (In Progress)
- **PDF Processing:** PyPDF2 for text extraction
- **Text Chunking:** 500-character chunks with overlap
- **Embeddings:** SentenceTransformers with all-MiniLM-L6-v2
- **Vector Store:** FAISS for similarity search
- **LLM Options:** 
  - Hugging Face Transformers (local)
  - Ollama integration (local)
- **Architecture:** Modular design with clear separation

### Project Structure
```
rag-assistant-phase1/
├── modules/
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── text_splitter.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── tests/
├── sample_data/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── config.py
```

## 🎯 Current Status
- [x] Project structure planning
- [x] Step 1: Foundation & Environment Setup
- [x] Step 2: PDF Text Extraction Engine
- [x] Step 3: Text Chunking System
- [x] Step 4: Embedding Generation Pipeline
- [x] Step 5: Semantic Search & Retrieval
- [ ] Step 6: Open Source LLM Integration
- [ ] Step 7: Core RAG Pipeline Assembly
- [ ] Step 8: Main Application & Demo
- [ ] Step 9: Robustness & Error Handling
- [ ] Step 10: Documentation & Deployment


