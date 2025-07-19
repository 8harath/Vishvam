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
- PyMuPDF
- sentence-transformers (`all-MiniLM-L6-v2`)
- faiss-cpu
- DeepSeek LLM API (use mock if API key is missing)


