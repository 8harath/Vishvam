"""
Configuration settings for RAG Assistant Phase 1
"""

import os
from typing import Optional

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

# SentenceTransformers model for generating embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Text chunking parameters
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between consecutive chunks
MAX_CHUNK_LENGTH = 1000  # maximum characters per chunk

# Vector search parameters
TOP_K_RETRIEVAL = 5  # number of chunks to retrieve for context
SIMILARITY_THRESHOLD = 0.3  # minimum similarity score for retrieval

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Choose LLM backend: "huggingface" or "ollama"
LLM_BACKEND = "huggingface"  # Change to "ollama" for Ollama integration

# Hugging Face Transformers configuration
HF_MODEL_NAME = "microsoft/DialoGPT-medium"  # Lightweight conversational model
# HF_MODEL_NAME = "distilgpt2"  # Even lighter option
# HF_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # More powerful (requires more RAM)

HF_DEVICE = "cpu"  # Use "cuda" if GPU available
HF_MAX_LENGTH = 512  # Maximum tokens for generation
HF_TEMPERATURE = 0.7  # Creativity vs consistency (0.1-1.0)

# Ollama configuration
OLLAMA_MODEL = "llama2:7b"  # Ollama model name
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL

# =============================================================================
# FILE AND PATH CONFIGURATION
# =============================================================================

# Default paths
DEFAULT_PDF_PATH = "sample_data/sample_document.pdf"
VECTOR_STORE_PATH = "vector_store_index"  # FAISS index save location
LOG_FILE_PATH = "logs/rag_assistant.log"

# File size limits
MAX_PDF_SIZE_MB = 50  # Maximum PDF file size in MB
MAX_CHUNKS_PER_DOCUMENT = 1000  # Maximum chunks to process per document

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Batch processing
EMBEDDING_BATCH_SIZE = 32  # Process embeddings in batches
PARALLEL_PROCESSING = True  # Enable multiprocessing where applicable

# Memory management
CLEAR_CACHE_AFTER_PROCESSING = True  # Clear model cache after each document
MAX_MEMORY_USAGE_GB = 8  # Maximum RAM usage (approximate)

# =============================================================================
# RAG PIPELINE CONFIGURATION
# =============================================================================

# Prompt templates
RAG_PROMPT_TEMPLATE = """
Based on the following context, please answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""

FALLBACK_RESPONSE = "I couldn't find relevant information in the document to answer your question. Please try rephrasing or ask about something else."

# Context processing
MAX_CONTEXT_LENGTH = 2000  # Maximum characters in combined context
CONTEXT_SEPARATOR = "\n\n---\n\n"  # Separator between retrieved chunks

# =============================================================================
# DEVELOPMENT AND DEBUG SETTINGS
# =============================================================================

DEBUG_MODE = False  # Enable verbose logging and debug features
SAVE_INTERMEDIATE_RESULTS = False  # Save embeddings, chunks, etc. for inspection
TIMING_ENABLED = True  # Track and log processing times

# Testing configuration
TEST_PDF_PATH = "sample_data/sample_document.pdf"
TEST_QUERIES = [
    "What is the main topic of this document?",
    "Can you summarize the key points?",
    "What are the most important findings?"
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config_summary() -> dict:
    """Return a summary of current configuration settings."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "llm_backend": LLM_BACKEND,
        "hf_model": HF_MODEL_NAME if LLM_BACKEND == "huggingface" else None,
        "ollama_model": OLLAMA_MODEL if LLM_BACKEND == "ollama" else None,
        "chunk_size": CHUNK_SIZE,
        "top_k_retrieval": TOP_K_RETRIEVAL,
        "debug_mode": DEBUG_MODE
    }

def validate_config() -> bool:
    """Validate configuration settings."""
    if LLM_BACKEND not in ["huggingface", "ollama"]:
        raise ValueError("LLM_BACKEND must be 'huggingface' or 'ollama'")
    
    if CHUNK_SIZE <= 0 or CHUNK_SIZE > MAX_CHUNK_LENGTH:
        raise ValueError(f"CHUNK_SIZE must be between 1 and {MAX_CHUNK_LENGTH}")
    
    if TOP_K_RETRIEVAL <= 0:
        raise ValueError("TOP_K_RETRIEVAL must be positive")
    
    return True

# Environment variable overrides
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", EMBEDDING_MODEL)
LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", LLM_BACKEND)
HF_MODEL_NAME = os.getenv("RAG_HF_MODEL", HF_MODEL_NAME)
DEBUG_MODE = os.getenv("RAG_DEBUG", str(DEBUG_MODE)).lower() == "true"
