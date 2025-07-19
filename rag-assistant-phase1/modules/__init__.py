"""
RAG Assistant Phase 1 - Core Modules
"""

__version__ = "1.0.0"
__author__ = "RAG Assistant Team"

# Module imports for easy access
from .pdf_parser import PDFParser
from .text_splitter import TextSplitter, chunk_text

__all__ = [
    "PDFParser",
    "TextSplitter",
    "chunk_text"
]
