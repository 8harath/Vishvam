"""
Core RAG Pipeline Assembly for RAG Assistant Phase 1

This module implements the complete end-to-end RAG functionality orchestrating
all components: PDF parsing → Chunking → Embedding → Retrieval → LLM query.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .pdf_parser import PDFParser
from .text_splitter import TextSplitter
from .embedder import TextEmbedder
from .llm_handler import create_llm_manager
from ..config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, FALLBACK_RESPONSE,
    CONTEXT_SEPARATOR, MAX_CONTEXT_LENGTH, RAG_PROMPT_TEMPLATE,
    LLM_BACKEND, HF_TEMPERATURE, MAX_PDF_SIZE_MB
)

# Configure logging
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) Pipeline.
    
    This class orchestrates all RAG components to provide end-to-end functionality
    from PDF documents to contextual answers using local LLMs.
    """
    
    def __init__(
        self,
        llm_backend: str = LLM_BACKEND,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K_RETRIEVAL,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_pdf_size_mb: float = MAX_PDF_SIZE_MB
    ):
        """
        Initialize the RAG pipeline with all components.
        
        Args:
            llm_backend (str): LLM backend to use ('huggingface' or 'ollama')
            embedding_model (str): Name of the embedding model
            chunk_size (int): Size of text chunks in characters
            chunk_overlap (int): Overlap between consecutive chunks
            top_k (int): Number of chunks to retrieve for context
            similarity_threshold (float): Minimum similarity score for retrieval
            max_pdf_size_mb (float): Maximum PDF file size in MB
        """
        logger.info("Initializing RAG Pipeline...")
        
        # Store configuration
        self.llm_backend = llm_backend
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_pdf_size_mb = max_pdf_size_mb
        
        # Initialize components
        self._initialize_components()
        
        # Document storage and state
        self.processed_documents: Dict[str, Dict] = {}
        self.current_document_path: Optional[str] = None
        self.is_ready = False
        
        logger.info("RAG Pipeline initialized successfully!")
    
    def _initialize_components(self):
        """Initialize all RAG pipeline components."""
        try:
            logger.info("Loading RAG components...")
            
            # PDF Parser
            self.pdf_parser = PDFParser()
            
            # Text Splitter
            self.text_splitter = TextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Text Embedder
            self.embedder = TextEmbedder(model_name=self.embedding_model)
            
            # LLM Manager (use lightweight model for reliability)
            self.llm_manager = create_llm_manager(
                backend=self.llm_backend,
                hf_model="distilgpt2",  # Lightweight and reliable
                device="cpu",
                use_quantization=False
            )
            
            logger.info("All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise RuntimeError(f"Pipeline initialization failed: {str(e)}")
    
    def load_document(self, pdf_path: str) -> bool:
        """
        Load and process a PDF document through the complete pipeline.
        
        Args:
            pdf_path (str): Path to the PDF file to process
            
        Returns:
            bool: True if document was successfully loaded and processed
        """
        try:
            logger.info(f"Loading document: {pdf_path}")
            
            # Validate file
            if not self._validate_pdf_file(pdf_path):
                return False
            
            # Extract text from PDF
            logger.info("Step 1: Extracting text from PDF...")
            text_content = self.pdf_parser.extract_text_from_pdf(pdf_path)
            
            if not text_content or not text_content.strip():
                logger.error("No text content extracted from PDF")
                return False
            
            logger.info(f"Extracted {len(text_content)} characters from PDF")
            
            # Split text into chunks
            logger.info("Step 2: Splitting text into chunks...")
            chunks = self.text_splitter.chunk_text(text_content)
            
            if not chunks:
                logger.error("No chunks created from text")
                return False
            
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Generate and store embeddings
            logger.info("Step 3: Generating embeddings...")
            self.embedder.store_embeddings(chunks)
            
            # Store document information
            self.processed_documents[pdf_path] = {
                "text_content": text_content,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "file_size_mb": os.path.getsize(pdf_path) / (1024 * 1024),
                "processed_at": time.time()
            }
            
            self.current_document_path = pdf_path
            self.is_ready = True
            
            logger.info(f"Document '{pdf_path}' loaded and processed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load document '{pdf_path}': {str(e)}")
            return False
    
    def generate_answer(self, question: str, max_response_length: int = 300) -> Dict[str, Any]:
        """
        Generate an answer to a question using the complete RAG pipeline.
        
        This is the main orchestration method that coordinates:
        1. Semantic retrieval of relevant chunks
        2. Context preparation and formatting
        3. LLM query with proper prompting
        4. Response generation and formatting
        
        Args:
            question (str): User question to answer
            max_response_length (int): Maximum length of generated response
            
        Returns:
            Dict[str, Any]: Complete response with answer, context, and metadata
        """
        if not self.is_ready:
            return self._create_error_response(
                "No documents have been loaded. Please load a document first.",
                question
            )
        
        if not question or not question.strip():
            return self._create_error_response(
                "Please provide a valid question.",
                question
            )
        
        try:
            start_time = time.time()
            logger.info(f"Generating answer for question: '{question[:50]}...'")
            
            # Step 1: Retrieve relevant context
            retrieval_start = time.time()
            context_data = self._retrieve_relevant_context(question)
            retrieval_time = time.time() - retrieval_start
            
            if not context_data["chunks"]:
                return {
                    "question": question,
                    "answer": FALLBACK_RESPONSE,
                    "context": [],
                    "document_path": self.current_document_path,
                    "retrieval_time": retrieval_time,
                    "generation_time": 0,
                    "total_time": time.time() - start_time,
                    "success": True
                }
            
            # Step 2: Generate response using LLM
            generation_start = time.time()
            answer = self._generate_llm_response(
                question, 
                context_data["formatted_context"],
                max_response_length
            )
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            logger.info(f"Answer generated successfully in {total_time:.2f}s")
            
            return {
                "question": question,
                "answer": answer,
                "context": context_data["chunks"],
                "document_path": self.current_document_path,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return self._create_error_response(
                f"An error occurred while processing your question: {str(e)}",
                question,
                time.time() - start_time if 'start_time' in locals() else 0
            )
    
    def _retrieve_relevant_context(self, question: str) -> Dict[str, Any]:
        """
        Retrieve and format relevant context for the question.
        
        Args:
            question (str): User question
            
        Returns:
            Dict[str, Any]: Context data with chunks and formatted text
        """
        logger.info("Retrieving relevant context...")
        
        # Get top-k most similar chunks
        relevant_chunks = self.embedder.get_top_k_chunks(
            query=question,
            k=self.top_k,
            min_similarity=self.similarity_threshold
        )
        
        if not relevant_chunks:
            return {"chunks": [], "formatted_context": ""}
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Format context for LLM
        context_parts = []
        chunk_metadata = []
        
        for i, (chunk_text, similarity, chunk_idx) in enumerate(relevant_chunks):
            # Add similarity info to context
            context_parts.append(f"[Context {i+1}] (Relevance: {similarity:.3f})\n{chunk_text}")
            
            # Store metadata
            chunk_metadata.append({
                "index": chunk_idx,
                "text": chunk_text,
                "similarity": similarity,
                "rank": i + 1
            })
        
        # Combine all context parts
        formatted_context = CONTEXT_SEPARATOR.join(context_parts)
        
        # Truncate if too long
        if len(formatted_context) > MAX_CONTEXT_LENGTH:
            logger.warning(f"Context truncated from {len(formatted_context)} to {MAX_CONTEXT_LENGTH} characters")
            formatted_context = formatted_context[:MAX_CONTEXT_LENGTH] + "..."
        
        return {
            "chunks": chunk_metadata,
            "formatted_context": formatted_context
        }
    
    def _generate_llm_response(
        self, 
        question: str, 
        context: str, 
        max_length: int
    ) -> str:
        """
        Generate response using LLM with proper prompting.
        
        Args:
            question (str): User question
            context (str): Formatted context from retrieval
            max_length (int): Maximum response length
            
        Returns:
            str: Generated answer
        """
        logger.info("Generating LLM response...")
        
        # Create RAG prompt using template
        rag_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        # Generate response
        answer = self.llm_manager.query_llm(
            rag_prompt,
            max_length=max_length,
            temperature=HF_TEMPERATURE
        )
        
        # Clean up response
        answer = self._clean_response(answer, question)
        
        return answer
    
    def _clean_response(self, response: str, question: str) -> str:
        """
        Clean and format the LLM response.
        
        Args:
            response (str): Raw LLM response
            question (str): Original question for context
            
        Returns:
            str: Cleaned response
        """
        if not response:
            return FALLBACK_RESPONSE
        
        # Remove common artifacts
        response = response.strip()
        
        # Remove repetitive content
        lines = response.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line.strip() not in unique_lines:
                unique_lines.append(line.strip())
        
        cleaned = ' '.join(unique_lines)
        
        # Ensure reasonable length
        if len(cleaned) < 10:
            return FALLBACK_RESPONSE
        
        return cleaned
    
    def _validate_pdf_file(self, pdf_path: str) -> bool:
        """
        Validate PDF file before processing.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            bool: True if file is valid for processing
        """
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"File not found: {pdf_path}")
                return False
            
            if not pdf_path.lower().endswith('.pdf'):
                logger.error(f"File is not a PDF: {pdf_path}")
                return False
            
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            if file_size_mb > self.max_pdf_size_mb:
                logger.error(f"File too large: {file_size_mb:.1f}MB (max: {self.max_pdf_size_mb}MB)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"File validation failed: {str(e)}")
            return False
    
    def _create_error_response(
        self, 
        error_message: str, 
        question: str, 
        total_time: float = 0
    ) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "question": question,
            "answer": error_message,
            "context": [],
            "document_path": self.current_document_path,
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": total_time,
            "success": False
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the RAG pipeline.
        
        Returns:
            Dict[str, Any]: Pipeline status and configuration
        """
        return {
            "is_ready": self.is_ready,
            "current_document": self.current_document_path,
            "processed_documents": len(self.processed_documents),
            "components": {
                "pdf_parser": "initialized",
                "text_splitter": f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}",
                "embedder": f"model={self.embedding_model}",
                "llm_manager": f"backend={self.llm_backend}"
            },
            "embedding_stats": self.embedder.get_embedding_stats() if self.is_ready else {},
            "llm_status": self.llm_manager.get_status() if hasattr(self.llm_manager, 'get_status') else {},
            "configuration": {
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "max_pdf_size_mb": self.max_pdf_size_mb
            }
        }
    
    def list_processed_documents(self) -> List[Dict[str, Any]]:
        """
        Get information about all processed documents.
        
        Returns:
            List[Dict[str, Any]]: List of document information
        """
        documents = []
        for path, info in self.processed_documents.items():
            documents.append({
                "path": path,
                "chunk_count": info["chunk_count"],
                "file_size_mb": info["file_size_mb"],
                "processed_at": time.strftime(
                    "%Y-%m-%d %H:%M:%S", 
                    time.localtime(info["processed_at"])
                ),
                "is_current": path == self.current_document_path
            })
        return documents
    
    def switch_document(self, pdf_path: str) -> bool:
        """
        Switch to a previously processed document.
        
        Args:
            pdf_path (str): Path to the document to switch to
            
        Returns:
            bool: True if switch was successful
        """
        if pdf_path in self.processed_documents:
            # Restore embeddings for this document
            doc_info = self.processed_documents[pdf_path]
            self.embedder.store_embeddings(doc_info["chunks"])
            self.current_document_path = pdf_path
            self.is_ready = True
            logger.info(f"Switched to document: {pdf_path}")
            return True
        else:
            logger.error(f"Document not found in processed documents: {pdf_path}")
            return False
    
    def clear_documents(self):
        """Clear all processed documents and reset pipeline."""
        self.processed_documents.clear()
        self.current_document_path = None
        self.is_ready = False
        # Clear embeddings
        if hasattr(self.embedder, 'clear_embeddings'):
            self.embedder.clear_embeddings()
        logger.info("All documents cleared from pipeline")


def create_rag_pipeline(**kwargs) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline with custom configuration.
    
    Args:
        **kwargs: Configuration parameters for RAGPipeline
        
    Returns:
        RAGPipeline: Configured pipeline instance
    """
    return RAGPipeline(**kwargs)
