"""
Enhanced RAG Pipeline with Production-Ready Error Handling - Step 9

This module provides the complete end-to-end RAG functionality with comprehensive
error handling, retry logic, progress tracking, and resource monitoring.
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from .pdf_parser import PDFParser
from .text_splitter import TextSplitter
from .embedder import TextEmbedder
from .llm_handler import create_llm_manager
from .error_handler import (
    RAGAssistantError, DocumentProcessingError, PDFProcessingError,
    EmbeddingGenerationError, LLMProcessingError, MemoryError,
    get_logger, handle_exceptions, retry_on_failure, error_context,
    memory_monitor, ProgressTracker, ValidationUtils
)

# Import config - handle both relative and absolute imports
try:
    from ..config import (
        EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
        TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, FALLBACK_RESPONSE,
        CONTEXT_SEPARATOR, MAX_CONTEXT_LENGTH, RAG_PROMPT_TEMPLATE,
        LLM_BACKEND, HF_TEMPERATURE, MAX_PDF_SIZE_MB
    )
except ImportError:
    # Fallback to absolute import for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (
        EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
        TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, FALLBACK_RESPONSE,
        CONTEXT_SEPARATOR, MAX_CONTEXT_LENGTH, RAG_PROMPT_TEMPLATE,
        LLM_BACKEND, HF_TEMPERATURE, MAX_PDF_SIZE_MB
    )

# Configure logging
logger = get_logger()


class EnhancedRAGPipeline:
    """
    Enhanced RAG Pipeline with comprehensive error handling and robustness features.
    
    This class provides production-ready RAG functionality with:
    - Comprehensive error handling and recovery
    - Resource monitoring and limits
    - Progress tracking for long operations
    - Retry logic for transient failures
    - Input validation and sanitization
    """
    
    def __init__(
        self,
        llm_backend: str = LLM_BACKEND,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K_RETRIEVAL,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_pdf_size_mb: float = MAX_PDF_SIZE_MB,
        enable_retry: bool = True,
        max_memory_gb: float = 8.0
    ):
        """
        Initialize the enhanced RAG pipeline.
        
        Args:
            llm_backend: LLM backend to use ('huggingface' or 'ollama')
            embedding_model: Name of the embedding model
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between consecutive chunks
            top_k: Number of chunks to retrieve for context
            similarity_threshold: Minimum similarity score for retrieval
            max_pdf_size_mb: Maximum PDF file size in MB
            enable_retry: Enable retry logic for failed operations
            max_memory_gb: Maximum memory usage limit in GB
        """
        logger.info("Initializing Enhanced RAG Pipeline...")
        
        # Store configuration
        self.llm_backend = llm_backend
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_pdf_size_mb = max_pdf_size_mb
        self.enable_retry = enable_retry
        self.max_memory_gb = max_memory_gb
        
        # Pipeline state
        self.processed_documents: Dict[str, Dict] = {}
        self.current_document_path: Optional[str] = None
        self.is_ready = False
        self.initialization_errors: List[str] = []
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.total_questions_processed = 0
        self.failed_operations = 0
        
        # Initialize components with error handling
        self._initialize_components_safely()
        
        logger.info(f"Enhanced RAG Pipeline initialized. Ready: {self.is_ready}")
    
    @handle_exceptions(default_return=False, log_errors=True)
    def _initialize_components_safely(self) -> bool:
        """Initialize all pipeline components with comprehensive error handling."""
        with error_context("Component Initialization"):
            try:
                progress = ProgressTracker(4, "Component Initialization")
                
                # Initialize PDF Parser
                progress.update(1, "Initializing PDF parser...")
                self.pdf_parser = PDFParser()
                
                # Initialize Text Splitter
                progress.update(1, "Initializing text splitter...")
                self.text_splitter = TextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
                # Initialize Text Embedder with retry
                progress.update(1, "Loading embedding model...")
                if self.enable_retry:
                    self.embedder = self._initialize_embedder_with_retry()
                else:
                    self.embedder = TextEmbedder(model_name=self.embedding_model)
                
                # Initialize LLM Manager
                progress.update(1, "Loading LLM...")
                if self.enable_retry:
                    self.llm_manager = self._initialize_llm_with_retry()
                else:
                    self.llm_manager = create_llm_manager(
                        backend=self.llm_backend,
                        hf_model="distilgpt2",
                        device="cpu",
                        use_quantization=False
                    )
                
                progress.finish("All components ready")
                self.is_ready = True
                return True
                
            except Exception as e:
                error_msg = f"Component initialization failed: {e}"
                self.initialization_errors.append(error_msg)
                logger.error(error_msg)
                self.is_ready = False
                return False
    
    @retry_on_failure(max_retries=3, delay=2.0, exceptions=(Exception,))
    def _initialize_embedder_with_retry(self) -> TextEmbedder:
        """Initialize embedder with retry logic."""
        try:
            return TextEmbedder(model_name=self.embedding_model)
        except Exception as e:
            logger.warning(f"Embedder initialization attempt failed: {e}")
            raise EmbeddingGenerationError(f"Failed to initialize embedder: {e}")
    
    @retry_on_failure(max_retries=2, delay=1.0, exceptions=(Exception,))
    def _initialize_llm_with_retry(self) -> Any:
        """Initialize LLM with retry logic."""
        try:
            return create_llm_manager(
                backend=self.llm_backend,
                hf_model="distilgpt2",
                device="cpu",
                use_quantization=False
            )
        except Exception as e:
            logger.warning(f"LLM initialization attempt failed: {e}")
            raise LLMProcessingError(f"Failed to initialize LLM: {e}")
    
    @handle_exceptions(default_return=False, reraise=True, error_type=DocumentProcessingError)
    def load_document(self, pdf_path: str) -> bool:
        """
        Load and process a PDF document with comprehensive error handling.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            bool: True if document was successfully loaded and processed
            
        Raises:
            DocumentProcessingError: If document processing fails
        """
        if not self.is_ready:
            raise DocumentProcessingError(f"Pipeline not ready. Errors: {self.initialization_errors}")
        
        with error_context(f"Document Loading: {pdf_path}"):
            try:
                # Validate input
                validated_path = ValidationUtils.validate_file_path(pdf_path, ['.pdf'])
                file_size_mb = ValidationUtils.validate_file_size(str(validated_path), self.max_pdf_size_mb)
                
                logger.info(f"Loading document: {pdf_path} ({file_size_mb:.1f} MB)")
                
                # Monitor memory usage during processing
                with memory_monitor(self.max_memory_gb):
                    return self._process_document_safely(str(validated_path))
                
            except (FileNotFoundError, ValueError) as e:
                raise PDFProcessingError(f"Document validation failed: {e}")
            except MemoryError as e:
                raise DocumentProcessingError(f"Memory limit exceeded while processing document: {e}")
            except Exception as e:
                self.failed_operations += 1
                raise DocumentProcessingError(f"Unexpected error loading document: {e}")
    
    def _process_document_safely(self, pdf_path: str) -> bool:
        """Process document with detailed progress tracking."""
        progress = ProgressTracker(5, f"Processing {Path(pdf_path).name}")
        
        try:
            # Step 1: Extract text
            progress.update(1, "Extracting text from PDF...")
            text_content = self._extract_text_safely(pdf_path)
            
            # Step 2: Split into chunks
            progress.update(1, "Splitting text into chunks...")
            chunks = self._split_text_safely(text_content)
            
            # Step 3: Generate embeddings
            progress.update(1, "Generating embeddings...")
            embeddings = self._generate_embeddings_safely(chunks)
            
            # Step 4: Build vector index
            progress.update(1, "Building search index...")
            self._build_vector_index_safely(chunks, embeddings)
            
            # Step 5: Store document information
            progress.update(1, "Finalizing document processing...")
            self._store_document_info(pdf_path, text_content, chunks, embeddings)
            
            progress.finish(f"Document ready: {len(chunks)} chunks, {len(embeddings)} embeddings")
            
            self.current_document_path = pdf_path
            return True
            
        except Exception as e:
            progress.finish(f"Failed: {e}")
            raise DocumentProcessingError(f"Document processing failed: {e}")
    
    @handle_exceptions(default_return="", reraise=True, error_type=PDFProcessingError)
    def _extract_text_safely(self, pdf_path: str) -> str:
        """Extract text with error handling and validation."""
        try:
            text_content = self.pdf_parser.extract_text_from_pdf(pdf_path)
            
            if not text_content:
                raise PDFProcessingError("No text content extracted from PDF")
            
            # Validate extracted text
            validated_text = ValidationUtils.validate_text_content(text_content, min_length=50)
            
            logger.info(f"Extracted {len(validated_text)} characters from PDF")
            return validated_text
            
        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Text extraction failed: {e}")
    
    @handle_exceptions(default_return=[], reraise=True, error_type=DocumentProcessingError)
    def _split_text_safely(self, text: str) -> List[str]:
        """Split text into chunks with validation."""
        try:
            chunks = self.text_splitter.chunk_text(text)
            
            if not chunks:
                raise DocumentProcessingError("No chunks generated from text")
            
            # Filter out very short chunks
            valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 20]
            
            if not valid_chunks:
                raise DocumentProcessingError("No valid chunks after filtering")
            
            logger.info(f"Generated {len(valid_chunks)} valid chunks from text")
            return valid_chunks
            
        except Exception as e:
            if isinstance(e, DocumentProcessingError):
                raise
            raise DocumentProcessingError(f"Text splitting failed: {e}")
    
    @handle_exceptions(default_return=[], reraise=True, error_type=EmbeddingGenerationError)
    def _generate_embeddings_safely(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings with error handling and retry."""
        try:
            if self.enable_retry:
                return self._generate_embeddings_with_retry(chunks)
            else:
                return self._generate_embeddings_direct(chunks)
                
        except Exception as e:
            if isinstance(e, EmbeddingGenerationError):
                raise
            raise EmbeddingGenerationError(f"Embedding generation failed: {e}")
    
    @retry_on_failure(max_retries=2, delay=3.0, exceptions=(Exception,))
    def _generate_embeddings_with_retry(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings with retry logic."""
        return self._generate_embeddings_direct(chunks)
    
    def _generate_embeddings_direct(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings directly."""
        embeddings = self.embedder.embed_batch(chunks)
        
        if not embeddings or len(embeddings) != len(chunks):
            raise EmbeddingGenerationError(
                f"Embedding generation failed: expected {len(chunks)}, got {len(embeddings) if embeddings else 0}"
            )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def _build_vector_index_safely(self, chunks: List[str], embeddings: List[List[float]]):
        """Build vector index with validation."""
        try:
            # Store chunks and embeddings for retrieval
            self.chunk_store = chunks
            self.embedding_store = embeddings
            
            logger.info("Vector index built successfully")
            
        except Exception as e:
            raise DocumentProcessingError(f"Vector index creation failed: {e}")
    
    def _store_document_info(self, pdf_path: str, text: str, chunks: List[str], embeddings: List[List[float]]):
        """Store document processing information."""
        self.processed_documents[pdf_path] = {
            'text_length': len(text),
            'num_chunks': len(chunks),
            'num_embeddings': len(embeddings),
            'processing_time': time.time(),
            'chunk_size': self.chunk_size,
            'overlap': self.chunk_overlap
        }
    
    @handle_exceptions(default_return=None, log_errors=True)
    def generate_answer(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Generate answer to a question with comprehensive error handling.
        
        Args:
            question: The question to answer
            
        Returns:
            Dict containing answer and metadata, or None if failed
        """
        if not self.is_ready:
            return self._create_error_response("Pipeline not initialized", question)
        
        if not self.current_document_path:
            return self._create_error_response("No document loaded", question)
        
        try:
            # Validate input
            validated_question = ValidationUtils.validate_text_content(question, min_length=3)
            
            start_time = time.time()
            
            with error_context(f"Answer Generation: {validated_question[:50]}..."):
                # Step 1: Retrieve relevant chunks
                retrieval_start = time.time()
                context_chunks = self._retrieve_context_safely(validated_question)
                retrieval_time = time.time() - retrieval_start
                
                # Step 2: Generate answer
                generation_start = time.time()
                answer = self._generate_llm_response_safely(validated_question, context_chunks)
                generation_time = time.time() - generation_start
                
                total_time = time.time() - start_time
                self.total_processing_time += total_time
                self.total_questions_processed += 1
                
                # Build response
                return self._create_success_response(
                    question=validated_question,
                    answer=answer,
                    context_chunks=context_chunks,
                    performance={
                        'total_time': total_time,
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time
                    }
                )
                
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"Answer generation failed: {e}")
            return self._create_error_response(str(e), question)
    
    @handle_exceptions(default_return=[], log_errors=True)
    def _retrieve_context_safely(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks safely."""
        try:
            if not hasattr(self, 'chunk_store') or not hasattr(self, 'embedding_store'):
                raise DocumentProcessingError("No processed document available for retrieval")
            
            # Generate query embedding
            query_embedding = self.embedder.embed_text(question)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            similarities = cosine_similarity(
                [query_embedding],
                self.embedding_store
            )[0]
            
            # Get top-k chunks
            top_indices = np.argsort(similarities)[::-1][:self.top_k]
            
            context_chunks = []
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold:
                    context_chunks.append({
                        'text': self.chunk_store[idx],
                        'score': float(similarities[idx]),
                        'index': int(idx)
                    })
            
            logger.info(f"Retrieved {len(context_chunks)} relevant chunks")
            return context_chunks
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    @handle_exceptions(default_return=FALLBACK_RESPONSE, log_errors=True)
    def _generate_llm_response_safely(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate LLM response with error handling and fallback."""
        try:
            if not context_chunks:
                logger.warning("No context chunks available, using fallback response")
                return FALLBACK_RESPONSE
            
            # Build context
            context_text = CONTEXT_SEPARATOR.join([
                chunk['text'] for chunk in context_chunks
            ])
            
            # Truncate context if too long
            if len(context_text) > MAX_CONTEXT_LENGTH:
                context_text = context_text[:MAX_CONTEXT_LENGTH] + "..."
                logger.warning(f"Context truncated to {MAX_CONTEXT_LENGTH} characters")
            
            # Build prompt
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context_text,
                question=question
            )
            
            # Generate response with retry if enabled
            if self.enable_retry:
                response = self._query_llm_with_retry(prompt)
            else:
                response = self.llm_manager.query_llm(prompt, temperature=HF_TEMPERATURE)
            
            # Validate and clean response
            if not response or not response.strip():
                logger.warning("LLM returned empty response, using fallback")
                return FALLBACK_RESPONSE
            
            # Basic response cleaning
            cleaned_response = response.strip()
            if len(cleaned_response) > 1000:
                cleaned_response = cleaned_response[:1000] + "..."
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return FALLBACK_RESPONSE
    
    @retry_on_failure(max_retries=2, delay=2.0, exceptions=(Exception,))
    def _query_llm_with_retry(self, prompt: str) -> str:
        """Query LLM with retry logic."""
        try:
            return self.llm_manager.query_llm(prompt, temperature=HF_TEMPERATURE)
        except Exception as e:
            logger.warning(f"LLM query attempt failed: {e}")
            raise LLMProcessingError(f"LLM query failed: {e}")
    
    def _create_success_response(
        self, question: str, answer: str, context_chunks: List[Dict[str, Any]],
        performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create a successful response dictionary."""
        return {
            'success': True,
            'question': question,
            'answer': answer,
            'context_chunks': context_chunks,
            'performance': performance,
            'timestamp': time.time(),
            'document': self.current_document_path
        }
    
    def _create_error_response(self, error: str, question: str) -> Dict[str, Any]:
        """Create an error response dictionary."""
        return {
            'success': False,
            'error': error,
            'question': question,
            'timestamp': time.time(),
            'document': self.current_document_path
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status information."""
        return {
            'is_ready': self.is_ready,
            'current_document': self.current_document_path,
            'documents_processed': len(self.processed_documents),
            'questions_processed': self.total_questions_processed,
            'failed_operations': self.failed_operations,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': (
                self.total_processing_time / self.total_questions_processed
                if self.total_questions_processed > 0 else 0
            ),
            'initialization_errors': self.initialization_errors,
            'configuration': {
                'llm_backend': self.llm_backend,
                'embedding_model': self.embedding_model,
                'chunk_size': self.chunk_size,
                'top_k': self.top_k,
                'similarity_threshold': self.similarity_threshold
            }
        }


# Factory function for backward compatibility
def create_enhanced_rag_pipeline(**kwargs) -> EnhancedRAGPipeline:
    """Create an enhanced RAG pipeline with error handling."""
    return EnhancedRAGPipeline(**kwargs)


# Maintain backward compatibility with original interface
RAGPipeline = EnhancedRAGPipeline
create_rag_pipeline = create_enhanced_rag_pipeline
