"""
Embedding Generation Pipeline for RAG Assistant Phase 1

This module provides functionality to convert text chunks into numerical 
vector representations using SentenceTransformer models for semantic search.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Text embedding generator using SentenceTransformer models.
    
    This class provides functionality to convert text chunks into dense vector
    representations that capture semantic meaning for similarity search in RAG systems.
    Enhanced with semantic search and retrieval capabilities.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the text embedder with specified model.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
                            Default: "all-MiniLM-L6-v2" (fast, lightweight, good performance)
            device (str, optional): Device to run model on ('cpu', 'cuda', 'mps')
                                   If None, automatically detects best available device
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():  # Apple Silicon
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        logger.info(f"Initializing TextEmbedder with model '{model_name}' on device '{device}'")
        
        try:
            # Load the SentenceTransformer model
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
            # Initialize storage for embeddings and chunks
            self.stored_embeddings: Optional[np.ndarray] = None
            self.stored_chunks: List[str] = []
            self.is_index_built = False
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")
    
    def embed_chunks(self, chunks: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Convert a list of text chunks into embeddings.
        
        Args:
            chunks (List[str]): List of text chunks to embed
            show_progress (bool): Whether to show progress bar during embedding
        
        Returns:
            np.ndarray: 2D array of embeddings with shape (num_chunks, embedding_dim)
        
        Raises:
            ValueError: If chunks list is empty or contains invalid data
            RuntimeError: If embedding generation fails
        """
        if not chunks:
            raise ValueError("Cannot embed empty list of chunks")
        
        # Validate chunks
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                logger.warning(f"Chunk {i} is not a string, converting to string")
                chunk = str(chunk)
            
            if not chunk.strip():
                logger.warning(f"Chunk {i} is empty or whitespace only, skipping")
                continue
                
            valid_chunks.append(chunk.strip())
        
        if not valid_chunks:
            raise ValueError("No valid chunks found after filtering")
        
        logger.info(f"Generating embeddings for {len(valid_chunks)} chunks...")
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                valid_chunks,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            
            logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text (str): Text to embed
        
        Returns:
            np.ndarray: 1D array embedding vector
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")
        
        try:
            embedding = self.model.encode(
                text.strip(),
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed single text: {str(e)}")
            raise RuntimeError(f"Single text embedding failed: {str(e)}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including name, dimension, device
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "model_max_seq_length": getattr(self.model, "max_seq_length", "Unknown")
        }
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
        
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError("Embeddings must have the same shape")
        
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def store_embeddings(self, chunks: List[str], embeddings: Optional[np.ndarray] = None) -> None:
        """
        Store text chunks and their embeddings for later retrieval.
        
        Args:
            chunks (List[str]): List of text chunks to store
            embeddings (np.ndarray, optional): Pre-computed embeddings. If None, will generate them.
        """
        if not chunks:
            raise ValueError("Cannot store empty list of chunks")
        
        # Generate embeddings if not provided
        if embeddings is None:
            logger.info("Generating embeddings for storage...")
            embeddings = self.embed_chunks(chunks)
        
        # Validate embeddings
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Store the data
        self.stored_chunks = chunks.copy()
        self.stored_embeddings = embeddings.copy()
        self.is_index_built = True
        
        logger.info(f"Stored {len(chunks)} chunks with embeddings for semantic search")
    
    def get_top_k_chunks(self, query: str, k: int = 5, min_similarity: float = 0.0) -> List[Tuple[str, float, int]]:
        """
        Retrieve top-K most similar chunks to the query.
        
        Args:
            query (str): Query text to search for
            k (int): Number of top similar chunks to retrieve
            min_similarity (float): Minimum similarity threshold (0.0 to 1.0)
        
        Returns:
            List[Tuple[str, float, int]]: List of (chunk_text, similarity_score, chunk_index)
                                         sorted by similarity score in descending order
        
        Raises:
            ValueError: If no embeddings are stored or invalid parameters
            RuntimeError: If query embedding fails
        """
        if not self.is_index_built or self.stored_embeddings is None:
            raise ValueError("No embeddings stored. Call store_embeddings() first.")
        
        if k <= 0:
            raise ValueError("k must be positive")
        
        if not (0.0 <= min_similarity <= 1.0):
            raise ValueError("min_similarity must be between 0.0 and 1.0")
        
        try:
            # Generate query embedding
            query_embedding = self.embed_single_text(query)
            
            # Compute similarities with all stored embeddings
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                self.stored_embeddings
            )[0]
            
            # Create list of (similarity, index) tuples
            chunk_similarities = [
                (float(sim), idx) for idx, sim in enumerate(similarities)
                if sim >= min_similarity
            ]
            
            # Sort by similarity (descending) and take top-k
            chunk_similarities.sort(key=lambda x: x[0], reverse=True)
            top_k_similarities = chunk_similarities[:k]
            
            # Format results as (chunk_text, similarity_score, chunk_index)
            results = [
                (self.stored_chunks[idx], sim_score, idx)
                for sim_score, idx in top_k_similarities
            ]
            
            logger.info(f"Retrieved {len(results)} chunks for query (min_sim={min_similarity:.3f})")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve top-k chunks: {str(e)}")
            raise RuntimeError(f"Chunk retrieval failed: {str(e)}")
    
    def search_similar_chunks(self, query: str, similarity_threshold: float = 0.3) -> List[Tuple[str, float, int]]:
        """
        Search for chunks above a similarity threshold.
        
        Args:
            query (str): Query text to search for
            similarity_threshold (float): Minimum similarity score threshold
        
        Returns:
            List[Tuple[str, float, int]]: All chunks above threshold, sorted by similarity
        """
        if not self.is_index_built:
            raise ValueError("No embeddings stored. Call store_embeddings() first.")
        
        # Use get_top_k_chunks with large k to get all results above threshold
        all_results = self.get_top_k_chunks(
            query=query,
            k=len(self.stored_chunks),  # Get all chunks
            min_similarity=similarity_threshold
        )
        
        return all_results
    
    def get_embedding_stats(self) -> dict:
        """
        Get statistics about stored embeddings.
        
        Returns:
            dict: Statistics including count, dimension, memory usage
        """
        if not self.is_index_built:
            return {"status": "No embeddings stored"}
        
        memory_mb = self.stored_embeddings.nbytes / (1024 * 1024)
        
        return {
            "status": "Embeddings stored",
            "chunk_count": len(self.stored_chunks),
            "embedding_dimension": self.stored_embeddings.shape[1],
            "memory_usage_mb": round(memory_mb, 2),
            "total_characters": sum(len(chunk) for chunk in self.stored_chunks),
            "avg_chunk_length": round(np.mean([len(chunk) for chunk in self.stored_chunks]), 1)
        }
    
    def clear_stored_embeddings(self) -> None:
        """Clear all stored embeddings and chunks to free memory."""
        self.stored_embeddings = None
        self.stored_chunks = []
        self.is_index_built = False
        logger.info("Cleared all stored embeddings and chunks")


def create_embedder(model_name: str = "all-MiniLM-L6-v2") -> TextEmbedder:
    """
    Factory function to create a TextEmbedder instance.
    
    Args:
        model_name (str): SentenceTransformer model name
    
    Returns:
        TextEmbedder: Initialized embedder instance
    """
    return TextEmbedder(model_name=model_name)


# Example usage and testing
if __name__ == "__main__":
    # Demo of the embedding and semantic search functionality
    print("=== Text Embedder with Semantic Search Demo ===")
    
    # Sample text chunks for a RAG system
    sample_chunks = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Natural language processing (NLP) focuses on the interaction between computers and human language, including text analysis and understanding.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Transformer models revolutionized NLP by using attention mechanisms to process sequences more effectively than traditional RNNs.",
        "Computer vision involves teaching machines to interpret and understand visual information from images and videos.",
        "Reinforcement learning trains agents to make decisions by interacting with an environment and receiving rewards or penalties.",
        "Data preprocessing is crucial for machine learning, involving cleaning, transforming, and preparing raw data for model training.",
        "Cross-validation is a technique used to evaluate machine learning models by splitting data into training and testing sets multiple times."
    ]
    
    try:
        # Create embedder
        embedder = create_embedder()
        
        # Display model info
        model_info = embedder.get_model_info()
        print(f"Model Info: {model_info}")
        
        # Store embeddings for semantic search
        print(f"\nStoring {len(sample_chunks)} chunks for semantic search...")
        embedder.store_embeddings(sample_chunks)
        
        # Display embedding stats
        stats = embedder.get_embedding_stats()
        print(f"Embedding Stats: {stats}")
        
        # Test queries for semantic search
        test_queries = [
            "What is deep learning and neural networks?",
            "How do transformer models work?",
            "Tell me about data preparation",
            "What is reinforcement learning?",
            "Computer vision applications"
        ]
        
        print(f"\n=== Semantic Search Results ===")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            # Get top-3 most similar chunks
            top_chunks = embedder.get_top_k_chunks(query, k=3, min_similarity=0.1)
            
            if top_chunks:
                for rank, (chunk, similarity, idx) in enumerate(top_chunks, 1):
                    print(f"   Rank {rank} (Similarity: {similarity:.3f}): {chunk[:80]}...")
            else:
                print("   No relevant chunks found.")
        
        # Test similarity threshold search
        print(f"\n=== Threshold-based Search ===")
        query = "artificial intelligence and machine learning"
        print(f"Query: '{query}' (threshold: 0.2)")
        
        similar_chunks = embedder.search_similar_chunks(query, similarity_threshold=0.2)
        print(f"Found {len(similar_chunks)} chunks above threshold:")
        
        for chunk, similarity, idx in similar_chunks:
            print(f"  - Similarity {similarity:.3f}: {chunk[:60]}...")
        
        # Performance comparison
        print(f"\n=== Performance Test ===")
        import time
        
        query = "What are neural networks?"
        start_time = time.time()
        results = embedder.get_top_k_chunks(query, k=5)
        search_time = time.time() - start_time
        
        print(f"Search completed in {search_time:.4f} seconds")
        print(f"Retrieved {len(results)} results")
        
        print("\n✅ Semantic search functionality working successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
