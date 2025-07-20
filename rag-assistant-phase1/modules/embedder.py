"""
Embedding Generation Pipeline for RAG Assistant Phase 1

This module provides functionality to convert text chunks into numerical 
vector representations using SentenceTransformer models for semantic search.
"""

import logging
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Text embedding generator using SentenceTransformer models.
    
    This class provides functionality to convert text chunks into dense vector
    representations that capture semantic meaning for similarity search in RAG systems.
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
    # Demo of the embedding functionality
    print("=== Text Embedder Demo ===")
    
    # Sample text chunks
    sample_chunks = [
        "This is the first chunk of text about machine learning.",
        "The second chunk discusses natural language processing techniques.",
        "This third chunk covers deep learning and neural networks.",
        "The final chunk explains transformer models and attention mechanisms."
    ]
    
    try:
        # Create embedder
        embedder = create_embedder()
        
        # Display model info
        model_info = embedder.get_model_info()
        print(f"Model Info: {model_info}")
        
        # Generate embeddings
        print(f"\nGenerating embeddings for {len(sample_chunks)} chunks...")
        embeddings = embedder.embed_chunks(sample_chunks)
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Test single text embedding
        single_text = "Query about artificial intelligence"
        query_embedding = embedder.embed_single_text(single_text)
        print(f"Single text embedding shape: {query_embedding.shape}")
        
        # Compute similarity between query and first chunk
        similarity = embedder.compute_similarity(query_embedding, embeddings[0])
        print(f"Similarity between query and first chunk: {similarity:.4f}")
        
        # Find most similar chunk
        similarities = [
            embedder.compute_similarity(query_embedding, chunk_emb)
            for chunk_emb in embeddings
        ]
        
        most_similar_idx = np.argmax(similarities)
        print(f"\nMost similar chunk (index {most_similar_idx}): {sample_chunks[most_similar_idx]}")
        print(f"Similarity score: {similarities[most_similar_idx]:.4f}")
        
        print("\n✅ Embedding generation pipeline working successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
