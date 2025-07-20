"""
Vector Store Module for RAG Assistant Phase 1

This module provides a FAISS-based vector store for efficient similarity search
and retrieval of text chunks in the RAG pipeline.
"""

import logging
import numpy as np
import pickle
import os
from typing import List, Optional, Tuple, Dict, Any
import faiss
from .embedder import TextEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for efficient similarity search and retrieval.
    
    This class provides high-performance vector similarity search using FAISS
    (Facebook AI Similarity Search) library with support for persistence.
    """
    
    def __init__(self, embedding_dimension: int, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            embedding_dimension (int): Dimension of the embedding vectors
            index_type (str): Type of FAISS index ("flat", "ivf", "hnsw")
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.index = None
        self.chunks = []
        self.is_built = False
        
        # Initialize FAISS index
        self._create_index()
        
        logger.info(f"VectorStore initialized with {index_type} index (dim={embedding_dimension})")
    
    def _create_index(self) -> None:
        """Create FAISS index based on the specified type."""
        if self.index_type == "flat":
            # Exact search using L2 distance
            self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            # Approximate search with IVF (Inverted File)
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World (HNSW) for very fast approximate search
            M = 16  # Number of connections
            self.index = faiss.IndexHNSWFlat(self.embedding_dimension, M)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str]) -> None:
        """
        Add embeddings and corresponding text chunks to the vector store.
        
        Args:
            embeddings (np.ndarray): 2D array of embeddings (num_chunks, embedding_dim)
            chunks (List[str]): List of text chunks corresponding to embeddings
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings")
        
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {embeddings.shape[1]}")
        
        # Normalize embeddings for cosine similarity with inner product
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(normalized_embeddings.astype(np.float32))
        
        # Add embeddings to index
        self.index.add(normalized_embeddings.astype(np.float32))
        
        # Store chunks
        self.chunks.extend(chunks)
        self.is_built = True
        
        logger.info(f"Added {len(chunks)} chunks to vector store (total: {len(self.chunks)})")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.0) -> List[Tuple[str, float, int]]:
        """
        Search for similar chunks using the query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of top results to return
            min_similarity (float): Minimum similarity threshold (0.0 to 1.0)
        
        Returns:
            List[Tuple[str, float, int]]: List of (chunk_text, similarity_score, chunk_index)
        """
        if not self.is_built:
            raise ValueError("Vector store is empty. Add embeddings first.")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query embedding
        normalized_query = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search for similar vectors
        similarities, indices = self.index.search(normalized_query.astype(np.float32), k)
        
        # Format results
        results = []
        for i in range(len(similarities[0])):
            similarity = float(similarities[0][i])
            idx = int(indices[0][i])
            
            # Skip invalid indices and low similarity scores
            if idx < 0 or idx >= len(self.chunks) or similarity < min_similarity:
                continue
            
            chunk_text = self.chunks[idx]
            results.append((chunk_text, similarity, idx))
        
        logger.debug(f"Found {len(results)} chunks above similarity threshold {min_similarity}")
        return results
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[str]:
        """
        Retrieve chunk by its ID.
        
        Args:
            chunk_id (int): ID of the chunk to retrieve
        
        Returns:
            str: The chunk text, or None if not found
        """
        if 0 <= chunk_id < len(self.chunks):
            return self.chunks[chunk_id]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict: Statistics including size, index type, memory usage
        """
        if not self.is_built:
            return {"status": "Empty vector store"}
        
        return {
            "status": "Active",
            "index_type": self.index_type,
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embedding_dimension,
            "is_trained": getattr(self.index, 'is_trained', True),
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if not self.is_built:
            return 0.0
        
        # Rough estimation based on index type and data
        base_memory = len(self.chunks) * self.embedding_dimension * 4 / (1024 * 1024)  # 4 bytes per float
        text_memory = sum(len(chunk) for chunk in self.chunks) / (1024 * 1024)
        
        return round(base_memory + text_memory, 2)
    
    def save(self, file_path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            file_path (str): Path to save the vector store
        """
        if not self.is_built:
            raise ValueError("Cannot save empty vector store")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save FAISS index
        index_path = f"{file_path}.faiss"
        faiss.write_index(self.index, index_path)
        
        # Save metadata and chunks
        metadata = {
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'chunks': self.chunks,
            'is_built': self.is_built
        }
        
        with open(f"{file_path}.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Vector store saved to {file_path}")
    
    def load(self, file_path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            file_path (str): Path to load the vector store from
        """
        # Load FAISS index
        index_path = f"{file_path}.faiss"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load metadata and chunks
        metadata_path = f"{file_path}.pkl"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.embedding_dimension = metadata['embedding_dimension']
        self.index_type = metadata['index_type']
        self.chunks = metadata['chunks']
        self.is_built = metadata['is_built']
        
        logger.info(f"Vector store loaded from {file_path} ({len(self.chunks)} chunks)")
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.chunks = []
        self.is_built = False
        self._create_index()
        logger.info("Vector store cleared")


class RAGRetriever:
    """
    High-level retriever that combines TextEmbedder and VectorStore for RAG systems.
    """
    
    def __init__(self, embedder: TextEmbedder, vector_store: Optional[VectorStore] = None):
        """
        Initialize the RAG retriever.
        
        Args:
            embedder (TextEmbedder): Text embedder instance
            vector_store (VectorStore, optional): Vector store instance
        """
        self.embedder = embedder
        
        if vector_store is None:
            # Create vector store with embedder's dimension
            self.vector_store = VectorStore(
                embedding_dimension=embedder.embedding_dimension,
                index_type="flat"  # Default to exact search
            )
        else:
            self.vector_store = vector_store
        
        logger.info("RAG Retriever initialized")
    
    def add_documents(self, chunks: List[str]) -> None:
        """
        Add documents (chunks) to the retrieval system.
        
        Args:
            chunks (List[str]): List of text chunks to add
        """
        logger.info(f"Adding {len(chunks)} chunks to retrieval system...")
        
        # Generate embeddings
        embeddings = self.embedder.embed_chunks(chunks, show_progress=True)
        
        # Add to vector store
        self.vector_store.add_embeddings(embeddings, chunks)
        
        logger.info("Documents added successfully")
    
    def retrieve(self, query: str, k: int = 5, min_similarity: float = 0.0) -> List[Tuple[str, float, int]]:
        """
        Retrieve top-K most similar chunks for a query.
        
        Args:
            query (str): Query text
            k (int): Number of top results to return
            min_similarity (float): Minimum similarity threshold
        
        Returns:
            List[Tuple[str, float, int]]: Retrieved chunks with scores
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_single_text(query)
        
        # Search in vector store
        return self.vector_store.search(query_embedding, k=k, min_similarity=min_similarity)
    
    def get_context(self, query: str, k: int = 5, separator: str = "\n\n") -> str:
        """
        Get formatted context string for RAG pipeline.
        
        Args:
            query (str): Query text
            k (int): Number of chunks to retrieve
            separator (str): Separator between chunks
        
        Returns:
            str: Formatted context string
        """
        results = self.retrieve(query, k=k)
        
        if not results:
            return ""
        
        context_chunks = [chunk for chunk, _, _ in results]
        return separator.join(context_chunks)
    
    def save_index(self, file_path: str) -> None:
        """Save the retrieval index to disk."""
        self.vector_store.save(file_path)
    
    def load_index(self, file_path: str) -> None:
        """Load the retrieval index from disk."""
        self.vector_store.load(file_path)


# Example usage and testing
if __name__ == "__main__":
    print("=== Vector Store and RAG Retriever Demo ===")
    
    # Sample documents for testing
    sample_documents = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is part of machine learning based on artificial neural networks.",
        "Natural language processing (NLP) is a subfield of linguistics and artificial intelligence.",
        "Computer vision is an interdisciplinary scientific field that deals with how computers can gain understanding from digital images or videos.",
        "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions.",
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from data.",
        "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans.",
        "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains."
    ]
    
    try:
        # Create embedder and retriever
        from .embedder import create_embedder
        
        embedder = create_embedder()
        retriever = RAGRetriever(embedder)
        
        # Add documents to the retrieval system
        retriever.add_documents(sample_documents)
        
        # Test retrieval
        test_queries = [
            "What is machine learning?",
            "Tell me about neural networks",
            "How does computer vision work?",
            "What is artificial intelligence?"
        ]
        
        print("\n=== Retrieval Results ===")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = retriever.retrieve(query, k=3, min_similarity=0.1)
            
            for i, (chunk, score, idx) in enumerate(results, 1):
                print(f"  {i}. Score: {score:.3f} - {chunk[:80]}...")
        
        # Test context generation
        print("\n=== Context Generation ===")
        query = "Explain machine learning and AI"
        context = retriever.get_context(query, k=3)
        print(f"Query: {query}")
        print(f"Context:\n{context}")
        
        # Display stats
        print(f"\n=== Vector Store Statistics ===")
        stats = retriever.vector_store.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n✅ Vector store and retrieval system working successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
