"""
Test suite for semantic search and retrieval functionality (Step 5)
"""

import sys
import os
import pytest
import numpy as np
from typing import List

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.embedder import TextEmbedder, create_embedder
from modules.vector_store import VectorStore, RAGRetriever


class TestSemanticSearch:
    """Test cases for semantic search and retrieval functionality."""
    
    @pytest.fixture
    def sample_chunks(self) -> List[str]:
        """Sample text chunks for testing."""
        return [
            "Machine learning is a subset of artificial intelligence focused on algorithms that improve through experience.",
            "Deep learning uses neural networks with multiple layers to learn complex patterns in data.",
            "Natural language processing enables computers to understand and interpret human language.",
            "Computer vision allows machines to interpret and analyze visual information from images and videos.",
            "Reinforcement learning trains agents through interaction with environments using rewards and penalties.",
            "Data preprocessing involves cleaning and preparing raw data for machine learning algorithms.",
            "Feature engineering is the process of selecting and transforming variables for machine learning models.",
            "Model evaluation assesses the performance of machine learning algorithms using various metrics."
        ]
    
    @pytest.fixture
    def embedder(self) -> TextEmbedder:
        """Create a TextEmbedder instance for testing."""
        return create_embedder()
    
    def test_embedder_initialization(self, embedder):
        """Test TextEmbedder initialization and basic properties."""
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.embedding_dimension > 0
        assert hasattr(embedder, 'model')
        assert hasattr(embedder, 'stored_embeddings')
        assert hasattr(embedder, 'stored_chunks')
        assert not embedder.is_index_built
    
    def test_store_embeddings(self, embedder, sample_chunks):
        """Test storing embeddings for semantic search."""
        # Store embeddings
        embedder.store_embeddings(sample_chunks)
        
        # Verify storage
        assert embedder.is_index_built
        assert len(embedder.stored_chunks) == len(sample_chunks)
        assert embedder.stored_embeddings is not None
        assert embedder.stored_embeddings.shape[0] == len(sample_chunks)
        assert embedder.stored_embeddings.shape[1] == embedder.embedding_dimension
    
    def test_get_top_k_chunks_basic(self, embedder, sample_chunks):
        """Test basic top-K chunk retrieval."""
        # Store embeddings
        embedder.store_embeddings(sample_chunks)
        
        # Test query
        query = "What is deep learning?"
        results = embedder.get_top_k_chunks(query, k=3)
        
        # Verify results
        assert len(results) <= 3
        assert all(isinstance(chunk, str) for chunk, _, _ in results)
        assert all(isinstance(score, float) for _, score, _ in results)
        assert all(isinstance(idx, int) for _, _, idx in results)
        
        # Results should be sorted by similarity (descending)
        scores = [score for _, score, _ in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_get_top_k_chunks_relevance(self, embedder, sample_chunks):
        """Test that semantic search returns relevant results."""
        embedder.store_embeddings(sample_chunks)
        
        # Test specific queries and verify relevance
        test_cases = [
            ("deep learning neural networks", "deep learning"),
            ("natural language processing NLP", "natural language"),
            ("computer vision images", "computer vision"),
            ("reinforcement learning rewards", "reinforcement learning")
        ]
        
        for query, expected_keyword in test_cases:
            results = embedder.get_top_k_chunks(query, k=3)
            assert len(results) > 0, f"No results for query: {query}"
            
            # Check that the top result contains the expected keyword
            top_chunk = results[0][0].lower()
            assert expected_keyword in top_chunk, f"Expected '{expected_keyword}' in top result for '{query}'"
    
    def test_similarity_threshold(self, embedder, sample_chunks):
        """Test similarity threshold filtering."""
        embedder.store_embeddings(sample_chunks)
        
        query = "deep learning neural networks"
        
        # Test with different thresholds
        results_low = embedder.get_top_k_chunks(query, k=10, min_similarity=0.1)
        results_high = embedder.get_top_k_chunks(query, k=10, min_similarity=0.5)
        
        # Higher threshold should return fewer or equal results
        assert len(results_high) <= len(results_low)
        
        # All results should meet the threshold
        for _, score, _ in results_high:
            assert score >= 0.5
    
    def test_search_similar_chunks(self, embedder, sample_chunks):
        """Test threshold-based search functionality."""
        embedder.store_embeddings(sample_chunks)
        
        query = "machine learning algorithms"
        results = embedder.search_similar_chunks(query, similarity_threshold=0.2)
        
        # Verify all results meet threshold
        for _, score, _ in results:
            assert score >= 0.2
        
        # Results should be sorted by similarity
        scores = [score for _, score, _ in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_embedding_stats(self, embedder, sample_chunks):
        """Test embedding statistics functionality."""
        # Test empty state
        stats_empty = embedder.get_embedding_stats()
        assert stats_empty["status"] == "No embeddings stored"
        
        # Test with embeddings
        embedder.store_embeddings(sample_chunks)
        stats = embedder.get_embedding_stats()
        
        assert stats["status"] == "Embeddings stored"
        assert stats["chunk_count"] == len(sample_chunks)
        assert stats["embedding_dimension"] == embedder.embedding_dimension
        assert "memory_usage_mb" in stats
        assert "total_characters" in stats
        assert "avg_chunk_length" in stats
    
    def test_clear_stored_embeddings(self, embedder, sample_chunks):
        """Test clearing stored embeddings."""
        # Store and verify
        embedder.store_embeddings(sample_chunks)
        assert embedder.is_index_built
        
        # Clear and verify
        embedder.clear_stored_embeddings()
        assert not embedder.is_index_built
        assert len(embedder.stored_chunks) == 0
        assert embedder.stored_embeddings is None
    
    def test_error_handling(self, embedder, sample_chunks):
        """Test error handling in semantic search."""
        # Test search without stored embeddings
        with pytest.raises(ValueError, match="No embeddings stored"):
            embedder.get_top_k_chunks("test query", k=5)
        
        # Test invalid parameters
        embedder.store_embeddings(sample_chunks)
        
        with pytest.raises(ValueError, match="k must be positive"):
            embedder.get_top_k_chunks("test query", k=0)
        
        with pytest.raises(ValueError, match="min_similarity must be between"):
            embedder.get_top_k_chunks("test query", k=5, min_similarity=1.5)


class TestVectorStore:
    """Test cases for VectorStore functionality."""
    
    @pytest.fixture
    def vector_store(self) -> VectorStore:
        """Create a VectorStore instance for testing."""
        return VectorStore(embedding_dimension=384)  # all-MiniLM-L6-v2 dimension
    
    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Sample embeddings for testing."""
        np.random.seed(42)
        return np.random.rand(5, 384).astype(np.float32)
    
    @pytest.fixture
    def sample_chunks(self) -> List[str]:
        """Sample chunks for testing."""
        return [
            "First sample chunk about machine learning",
            "Second chunk discussing deep learning",
            "Third chunk on natural language processing",
            "Fourth chunk about computer vision",
            "Fifth chunk on reinforcement learning"
        ]
    
    def test_vector_store_initialization(self, vector_store):
        """Test VectorStore initialization."""
        assert vector_store.embedding_dimension == 384
        assert vector_store.index_type == "flat"
        assert not vector_store.is_built
        assert len(vector_store.chunks) == 0
        assert vector_store.index is not None
    
    def test_add_embeddings(self, vector_store, sample_embeddings, sample_chunks):
        """Test adding embeddings to vector store."""
        vector_store.add_embeddings(sample_embeddings, sample_chunks)
        
        assert vector_store.is_built
        assert len(vector_store.chunks) == len(sample_chunks)
        assert vector_store.index.ntotal == len(sample_embeddings)
    
    def test_vector_store_search(self, vector_store, sample_embeddings, sample_chunks):
        """Test searching in vector store."""
        vector_store.add_embeddings(sample_embeddings, sample_chunks)
        
        # Use first embedding as query
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, k=3)
        
        assert len(results) <= 3
        assert all(isinstance(chunk, str) for chunk, _, _ in results)
        assert all(isinstance(score, float) for _, score, _ in results)
        
        # First result should be the query itself (highest similarity)
        assert results[0][2] == 0  # First chunk index
    
    def test_vector_store_stats(self, vector_store, sample_embeddings, sample_chunks):
        """Test vector store statistics."""
        # Empty store
        stats_empty = vector_store.get_stats()
        assert stats_empty["status"] == "Empty vector store"
        
        # With data
        vector_store.add_embeddings(sample_embeddings, sample_chunks)
        stats = vector_store.get_stats()
        
        assert stats["status"] == "Active"
        assert stats["total_chunks"] == len(sample_chunks)
        assert stats["embedding_dimension"] == 384
        assert "memory_usage_mb" in stats


class TestRAGRetriever:
    """Test cases for RAGRetriever functionality."""
    
    @pytest.fixture
    def embedder(self) -> TextEmbedder:
        """Create TextEmbedder for testing."""
        return create_embedder()
    
    @pytest.fixture
    def retriever(self, embedder) -> RAGRetriever:
        """Create RAGRetriever for testing."""
        return RAGRetriever(embedder)
    
    @pytest.fixture
    def sample_documents(self) -> List[str]:
        """Sample documents for testing."""
        return [
            "Machine learning is a powerful tool for data analysis and pattern recognition.",
            "Deep learning networks can learn complex representations from raw data automatically.",
            "Natural language processing enables computers to understand human communication.",
            "Computer vision systems can analyze and interpret visual information effectively.",
            "Reinforcement learning agents learn optimal behaviors through environmental interaction."
        ]
    
    def test_rag_retriever_initialization(self, retriever):
        """Test RAGRetriever initialization."""
        assert retriever.embedder is not None
        assert retriever.vector_store is not None
        assert retriever.vector_store.embedding_dimension == retriever.embedder.embedding_dimension
    
    def test_add_documents(self, retriever, sample_documents):
        """Test adding documents to retriever."""
        retriever.add_documents(sample_documents)
        
        assert retriever.vector_store.is_built
        assert len(retriever.vector_store.chunks) == len(sample_documents)
    
    def test_retrieve(self, retriever, sample_documents):
        """Test document retrieval."""
        retriever.add_documents(sample_documents)
        
        query = "What is machine learning?"
        results = retriever.retrieve(query, k=3)
        
        assert len(results) <= 3
        assert all(isinstance(chunk, str) for chunk, _, _ in results)
        assert all(isinstance(score, float) for _, score, _ in results)
        
        # Check relevance - first result should contain "machine learning"
        top_chunk = results[0][0].lower()
        assert "machine learning" in top_chunk
    
    def test_get_context(self, retriever, sample_documents):
        """Test context generation for RAG."""
        retriever.add_documents(sample_documents)
        
        query = "Tell me about AI and machine learning"
        context = retriever.get_context(query, k=3, separator="\n\n")
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert "\n\n" in context  # Check separator is used
    
    def test_performance(self, retriever, sample_documents):
        """Test performance with larger dataset."""
        # Create larger dataset
        extended_docs = sample_documents * 20  # 100 documents
        
        import time
        start_time = time.time()
        retriever.add_documents(extended_docs)
        add_time = time.time() - start_time
        
        start_time = time.time()
        results = retriever.retrieve("machine learning deep learning", k=10)
        search_time = time.time() - start_time
        
        # Performance assertions (should be fast)
        assert add_time < 30  # Adding 100 docs should take less than 30 seconds
        assert search_time < 1  # Search should take less than 1 second
        assert len(results) <= 10


# Integration test
def test_end_to_end_semantic_search():
    """Test complete end-to-end semantic search pipeline."""
    
    # Sample knowledge base
    knowledge_base = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning algorithms can automatically improve through experience and data.",
        "Deep learning uses artificial neural networks with multiple layers for complex pattern recognition.",
        "Natural language processing involves computational techniques for analyzing human language.",
        "Data science combines statistics, programming, and domain expertise to extract insights.",
        "Artificial intelligence aims to create systems that can perform tasks requiring human-like intelligence.",
        "Computer vision enables machines to interpret and understand visual information from images.",
        "Web development involves creating websites and web applications using various technologies."
    ]
    
    # Initialize system
    embedder = create_embedder()
    retriever = RAGRetriever(embedder)
    
    # Add knowledge base
    retriever.add_documents(knowledge_base)
    
    # Test queries with expected results
    test_queries = [
        ("What is Python?", "python"),
        ("How does machine learning work?", "machine learning"),
        ("Tell me about neural networks", "neural networks"),
        ("What is NLP?", "natural language"),
        ("Computer vision applications", "computer vision")
    ]
    
    for query, expected_keyword in test_queries:
        results = retriever.retrieve(query, k=3, min_similarity=0.1)
        
        assert len(results) > 0, f"No results for query: {query}"
        
        # Check that top result is relevant
        top_chunk = results[0][0].lower()
        assert expected_keyword in top_chunk, f"Expected '{expected_keyword}' in result for '{query}'"
        
        # Check similarity scores are reasonable
        top_score = results[0][1]
        assert top_score > 0.1, f"Top score too low for '{query}': {top_score}"


if __name__ == "__main__":
    # Run basic tests if called directly
    print("Running basic semantic search tests...")
    
    try:
        # Test embedder functionality
        embedder = create_embedder()
        sample_chunks = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "NLP processes human language",
            "Computer vision analyzes images"
        ]
        
        embedder.store_embeddings(sample_chunks)
        results = embedder.get_top_k_chunks("What is deep learning?", k=2)
        
        print(f"✅ Embedder test passed - found {len(results)} results")
        
        # Test retriever functionality
        retriever = RAGRetriever(embedder)
        retriever.add_documents(sample_chunks)
        context = retriever.get_context("neural networks", k=2)
        
        print(f"✅ Retriever test passed - generated context: {len(context)} characters")
        
        print("\n🎉 All basic tests passed! Run 'pytest test_semantic_search.py -v' for comprehensive testing.")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
