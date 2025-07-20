"""
Demo script for Step 5: Semantic Search & Retrieval

This script demonstrates the enhanced embedding module with similarity search
and the new vector store capabilities for efficient document retrieval.
"""

import sys
import os
import time
from typing import List

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.embedder import create_embedder
from modules.vector_store import VectorStore, RAGRetriever
from modules.text_splitter import TextSplitter
import config


def load_sample_documents() -> List[str]:
    """Load sample documents for demonstration."""
    
    # Sample documents covering various AI/ML topics
    documents = [
        # Machine Learning
        """Machine learning is a subset of artificial intelligence that enables computers to learn and 
        make decisions from data without being explicitly programmed. It involves algorithms that can 
        identify patterns, make predictions, and improve their performance through experience. Common 
        applications include recommendation systems, image recognition, and natural language processing.""",
        
        # Deep Learning
        """Deep learning is a specialized branch of machine learning that uses artificial neural networks 
        with multiple layers to model and understand complex patterns in data. These deep neural networks 
        can automatically learn hierarchical representations, making them particularly effective for tasks 
        like computer vision, speech recognition, and language translation.""",
        
        # Natural Language Processing
        """Natural language processing (NLP) is a field that focuses on enabling computers to understand, 
        interpret, and generate human language. NLP combines computational linguistics with machine learning 
        to process text and speech data. Applications include sentiment analysis, machine translation, 
        chatbots, and text summarization.""",
        
        # Computer Vision
        """Computer vision is an interdisciplinary field that teaches machines to interpret and understand 
        visual information from the world. Using digital images and videos, computer vision systems can 
        identify objects, recognize faces, track movements, and analyze scenes. It's widely used in 
        autonomous vehicles, medical imaging, and security systems.""",
        
        # Reinforcement Learning
        """Reinforcement learning is a type of machine learning where agents learn to make decisions by 
        interacting with an environment. The agent receives rewards or penalties based on its actions, 
        gradually learning optimal strategies through trial and error. This approach has achieved 
        remarkable success in game playing, robotics, and resource management.""",
        
        # Data Science
        """Data science is an interdisciplinary field that combines statistics, programming, and domain 
        expertise to extract meaningful insights from data. Data scientists use various tools and 
        techniques including machine learning, data visualization, and statistical analysis to solve 
        complex business problems and make data-driven decisions.""",
        
        # Neural Networks
        """Artificial neural networks are computing systems inspired by biological neural networks. 
        They consist of interconnected nodes (neurons) that process information through weighted 
        connections. Neural networks can learn complex mappings between inputs and outputs, making 
        them versatile tools for pattern recognition, function approximation, and decision making.""",
        
        # Big Data
        """Big data refers to extremely large datasets that require specialized tools and techniques 
        to store, process, and analyze effectively. The characteristics of big data are often described 
        by the "3 Vs": Volume (large amounts), Velocity (rapid generation), and Variety (different types). 
        Technologies like Hadoop and Spark have been developed to handle big data challenges."""
    ]
    
    return documents


def chunk_documents(documents: List[str]) -> List[str]:
    """Split documents into smaller chunks for better retrieval."""
    splitter = TextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    all_chunks = []
    for doc in documents:
        chunks = splitter.chunk_text(doc)
        all_chunks.extend(chunks)
    
    return all_chunks


def demonstrate_embedder_search():
    """Demonstrate the enhanced TextEmbedder with semantic search."""
    print("=" * 60)
    print("ENHANCED TEXTEMBEDDER SEMANTIC SEARCH DEMO")
    print("=" * 60)
    
    # Load and prepare data
    documents = load_sample_documents()
    chunks = chunk_documents(documents)
    
    print(f"Loaded {len(documents)} documents")
    print(f"Split into {len(chunks)} chunks")
    
    # Initialize embedder
    embedder = create_embedder()
    print(f"\nModel: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.embedding_dimension}")
    
    # Store embeddings
    print(f"\nGenerating and storing embeddings...")
    start_time = time.time()
    embedder.store_embeddings(chunks)
    embedding_time = time.time() - start_time
    
    print(f"Embeddings generated in {embedding_time:.2f} seconds")
    
    # Display stats
    stats = embedder.get_embedding_stats()
    print(f"\nEmbedding Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test queries
    test_queries = [
        "What is machine learning and how does it work?",
        "Explain deep learning and neural networks",
        "How does natural language processing work?",
        "What are the applications of computer vision?",
        "Tell me about reinforcement learning agents",
        "What is big data and how is it processed?"
    ]
    
    print(f"\n" + "=" * 40)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        start_time = time.time()
        results = embedder.get_top_k_chunks(query, k=3, min_similarity=0.1)
        search_time = time.time() - start_time
        
        print(f"   Search time: {search_time:.4f} seconds")
        print(f"   Found {len(results)} relevant chunks:")
        
        for rank, (chunk, similarity, idx) in enumerate(results, 1):
            # Truncate long chunks for display
            display_chunk = chunk[:100] + "..." if len(chunk) > 100 else chunk
            print(f"     {rank}. Score: {similarity:.3f} - {display_chunk}")


def demonstrate_vector_store():
    """Demonstrate FAISS-based VectorStore functionality."""
    print(f"\n" + "=" * 60)
    print("VECTOR STORE DEMO")
    print("=" * 60)
    
    documents = load_sample_documents()
    chunks = chunk_documents(documents)
    
    # Initialize components
    embedder = create_embedder()
    vector_store = VectorStore(
        embedding_dimension=embedder.embedding_dimension,
        index_type="flat"  # Exact search
    )
    
    print(f"VectorStore initialized with {vector_store.index_type} index")
    
    # Generate embeddings and add to store
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedder.embed_chunks(chunks)
    
    start_time = time.time()
    vector_store.add_embeddings(embeddings, chunks)
    add_time = time.time() - start_time
    
    print(f"Added embeddings in {add_time:.4f} seconds")
    
    # Display stats
    stats = vector_store.get_stats()
    print(f"\nVector Store Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search performance
    query = "deep learning neural networks artificial intelligence"
    query_embedding = embedder.embed_single_text(query)
    
    print(f"\nTesting search performance with query: '{query}'")
    
    # Benchmark different k values
    k_values = [1, 5, 10, 20]
    for k in k_values:
        start_time = time.time()
        results = vector_store.search(query_embedding, k=k, min_similarity=0.0)
        search_time = time.time() - start_time
        
        print(f"  Top-{k} search: {search_time:.4f}s, found {len(results)} results")


def demonstrate_rag_retriever():
    """Demonstrate the complete RAG retrieval system."""
    print(f"\n" + "=" * 60)
    print("RAG RETRIEVER SYSTEM DEMO")
    print("=" * 60)
    
    documents = load_sample_documents()
    chunks = chunk_documents(documents)
    
    # Initialize RAG retriever
    embedder = create_embedder()
    retriever = RAGRetriever(embedder)
    
    print(f"RAG Retriever initialized")
    
    # Add documents
    print(f"Adding {len(chunks)} chunks to retrieval system...")
    start_time = time.time()
    retriever.add_documents(chunks)
    setup_time = time.time() - start_time
    
    print(f"Setup completed in {setup_time:.2f} seconds")
    
    # Test various retrieval scenarios
    test_scenarios = [
        {
            "name": "High Precision Search",
            "query": "What is machine learning?",
            "k": 3,
            "min_similarity": 0.3
        },
        {
            "name": "Broad Context Retrieval",
            "query": "artificial intelligence and neural networks",
            "k": 5,
            "min_similarity": 0.1
        },
        {
            "name": "Specific Technical Query",
            "query": "reinforcement learning agents and rewards",
            "k": 2,
            "min_similarity": 0.2
        }
    ]
    
    print(f"\n" + "-" * 50)
    print("RETRIEVAL SCENARIOS")
    print("-" * 50)
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Query: '{scenario['query']}'")
        print(f"  Parameters: k={scenario['k']}, min_sim={scenario['min_similarity']}")
        
        start_time = time.time()
        results = retriever.retrieve(
            scenario['query'], 
            k=scenario['k'], 
            min_similarity=scenario['min_similarity']
        )
        retrieval_time = time.time() - start_time
        
        print(f"  Retrieval time: {retrieval_time:.4f} seconds")
        print(f"  Results found: {len(results)}")
        
        for i, (chunk, score, idx) in enumerate(results, 1):
            display_chunk = chunk[:80] + "..." if len(chunk) > 80 else chunk
            print(f"    {i}. Score: {score:.3f} - {display_chunk}")
    
    # Demonstrate context generation
    print(f"\n" + "-" * 50)
    print("CONTEXT GENERATION FOR RAG PIPELINE")
    print("-" * 50)
    
    rag_query = "How do neural networks work in deep learning?"
    context = retriever.get_context(rag_query, k=config.TOP_K_RETRIEVAL)
    
    print(f"Query: '{rag_query}'")
    print(f"Generated context ({len(context)} characters):")
    print(f"{'=' * 50}")
    print(context[:300] + "..." if len(context) > 300 else context)
    print(f"{'=' * 50}")


def performance_benchmark():
    """Benchmark the semantic search performance."""
    print(f"\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create larger dataset for benchmarking
    base_documents = load_sample_documents()
    large_documents = base_documents * 5  # 40 documents
    chunks = chunk_documents(large_documents)
    
    print(f"Benchmarking with {len(chunks)} chunks")
    
    embedder = create_embedder()
    retriever = RAGRetriever(embedder)
    
    # Benchmark setup time
    start_time = time.time()
    retriever.add_documents(chunks)
    setup_time = time.time() - start_time
    
    print(f"Setup time: {setup_time:.2f} seconds")
    print(f"Setup rate: {len(chunks)/setup_time:.1f} chunks/second")
    
    # Benchmark search times
    queries = [
        "machine learning algorithms",
        "deep learning neural networks", 
        "natural language processing",
        "computer vision applications",
        "reinforcement learning"
    ]
    
    search_times = []
    for query in queries:
        start_time = time.time()
        results = retriever.retrieve(query, k=5)
        search_time = time.time() - start_time
        search_times.append(search_time)
    
    avg_search_time = sum(search_times) / len(search_times)
    print(f"\nSearch Performance:")
    print(f"  Average search time: {avg_search_time:.4f} seconds")
    print(f"  Queries per second: {1/avg_search_time:.1f}")
    print(f"  Min search time: {min(search_times):.4f} seconds")
    print(f"  Max search time: {max(search_times):.4f} seconds")


def main():
    """Run the complete Step 5 demonstration."""
    print("🚀 STEP 5: SEMANTIC SEARCH & RETRIEVAL DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Demonstrate each component
        demonstrate_embedder_search()
        demonstrate_vector_store()
        demonstrate_rag_retriever()
        performance_benchmark()
        
        print(f"\n" + "=" * 80)
        print("🎉 STEP 5 IMPLEMENTATION COMPLETE!")
        print("✅ Enhanced TextEmbedder with semantic search")
        print("✅ FAISS-based VectorStore for efficient retrieval")
        print("✅ RAGRetriever for complete document retrieval")
        print("✅ Top-K chunk retrieval with similarity thresholds")
        print("✅ Performance optimization for larger datasets")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
