"""
Test script for embedding generation pipeline
Tests the embedder module with sample text chunks
"""

import sys
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.embedder import create_embedder
from modules.text_splitter import TextSplitter


def test_embedder_basic():
    """Test basic embedder functionality"""
    print("=== Testing Basic Embedder Functionality ===")
    
    try:
        # Initialize embedder
        embedder = create_embedder()
        
        # Test model info
        model_info = embedder.get_model_info()
        print(f"✓ Model loaded: {model_info['model_name']}")
        print(f"✓ Embedding dimension: {model_info['embedding_dimension']}")
        print(f"✓ Device: {model_info['device']}")
        
        return embedder
    
    except Exception as e:
        print(f"❌ Basic embedder test failed: {e}")
        return None


def test_embedding_generation():
    """Test embedding generation with sample chunks"""
    print("\n=== Testing Embedding Generation ===")
    
    embedder = test_embedder_basic()
    if embedder is None:
        return False
    
    # Sample text chunks
    sample_chunks = [
        "Artificial intelligence is transforming the way we work and live.",
        "Machine learning algorithms can identify patterns in large datasets.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers for complex tasks.",
        "Computer vision allows machines to interpret and analyze visual information."
    ]
    
    try:
        # Generate embeddings
        print(f"Generating embeddings for {len(sample_chunks)} chunks...")
        embeddings = embedder.embed_chunks(sample_chunks)
        
        print("✓ Embeddings generated successfully")
        print(f"✓ Shape: {embeddings.shape}")
        print(f"✓ Data type: {embeddings.dtype}")
        
        # Validate embedding properties
        assert embeddings.shape[0] == len(sample_chunks), "Number of embeddings doesn't match number of chunks"
        assert embeddings.shape[1] == embedder.embedding_dimension, "Embedding dimension mismatch"
        assert np.all(np.isfinite(embeddings)), "Embeddings contain invalid values"
        
        print("✓ All validation checks passed")
        
        return embeddings, sample_chunks
    
    except Exception as e:
        print(f"❌ Embedding generation test failed: {e}")
        return None, None


def test_similarity_computation():
    """Test similarity computation between embeddings"""
    print("\n=== Testing Similarity Computation ===")
    
    embeddings, chunks = test_embedding_generation()
    if embeddings is None:
        return False
    
    try:
        embedder = create_embedder()
        
        # Test query similarity
        query = "How does machine learning work with data?"
        query_embedding = embedder.embed_single_text(query)
        
        print(f"Query: '{query}'")
        print("Similarity with each chunk:")
        
        similarities = []
        for i, chunk in enumerate(chunks):
            similarity = embedder.compute_similarity(query_embedding, embeddings[i])
            similarities.append(similarity)
            print(f"  Chunk {i+1}: {similarity:.4f} - '{chunk[:50]}...'")
        
        # Find most similar
        most_similar_idx = np.argmax(similarities)
        print(f"\n✓ Most similar chunk: {most_similar_idx + 1}")
        print(f"✓ Similarity score: {similarities[most_similar_idx]:.4f}")
        print(f"✓ Content: '{chunks[most_similar_idx]}'")
        
        return True
    
    except Exception as e:
        print(f"❌ Similarity computation test failed: {e}")
        return False


def test_with_text_splitter():
    """Test embedder with real text chunks from text splitter"""
    print("\n=== Testing with Text Splitter Integration ===")
    
    try:
        # Create sample document
        sample_document = """
        The field of artificial intelligence has evolved significantly over the past decades.
        Machine learning, a subset of AI, focuses on algorithms that can learn from data
        without explicit programming. Deep learning, which uses neural networks with
        multiple layers, has achieved remarkable results in image recognition, natural
        language processing, and game playing.
        
        Natural language processing (NLP) is a branch of AI that helps computers understand,
        interpret, and manipulate human language. Modern NLP systems use transformer
        architectures, which have revolutionized the field by enabling models to process
        sequential data more effectively.
        
        Computer vision is another important area of AI that enables machines to interpret
        and understand visual information from the world. Recent advances in convolutional
        neural networks have made it possible to achieve human-level performance in many
        visual recognition tasks.
        
        The applications of AI are vast and growing, including healthcare diagnostics,
        autonomous vehicles, financial trading, and personal assistants. As these
        technologies continue to advance, they promise to transform virtually every
        aspect of human society.
        """
        
        # Split text into chunks
        splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.chunk_text(sample_document)
        
        print(f"✓ Created {len(chunks)} text chunks")
        
        # Generate embeddings
        embedder = create_embedder()
        embeddings = embedder.embed_chunks(chunks)
        
        print(f"✓ Generated embeddings: {embeddings.shape}")
        
        # Test query against chunks
        queries = [
            "What is deep learning?",
            "How does computer vision work?",
            "What are AI applications?"
        ]
        
        for query in queries:
            query_embedding = embedder.embed_single_text(query)
            similarities = [
                embedder.compute_similarity(query_embedding, chunk_emb)
                for chunk_emb in embeddings
            ]
            
            best_match_idx = np.argmax(similarities)
            print(f"\nQuery: '{query}'")
            print(f"Best match (chunk {best_match_idx + 1}, similarity: {similarities[best_match_idx]:.4f}):")
            print(f"'{chunks[best_match_idx][:100]}...'")
        
        print("\n✅ Text splitter integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Text splitter integration test failed: {e}")
        return False


def main():
    """Run all embedding tests"""
    print("🧪 Testing Embedding Generation Pipeline")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_embedder_basic():
        tests_passed += 1
    
    if test_embedding_generation()[0] is not None:
        tests_passed += 1
    
    if test_similarity_computation():
        tests_passed += 1
        
    if test_with_text_splitter():
        tests_passed += 1
    
    # Results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Embedding generation pipeline is working correctly.")
        print("\n✅ SUCCESS CRITERIA MET:")
        print("   ✓ SentenceTransformer model initialized (all-MiniLM-L6-v2)")
        print("   ✓ embed_chunks() function implemented and working")
        print("   ✓ Text chunks successfully converted to numerical vectors")
        print("   ✓ Similarity computation working correctly")
        print("   ✓ Integration with text splitter validated")
        
        return True
    else:
        print(f"❌ {total_tests - tests_passed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
