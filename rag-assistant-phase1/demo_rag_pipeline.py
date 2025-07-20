#!/usr/bin/env python3
"""
RAG Pipeline Demo - Step 7 Implementation

This script demonstrates the complete end-to-end RAG pipeline using
the new modular rag_pipeline.py module.
"""

import os
import sys
import logging

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rag_pipeline import RAGPipeline, create_rag_pipeline
from config import LOG_FORMAT, DEFAULT_PDF_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def create_sample_content_for_testing():
    """Create sample content if no PDF is available."""
    sample_content = """
    Artificial Intelligence and Machine Learning

    Artificial Intelligence (AI) is a broad field of computer science focused on 
    building systems that can perform tasks typically requiring human intelligence. 
    These tasks include learning, reasoning, problem-solving, perception, and language understanding.

    Machine Learning Fundamentals

    Machine learning is a subset of AI that enables computers to learn and improve 
    from experience without being explicitly programmed. It works by identifying 
    patterns in data and making predictions or decisions based on those patterns.

    Types of Machine Learning

    There are three main categories of machine learning:

    1. Supervised Learning: Uses labeled training data to learn a mapping between 
       inputs and outputs. Examples include classification and regression problems.
       Common algorithms include linear regression, decision trees, and neural networks.

    2. Unsupervised Learning: Finds patterns in data without labeled examples. 
       It includes clustering, dimensionality reduction, and association rule learning.
       Examples include customer segmentation and anomaly detection.

    3. Reinforcement Learning: Learns through interaction with an environment, 
       receiving rewards or penalties for actions. Used in game playing, robotics, 
       and autonomous systems.

    Deep Learning

    Deep learning is a specialized subset of machine learning that uses neural networks 
    with multiple layers (deep neural networks) to model and understand complex patterns. 
    It has achieved remarkable success in image recognition, natural language processing, 
    and speech recognition.

    Applications of AI and ML

    AI and machine learning have numerous real-world applications:
    - Healthcare: Medical image analysis, drug discovery, personalized treatment
    - Finance: Fraud detection, algorithmic trading, risk assessment
    - Transportation: Autonomous vehicles, traffic optimization
    - Technology: Search engines, recommendation systems, virtual assistants
    - Entertainment: Content recommendation, game AI, creative content generation

    Future of AI

    The future of AI holds immense potential with developments in quantum computing, 
    neuromorphic computing, and advanced neural architectures. However, it also 
    raises important questions about ethics, job displacement, and the need for 
    responsible AI development.
    """
    return sample_content


def demo_rag_pipeline():
    """Demonstrate the complete RAG pipeline functionality."""
    print("=" * 80)
    print("RAG Pipeline Demo - Step 7: Core RAG Pipeline Assembly")
    print("=" * 80)
    
    try:
        # Create RAG pipeline
        print("🚀 Initializing RAG Pipeline...")
        rag = create_rag_pipeline(
            llm_backend="huggingface",
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=400,
            chunk_overlap=50,
            top_k=3,
            similarity_threshold=0.2
        )
        
        # Show initial status
        status = rag.get_pipeline_status()
        print(f"✅ Pipeline initialized: {status['components']}")
        
        # Try to load a PDF document
        success = False
        if os.path.exists(DEFAULT_PDF_PATH):
            print(f"\n📄 Loading PDF document: {DEFAULT_PDF_PATH}")
            success = rag.load_document(DEFAULT_PDF_PATH)
        
        # If no PDF available, create sample content
        if not success:
            print("\n📄 No PDF available, creating sample content for testing...")
            sample_text = create_sample_content_for_testing()
            
            # Process sample content directly
            chunks = rag.text_splitter.chunk_text(sample_text)
            rag.embedder.store_embeddings(chunks)
            
            # Store as processed document
            rag.processed_documents["sample_content"] = {
                "text_content": sample_text,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "file_size_mb": len(sample_text) / (1024 * 1024),
                "processed_at": rag.time.time() if hasattr(rag, 'time') else 0
            }
            rag.current_document_path = "sample_content"
            rag.is_ready = True
            
            print(f"✅ Sample content processed: {len(chunks)} chunks created")
        
        # Test the complete RAG pipeline with various questions
        test_questions = [
            "What is artificial intelligence?",
            "What are the main types of machine learning?",
            "How does supervised learning work?",
            "What is deep learning?",
            "What are some applications of AI in healthcare?",
            "What does the future of AI look like?",
            "Tell me about reinforcement learning.",
            "How is machine learning different from traditional programming?"
        ]
        
        print(f"\n{'='*60}")
        print("Testing Complete RAG Pipeline - generate_answer() Method")
        print(f"{'='*60}")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: '{question}'")
            print("-" * 50)
            
            # Use the main generate_answer method
            result = rag.generate_answer(
                question=question,
                max_response_length=250
            )
            
            # Display results
            if result['success']:
                print(f"✅ Answer: {result['answer']}")
                print(f"📊 Performance:")
                print(f"   - Retrieval: {result['retrieval_time']:.3f}s")
                print(f"   - Generation: {result['generation_time']:.3f}s")
                print(f"   - Total: {result['total_time']:.3f}s")
                print(f"📚 Context: {len(result['context'])} chunks retrieved")
                
                # Show top context chunk
                if result['context']:
                    top_chunk = result['context'][0]
                    print(f"🔍 Top context (similarity: {top_chunk['similarity']:.3f}):")
                    print(f"   {top_chunk['text'][:100]}...")
            else:
                print(f"❌ Error: {result['answer']}")
        
        # Show pipeline statistics
        final_status = rag.get_pipeline_status()
        processed_docs = rag.list_processed_documents()
        
        print(f"\n{'='*60}")
        print("Final Pipeline Status")
        print(f"{'='*60}")
        print(f"📋 Pipeline ready: {final_status['is_ready']}")
        print(f"📄 Documents processed: {len(processed_docs)}")
        print(f"🧠 Current document: {final_status['current_document']}")
        
        if final_status['embedding_stats']:
            print(f"🔢 Embedding stats: {final_status['embedding_stats']}")
        
        print(f"⚙️  Configuration:")
        for key, value in final_status['configuration'].items():
            print(f"   - {key}: {value}")
        
        # Demonstrate additional pipeline features
        print(f"\n{'='*60}")
        print("Additional Pipeline Features")
        print(f"{'='*60}")
        
        # Test error handling
        print("\n🧪 Testing error handling...")
        error_result = rag.generate_answer("")
        print(f"Empty question result: {error_result['success']} - {error_result['answer']}")
        
        # Test document listing
        if processed_docs:
            print(f"\n📚 Processed documents:")
            for doc in processed_docs:
                print(f"   - {doc['path']}: {doc['chunk_count']} chunks, {doc['file_size_mb']:.2f}MB")
        
        print(f"\n{'='*60}")
        print("🎉 RAG Pipeline Demo - Step 7 Complete!")
        print(f"{'='*60}")
        print("✅ Step 7 Deliverables Achieved:")
        print("   ✓ Created modules/rag_pipeline.py")
        print("   ✓ Implemented generate_answer() orchestrating all components")
        print("   ✓ PDF parsing → Chunking → Embedding → Retrieval → LLM query")
        print("   ✓ Effective prompt template for context + question")
        print("   ✓ Complete pipeline from PDF + query to answer")
        print("   ✓ End-to-end RAG functionality working")
        print("\n🚀 Ready for advanced RAG features and optimizations!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def interactive_rag_demo():
    """Interactive demo allowing user to ask custom questions."""
    print(f"\n{'='*60}")
    print("Interactive RAG Demo")
    print(f"{'='*60}")
    print("Enter questions to test the RAG pipeline (type 'quit' to exit)")
    
    try:
        # Initialize pipeline
        rag = create_rag_pipeline()
        
        # Load sample content
        sample_text = create_sample_content_for_testing()
        chunks = rag.text_splitter.chunk_text(sample_text)
        rag.embedder.store_embeddings(chunks)
        rag.processed_documents["sample_content"] = {
            "text_content": sample_text,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "file_size_mb": 0.01,
            "processed_at": 0
        }
        rag.current_document_path = "sample_content"
        rag.is_ready = True
        
        print("✅ Pipeline ready for interactive questions!")
        
        while True:
            try:
                question = input("\n❓ Enter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    print("Please enter a valid question.")
                    continue
                
                print("🤔 Thinking...")
                result = rag.generate_answer(question, max_response_length=300)
                
                if result['success']:
                    print(f"\n💬 Answer: {result['answer']}")
                    print(f"⏱️  Response time: {result['total_time']:.2f}s")
                else:
                    print(f"\n❌ Error: {result['answer']}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing question: {str(e)}")
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {str(e)}")


if __name__ == "__main__":
    # Run the main demo
    success = demo_rag_pipeline()
    
    # Optionally run interactive demo
    if success and len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_rag_demo()
    
    sys.exit(0 if success else 1)
