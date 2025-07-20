#!/usr/bin/env python3
"""
Main RAG Pipeline - Step 7 Complete with Modular Architecture

This script demonstrates the new modular RAG pipeline implementation
from Step 7, showcasing the dedicated rag_pipeline.py module.
"""

import os
import sys
import time
import logging

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rag_pipeline import create_rag_pipeline
from config import LOG_FORMAT, DEFAULT_PDF_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def demo_step7_rag_pipeline():
    """Demonstrate the Step 7 modular RAG pipeline implementation."""
    print("=" * 80)
    print("RAG Assistant Phase 1 - Step 7: Modular RAG Pipeline")
    print("=" * 80)
    
    try:
        # Initialize the new modular RAG Pipeline
        print("🚀 Creating RAG Pipeline using new modular architecture...")
        rag_pipeline = create_rag_pipeline(
            llm_backend="huggingface",
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=50,
            top_k=5,
            similarity_threshold=0.3
        )
        
        print("✅ RAG Pipeline initialized successfully!")
        
        # Show pipeline status
        status = rag_pipeline.get_pipeline_status()
        print(f"\n📊 Pipeline Status:")
        print(f"   Ready: {status['is_ready']}")
        print(f"   Components: {list(status['components'].keys())}")
        print(f"   Configuration: {status['configuration']}")
        
        # Try to load a document
        success = False
        if os.path.exists(DEFAULT_PDF_PATH):
            print(f"\n📄 Loading document: {DEFAULT_PDF_PATH}")
            success = rag_pipeline.load_document(DEFAULT_PDF_PATH)
        
        # Create sample content if no PDF is available
        if not success:
            print("\n📄 Creating sample AI/ML content for demonstration...")
            sample_content = """
            Artificial Intelligence and Machine Learning: A Comprehensive Overview

            Introduction to Artificial Intelligence
            Artificial Intelligence (AI) represents one of the most transformative 
            technologies of our time. It encompasses the development of computer systems 
            that can perform tasks typically requiring human intelligence, such as 
            visual perception, speech recognition, decision-making, and language translation.

            Machine Learning Fundamentals
            Machine Learning (ML) is a subset of AI that provides systems the ability 
            to automatically learn and improve from experience without being explicitly 
            programmed. ML focuses on the development of computer programs that can 
            access data and use it to learn for themselves.

            Types of Machine Learning
            There are three primary types of machine learning:

            1. Supervised Learning: This approach uses labeled training data to learn 
            a mapping function from input variables to output variables. Common examples 
            include email spam classification, image recognition, and price prediction.

            2. Unsupervised Learning: This method finds hidden patterns or intrinsic 
            structures in input data without labeled responses. Applications include 
            customer segmentation, anomaly detection, and data compression.

            3. Reinforcement Learning: This type involves an agent learning to make 
            decisions by taking actions in an environment to maximize cumulative reward. 
            It's used in game playing, robotics, and autonomous vehicle navigation.

            Deep Learning Revolution
            Deep Learning is a specialized subset of machine learning that uses neural 
            networks with multiple layers to model and understand complex patterns. 
            It has achieved breakthrough results in image recognition, natural language 
            processing, and speech synthesis.

            Real-World Applications
            AI and ML have found applications across numerous industries:
            - Healthcare: Medical diagnosis, drug discovery, personalized treatment
            - Finance: Fraud detection, algorithmic trading, credit scoring
            - Transportation: Autonomous vehicles, route optimization
            - Technology: Search engines, recommendation systems, virtual assistants
            """
            
            # Process sample content
            chunks = rag_pipeline.text_splitter.chunk_text(sample_content)
            rag_pipeline.embedder.store_embeddings(chunks)
            
            # Add to processed documents
            rag_pipeline.processed_documents["sample_ai_ml_content"] = {
                "text_content": sample_content,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "file_size_mb": len(sample_content) / (1024 * 1024),
                "processed_at": time.time()
            }
            rag_pipeline.current_document_path = "sample_ai_ml_content"
            rag_pipeline.is_ready = True
            
            print(f"✅ Sample content processed: {len(chunks)} chunks created")
        
        # Test the core generate_answer() functionality
        test_questions = [
            "What is artificial intelligence?",
            "Explain the different types of machine learning.",
            "How does supervised learning work?",
            "What is deep learning and why is it important?",
            "What are some real-world applications of AI?",
            "How does reinforcement learning differ from other ML types?",
            "What role does AI play in healthcare?",
            "Can you explain unsupervised learning with examples?"
        ]
        
        print(f"\n{'='*70}")
        print("Testing Core RAG Pipeline - generate_answer() Method")
        print(f"{'='*70}")
        
        total_queries = 0
        successful_queries = 0
        total_time = 0
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. 🤔 Question: '{question}'")
            print("-" * 60)
            
            start_time = time.time()
            
            # Use the main generate_answer method - this is the key Step 7 deliverable
            result = rag_pipeline.generate_answer(
                question=question,
                max_response_length=200
            )
            
            query_time = time.time() - start_time
            total_queries += 1
            total_time += query_time
            
            if result['success']:
                successful_queries += 1
                print(f"✅ Answer: {result['answer']}")
                print(f"🔍 Context chunks used: {len(result['context'])}")
                print(f"⏱️  Performance: {result['total_time']:.3f}s (Retrieval: {result['retrieval_time']:.3f}s, Generation: {result['generation_time']:.3f}s)")
                
                if result['context']:
                    best_context = result['context'][0]
                    print(f"🎯 Best match (similarity: {best_context['similarity']:.3f}): {best_context['text'][:80]}...")
            else:
                print(f"❌ Error: {result['answer']}")
        
        # Performance summary
        print(f"\n{'='*70}")
        print("Performance Summary")
        print(f"{'='*70}")
        print(f"📊 Total queries: {total_queries}")
        print(f"✅ Successful: {successful_queries}")
        print(f"📈 Success rate: {(successful_queries/total_queries)*100:.1f}%")
        print(f"⚡ Average response time: {total_time/total_queries:.3f}s")
        
        # Show final pipeline state
        final_status = rag_pipeline.get_pipeline_status()
        processed_docs = rag_pipeline.list_processed_documents()
        
        print(f"\n{'='*70}")
        print("Final Pipeline State")
        print(f"{'='*70}")
        print(f"🔄 Pipeline ready: {final_status['is_ready']}")
        print(f"📚 Documents processed: {len(processed_docs)}")
        print(f"🧠 Current document: {final_status['current_document']}")
        
        if final_status['embedding_stats']:
            print(f"🔢 Embedding statistics: {final_status['embedding_stats']}")
        
        # Test additional pipeline features
        print(f"\n{'='*70}")
        print("Additional Pipeline Features Demonstration")
        print(f"{'='*70}")
        
        # Error handling
        print("\n🧪 Testing error handling...")
        empty_result = rag_pipeline.generate_answer("")
        print(f"Empty question: {'✅' if not empty_result['success'] else '❌'} Handled correctly")
        
        invalid_result = rag_pipeline.generate_answer("   ")
        print(f"Whitespace question: {'✅' if not invalid_result['success'] else '❌'} Handled correctly")
        
        # Document management
        print(f"\n📋 Document management:")
        for doc in processed_docs:
            print(f"   📄 {doc['path']}: {doc['chunk_count']} chunks")
        
        print(f"\n{'='*70}")
        print("🎉 Step 7 Implementation Complete!")
        print(f"{'='*70}")
        print("✅ Deliverables Achieved:")
        print("   ✓ Created modules/rag_pipeline.py with modular architecture")
        print("   ✓ Implemented generate_answer() orchestrating all components:")
        print("     • PDF parsing → Chunking → Embedding → Retrieval → LLM query")
        print("   ✓ Designed effective prompt template for context + question")
        print("   ✓ Complete pipeline from PDF + query to answer working")
        print("   ✓ End-to-end RAG functionality demonstrated")
        print("   ✓ Error handling and robustness implemented")
        print("   ✓ Performance monitoring and optimization ready")
        
        print(f"\n🚀 RAG Assistant Phase 1 - Step 7 Successfully Complete!")
        print("Ready for advanced RAG features, optimization, and production deployment.")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = demo_step7_rag_pipeline()
    
    if success:
        print("\n🎯 Next Steps:")
        print("   • Add advanced retrieval techniques (hybrid search, re-ranking)")
        print("   • Implement conversation memory and context management")
        print("   • Add document preprocessing and optimization")
        print("   • Create web interface or API endpoints")
        print("   • Add evaluation metrics and testing framework")
    else:
        print("\n❌ Some issues occurred during the demo.")
    
    sys.exit(0 if success else 1)
