"""
Integration test for Step 5 with previous modules
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.pdf_parser import PDFParser
from modules.text_splitter import TextSplitter
from modules.embedder import create_embedder
from modules.vector_store import RAGRetriever
import config


def test_full_pipeline_integration():
    """Test the complete pipeline from PDF to semantic search."""
    
    print("🧪 FULL PIPELINE INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Step 1: Parse PDF
        parser = PDFParser()
        pdf_path = config.DEFAULT_PDF_PATH
        
        if os.path.exists(pdf_path):
            print("✅ Found sample PDF, parsing...")
            text = parser.extract_text_from_pdf(pdf_path)
            print(f"✅ PDF parsed: {len(text)} characters")
        else:
            print("⚠️  Sample PDF not found, using test text...")
            text = """
            Machine learning is a powerful tool for creating intelligent systems. 
            Deep learning uses neural networks to solve complex problems.
            Natural language processing helps computers understand human language.
            Computer vision enables machines to interpret visual information.
            Reinforcement learning trains agents through environmental interaction.
            """
        
        # Step 2: Split text into chunks
        splitter = TextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        chunks = splitter.chunk_text(text)
        print(f"✅ Text split into {len(chunks)} chunks")
        
        # Step 3: Create embedder and retriever
        embedder = create_embedder()
        retriever = RAGRetriever(embedder)
        
        print(f"✅ Embedder and retriever initialized")
        
        # Step 4: Add documents to retrieval system
        retriever.add_documents(chunks)
        print(f"✅ Documents added to retrieval system")
        
        # Step 5: Test semantic search
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Tell me about computer vision"
        ]
        
        print(f"\n🔍 Testing semantic search with {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            # Test retrieval
            results = retriever.retrieve(query, k=3, min_similarity=0.1)
            print(f"   Found {len(results)} relevant chunks")
            
            if results:
                top_chunk, top_score, _ = results[0]
                print(f"   Top result (score: {top_score:.3f}): {top_chunk[:80]}...")
            
            # Test context generation
            context = retriever.get_context(query, k=2)
            print(f"   Generated context: {len(context)} characters")
        
        print(f"\n🎉 FULL PIPELINE INTEGRATION TEST PASSED!")
        print("All components working together successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_pipeline_integration()
    if success:
        print("\n✅ Step 5 successfully integrated with previous modules!")
    else:
        print("\n❌ Integration issues detected!")
