"""
Validation test for Step 3: Text Chunking System Success Criteria
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.text_splitter import TextSplitter, chunk_text
from sample_pdf_generator import get_sample_content


def validate_success_criteria():
    """Validate all success criteria for Step 3."""
    print("🎯 Validating Step 3: Text Chunking System Success Criteria")
    print("=" * 70)
    
    success_count = 0
    total_criteria = 5
    
    # 1. Text splitter with configurable chunk size
    print("✅ Criterion 1: Text splitter with configurable chunk size")
    try:
        # Test different chunk sizes
        splitter_100 = TextSplitter(chunk_size=100)
        splitter_500 = TextSplitter(chunk_size=500) 
        splitter_1000 = TextSplitter(chunk_size=1000)
        
        assert splitter_100.chunk_size == 100
        assert splitter_500.chunk_size == 500
        assert splitter_1000.chunk_size == 1000
        
        print("   ✓ Configurable chunk sizes working correctly")
        success_count += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 2. Default 500-character chunks implemented
    print("\n✅ Criterion 2: 500-character default chunks implemented")
    try:
        # Test default chunk size
        default_splitter = TextSplitter()
        assert default_splitter.chunk_size == 500
        
        # Test convenience function with defaults
        sample_text = get_sample_content()
        chunks = chunk_text(sample_text)
        
        # Verify chunks are approximately 500 characters (with overlap and word boundaries)
        avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)
        assert 400 <= avg_size <= 600  # Allow for overlap and word boundaries
        
        print(f"   ✓ Default 500-char chunks working (avg: {avg_size:.1f} chars)")
        success_count += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 3. Chunking tested with extracted PDF text
    print("\n✅ Criterion 3: Chunking tested with extracted PDF text")
    try:
        pdf_content = get_sample_content()
        
        # Verify it's substantial PDF-like content
        assert len(pdf_content) > 3000
        assert "SMART HOME LED CONTROLLER" in pdf_content
        assert "TECHNICAL SPECIFICATIONS" in pdf_content
        
        # Test chunking with PDF content
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.chunk_text(pdf_content)
        
        # Verify reasonable chunking results
        assert len(chunks) > 5  # Should create multiple chunks
        assert all(len(chunk) > 0 for chunk in chunks)  # All chunks have content
        
        print(f"   ✓ PDF content chunked successfully ({len(chunks)} chunks)")
        success_count += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 4. Future enhancement prepared (sentence-based chunking)
    print("\n✅ Criterion 4: Future enhancement prepared (sentence-based chunking)")
    try:
        # Verify sentence-based chunking method exists and works
        splitter = TextSplitter(chunk_size=200)
        test_text = """
        This is sentence one. This is sentence two! This is sentence three?
        This is a longer sentence that contains multiple clauses and should be handled properly.
        Short sentence. Another short one.
        """
        
        sentence_chunks = splitter.chunk_text_by_sentences(test_text)
        
        # Verify method works
        assert len(sentence_chunks) > 0
        assert all(len(chunk.strip()) > 0 for chunk in sentence_chunks)
        
        print(f"   ✓ Sentence-based chunking ready ({len(sentence_chunks)} chunks)")
        success_count += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 5. Large text splits into manageable chunks
    print("\n✅ Criterion 5: Large text splits into manageable chunks")
    try:
        # Create large text content
        large_text = get_sample_content() * 5  # Make it even larger
        assert len(large_text) > 15000  # Verify it's large
        
        # Test chunking large content
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.chunk_text(large_text)
        stats = splitter.get_chunk_stats(chunks)
        
        # Verify manageable chunk sizes
        assert stats['max_chunk_size'] <= 550  # Should not exceed chunk_size by much
        assert stats['min_chunk_size'] > 0     # All chunks should have content
        assert len(chunks) > 20                # Should create many chunks from large text
        
        print(f"   ✓ Large text ({len(large_text)} chars) split into {len(chunks)} manageable chunks")
        print(f"     Chunk size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} characters")
        success_count += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Summary
    print(f"\n🎯 SUCCESS CRITERIA VALIDATION COMPLETE")
    print("-" * 50)
    print(f"✅ Passed: {success_count}/{total_criteria} criteria")
    
    if success_count == total_criteria:
        print("🎉 ALL SUCCESS CRITERIA MET! Step 3 is complete.")
        return True
    else:
        print(f"❌ {total_criteria - success_count} criteria failed. Review implementation.")
        return False


def demonstrate_key_features():
    """Demonstrate key features of the text chunking system."""
    print("\n🔧 Key Features Demonstration")
    print("-" * 50)
    
    sample_text = get_sample_content()[:1500]  # Use first 1500 chars
    
    # Feature 1: Configurable chunk size
    print("1. Configurable Chunk Size:")
    for size in [200, 500, 800]:
        chunks = chunk_text(sample_text, chunk_size=size)
        print(f"   Size {size}: {len(chunks)} chunks (avg: {sum(len(c) for c in chunks) / len(chunks):.1f} chars)")
    
    # Feature 2: Overlap for context preservation  
    print("\n2. Configurable Overlap:")
    for overlap in [0, 25, 50]:
        chunks = chunk_text(sample_text, chunk_size=300, chunk_overlap=overlap)
        print(f"   Overlap {overlap}: {len(chunks)} chunks")
    
    # Feature 3: Word boundary preservation
    print("\n3. Word Boundary Preservation:")
    splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.chunk_text("This is a test sentence with multiple words that should not be split inappropriately.")
    
    for i, chunk in enumerate(chunks, 1):
        # Check if chunk starts/ends with complete words
        starts_clean = chunk[0].isspace() or not chunk[0].isalnum() or i == 1
        ends_clean = chunk[-1].isspace() or not chunk[-1].isalnum() or chunk.endswith(('.', '!', '?'))
        print(f"   Chunk {i}: {'✓' if starts_clean and ends_clean else '?'} '{chunk}'")
    
    # Feature 4: Statistics and monitoring
    print("\n4. Comprehensive Statistics:")
    splitter = TextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.chunk_text(sample_text)
    stats = splitter.get_chunk_stats(chunks)
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key.replace('_', ' ').title()}: {value:.1f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    all_passed = validate_success_criteria()
    demonstrate_key_features()
    
    if all_passed:
        print(f"\n🏆 STEP 3 COMPLETION CONFIRMED")
        print("   The Text Chunking System is ready for the RAG pipeline!")
    else:
        print(f"\n⚠️  STEP 3 NEEDS ATTENTION")
        print("   Please review failed criteria before proceeding.")
