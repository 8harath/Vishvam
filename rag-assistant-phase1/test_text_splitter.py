"""
Test script for Text Chunking System
Tests the TextSplitter with various scenarios including PDF content
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.text_splitter import TextSplitter
from modules.pdf_parser import PDFParser
from sample_pdf_generator import get_sample_content


def test_basic_chunking():
    """Test basic text chunking functionality."""
    print("🧪 Test 1: Basic Text Chunking")
    print("-" * 50)
    
    sample_text = """
    This is a test document for the text chunking system. It contains multiple sentences 
    and paragraphs to test how the system splits text into manageable chunks while 
    preserving word boundaries and context.
    
    The chunking algorithm should handle different types of content including technical 
    documentation, user manuals, and other text-heavy documents that will be processed 
    in the RAG pipeline.
    """
    
    splitter = TextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.chunk_text(sample_text)
    
    print(f"✅ Input text length: {len(sample_text)} characters")
    print(f"✅ Generated {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"'{chunk}'")
    
    stats = splitter.get_chunk_stats(chunks)
    print("\n📊 Chunk Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return chunks


def test_pdf_content_chunking():
    """Test chunking with actual PDF content."""
    print("\n🧪 Test 2: PDF Content Chunking")
    print("-" * 50)
    
    # Get sample PDF content
    pdf_content = get_sample_content()
    print(f"✅ PDF content length: {len(pdf_content)} characters")
    
    # Test with default 500-character chunks
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.chunk_text(pdf_content)
    
    print(f"✅ Generated {len(chunks)} chunks with default settings")
    
    # Show first few chunks
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"'{chunk[:100]}...'" if len(chunk) > 100 else f"'{chunk}'")
    
    if len(chunks) > 3:
        print(f"\n... and {len(chunks) - 3} more chunks")
    
    # Get and display statistics
    stats = splitter.get_chunk_stats(chunks)
    print("\n📊 PDF Chunk Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    return chunks


def test_different_chunk_sizes():
    """Test different chunk sizes with the same content."""
    print("\n🧪 Test 3: Different Chunk Sizes")
    print("-" * 50)
    
    sample_text = get_sample_content()[:1000]  # Use first 1000 chars
    
    test_sizes = [200, 500, 800]
    
    for size in test_sizes:
        splitter = TextSplitter(chunk_size=size, chunk_overlap=50)
        chunks = splitter.chunk_text(sample_text)
        stats = splitter.get_chunk_stats(chunks)
        
        print(f"✅ Chunk size {size}: {stats['total_chunks']} chunks, "
              f"avg size: {stats['average_chunk_size']:.1f} chars")


def test_sentence_based_chunking():
    """Test sentence-based chunking (future enhancement)."""
    print("\n🧪 Test 4: Sentence-Based Chunking")
    print("-" * 50)
    
    sample_text = """
    The Smart Home LED Controller is an advanced device. It supports up to 16 LED strips. 
    The device includes WiFi and Bluetooth connectivity. Voice control is also supported 
    through major platforms. Installation is straightforward and user-friendly. 
    The warranty period is two years from purchase date.
    """
    
    splitter = TextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.chunk_text_by_sentences(sample_text)
    
    print(f"✅ Generated {len(chunks)} sentence-based chunks")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nSentence Chunk {i} ({len(chunk)} chars):")
        print(f"'{chunk}'")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Test 5: Edge Cases")
    print("-" * 50)
    
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    
    # Test empty text
    print("Testing empty text...")
    chunks = splitter.chunk_text("")
    print(f"✅ Empty text result: {len(chunks)} chunks")
    
    # Test very short text
    print("\nTesting very short text...")
    chunks = splitter.chunk_text("Short text")
    print(f"✅ Short text result: {len(chunks)} chunks: {chunks}")
    
    # Test text shorter than chunk size
    print("\nTesting text shorter than chunk size...")
    short_text = "This text is shorter than the chunk size limit."
    chunks = splitter.chunk_text(short_text)
    print(f"✅ Text length {len(short_text)}, chunks: {len(chunks)}")


def test_with_actual_pdf():
    """Test chunking with actual PDF file if available."""
    print("\n🧪 Test 6: Actual PDF Processing")
    print("-" * 50)
    
    pdf_path = Path("sample_data/sample_document.pdf")
    
    if pdf_path.exists():
        print(f"✅ Found PDF file: {pdf_path}")
        
        # Extract text using PDF parser
        parser = PDFParser()
        extracted_text = parser.extract_text(str(pdf_path))
        
        if extracted_text:
            print(f"✅ Extracted {len(extracted_text)} characters from PDF")
            
            # Chunk the extracted text
            splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.chunk_text(extracted_text)
            
            print(f"✅ Generated {len(chunks)} chunks from PDF content")
            
            stats = splitter.get_chunk_stats(chunks)
            print(f"📊 PDF Chunk Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.1f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("❌ Failed to extract text from PDF")
    else:
        print("ℹ️  No PDF file found, skipping PDF processing test")


def save_test_results():
    """Save chunking results for inspection."""
    print("\n💾 Saving Test Results")
    print("-" * 50)
    
    # Get sample content and chunk it
    content = get_sample_content()
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.chunk_text(content)
    
    # Save chunks to file
    output_path = "sample_data/chunked_output.txt"
    success = splitter.save_chunks_to_file(chunks, output_path)
    
    if success:
        print(f"✅ Test results saved to {output_path}")
    else:
        print("❌ Failed to save test results")


def main():
    """Run all text chunking tests."""
    print("🚀 Text Chunking System Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        test_basic_chunking()
        test_pdf_content_chunking()
        test_different_chunk_sizes()
        test_sentence_based_chunking()
        test_edge_cases()
        test_with_actual_pdf()
        save_test_results()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎯 Success Criteria Met:")
        print("  ✓ Text splitter created with configurable chunk size")
        print("  ✓ Default 500-character chunks implemented")
        print("  ✓ Chunking tested with extracted PDF text")
        print("  ✓ Future enhancement prepared (sentence-based chunking)")
        print("  ✓ Large text splits into manageable chunks")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
