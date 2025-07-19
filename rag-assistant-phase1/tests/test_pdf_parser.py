"""
Tests for PDF Parser Module
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pdf_parser import PDFParser, extract_text_from_pdf

def test_pdf_parser():
    """Test PDF parser with various scenarios."""
    print("🧪 Testing PDF Parser Module...")
    
    parser = PDFParser()
    
    # Test 1: Valid PDF file
    print("\n1. Testing with valid PDF...")
    pdf_path = "sample_data/sample_document.pdf"
    
    if os.path.exists(pdf_path):
        text = parser.extract_text_from_pdf(pdf_path)
        if text and len(text) > 100:
            print("✅ Successfully extracted text from valid PDF")
            print(f"   Extracted {len(text)} characters")
        else:
            print("❌ Failed to extract sufficient text from valid PDF")
    else:
        print("⚠️  Sample PDF not found, skipping valid PDF test")
    
    # Test 2: Non-existent file
    print("\n2. Testing with non-existent file...")
    result = parser.extract_text_from_pdf("non_existent_file.pdf")
    if result is None:
        print("✅ Correctly handled non-existent file")
    else:
        print("❌ Should return None for non-existent file")
    
    # Test 3: Non-PDF file
    print("\n3. Testing with non-PDF file...")
    result = parser.extract_text_from_pdf("README.md")
    if result is None:
        print("✅ Correctly rejected non-PDF file")
    else:
        print("❌ Should reject non-PDF files")
    
    # Test 4: PDF info
    print("\n4. Testing PDF info extraction...")
    if os.path.exists(pdf_path):
        info = parser.get_pdf_info(pdf_path)
        if info and 'num_pages' in info:
            print("✅ Successfully extracted PDF metadata")
            print(f"   Pages: {info.get('num_pages', 'Unknown')}")
            print(f"   Size: {info.get('file_size_bytes', 'Unknown')} bytes")
        else:
            print("❌ Failed to extract PDF metadata")
    
    # Test 5: Page-by-page extraction
    print("\n5. Testing page-by-page extraction...")
    if os.path.exists(pdf_path):
        pages = parser.extract_text_by_page(pdf_path)
        if pages and len(pages) > 0:
            print(f"✅ Successfully extracted {len(pages)} pages")
        else:
            print("❌ Failed to extract pages")
    
    print("\n🎉 PDF Parser tests completed!")

def test_convenience_function():
    """Test the convenience function."""
    print("\n🧪 Testing convenience function...")
    
    pdf_path = "sample_data/sample_document.pdf"
    if os.path.exists(pdf_path):
        text = extract_text_from_pdf(pdf_path)
        if text and len(text) > 100:
            print("✅ Convenience function works correctly")
        else:
            print("❌ Convenience function failed")
    else:
        print("⚠️  Sample PDF not found for convenience function test")

if __name__ == "__main__":
    test_pdf_parser()
    test_convenience_function()
