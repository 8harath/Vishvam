"""
RAG Assistant Phase 1 - Main CLI Application
Now with semantic search and retrieval capabilities (Step 5)
"""

import argparse
import sys
from pathlib import Path

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.pdf_parser import PDFParser
from modules.text_splitter import TextSplitter
from modules.embedder import create_embedder
from modules.vector_store import RAGRetriever


def semantic_search_demo():
    """Demonstrate semantic search capabilities (Step 5)."""
    print("\n🔍 SEMANTIC SEARCH DEMO (Step 5)")
    print("=" * 50)
    
    try:
        # Sample documents
        documents = [
            "Machine learning is a powerful subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information."
        ]
        
        # Initialize components
        embedder = create_embedder()
        retriever = RAGRetriever(embedder)
        
        # Add documents
        retriever.add_documents(documents)
        print(f"✅ Added {len(documents)} documents to search index")
        
        # Test queries
        queries = ["What is machine learning?", "How do neural networks work?"]
        
        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            results = retriever.retrieve(query, k=2)
            
            for i, (chunk, score, _) in enumerate(results, 1):
                print(f"  {i}. Score: {score:.3f} - {chunk}")
        
        print("\n✅ Semantic search demo completed!")
        
    except Exception as e:
        print(f"❌ Semantic search demo failed: {e}")


def main():
    """Main CLI application."""
    parser = argparse.ArgumentParser(
        description="RAG Assistant Phase 1 - PDF Processing with Semantic Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py sample_data/sample_document.pdf
  python main.py document.pdf --chunk-size 300
  python main.py --demo-search  # Demo semantic search
        """
    )
    
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to the PDF file to process"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Size of text chunks (default: 500)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save output files (default: output/)"
    )
    
    parser.add_argument(
        "--demo-search",
        action="store_true",
        help="Run semantic search demo instead of processing PDF"
    )
    
    args = parser.parse_args()
    
    # Run semantic search demo if requested
    if args.demo_search:
        semantic_search_demo()
        return
    
    # Require PDF path if not running demo
    if not args.pdf_path:
        parser.error("PDF path is required unless using --demo-search")
    
    pdf_path = Path(args.pdf_path)
    
    try:
        print(f"📄 Processing PDF: {pdf_path}")
        print(f"⚙️  Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
        
        # Initialize components
        pdf_parser = PDFParser()
        text_splitter = TextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Parse PDF
        text = pdf_parser.extract_text_from_pdf(str(pdf_path))
        if not text:
            print("❌ No text could be extracted from the PDF")
            return
        
        print(f"📝 Extracted {len(text)} characters from PDF")
        
        # Split into chunks
        chunks = text_splitter.chunk_text(text)
        print(f"🔀 Split into {len(chunks)} chunks")
        
        # Save output
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save full text
        text_file = output_dir / f"{pdf_path.stem}_full_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"💾 Full text saved to: {text_file}")
        
        # Save chunks
        chunks_file = output_dir / f"{pdf_path.stem}_chunks.txt"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"=== CHUNK {i} ===\n")
                f.write(chunk)
                f.write(f"\n\n")
        print(f"📦 Chunks saved to: {chunks_file}")
        
        # NEW: Create semantic search index
        print("\n🔍 Creating semantic search index...")
        embedder = create_embedder()
        retriever = RAGRetriever(embedder)
        retriever.add_documents(chunks)
        print(f"✅ Search index created with {len(chunks)} chunks")
        
        # Demo search on the processed document
        print("\n🔍 Testing search on processed document:")
        test_queries = [
            "What is the main topic?",
            "Key features and capabilities",
            "How does it work?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = retriever.retrieve(query, k=2, min_similarity=0.1)
            
            if results:
                for i, (chunk, score, _) in enumerate(results, 1):
                    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    print(f"  {i}. Score: {score:.3f} - {preview}")
            else:
                print("  No relevant chunks found")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    
    print("\n✅ PDF processing with semantic search completed!")


if __name__ == "__main__":
    main()
