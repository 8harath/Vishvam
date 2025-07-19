"""
RAG Assistant Phase 1 - Main CLI Application
Simple demonstration of PDF text extraction
"""

import argparse
import sys
from pathlib import Path

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.pdf_parser import PDFParser

def main():
    """Main CLI application."""
    parser = argparse.ArgumentParser(
        description="RAG Assistant Phase 1 - PDF Text Extraction Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pdf sample_data/sample_document.pdf
  python main.py --pdf sample_data/sample_document.pdf --info
  python main.py --help
        """
    )
    
    parser.add_argument(
        "--pdf", 
        type=str, 
        help="Path to PDF file to process"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Display PDF metadata information"
    )
    
    parser.add_argument(
        "--pages", 
        action="store_true", 
        help="Extract text page by page"
    )
    
    args = parser.parse_args()
    
    if not args.pdf:
        parser.print_help()
        return
    
    # Initialize PDF parser
    pdf_parser = PDFParser()
    
    print(f"🔍 Processing PDF: {args.pdf}")
    print("-" * 50)
    
    try:
        # Show PDF info if requested
        if args.info:
            print("📄 PDF Information:")
            info = pdf_parser.get_pdf_info(args.pdf)
            for key, value in info.items():
                print(f"  {key}: {value}")
            print()
        
        # Extract text page by page if requested
        if args.pages:
            print("📑 Extracting text page by page:")
            pages = pdf_parser.extract_text_by_page(args.pdf)
            for i, page_text in enumerate(pages, 1):
                print(f"\n--- Page {i} ---")
                print(page_text[:200] + "..." if len(page_text) > 200 else page_text)
        else:
            # Extract all text
            print("📝 Extracted Text:")
            text = pdf_parser.extract_text_from_pdf(args.pdf)
            
            if text:
                print(f"✅ Successfully extracted {len(text)} characters")
                print("-" * 50)
                
                # Show first 500 characters as preview
                preview = text[:500] + "..." if len(text) > 500 else text
                print(preview)
                
                if len(text) > 500:
                    print(f"\n[Showing first 500 characters of {len(text)} total]")
            else:
                print("❌ No text could be extracted from the PDF")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    
    print("\n✅ PDF processing completed!")

if __name__ == "__main__":
    main()
