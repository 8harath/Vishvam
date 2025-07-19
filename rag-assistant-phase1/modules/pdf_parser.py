"""
PDF Text Extraction Engine for RAG Assistant Phase 1

This module provides functionality to extract text from PDF documents
using PyPDF2 library with comprehensive error handling.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    raise ImportError(
        "PyPDF2 is required for PDF parsing. Install it with: pip install PyPDF2>=3.0.1"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFParser:
    """
    A robust PDF text extraction engine with error handling and validation.
    
    Features:
    - Extract text from PDF files using PyPDF2
    - Handle corrupted and encrypted PDFs
    - Validate file existence and format
    - Provide detailed logging and error messages
    """
    
    def __init__(self):
        """Initialize the PDF parser."""
        self.supported_extensions = ['.pdf']
        logger.info("PDFParser initialized successfully")
    
    def validate_file(self, pdf_path: str) -> bool:
        """
        Validate if the file exists and is a PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            file_path = Path(pdf_path)
            
            # Check if file exists
            if not file_path.exists():
                logger.error(f"File not found: {pdf_path}")
                return False
            
            # Check if file has PDF extension
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
            
            # Check if file is not empty
            if file_path.stat().st_size == 0:
                logger.error(f"File is empty: {pdf_path}")
                return False
                
            logger.info(f"File validation successful: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {pdf_path}: {str(e)}")
            return False
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from a PDF file with comprehensive error handling.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Optional[str]: Extracted text if successful, None if failed
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the file is not a valid PDF
            Exception: For other PDF processing errors
        """
        
        # Validate file first
        if not self.validate_file(pdf_path):
            return None
        
        extracted_text = ""
        
        try:
            logger.info(f"Starting text extraction from: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF is encrypted: {pdf_path}")
                    # Try to decrypt with empty password (common case)
                    try:
                        pdf_reader.decrypt("")
                        logger.info("Successfully decrypted PDF with empty password")
                    except Exception as e:
                        logger.error(f"Could not decrypt PDF - password required: {str(e)}")
                        raise ValueError("PDF is password protected and cannot be decrypted")
                
                # Get number of pages
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {num_pages} pages")
                
                if num_pages == 0:
                    logger.warning("PDF has no pages")
                    return ""
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n\n"
                        logger.debug(f"Extracted text from page {page_num}")
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num}: {str(e)}")
                        continue
                
                # Clean up extracted text
                extracted_text = self._clean_text(extracted_text)
                
                if not extracted_text.strip():
                    logger.warning("No text could be extracted from the PDF")
                    return ""
                
                logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF")
                return extracted_text
                
        except FileNotFoundError:
            logger.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"The file {pdf_path} was not found")
            
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Invalid or corrupted PDF file {pdf_path}: {str(e)}")
            raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace and skip empty lines
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Join lines with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Replace multiple spaces with single space
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get metadata information about the PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Dictionary containing PDF metadata
        """
        if not self.validate_file(pdf_path):
            return {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                info = {
                    'num_pages': len(pdf_reader.pages),
                    'is_encrypted': pdf_reader.is_encrypted,
                    'file_size_bytes': os.path.getsize(pdf_path),
                    'file_path': pdf_path
                }
                
                # Try to get metadata
                if pdf_reader.metadata:
                    info.update({
                        'title': pdf_reader.metadata.get('/Title', 'Unknown'),
                        'author': pdf_reader.metadata.get('/Author', 'Unknown'),
                        'subject': pdf_reader.metadata.get('/Subject', 'Unknown'),
                        'creator': pdf_reader.metadata.get('/Creator', 'Unknown'),
                        'producer': pdf_reader.metadata.get('/Producer', 'Unknown'),
                    })
                
                return info
                
        except Exception as e:
            logger.error(f"Error getting PDF info for {pdf_path}: {str(e)}")
            return {'error': str(e)}
    
    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        """
        Extract text from PDF and return as a list of pages.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[str]: List of text content for each page
        """
        if not self.validate_file(pdf_path):
            return []
        
        pages_text = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.is_encrypted:
                    pdf_reader.decrypt("")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        cleaned_text = self._clean_text(page_text) if page_text else ""
                        pages_text.append(cleaned_text)
                        logger.debug(f"Extracted text from page {page_num}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {str(e)}")
                        pages_text.append("")
                
                return pages_text
                
        except Exception as e:
            logger.error(f"Error extracting pages from {pdf_path}: {str(e)}")
            return []


# Convenience function for simple usage
def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Convenience function to extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Optional[str]: Extracted text if successful, None if failed
    """
    parser = PDFParser()
    return parser.extract_text_from_pdf(pdf_path)


if __name__ == "__main__":
    # Simple test when run directly
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        print(f"Testing PDF extraction with: {pdf_file}")
        
        parser = PDFParser()
        
        # Get PDF info
        info = parser.get_pdf_info(pdf_file)
        print(f"\nPDF Info: {info}")
        
        # Extract text
        text = parser.extract_text_from_pdf(pdf_file)
        
        if text:
            print(f"\nExtracted text ({len(text)} characters):")
            print("=" * 50)
            print(text[:500] + "..." if len(text) > 500 else text)
        else:
            print("Failed to extract text from PDF")
    else:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        print("Example: python pdf_parser.py sample_data/sample_document.pdf")
