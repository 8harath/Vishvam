"""
Text Chunking System for RAG Assistant Phase 1

This module provides functionality to split large text documents into
manageable chunks for processing with configurable chunk sizes.
"""

import re
import logging
from typing import List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSplitter:
    """
    A configurable text chunking system for breaking down large documents.
    
    Features:
    - Character-based chunking with configurable size
    - Overlap support to preserve context between chunks
    - Word boundary preservation
    - Future-ready for sentence-based chunking enhancement
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters (default: 500)
            chunk_overlap (int): Number of characters to overlap between chunks (default: 50)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"TextSplitter initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text: str, preserve_word_boundaries: bool = True) -> List[str]:
        """
        Split text into chunks of specified size.
        
        Args:
            text (str): The text to be chunked
            preserve_word_boundaries (bool): Whether to avoid splitting words (default: True)
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return []
        
        # Clean and normalize the text
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            logger.info(f"Text length ({len(text)}) is within chunk size, returning single chunk")
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If we're at the end of the text, take everything
            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Find the best split point
            if preserve_word_boundaries:
                # Look for word boundaries within the last 10% of the chunk
                search_start = max(start + int(self.chunk_size * 0.9), start + self.chunk_size - 100)
                split_point = self._find_best_split_point(text, search_start, end)
                if split_point > start:
                    end = split_point
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position considering overlap
            start = max(end - self.chunk_overlap, start + 1)
        
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    
    def chunk_text_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks by sentences (future enhancement).
        
        Args:
            text (str): The text to be chunked
            
        Returns:
            List[str]: List of text chunks based on sentence boundaries
        """
        # Split text into sentences using regex
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap if configured
                    if self.chunk_overlap > 0 and chunks:
                        current_chunk = self._get_overlap_text(current_chunk, sentence)
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is longer than chunk size
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Text split into {len(chunks)} sentence-based chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces and tabs
        text = text.strip()
        return text
    
    def _find_best_split_point(self, text: str, search_start: int, max_end: int) -> int:
        """
        Find the best point to split text while preserving word boundaries.
        
        Args:
            text (str): The text to search in
            search_start (int): Start position for searching
            max_end (int): Maximum end position
            
        Returns:
            int: Best split position
        """
        # Priority order for split points
        split_chars = ['\n\n', '\n', '.', '!', '?', ';', ',', ' ']
        
        for split_char in split_chars:
            pos = text.rfind(split_char, search_start, max_end)
            if pos != -1:
                # For sentence endings, include the punctuation
                if split_char in '.!?':
                    return pos + 1
                # For other splits, split after the character
                return pos + len(split_char) if split_char != ' ' else pos
        
        # If no good split point found, use max_end
        return max_end
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text (str): Text to split into sentences
            
        Returns:
            List[str]: List of sentences
        """
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _get_overlap_text(self, previous_chunk: str, next_sentence: str) -> str:
        """
        Get overlap text from previous chunk for context preservation.
        
        Args:
            previous_chunk (str): The previous chunk
            next_sentence (str): The next sentence to start with
            
        Returns:
            str: Text with appropriate overlap
        """
        if self.chunk_overlap <= 0:
            return next_sentence
        
        # Get last part of previous chunk for overlap
        overlap_text = previous_chunk[-self.chunk_overlap:].strip()
        
        # Find word boundary for clean overlap
        space_pos = overlap_text.find(' ')
        if space_pos != -1:
            overlap_text = overlap_text[space_pos:].strip()
        
        return f"{overlap_text} {next_sentence}" if overlap_text else next_sentence
    
    def get_chunk_stats(self, chunks: List[str]) -> dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            
        Returns:
            dict: Statistics about the chunks
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'average_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'average_chunk_size': sum(chunk_sizes) / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }
        
        return stats
    
    def save_chunks_to_file(self, chunks: List[str], output_path: str) -> bool:
        """
        Save chunks to a file for inspection or debugging.
        
        Args:
            chunks (List[str]): List of chunks to save
            output_path (str): Path to save the chunks
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"=== CHUNK {i} ===\n")
                    f.write(f"Length: {len(chunk)} characters\n")
                    f.write("-" * 50 + "\n")
                    f.write(chunk)
                    f.write("\n" + "=" * 60 + "\n\n")
            
            logger.info(f"Chunks saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            return False


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Convenience function to chunk text with default parameters.
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Size of each chunk (default: 500)
        chunk_overlap (int): Overlap between chunks (default: 50)
        
    Returns:
        List[str]: List of text chunks
    """
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.chunk_text(text)


if __name__ == "__main__":
    # Quick test with sample text
    sample_text = """
    This is a sample document for testing the text chunking system. 
    It contains multiple paragraphs and sentences to demonstrate how 
    the chunking algorithm works with different text structures.
    
    The text splitter should be able to handle various types of content
    and split them into manageable chunks while preserving word boundaries
    and maintaining context through overlapping sections.
    """
    
    print("🧪 Testing Text Splitter...")
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.chunk_text(sample_text)
    
    print(f"✅ Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars): {chunk[:50]}...")
    
    stats = splitter.get_chunk_stats(chunks)
    print(f"\n📊 Stats: {stats}")
