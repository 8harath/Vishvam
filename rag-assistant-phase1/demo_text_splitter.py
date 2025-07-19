"""
Simple demonstration of the Text Chunking System
"""

from modules.text_splitter import TextSplitter, chunk_text


def demo_simple_chunking():
    """Demonstrate the simple chunk_text function."""
    print("🧪 Simple Text Chunking Demo")
    print("-" * 40)
    
    sample_text = """
    The Smart Home LED Controller SH-LED-2024 is designed for modern home automation.
    It features WiFi connectivity, voice control integration, and supports up to 16 LED channels.
    The device comes with a comprehensive warranty and user manual for easy installation and setup.
    """
    
    # Use the convenience function
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    
    print(f"✅ Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: '{chunk[:60]}...'" if len(chunk) > 60 else f"  Chunk {i}: '{chunk}'")


def demo_advanced_chunking():
    """Demonstrate advanced TextSplitter features."""
    print("\n🧪 Advanced Text Splitter Demo")
    print("-" * 40)
    
    text = """
    INSTALLATION GUIDE
    
    1. Mounting the Controller
    Mount the controller in a dry, well-ventilated area away from direct sunlight and heat sources.
    Use the provided mounting screws to secure the unit to a wall or surface.
    
    2. Connecting LED Strips
    Connect your LED strips to the output terminals. Each channel supports up to 2A current draw.
    Ensure proper polarity when connecting (red = positive, black = negative).
    
    3. Power Connection
    Connect the provided power adapter to the controller and plug into a standard wall outlet.
    The power LED will illuminate green when properly connected.
    """
    
    splitter = TextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.chunk_text(text)
    stats = splitter.get_chunk_stats(chunks)
    
    print(f"✅ Text split into {stats['total_chunks']} chunks")
    print(f"   Average chunk size: {stats['average_chunk_size']:.1f} characters")
    print(f"   Size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} characters")
    
    print("\n📄 Sample chunks:")
    for i, chunk in enumerate(chunks[:2], 1):
        print(f"  Chunk {i} ({len(chunk)} chars): {chunk[:80]}...")


if __name__ == "__main__":
    print("🚀 Text Chunking System Demo")
    print("=" * 50)
    
    demo_simple_chunking()
    demo_advanced_chunking()
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Key Features:")
    print("  • Configurable chunk size (default: 500 characters)")
    print("  • Configurable overlap for context preservation")
    print("  • Word boundary preservation")
    print("  • Sentence-based chunking (future enhancement)")
    print("  • Comprehensive statistics and error handling")
