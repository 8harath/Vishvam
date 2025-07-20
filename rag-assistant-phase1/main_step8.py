#!/usr/bin/env python3
"""
RAG Assistant Phase 1 - Step 8: Main Application & Demo
Complete CLI interface with interactive Q&A functionality
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.rag_pipeline import RAGPipeline, create_rag_pipeline

# ANSI color codes for better terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠️ {message}{Colors.ENDC}")

def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.OKBLUE}ℹ️ {message}{Colors.ENDC}")

def format_time(seconds: float) -> str:
    """Format time duration for display."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def display_answer_result(result: Dict[str, Any]):
    """Display the answer result in a formatted way."""
    print(f"\n{Colors.OKCYAN}💬 Question:{Colors.ENDC}")
    print(f"   {result['question']}")
    
    print(f"\n{Colors.OKGREEN}🤖 Answer:{Colors.ENDC}")
    print(f"   {result['answer']}")
    
    # Performance metrics
    print(f"\n{Colors.OKBLUE}📊 Performance Metrics:{Colors.ENDC}")
    print(f"   • Total Time: {format_time(result['total_time'])}")
    print(f"   • Retrieval Time: {format_time(result['retrieval_time'])}")
    print(f"   • Generation Time: {format_time(result['generation_time'])}")
    
    # Context information
    if result.get('context'):
        print(f"\n{Colors.OKBLUE}📋 Context Used:{Colors.ENDC}")
        for i, ctx in enumerate(result['context'], 1):
            score = ctx.get('score', 0)
            preview = ctx.get('text', '')[:100] + "..." if len(ctx.get('text', '')) > 100 else ctx.get('text', '')
            print(f"   {i}. (Score: {score:.3f}) {preview}")

def interactive_mode(rag_pipeline: RAGPipeline):
    """Run the RAG assistant in interactive mode."""
    print_header("🚀 RAG Assistant - Interactive Mode")
    print_info("Type 'exit', 'quit', or 'q' to leave interactive mode")
    print_info("Type 'help' for available commands")
    print_info("Type 'status' to see pipeline status")
    print_info("Type 'docs' to see loaded documents")
    print()
    
    while True:
        try:
            # Get user input
            question = input(f"{Colors.OKCYAN}❓ Enter your question: {Colors.ENDC}").strip()
            
            # Handle special commands
            if question.lower() in ['exit', 'quit', 'q']:
                print_success("Goodbye! 👋")
                break
            elif question.lower() == 'help':
                print_commands_help()
                continue
            elif question.lower() == 'status':
                display_pipeline_status(rag_pipeline)
                continue
            elif question.lower() == 'docs':
                display_loaded_documents(rag_pipeline)
                continue
            elif not question:
                print_warning("Please enter a question or command")
                continue
            
            # Process the question
            print(f"\n{Colors.OKBLUE}🔍 Searching for relevant information...{Colors.ENDC}")
            
            result = rag_pipeline.generate_answer(question)
            
            if result['success']:
                display_answer_result(result)
            else:
                print_error(f"Failed to generate answer: {result.get('error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
            break
        except Exception as e:
            print_error(f"Error processing question: {str(e)}")

def print_commands_help():
    """Print available commands in interactive mode."""
    print(f"\n{Colors.OKBLUE}📖 Available Commands:{Colors.ENDC}")
    print("   • Ask any question about the loaded document")
    print("   • 'help' - Show this help message")
    print("   • 'status' - Show pipeline status")
    print("   • 'docs' - Show loaded documents")
    print("   • 'exit', 'quit', 'q' - Exit interactive mode")

def display_pipeline_status(rag_pipeline: RAGPipeline):
    """Display current pipeline status."""
    status = rag_pipeline.get_pipeline_status()
    
    print(f"\n{Colors.OKBLUE}📊 Pipeline Status:{Colors.ENDC}")
    print(f"   • Ready: {rag_pipeline.is_ready}")
    print(f"   • Components: {', '.join(status.get('components', []))}")
    if 'configuration' in status:
        print(f"   • Configuration: {status['configuration']}")
    
    if status.get('embedding_stats'):
        stats = status['embedding_stats']
        print(f"   • Chunks: {stats['chunk_count']}")
        print(f"   • Embedding Dimension: {stats['embedding_dimension']}")

def display_loaded_documents(rag_pipeline: RAGPipeline):
    """Display information about loaded documents."""
    docs = rag_pipeline.list_processed_documents()
    
    if not docs:
        print_warning("No documents currently loaded")
        return
    
    print(f"\n{Colors.OKBLUE}📚 Loaded Documents:{Colors.ENDC}")
    for i, doc in enumerate(docs, 1):
        print(f"   {i}. {doc['path']}")
        print(f"      • Size: {doc['size_mb']:.2f} MB")
        print(f"      • Chunks: {doc['chunk_count']}")
        print(f"      • Processing Time: {format_time(doc['processing_time'])}")

def run_demo_questions(rag_pipeline: RAGPipeline):
    """Run a set of demo questions to showcase the system."""
    print_header("🎯 Demo Questions - Product Manual Test")
    
    demo_questions = [
        "What is the warranty period for this router?",
        "How do I set up the WiFi router?",
        "What should I do if my internet connection is not working?",
        "What is the maximum speed of this router?",
        "How can I contact customer support?",
        "What is not covered under the warranty?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{Colors.OKCYAN}Demo Question {i}/{len(demo_questions)}:{Colors.ENDC}")
        print(f"❓ {question}")
        
        try:
            result = rag_pipeline.generate_answer(question)
            
            if result['success']:
                print(f"\n{Colors.OKGREEN}🤖 Answer:{Colors.ENDC}")
                print(f"   {result['answer']}")
                print(f"\n{Colors.OKBLUE}⏱️ Time: {format_time(result['total_time'])}{Colors.ENDC}")
            else:
                print_error(f"Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print_error(f"Error: {str(e)}")
        
        # Add small delay between questions for better readability
        time.sleep(1)

def main():
    """Main CLI application entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Assistant Phase 1 - Step 8: Main Application & Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with sample document
  python main_step8.py --pdf sample_data/sample_document.pdf --interactive
  
  # Interactive mode with product manual
  python main_step8.py --pdf sample_data/product_manual.pdf --interactive
  
  # Run demo questions
  python main_step8.py --pdf sample_data/product_manual.pdf --demo
  
  # Single question mode
  python main_step8.py --pdf sample_data/product_manual.pdf --question "What is the warranty period?"
  
  # Quick status check
  python main_step8.py --pdf sample_data/product_manual.pdf --status
        """
    )
    
    parser.add_argument(
        "--pdf", 
        type=str,
        required=True,
        help="Path to PDF file to load and query"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true", 
        help="Run in interactive Q&A mode"
    )
    
    parser.add_argument(
        "--demo", "-d",
        action="store_true", 
        help="Run demo questions showcase"
    )
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Ask a single question and exit"
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show pipeline status and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Print application header
    print_header("🚀 RAG Assistant Phase 1 - Step 8 Demo")
    
    try:
        # Initialize RAG Pipeline
        print_info("Initializing RAG Pipeline...")
        rag_pipeline = create_rag_pipeline()
        print_success("RAG Pipeline initialized!")
        
        # Load the PDF document
        print_info(f"Loading PDF document: {args.pdf}")
        if not os.path.exists(args.pdf):
            print_error(f"PDF file not found: {args.pdf}")
            sys.exit(1)
        
        success = rag_pipeline.load_document(args.pdf)
        if not success:
            print_error("Failed to load and process PDF document")
            sys.exit(1)
        
        print_success("Document loaded and processed successfully!")
        
        # Handle different modes
        if args.status:
            display_pipeline_status(rag_pipeline)
            display_loaded_documents(rag_pipeline)
        
        elif args.question:
            print_info(f"Processing question: {args.question}")
            result = rag_pipeline.generate_answer(args.question)
            
            if result['success']:
                display_answer_result(result)
            else:
                print_error(f"Failed to generate answer: {result.get('error', 'Unknown error')}")
        
        elif args.demo:
            run_demo_questions(rag_pipeline)
        
        elif args.interactive:
            interactive_mode(rag_pipeline)
        
        else:
            # Default: show status and enter interactive mode
            display_pipeline_status(rag_pipeline)
            print_info("No specific mode selected, entering interactive mode...")
            interactive_mode(rag_pipeline)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Application interrupted by user{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Application error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
