#!/usr/bin/env python3
"""
RAG Assistant Phase 1 - Step 9: Production-Ready Main Application
Enhanced with comprehensive error handling, logging, and robustness features
"""

import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.rag_pipeline import RAGPipeline, create_rag_pipeline
from modules.error_handler import (
    RAGAssistantError, DocumentProcessingError, PDFProcessingError,
    LLMProcessingError, MemoryError,
    get_logger, handle_exceptions, error_context, memory_monitor,
    ProgressTracker, ValidationUtils, HealthChecker
)

# Enhanced ANSI color codes for better terminal output
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


class RobustMainApp:
    """Production-ready RAG Assistant application with comprehensive error handling."""
    
    def __init__(self):
        """Initialize the application."""
        self.logger = get_logger()
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.health_checker = HealthChecker()
        self.start_time = time.time()
        
        # Application state
        self.is_initialized = False
        self.current_document = None
        self.total_questions_asked = 0
        self.successful_answers = 0
        
        self.logger.info("RobustMainApp initialized")
    
    def print_header(self, title: str):
        """Print a formatted header with error handling."""
        try:
            print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}{title.center(70)}{Colors.ENDC}")
            print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
        except Exception as e:
            self.logger.error(f"Error printing header: {e}")
            print(f"\n=== {title} ===\n")  # Fallback
    
    def print_success(self, message: str):
        """Print a success message."""
        try:
            print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")
            self.logger.info(f"SUCCESS: {message}")
        except Exception as e:
            self.logger.error(f"Error printing success message: {e}")
            print(f"✅ {message}")
    
    def print_error(self, message: str, exception: Optional[Exception] = None):
        """Print an error message with optional exception details."""
        try:
            print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")
            self.logger.error(f"ERROR: {message}")
            if exception:
                self.logger.debug(f"Exception details: {str(exception)}")
        except Exception as e:
            self.logger.critical(f"Error printing error message: {e}")
            print(f"❌ {message}")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        try:
            print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")
            self.logger.warning(message)
        except Exception as e:
            self.logger.error(f"Error printing warning: {e}")
            print(f"⚠️  {message}")
    
    def print_info(self, message: str):
        """Print an info message."""
        try:
            print(f"{Colors.OKBLUE}ℹ️  {message}{Colors.ENDC}")
            self.logger.info(message)
        except Exception as e:
            self.logger.error(f"Error printing info: {e}")
            print(f"ℹ️  {message}")
    
    @handle_exceptions(default_return=False, log_errors=True)
    def run_health_check(self) -> bool:
        """Run comprehensive system health check."""
        with error_context("System Health Check"):
            self.print_info("Running system health check...")
            
            health_results = self.health_checker.run_health_check()
            
            # Display results
            print("\n" + "="*50)
            print("🏥 SYSTEM HEALTH CHECK")
            print("="*50)
            
            # System resources
            sys_resources = health_results.get('system_resources', {})
            if sys_resources.get('monitoring_available', True):
                memory_gb = sys_resources.get('memory_available_gb', 0)
                memory_percent = sys_resources.get('memory_percent_used', 0)
                disk_gb = sys_resources.get('disk_free_gb', 0)
                cpu_percent = sys_resources.get('cpu_percent', 0)
                
                print(f"💾 Memory: {memory_gb:.1f} GB available ({memory_percent:.1f}% used)")
                print(f"💽 Disk Space: {disk_gb:.1f} GB free")
                print(f"🖥️  CPU Usage: {cpu_percent:.1f}%")
            else:
                print("💾 Resource monitoring not available")
            
            # Model availability
            model_status = health_results.get('model_availability', {})
            if model_status.get('sentence_transformers', False):
                print("🤖 Embedding models: ✅ Available")
            else:
                print(f"🤖 Embedding models: ❌ Issue - {model_status.get('error', 'Unknown')}")
            
            # Overall status
            overall_healthy = health_results.get('overall_healthy', False)
            status_color = Colors.OKGREEN if overall_healthy else Colors.FAIL
            status_icon = "✅" if overall_healthy else "❌"
            print(f"\n{status_color}Overall Status: {status_icon} {'Healthy' if overall_healthy else 'Issues Detected'}{Colors.ENDC}")
            
            if not overall_healthy:
                self.print_warning("System health issues detected. Some features may not work properly.")
            
            return overall_healthy
    
    @handle_exceptions(default_return=False, reraise=True, error_type=DocumentProcessingError)
    def initialize_rag_pipeline(self) -> bool:
        """Initialize the RAG pipeline with error handling."""
        with error_context("RAG Pipeline Initialization"):
            try:
                self.print_info("Initializing RAG Pipeline...")
                
                # Use memory monitoring
                with memory_monitor(max_memory_gb=8.0):
                    self.rag_pipeline = create_rag_pipeline()
                
                if self.rag_pipeline is None:
                    raise DocumentProcessingError("Failed to create RAG pipeline")
                
                self.is_initialized = True
                self.print_success("RAG Pipeline initialized successfully!")
                return True
                
            except MemoryError as e:
                raise DocumentProcessingError(f"Memory limit exceeded during initialization: {e}")
            except Exception as e:
                raise DocumentProcessingError(f"Pipeline initialization failed: {e}")
    
    @handle_exceptions(default_return=False, reraise=True)
    def load_and_validate_document(self, pdf_path: str) -> bool:
        """Load and validate PDF document with comprehensive error handling."""
        with error_context(f"Document Loading: {pdf_path}"):
            try:
                # Validate file path and size
                validated_path = ValidationUtils.validate_file_path(pdf_path, ['.pdf'])
                file_size_mb = ValidationUtils.validate_file_size(str(validated_path), max_size_mb=50.0)
                
                self.print_info(f"Loading PDF: {pdf_path} ({file_size_mb:.1f} MB)")
                
                if not self.rag_pipeline:
                    raise DocumentProcessingError("RAG pipeline not initialized")
                
                # Load document with progress tracking
                progress = ProgressTracker(total_steps=4, description="Document Processing")
                
                progress.update(1, "Validating PDF structure...")
                success = self.rag_pipeline.load_document(str(validated_path))
                
                progress.update(1, "Extracting text content...")
                if not success:
                    raise PDFProcessingError("Failed to load and process PDF document")
                
                progress.update(1, "Generating embeddings...")
                # Validate that document was actually processed
                if not hasattr(self.rag_pipeline, 'processed_documents') or not self.rag_pipeline.processed_documents:
                    raise DocumentProcessingError("Document processed but no content found")
                
                progress.update(1, "Building search index...")
                progress.finish("Document ready for querying")
                
                self.current_document = pdf_path
                self.print_success(f"Document loaded successfully: {Path(pdf_path).name}")
                
                return True
                
            except (FileNotFoundError, ValueError) as e:
                raise PDFProcessingError(f"File validation failed: {e}")
            except PDFProcessingError:
                raise  # Re-raise PDF-specific errors
            except Exception as e:
                raise DocumentProcessingError(f"Unexpected error loading document: {e}")
    
    @handle_exceptions(default_return=None, log_errors=True)
    def process_question_safely(self, question: str) -> Optional[Dict[str, Any]]:
        """Process a question with comprehensive error handling."""
        try:
            # Validate input
            validated_question = ValidationUtils.validate_text_content(question, min_length=3)
            
            with error_context(f"Question Processing: {validated_question[:50]}..."):
                if not self.rag_pipeline:
                    raise DocumentProcessingError("RAG pipeline not initialized")
                
                if not self.current_document:
                    raise DocumentProcessingError("No document loaded")
                
                self.total_questions_asked += 1
                self.print_info(f"Processing question: {validated_question}")
                
                # Process with timeout monitoring
                start_time = time.time()
                result = self.rag_pipeline.generate_answer(validated_question)
                processing_time = time.time() - start_time
                
                # Validate result
                if not isinstance(result, dict):
                    raise LLMProcessingError("Invalid response format from pipeline")
                
                if not result.get('success', False):
                    error_msg = result.get('error', 'Unknown error occurred')
                    raise LLMProcessingError(f"Answer generation failed: {error_msg}")
                
                # Log performance metrics
                self.logger.info(f"Question processed successfully in {processing_time:.2f}s")
                
                self.successful_answers += 1
                return result
                
        except ValueError as e:
            self.print_error(f"Invalid question: {e}")
            return None
        except LLMProcessingError as e:
            self.print_error(f"LLM processing error: {e}")
            return None
        except Exception as e:
            self.print_error(f"Unexpected error processing question: {e}", e)
            return None
    
    def display_answer_result(self, result: Dict[str, Any]):
        """Display answer result with error handling."""
        try:
            print(f"\n{Colors.OKGREEN}💬 Question:{Colors.ENDC} {result.get('question', 'N/A')}")
            print(f"\n{Colors.OKCYAN}🤖 Answer:{Colors.ENDC}")
            print(f"{result.get('answer', 'No answer available')}")
            
            # Performance metrics
            if 'performance' in result:
                perf = result['performance']
                print(f"\n{Colors.OKBLUE}📊 Performance Metrics:{Colors.ENDC}")
                print(f"   • Total Time: {perf.get('total_time', 0):.2f}s")
                print(f"   • Retrieval Time: {perf.get('retrieval_time', 0):.3f}s")
                print(f"   • Generation Time: {perf.get('generation_time', 0):.2f}s")
            
            # Context information
            if 'context_chunks' in result:
                chunks = result['context_chunks']
                print(f"\n{Colors.HEADER}📋 Context Used ({len(chunks)} chunks):{Colors.ENDC}")
                for i, chunk in enumerate(chunks[:3], 1):  # Show top 3
                    score = chunk.get('score', 0)
                    text_preview = chunk.get('text', '')[:100] + "..."
                    print(f"   {i}. (Score: {score:.3f}) {text_preview}")
            
        except Exception as e:
            self.logger.error(f"Error displaying result: {e}")
            print(f"Answer: {result.get('answer', 'Error displaying answer')}")
    
    def display_pipeline_status(self):
        """Display comprehensive pipeline status."""
        try:
            print("\n" + "="*60)
            print("📊 RAG PIPELINE STATUS")
            print("="*60)
            
            if not self.rag_pipeline:
                print("❌ Pipeline: Not initialized")
                return
            
            # Pipeline status
            print("✅ Pipeline: Initialized and ready")
            
            # Document status
            if self.current_document:
                print(f"📄 Current Document: {Path(self.current_document).name}")
                
                # Document details
                if hasattr(self.rag_pipeline, 'processed_documents'):
                    doc_info = self.rag_pipeline.processed_documents.get(self.current_document, {})
                    if doc_info:
                        print(f"   • Text Length: {doc_info.get('text_length', 0)} characters")
                        print(f"   • Chunks: {doc_info.get('num_chunks', 0)}")
                        print(f"   • Embeddings: {doc_info.get('num_embeddings', 0)}")
            else:
                print("📄 Document: None loaded")
            
            # Usage statistics
            print("\n📈 Usage Statistics:")
            print(f"   • Questions Asked: {self.total_questions_asked}")
            print(f"   • Successful Answers: {self.successful_answers}")
            if self.total_questions_asked > 0:
                success_rate = (self.successful_answers / self.total_questions_asked) * 100
                print(f"   • Success Rate: {success_rate:.1f}%")
            
            # Runtime information
            runtime = time.time() - self.start_time
            print(f"   • Runtime: {runtime:.1f}s")
            
        except Exception as e:
            self.logger.error(f"Error displaying status: {e}")
            print("❌ Error retrieving pipeline status")
    
    @handle_exceptions(default_return=False, log_errors=True)
    def run_demo_questions(self) -> bool:
        """Run demo questions with error handling."""
        if not self.rag_pipeline or not self.current_document:
            self.print_error("Pipeline not initialized or no document loaded")
            return False
        
        demo_questions = [
            "What is the main topic of this document?",
            "What are the key features mentioned?",
            "How does this system work?",
            "What are the benefits described?",
            "Are there any limitations mentioned?"
        ]
        
        self.print_header("🎯 Demo Questions")
        successful_demos = 0
        
        for i, question in enumerate(demo_questions, 1):
            try:
                print(f"\n{Colors.OKBLUE}Demo Question {i}/{len(demo_questions)}:{Colors.ENDC} {question}")
                
                result = self.process_question_safely(question)
                if result:
                    self.display_answer_result(result)
                    successful_demos += 1
                else:
                    self.print_warning(f"Failed to process demo question {i}")
                
                # Small delay between questions
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                self.print_warning("Demo interrupted by user")
                break
            except Exception as e:
                self.print_error(f"Error in demo question {i}: {e}")
                continue
        
        success_rate = (successful_demos / len(demo_questions)) * 100
        self.print_info(f"Demo completed: {successful_demos}/{len(demo_questions)} successful ({success_rate:.1f}%)")
        
        return successful_demos > 0
    
    @handle_exceptions(default_return=False, log_errors=True)
    def interactive_mode(self) -> bool:
        """Run interactive Q&A mode with robust error handling."""
        self.print_header("💬 Interactive Q&A Mode")
        print("Enter your questions (type 'help', 'status', 'exit', or 'quit' for commands)\n")
        
        try:
            while True:
                try:
                    # Get user input with prompt
                    question = input(f"{Colors.OKBLUE}❓ Enter your question:{Colors.ENDC} ").strip()
                    
                    if not question:
                        continue
                    
                    # Handle special commands
                    if question.lower() in ['exit', 'quit', 'q']:
                        self.print_info("Exiting interactive mode...")
                        break
                    elif question.lower() == 'help':
                        self.show_interactive_help()
                        continue
                    elif question.lower() == 'status':
                        self.display_pipeline_status()
                        continue
                    elif question.lower() == 'health':
                        self.run_health_check()
                        continue
                    
                    # Process the question
                    result = self.process_question_safely(question)
                    if result:
                        self.display_answer_result(result)
                    else:
                        self.print_warning("Could not generate an answer. Please try rephrasing your question.")
                    
                    print()  # Add spacing
                    
                except KeyboardInterrupt:
                    print(f"\n{Colors.WARNING}Use 'exit' or 'quit' to leave interactive mode{Colors.ENDC}")
                    continue
                except EOFError:
                    print(f"\n{Colors.INFO}Exiting interactive mode...{Colors.ENDC}")
                    break
                except Exception as e:
                    self.print_error(f"Error in interactive mode: {e}")
                    continue
        
        except Exception as e:
            self.print_error(f"Critical error in interactive mode: {e}")
            return False
        
        return True
    
    def show_interactive_help(self):
        """Show help for interactive mode."""
        print(f"\n{Colors.HEADER}📚 Interactive Mode Commands:{Colors.ENDC}")
        print("  • Type any question to get an answer")
        print("  • 'help' - Show this help message")
        print("  • 'status' - Show pipeline status and statistics")
        print("  • 'health' - Run system health check")
        print("  • 'exit', 'quit', or 'q' - Exit interactive mode")
        print()


def main():
    """Enhanced main function with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="RAG Assistant Phase 1 - Step 9: Production-Ready Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with health check
  python main_step9.py --pdf sample_data/sample_document.pdf --interactive
  
  # Single question with verbose logging
  python main_step9.py --pdf document.pdf --question "What is this about?" --verbose
  
  # Run system health check
  python main_step9.py --health-check
  
  # Demo mode with error recovery
  python main_step9.py --pdf document.pdf --demo --verbose
        """
    )
    
    parser.add_argument(
        "--pdf", 
        type=str,
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
        "--health-check",
        action="store_true",
        help="Run system health check and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output and debug logging"
    )
    
    args = parser.parse_args()
    
    # Initialize the robust application
    app = RobustMainApp()
    
    try:
        # Set up logging level based on verbosity
        if args.verbose:
            logging.getLogger('rag_assistant').setLevel(logging.DEBUG)
            app.print_info("Verbose logging enabled")
        
        app.print_header("🚀 RAG Assistant Phase 1 - Step 9: Production Ready")
        
        # Run health check if requested or if verbose
        if args.health_check or args.verbose:
            health_ok = app.run_health_check()
            if args.health_check:
                sys.exit(0 if health_ok else 1)
            
            if not health_ok and not args.verbose:
                app.print_warning("Health check failed. Use --verbose for details.")
        
        # Validate arguments
        if not args.health_check and not args.pdf:
            app.print_error("PDF file is required unless running --health-check")
            parser.print_help()
            sys.exit(1)
        
        # Initialize RAG pipeline
        app.initialize_rag_pipeline()
        
        # Load document if provided
        if args.pdf:
            app.load_and_validate_document(args.pdf)
        
        # Handle different modes
        if args.status:
            app.display_pipeline_status()
        
        elif args.question:
            result = app.process_question_safely(args.question)
            if result:
                app.display_answer_result(result)
            else:
                app.print_error("Failed to generate answer")
                sys.exit(1)
        
        elif args.demo:
            success = app.run_demo_questions()
            if not success:
                app.print_warning("Demo completed with some issues")
        
        elif args.interactive:
            app.interactive_mode()
        
        else:
            # Default: show status and enter interactive mode
            app.display_pipeline_status()
            app.print_info("Entering interactive mode... (use 'help' for commands)")
            app.interactive_mode()
        
        # Final statistics
        if app.total_questions_asked > 0:
            app.print_success(f"Session completed: {app.successful_answers}/{app.total_questions_asked} successful answers")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Application interrupted by user{Colors.ENDC}")
        sys.exit(130)  # Standard exit code for Ctrl+C
    
    except RAGAssistantError as e:
        app.print_error(f"RAG Assistant error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    except Exception as e:
        app.print_error(f"Critical application error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
