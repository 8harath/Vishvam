"""
Comprehensive Error Handling and Logging System for RAG Assistant Phase 1

This module provides robust error handling, logging, and progress tracking
for production-ready RAG applications.
"""

import sys
import logging
import traceback
import time
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from contextlib import contextmanager
from functools import wraps


# =============================================================================
# CUSTOM EXCEPTION CLASSES
# =============================================================================

class RAGAssistantError(Exception):
    """Base exception for RAG Assistant errors."""
    pass


class DocumentProcessingError(RAGAssistantError):
    """Raised when document processing fails."""
    pass


class PDFProcessingError(DocumentProcessingError):
    """Raised when PDF processing fails."""
    pass


class EmbeddingGenerationError(RAGAssistantError):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGAssistantError):
    """Raised when vector store operations fail."""
    pass


class LLMProcessingError(RAGAssistantError):
    """Raised when LLM processing fails."""
    pass


class MemoryError(RAGAssistantError):
    """Raised when memory limits are exceeded."""
    pass


class ConfigurationError(RAGAssistantError):
    """Raised when configuration is invalid."""
    pass


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class RAGLogger:
    """Enhanced logging system for RAG Assistant."""
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
        max_file_size_mb: int = 10,
        backup_count: int = 5
    ):
        """
        Initialize logging system.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (None for console only)
            console_output: Enable console output
            max_file_size_mb: Maximum log file size in MB before rotation
            backup_count: Number of backup log files to keep
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.console_output = console_output
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logging system."""
        # Create logger
        logger = logging.getLogger('rag_assistant')
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.log_file:
            from logging.handlers import RotatingFileHandler
            
            # Create log directory if it doesn't exist
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


# Global logger instance
_global_logger = None


def get_logger() -> logging.Logger:
    """Get the global RAG logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = RAGLogger(
            log_level="INFO",
            log_file="logs/rag_assistant.log",
            console_output=True
        )
    return _global_logger.logger


# =============================================================================
# ERROR HANDLING DECORATORS
# =============================================================================

def handle_exceptions(
    default_return=None,
    reraise: bool = False,
    log_errors: bool = True,
    error_type: Optional[type] = None
):
    """
    Decorator for comprehensive exception handling.
    
    Args:
        default_return: Value to return on error
        reraise: Whether to reraise the exception after logging
        log_errors: Whether to log errors
        error_type: Custom exception type to raise
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = get_logger()
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(f"Traceback:\n{traceback.format_exc()}")
                
                if error_type:
                    raise error_type(f"Error in {func.__name__}: {str(e)}") from e
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying operations on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    logger = get_logger()
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    logger.info(f"Retrying in {current_delay:.2f} seconds...")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # If all retries failed, raise the last exception
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Description of the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.logger = get_logger()
        
        self.logger.info(f"Starting {description} - {total_steps} steps")
    
    def update(self, step_increment: int = 1, message: str = ""):
        """Update progress."""
        self.current_step += step_increment
        progress = (self.current_step / self.total_steps) * 100
        
        elapsed_time = time.time() - self.start_time
        if self.current_step > 0:
            estimated_total_time = (elapsed_time / self.current_step) * self.total_steps
            estimated_remaining = estimated_total_time - elapsed_time
        else:
            estimated_remaining = 0
        
        status_msg = f"{self.description}: {self.current_step}/{self.total_steps} ({progress:.1f}%)"
        if estimated_remaining > 0:
            status_msg += f" - ETA: {estimated_remaining:.1f}s"
        if message:
            status_msg += f" - {message}"
        
        self.logger.info(status_msg)
    
    def finish(self, message: str = ""):
        """Mark progress as complete."""
        total_time = time.time() - self.start_time
        completion_msg = f"{self.description} completed in {total_time:.2f}s"
        if message:
            completion_msg += f" - {message}"
        
        self.logger.info(completion_msg)


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def error_context(operation_name: str, log_start: bool = True, log_end: bool = True):
    """Context manager for operation error handling."""
    logger = get_logger()
    
    if log_start:
        logger.info(f"Starting: {operation_name}")
    
    start_time = time.time()
    try:
        yield
        if log_end:
            elapsed = time.time() - start_time
            logger.info(f"Completed: {operation_name} (took {elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed: {operation_name} (after {elapsed:.2f}s) - {str(e)}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        raise


@contextmanager
def memory_monitor(max_memory_gb: float = 8.0):
    """Context manager for monitoring memory usage."""
    import psutil
    
    process = psutil.Process()
    logger = get_logger()
    
    initial_memory = process.memory_info().rss / (1024 ** 3)  # GB
    logger.debug(f"Initial memory usage: {initial_memory:.2f} GB")
    
    try:
        yield
    finally:
        final_memory = process.memory_info().rss / (1024 ** 3)  # GB
        memory_increase = final_memory - initial_memory
        
        logger.debug(f"Final memory usage: {final_memory:.2f} GB")
        logger.debug(f"Memory increase: {memory_increase:.2f} GB")
        
        if final_memory > max_memory_gb:
            logger.warning(f"Memory usage ({final_memory:.2f} GB) exceeded threshold ({max_memory_gb:.2f} GB)")
            raise MemoryError(f"Memory usage exceeded {max_memory_gb:.2f} GB limit")


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class ValidationUtils:
    """Utility functions for input validation."""
    
    @staticmethod
    def validate_file_path(file_path: str, extensions: Optional[list] = None) -> Path:
        """
        Validate file path and existence.
        
        Args:
            file_path: Path to validate
            extensions: List of allowed extensions (e.g., ['.pdf', '.txt'])
            
        Returns:
            Path: Validated Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension not allowed
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if extensions and path.suffix.lower() not in extensions:
            raise ValueError(f"File extension {path.suffix} not allowed. Allowed: {extensions}")
        
        return path
    
    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: float) -> float:
        """
        Validate file size.
        
        Args:
            file_path: Path to file
            max_size_mb: Maximum allowed size in MB
            
        Returns:
            float: File size in MB
            
        Raises:
            ValueError: If file too large
        """
        path = Path(file_path)
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 ** 2)
        
        if size_mb > max_size_mb:
            raise ValueError(f"File too large ({size_mb:.2f} MB). Maximum allowed: {max_size_mb} MB")
        
        return size_mb
    
    @staticmethod
    def validate_text_content(text: str, min_length: int = 10) -> str:
        """
        Validate text content.
        
        Args:
            text: Text to validate
            min_length: Minimum required length
            
        Returns:
            str: Validated text
            
        Raises:
            ValueError: If text invalid
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        
        text = text.strip()
        
        if len(text) < min_length:
            raise ValueError(f"Text too short ({len(text)} chars). Minimum: {min_length} chars")
        
        return text
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: list) -> Dict[str, Any]:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            
        Returns:
            Dict: Validated configuration
            
        Raises:
            ConfigurationError: If configuration invalid
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ConfigurationError(f"Missing required configuration keys: {missing_keys}")
        
        return config


# =============================================================================
# HEALTH CHECK SYSTEM
# =============================================================================

class HealthChecker:
    """System health monitoring for RAG Assistant."""
    
    def __init__(self):
        """Initialize health checker."""
        self.logger = get_logger()
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_available_gb = memory.available / (1024 ** 3)
            
            # Disk space check
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024 ** 3)
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'memory_available_gb': memory_available_gb,
                'memory_percent_used': memory.percent,
                'disk_free_gb': disk_free_gb,
                'cpu_percent': cpu_percent,
                'healthy': memory.percent < 90 and disk_free_gb > 1.0
            }
        except ImportError:
            self.logger.warning("psutil not available for system monitoring")
            return {'healthy': True, 'monitoring_available': False}
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def check_model_availability(self) -> Dict[str, Any]:
        """Check if required models are available."""
        try:
            # Check if sentence-transformers is available
            from sentence_transformers import SentenceTransformer
            
            # Try loading a small model
            model_name = "all-MiniLM-L6-v2"
            _ = SentenceTransformer(model_name)  # Just test loading, don't store
            
            return {
                'sentence_transformers': True,
                'test_model_loaded': True,
                'model_name': model_name
            }
        except ImportError:
            return {
                'sentence_transformers': False,
                'error': 'sentence-transformers not installed'
            }
        except Exception as e:
            return {
                'sentence_transformers': True,
                'test_model_loaded': False,
                'error': str(e)
            }
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        self.logger.info("Running system health check...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'system_resources': self.check_system_resources(),
            'model_availability': self.check_model_availability()
        }
        
        # Overall health status
        results['overall_healthy'] = (
            results['system_resources'].get('healthy', False) and
            results['model_availability'].get('sentence_transformers', False)
        )
        
        self.logger.info(f"Health check completed. Status: {'✅ Healthy' if results['overall_healthy'] else '❌ Issues detected'}")
        
        return results


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Initialize global logger on module import
logger = get_logger()
logger.info("Error handling and logging system initialized")
