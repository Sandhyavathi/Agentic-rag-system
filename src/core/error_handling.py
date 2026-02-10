"""Comprehensive error handling for the Agentic RAG System."""

import logging
import traceback
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    error_message: str
    component: str
    timestamp: datetime
    details: Dict[str, Any]
    stack_trace: Optional[str] = None

class RAGError(Exception):
    """Base exception for RAG system errors."""
    def __init__(self, message: str, component: str, details: Dict[str, Any] = None):
        self.message = message
        self.component = component
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)

class DocumentProcessingError(RAGError):
    """Error during document processing."""
    pass

class VectorStoreError(RAGError):
    """Error during vector store operations."""
    pass

class RetrievalError(RAGError):
    """Error during document retrieval."""
    pass

class GenerationError(RAGError):
    """Error during answer generation."""
    pass

class LLMConnectionError(RAGError):
    """Error connecting to LLM provider."""
    pass

class ConfigurationError(RAGError):
    """Error in system configuration."""
    pass

def error_handler(component: str):
    """Decorator for error handling in functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RAGError as e:
                # Re-raise RAG errors with component info
                if not e.component:
                    e.component = component
                logger.error(f"RAG error in {component}: {e.message}")
                raise e
            except Exception as e:
                # Convert generic exceptions to RAG errors
                logger.exception(f"Unexpected error in {component}: {e}")
                rag_error = RAGError(
                    message=str(e),
                    component=component,
                    details={"original_error": str(e)}
                )
                raise rag_error
        return wrapper
    return decorator

def handle_llm_errors(func):
    """Decorator to handle LLM-related errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"LLM error in {func.__name__}: {e}")
            raise LLMConnectionError(f"LLM service error: {e}", "llm")
    return wrapper

def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )