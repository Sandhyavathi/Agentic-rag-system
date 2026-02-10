"""Embedding generation for the Agentic RAG System."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from ..core.config import config
from ..core.error_handling import error_handler, GenerationError
from .base_llm import BaseLLM
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Embedding generator using sentence transformers."""
    
    def __init__(self):
        self.model = None
        self.device = config.embedding.device
        self.model_name = config.embedding.model_name
        self.batch_size = config.embedding.batch_size
        self.normalize_embeddings = config.embedding.normalize_embeddings
        self._initialized = False
    
    def initialize(self):
        """Initialize the embedding model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            self._initialized = True
            logger.info("Embedding model initialized successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Please install it with: pip install sentence-transformers")
            raise GenerationError(
                "sentence-transformers not installed",
                "embeddings"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise GenerationError(
                f"Failed to initialize embedding model: {e}",
                "embeddings"
            )
    
    @error_handler("embeddings")
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self._initialized:
            self.initialize()
        
        if not texts:
            return []
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise GenerationError(
                f"Failed to generate embeddings: {e}",
                "embeddings",
                {"text_count": len(texts), "model": self.model_name}
            )
    
    @error_handler("embeddings")
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.get_embeddings([text])[0]
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        if not self._initialized:
            self.initialize()
        
        # Generate a dummy embedding to get dimension
        dummy_embedding = self.get_embedding("test")
        return len(dummy_embedding)

# Global embedding generator instance
embedding_generator = EmbeddingGenerator()

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Convenience function to get embeddings."""
    return embedding_generator.get_embeddings(texts)

def get_embedding(text: str) -> List[float]:
    """Convenience function to get a single embedding."""
    return embedding_generator.get_embedding(text)