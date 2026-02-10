"""Configuration validation and startup checks."""

import logging
import os
from typing import Dict, Any, List, Optional

from .config import get_config, SystemConfig
from ..llm.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validate configuration and perform startup checks."""
    
    def __init__(self):
        self.config = get_config()
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> bool:
        """Validate all configuration settings."""
        logger.info("Starting configuration validation...")
        
        # Basic configuration validation
        self._validate_basic_config()
        
        # LLM configuration validation
        self._validate_llm_config()
        
        # Milvus configuration validation
        self._validate_milvus_config()
        
        # Embedding configuration validation
        self._validate_embedding_config()
        
        # Report results
        self._report_validation_results()
        
        return len(self.errors) == 0
    
    def _validate_basic_config(self):
        """Validate basic configuration settings."""
        # Check required directories exist
        required_dirs = [
            self.config.upload_dir,
            self.config.temp_dir
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"Created directory: {directory}")
                except Exception as e:
                    self.errors.append(f"Cannot create directory {directory}: {e}")
    
    def _validate_llm_config(self):
        """Validate LLM configuration."""
        llm_config = self.config.llm
        
        # Check provider
        if llm_config.provider != "ollama":
            self.errors.append(f"Invalid LLM provider: {llm_config.provider}. Only Ollama is supported.")
        
        # Validate Ollama configuration
        if llm_config.provider == "ollama":
            if not llm_config.ollama_base_url:
                self.errors.append("Ollama base URL is required")
            if not llm_config.model:
                self.errors.append("Ollama model is required")
            
            try:
                provider = OllamaProvider()
                # Test connection
                models = provider.list_models()
                logger.info(f"✓ Ollama connection successful, {len(models)} models available")
            except Exception as e:
                self.errors.append(f"Ollama connection failed: {e}")
    
    def _validate_milvus_config(self):
        """Validate Milvus configuration."""
        milvus_config = self.config.milvus
        
        # Basic validation
        if not milvus_config.host:
            self.errors.append("Milvus host not configured")
        
        if milvus_config.port <= 0 or milvus_config.port > 65535:
            self.errors.append(f"Invalid Milvus port: {milvus_config.port}")
        
        if milvus_config.embedding_dim <= 0:
            self.errors.append(f"Invalid embedding dimension: {milvus_config.embedding_dim}")
    
    def _validate_embedding_config(self):
        """Validate embedding configuration."""
        embedding_config = self.config.embedding
        
        # Check embedding dimension matches model
        try:
            from ..llm.embeddings import embedding_generator
            actual_dim = embedding_generator.get_embedding_dim()
            expected_dim = self.config.milvus.embedding_dim
            
            if actual_dim != expected_dim:
                self.errors.append(
                    f"Embedding dimension mismatch: "
                    f"Model produces {actual_dim} dimensions, "
                    f"but Milvus expects {expected_dim} dimensions. "
                    f"Please update MILVUS_EMBEDDING_DIM to {actual_dim}"
                )
        except Exception as e:
            self.errors.append(f"Cannot validate embedding dimension: {e}")
    
    def _report_validation_results(self):
        """Report validation results."""
        if self.errors:
            logger.error("Configuration validation failed with errors:")
            for error in self.errors:
                logger.error(f"  ✗ {error}")
        
        if self.warnings:
            logger.warning("Configuration validation completed with warnings:")
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")
        
        if not self.errors and not self.warnings:
            logger.info("✓ Configuration validation passed")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "config": {
                "llm_provider": self.config.llm.provider,
                "milvus_host": self.config.milvus.host,
                "milvus_port": self.config.milvus.port,
                "embedding_model": self.config.embedding.model_name,
                "embedding_dim": self.config.milvus.embedding_dim
            }
        }


def validate_startup() -> bool:
    """Validate configuration on startup."""
    validator = ConfigValidator()
    return validator.validate_all()


def get_startup_summary() -> Dict[str, Any]:
    """Get startup validation summary."""
    validator = ConfigValidator()
    validator.validate_all()
    return validator.get_validation_summary()