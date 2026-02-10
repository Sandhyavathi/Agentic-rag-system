"""Configuration management for the Agentic RAG System."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class MilvusConfig:
    """Milvus database configuration."""
    host: str = os.getenv("MILVUS_HOST", "localhost")
    port: int = int(os.getenv("MILVUS_PORT", "19530"))
    collection_name: str = os.getenv("MILVUS_COLLECTION", "documents")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    index_type: str = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
    metric_type: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
    nlist: int = int(os.getenv("MILVUS_NLIST", "1024"))
    nprobe: int = int(os.getenv("MILVUS_NPROBE", "10"))
    
@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = os.getenv("LLM_PROVIDER", "ollama")  # ollama only
    model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    
    # Ollama specific
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu or cuda
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    normalize_embeddings: bool = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"

@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    separators: list = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]

@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    rerank_k: int = int(os.getenv("RERANK_K", "3"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

@dataclass
class SystemConfig:
    """Complete system configuration."""
    milvus: MilvusConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    
    # Application settings
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Paths
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    temp_dir: str = os.getenv("TEMP_DIR", "./temp")
    
    def __post_init__(self):
        # Ensure directories exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

def get_config() -> SystemConfig:
    """Get the system configuration."""
    return SystemConfig(
        milvus=MilvusConfig(),
        llm=LLMConfig(),
        embedding=EmbeddingConfig(),
        chunking=ChunkingConfig(),
        retrieval=RetrievalConfig()
    )

# Global configuration instance
config = get_config()