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
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    nlist: int = 1024
    nprobe: int = 10
    
@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = os.getenv("LLM_PROVIDER", "gemini")
    model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    top_k: int = 5
    
    # Gemini specific
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    # Ollama specific
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    normalize_embeddings: bool = True

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
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    rerank_k: int = int(os.getenv("RERANK_K", "3"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))  # Lowered from 0.6

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
