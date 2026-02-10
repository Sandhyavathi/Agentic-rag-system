# API Reference

## Core Modules

### Configuration (`src/core/config.py`)

#### SystemConfig
Main configuration class containing all system settings.

```python
@dataclass
class SystemConfig:
    milvus: MilvusConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    debug: bool
    log_level: str
    upload_dir: str
    temp_dir: str
```

#### MilvusConfig
Milvus database configuration.

```python
@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "documents"
    embedding_dim: int = 384
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 1024
    nprobe: int = 10
```

#### LLMConfig
LLM provider configuration.

```python
@dataclass
class LLMConfig:
    provider: str = "gemini"  # "gemini" or "ollama"
    model: str = "gemini-pro"
    temperature: float = 0.1
    max_tokens: int = 2000
    top_k: int = 5
    google_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
```

### State Management (`src/core/state_management.py`)

#### AgentState
Complete state for LangGraph agents.

```python
class AgentState(TypedDict):
    user_query: str
    conversation_history: List[Message]
    query_analysis: Optional[QueryAnalysis]
    routing_decision: Optional[RetrievalStrategy]
    retrieved_chunks: List[Chunk]
    graded_chunks: List[GradedChunk]
    generated_answer: str
    validation_result: Optional[ValidationResult]
    iteration_count: int
    max_iterations: int
    agent_trace: List[AgentTrace]
    errors: List[str]
    retry_reasons: List[str]
```

#### QueryAnalysis
Result of query analysis.

```python
@dataclass
class QueryAnalysis:
    query_type: QueryType  # FACTUAL, COMPARISON, SUMMARY, MULTI_HOP
    complexity: int  # 1-5 scale
    entities: List[str]
    requires_filters: Dict[str, Any]
    reasoning: str
```

#### RetrievalStrategy
Available retrieval strategies.

```python
class RetrievalStrategy(str, Enum):
    SIMPLE = "simple"
    HYBRID = "hybrid"
    PARALLEL = "parallel"
    MULTI_HOP = "multi_hop"
```

### Document Processing

#### ParsedDocument
Result of structured document parsing.

```python
@dataclass
class ParsedDocument:
    content: str
    metadata: Dict[str, Any]
    sections: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
```

#### TabularDocument
Result of tabular document parsing.

```python
@dataclass
class TabularDocument:
    dataframes: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any]
    representations: List[Dict[str, Any]]
```

#### Chunk
Document chunk with metadata.

```python
@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
```

### Vector Store (`src/retrieval/vector_store.py`)

#### MilvusVectorStore
Milvus vector store implementation.

```python
class MilvusVectorStore:
    def connect(self) -> bool
    def disconnect(self)
    def create_collection(self) -> bool
    def add_chunks(self, chunks: List[Chunk]) -> bool
    def search(self, query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]
    def delete_by_source(self, source_file: str) -> bool
    def get_collection_stats(self) -> Dict[str, Any]
    def clear_collection(self) -> bool
```

#### SearchResult
Result of a vector search.

```python
@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    distance: float
```

### Agents

#### QueryAnalyzerAgent
Analyzes user queries to determine type and complexity.

```python
class QueryAnalyzerAgent:
    def analyze_query(self, state: AgentState) -> AgentState
```

#### RoutingAgent
Routes queries to appropriate retrieval strategies.

```python
class RoutingAgent:
    def route_query(self, state: AgentState) -> AgentState
```

#### RetrievalAgent
Base class for retrieval agents.

```python
class RetrievalAgent:
    def retrieve(self, state: AgentState) -> AgentState
```

#### GradingAgent
Grades retrieved chunks for relevance.

```python
class GradingAgent:
    def grade_chunks(self, state: AgentState) -> AgentState
```

#### GenerationAgent
Generates answers from graded chunks.

```python
class GenerationAgent:
    def generate_answer(self, state: AgentState) -> AgentState
```

#### ValidationAgent
Validates generated answers for quality.

```python
class ValidationAgent:
    def validate_answer(self, state: AgentState) -> AgentState
```

### Document Pipeline (`src/ingestion/pipeline.py`)

#### DocumentPipeline
Complete document ingestion pipeline.

```python
class DocumentPipeline:
    def process_document(self, file_path: str) -> Dict[str, Any]
    def process_multiple_documents(self, file_paths: List[str]) -> Dict[str, Any]
    def delete_document(self, source_file: str) -> bool
    def get_pipeline_stats(self) -> Dict[str, Any]
```

### Embeddings (`src/llm/embeddings.py`)

#### EmbeddingGenerator
Generates embeddings for text.

```python
class EmbeddingGenerator:
    def get_embeddings(self, texts: List[str]) -> List[List[float]]
    def get_embedding(self, text: str) -> List[float]
    def get_embedding_dim(self) -> int
```

### Hybrid Search (`src/retrieval/hybrid_search.py`)

#### HybridSearch
Combines semantic and keyword search.

```python
class HybridSearch:
    def search(self, query: str, query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]
```

### Error Handling (`src/core/error_handling.py`)

#### Custom Exceptions
```python
class RAGError(Exception)
class DocumentProcessingError(RAGError)
class VectorStorageError(RAGError)
class RetrievalError(RAGError)
class GenerationError(RAGError)
class LLMConnectionError(RAGError)
class ConfigurationError(RAGError)
```

#### ErrorHandler
Centralized error handling.

```python
class ErrorHandler:
    def register_handler(self, error_type: str, handler: Callable)
    def register_fallback(self, component: str, fallback: Callable)
    def handle_error(self, error: RAGError, context: Dict[str, Any] = None) -> Dict[str, Any]
```

## Usage Examples

### Basic Configuration

```python
from src.core.config import get_config

config = get_config()
print(f"Milvus host: {config.milvus.host}")
print(f"LLM provider: {config.llm.provider}")
```

### Document Processing

```python
from src.ingestion.pipeline import document_pipeline

# Process a single document
result = document_pipeline.process_document("path/to/document.pdf")
print(f"Success: {result['success']}")
print(f"Chunks created: {result['chunk_count']}")

# Process multiple documents
results = document_pipeline.process_multiple_documents([
    "doc1.pdf", "doc2.xlsx", "doc3.csv"
])
print(f"Successful: {results['successful_files']}")
```

### Query Processing

```python
from src.core.orchestration import agentic_orchestrator

# Process a query
result = agentic_orchestrator.run_sync(
    user_query="What are the Q3 sales figures?",
    conversation_history=[]
)

print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
print(f"Agent trace: {result['agent_trace']}")
```

### Vector Operations

```python
from src.retrieval.vector_store import milvus_vector_store

# Connect to vector store
milvus_vector_store.connect()

# Search for similar chunks
results = milvus_vector_store.search(
    query_vector=[0.1, 0.2, 0.3, ...],
    top_k=5,
    filters={"source_file": "report.pdf"}
)

for result in results:
    print(f"Chunk: {result.chunk.text[:100]}...")
    print(f"Score: {result.score}")
```

### Error Handling

```python
from src.core.error_handling import error_handler, ErrorHandler

# Custom error handler
def custom_handler(error, context):
    return {
        "success": False,
        "error": str(error),
        "fallback_used": True
    }

# Register handler
error_handler_instance = ErrorHandler()
error_handler_instance.register_handler("DocumentProcessingError", custom_handler)
```

## Environment Variables

### Required Variables

```env
# For Gemini LLM
GOOGLE_API_KEY=your_api_key_here

# For Ollama LLM
OLLAMA_BASE_URL=http://localhost:11434

# For Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### Optional Variables

```env
# Application settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
MAX_CHUNK_SIZE=1000
DEBUG=false
LOG_LEVEL=INFO

# Milvus settings
MILVUS_COLLECTION=documents
MILVUS_EMBEDDING_DIM=384
MILVUS_INDEX_TYPE=IVF_FLAT
MILVUS_METRIC_TYPE=COSINE

# LLM settings
LLM_PROVIDER=gemini
LLM_MODEL=gemini-pro
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000

# Retrieval settings
RETRIEVAL_TOP_K=5
RETRIEVAL_RERANK_K=3
BM25_WEIGHT=0.3
SEMANTIC_WEIGHT=0.7
CONFIDENCE_THRESHOLD=0.6
```

This API reference provides comprehensive documentation for all major components of the Agentic RAG System, enabling developers to understand and extend the system effectively.