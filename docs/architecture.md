# Agentic RAG System Architecture

## Overview

The Agentic RAG System is a sophisticated Retrieval-Augmented Generation system that uses LangGraph for agentic orchestration, Milvus for vector storage, and advanced document processing capabilities.

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)                    │
│  - Document Upload                                               │
│  - Chat Interface                                                │
│  - Agent Visualization (shows decision-making process)           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              AGENTIC ORCHESTRATOR (LangGraph)                    │
│                                                                   │
│  State Management → Agent Nodes → Conditional Routing            │
└────────────────┬────────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Ingest  │  │ Retrieval│  │ Response│
│ Pipeline│  │ Agents   │  │ Generation│
└─────────┘  └─────────┘  └─────────┘
    │            │            │
    ▼            ▼            ▼
┌─────────────────────────────────────┐
│     VECTOR DATABASE (Milvus)         │
│  - Document chunks                   │
│  - Embeddings                        │
│  - Metadata (source, page, section)  │
└─────────────────────────────────────┘
```

## Core Components

### 1. Document Ingestion Pipeline

**Purpose**: Process and index documents for retrieval

**Components**:
- **Parsers**: Docling (PDF/DOCX/PPTX) and Tabular (CSV/Excel) parsers
- **Chunking**: Smart chunking with context awareness
- **Embeddings**: Sentence transformers for semantic representation
- **Vector Storage**: Milvus for efficient similarity search

**Workflow**:
1. File upload and validation
2. Document parsing (structured or tabular)
3. Smart chunking with metadata enrichment
4. Embedding generation
5. Vector storage in Milvus

### 2. Agentic Retrieval Pipeline

**Purpose**: Intelligently retrieve relevant document chunks

**Components**:
- **Query Analyzer**: Analyzes query type and complexity
- **Routing Agent**: Routes to appropriate retrieval strategy
- **Retrieval Agents**: Multiple specialized retrieval strategies
- **Grading Agent**: Validates chunk relevance
- **Generation Agent**: Synthesizes answers
- **Validation Agent**: Ensures answer quality

**Retrieval Strategies**:
- **Simple**: Basic vector search for straightforward queries
- **Hybrid**: Combines semantic and keyword search
- **Parallel**: For comparison queries across entities
- **Multi-hop**: For complex reasoning across documents

### 3. LangGraph Orchestration

**Purpose**: Coordinate agent interactions with state management

**Features**:
- **State Management**: Persistent state across agent calls
- **Conditional Routing**: Dynamic workflow based on query analysis
- **Self-Correction**: Validation and retry loops
- **Transparency**: Full traceability of decision-making

**State Schema**:
```python
AgentState = TypedDict({
    "user_query": str,
    "conversation_history": List[Message],
    "query_analysis": QueryAnalysis,
    "routing_decision": RetrievalStrategy,
    "retrieved_chunks": List[Chunk],
    "graded_chunks": List[GradedChunk],
    "generated_answer": str,
    "validation_result": ValidationResult,
    "agent_trace": List[AgentTrace]
})
```

### 4. Vector Database (Milvus)

**Purpose**: Efficient storage and retrieval of document embeddings

**Schema**:
- **id**: Primary key (chunk identifier)
- **embedding**: Vector embedding (384 dimensions)
- **text**: Original chunk text
- **metadata**: JSON with source, page, section, etc.

**Indexes**:
- **Vector Index**: IVF_FLAT or HNSW for similarity search
- **Scalar Indexes**: For metadata filtering

### 5. User Interface (Streamlit)

**Purpose**: User-friendly interface for document management and chat

**Features**:
- **Document Management**: Upload, view, and delete documents
- **Chat Interface**: Conversational interface with streaming responses
- **Agent Visualization**: Real-time view of agent decision-making
- **Settings**: Configure LLM providers and retrieval parameters

## Data Flow

### Document Processing Flow

1. **Upload**: User uploads document via Streamlit
2. **Parse**: Appropriate parser extracts content and metadata
3. **Chunk**: Smart chunking creates manageable text segments
4. **Embed**: Sentence transformers generate semantic embeddings
5. **Store**: Chunks stored in Milvus with metadata

### Query Processing Flow

1. **Input**: User submits query via chat interface
2. **Analyze**: Query analyzer determines type and complexity
3. **Route**: Routing agent selects appropriate retrieval strategy
4. **Retrieve**: Retrieval agent performs search and returns chunks
5. **Grade**: Grading agent validates chunk relevance
6. **Generate**: Generation agent creates response with citations
7. **Validate**: Validation agent checks answer quality
8. **Output**: Response displayed with agent trace

## Error Handling Strategy

### Layer 1: Document Processing
- **Invalid file type**: Skip with warning
- **Parsing failure**: Try fallback parser
- **Empty document**: Warning, don't add to DB

### Layer 2: Vector Storage
- **Milvus connection failure**: Retry 3 times
- **Milvus down**: Fall back to in-memory FAISS
- **Embedding generation failure**: Skip chunk

### Layer 3: Retrieval
- **No results found**: Expand search, reduce filters
- **Low quality results**: Trigger query refinement
- **Timeout**: Return partial results with warning

### Layer 4: Generation
- **LLM API failure**: Retry with exponential backoff
- **Hallucination detected**: Add grounding warning
- **Incomplete answer**: Mark as partial

## Configuration

### Environment Variables

```env
# LLM Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
OLLAMA_BASE_URL=http://localhost:11434

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Application Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
MAX_CHUNK_SIZE=1000
```

### Configuration Structure

```python
SystemConfig = {
    "milvus": {
        "host": "localhost",
        "port": 19530,
        "collection_name": "documents",
        "embedding_dim": 384
    },
    "llm": {
        "provider": "gemini",
        "model": "gemini-pro",
        "temperature": 0.1,
        "top_k": 5
    },
    "retrieval": {
        "top_k": 5,
        "rerank_k": 3,
        "bm25_weight": 0.3,
        "semantic_weight": 0.7
    }
}
```

## Performance Considerations

### Scalability
- **Horizontal Scaling**: Multiple Milvus nodes
- **Caching**: Redis for frequent queries
- **Load Balancing**: Multiple LLM instances

### Optimization
- **Batch Processing**: For document ingestion
- **Index Optimization**: Appropriate Milvus parameters
- **Memory Management**: Efficient chunking strategies

### Monitoring
- **Query Latency**: Track response times
- **Retrieval Quality**: Monitor relevance scores
- **System Health**: Monitor component status

## Security Considerations

### Data Privacy
- **Local Processing**: Option for self-hosted LLMs
- **Encryption**: Secure data transmission
- **Access Control**: Document-level permissions

### API Security
- **Rate Limiting**: Prevent abuse
- **Authentication**: Secure API access
- **Audit Logging**: Track system usage

## Future Enhancements

### Advanced Features
- **Multi-modal Support**: Images and audio processing
- **Real-time Updates**: Live document synchronization
- **Collaborative Features**: Multi-user document sharing

### AI Improvements
- **Better Hallucination Detection**: Advanced validation
- **Contextual Understanding**: Improved query analysis
- **Personalization**: User-specific response tuning

This architecture provides a robust, scalable, and extensible foundation for advanced document intelligence applications.