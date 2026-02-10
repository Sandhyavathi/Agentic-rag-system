# Agentic RAG System

A powerful Retrieval-Augmented Generation (RAG) system with advanced AI agents for document processing and intelligent chat.

## Features

### 🤖 **Agentic Architecture**
- **Query Analyzer Agent**: Intelligently analyzes and routes user queries
- **Retrieval Agents**: Multiple specialized agents for document retrieval
- **Generation Agent**: Advanced response generation with source attribution
- **Validation Agent**: Ensures response quality and accuracy
- **Grading Agent**: Evaluates response quality and provides feedback

### **Document Processing**
- **Multi-format Support**: PDF, DOCX, PPTX, CSV, Excel files
- **Smart Chunking**: Context-aware document segmentation
- **Metadata Extraction**: Rich metadata from structured documents
- **Table Processing**: Advanced table and figure extraction
- **OCR Support**: Text extraction from scanned documents

### **Advanced Retrieval**
- **Hybrid Search**: BM25 + Semantic search combination
- **Re-ranking**: Multi-stage relevance ranking
- **Query Expansion**: Automatic query enhancement
- **Multi-vector Retrieval**: Multiple embedding strategies
- **Confidence Scoring**: Response confidence estimation

### **LLM Integration**
- **Ollama Support**: Local LLM integration with llama3.2:3b
- **Structured Output**: JSON response generation
- **Streaming Support**: Real-time response generation
- **Error Handling**: Robust error handling

### **Vector Database**
- **Milvus Integration**: High-performance vector storage
- **Auto-scaling**: Dynamic resource management
- **Multi-tenancy**: Support for multiple document collections
- **Real-time Updates**: Live document indexing

### **Streamlit Frontend**
- **File Upload Interface**: Drag-and-drop file upload
- **Real-time Chat**: Interactive chat interface
- **Processing Status**: Live file processing updates
- **Source Attribution**: Detailed source information
- **Responsive Design**: Works on desktop and mobile

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your configuration:

```bash
# Copy the example configuration
cp .env.example .env


# Optional: Customize other settings
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 3. Run the Application

#### Option A: Streamlit Frontend with Direct Integration (Recommended)

```bash
# Start the Streamlit application
streamlit run frontend/app.py

# Open your browser to http://localhost:8501
```

#### Option B: API-based Frontend (Clean Separation)

```bash
# 1. Start the backend API server
cd backend
python api.py

# 2. In another terminal, start the frontend
cd frontend
streamlit run api_app.py

# Open your browser to http://localhost:8501
```

#### Option B: Local Testing

```bash
# Run the local setup test
python test_local_setup.py

# This will test all components and provide status
```

#### Option C: Docker (Optional)

```bash
# Build and run with Docker
docker-compose up --build

# Access the application at http://localhost:8501
```

## Usage Guide

### 1. Upload Documents

1. Open the Streamlit application in your browser
2. Go to the "Upload Documents" tab
3. Drag and drop or click to upload files
4. Supported formats: PDF, DOCX, PPTX, CSV, Excel
5. Monitor processing status in real-time

### 2. Chat with Your Documents

1. Switch to the "Chat" tab
2. Ensure you have processed documents
3. Type your question in the input box
4. The system will:
   - Analyze your query
   - Retrieve relevant document chunks
   - Generate a response with sources
   - Provide confidence scoring

### 3. Understanding Responses

Each response includes:
- **Answer**: The generated response
- **Sources**: Which documents were used
- **Confidence**: How confident the system is in the answer
- **Content Preview**: Excerpts from source documents

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (gemini/ollama) | 
| `LLM_MODEL` | LLM model name | llama3.2:3b |
| `MILVUS_HOST` | Milvus database host | localhost |
| `MILVUS_PORT` | Milvus database port | 19530 |
| `EMBEDDING_DIM` | Embedding dimension | 384 |
| `CHUNK_SIZE` | Document chunk size | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap size | 200 |

### Advanced Configuration

See `src/core/config.py` for all configuration options and their descriptions.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   RAG Orchestrator│    │   Document      │
│   Frontend      │    │                  │    │   Pipeline      │
│                 │    │  ┌─────────────┐  │    │                 │
│  ┌───────────┐  │    │  │Query Analyzer│  │    │  ┌───────────┐  │
│  │File Upload│  │◄──►│  └─────────────┘  │◄──►│  │Docling    │  │
│  └───────────┘  │    │         │         │    │  │Parser     │  │
│                 │    │         ▼         │    │  └───────────┘  │
│  ┌───────────┐  │    │  ┌─────────────┐  │    │                 │
│  │   Chat    │  │    │  │Retrieval    │  │    │  ┌───────────┐  │
│  │Interface  │  │◄──►│  │Agents       │  │◄──►│  │Tabular    │  │
│  └───────────┘  │    │  └─────────────┘  │    │  │Parser     │  │
│                 │    │         │         │    │  └───────────┘  │
│                 │    │         ▼         │    │                 │
│                 │    │  ┌─────────────┐  │    │  ┌───────────┐  │
│                 │    │  │Generation   │  │    │  │Chunker    │  │
│                 │    │  │Agent        │  │    │  └───────────┘  │
│                 │    │  └─────────────┘  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Vector Store  │
                       │   (Milvus)      │
                       └─────────────────┘
```

## Troubleshooting

### Common Issues


1. **Milvus Connection Issues**
   - Ensure Milvus is running (or use Docker Compose)
   - Check `MILVUS_HOST` and `MILVUS_PORT` settings

2. **Document Processing Errors**
   - Check file format compatibility
   - Ensure files are not corrupted
   - Check available disk space

3. **Streamlit Not Starting**
   - Ensure all dependencies are installed
   - Check Python version compatibility
   - Try running with `--server.port` option

### Debug Mode

Enable debug logging by setting:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

### Testing

Run the comprehensive test suite:
```bash
python test_local_setup.py
```

## Development

### Adding New Features

1. **New Document Types**: Extend parsers in `src/ingestion/parsers/`
2. **New LLM Providers**: Add providers in `src/llm/`
3. **New Agents**: Create agents in `src/agents/`
4. **New Retrieval Strategies**: Add strategies in `src/retrieval/`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Performance Optimization

### For Large Document Collections

1. **Increase Chunk Size**: Adjust `CHUNK_SIZE` for longer documents
2. **Optimize Retrieval**: Tune `TOP_K` and `RERANK_K` parameters
3. **Use GPU**: Set `EMBEDDING_DEVICE=cuda` if available
4. **Batch Processing**: Process files in batches for better performance

### For Better Response Quality

1. **Query Expansion**: Enable query expansion in retrieval config
2. **Multiple Retrievers**: Use multiple retrieval strategies
3. **Confidence Thresholding**: Filter low-confidence responses
4. **Source Attribution**: Always show sources for transparency

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the test files for usage examples
