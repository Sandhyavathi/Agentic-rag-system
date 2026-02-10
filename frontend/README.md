# Agentic RAG System - Frontend

This directory contains the Streamlit frontend application for the Agentic RAG System with Ollama integration.

## Overview

The frontend provides a user-friendly interface for:
- **Document Upload**: Drag-and-drop file upload for multiple formats
- **Real-time Processing**: Live status updates during document processing
- **Interactive Chat**: Natural language interface to query your documents
- **Source Attribution**: Detailed information about which documents were used
- **System Monitoring**: Real-time system status and configuration

## Architecture

```
frontend/
├── app.py              # Main Streamlit application
└── README.md           # This file

src/                    # Backend modules (separate from frontend)
├── core/              # Core system components
├── agents/            # AI agent implementations
├── ingestion/         # Document processing pipeline
├── llm/               # LLM providers and interfaces
└── retrieval/         # Vector search and retrieval
```

## Key Features

### 📁 **File Upload Interface**
- **Multi-format Support**: PDF, DOCX, PPTX, CSV, Excel files
- **Drag & Drop**: Intuitive file upload experience
- **Real-time Status**: Live processing progress updates
- **Error Handling**: Clear error messages and recovery options

### 💬 **Chat Interface**
- **Natural Language**: Conversational interface for document queries
- **Source Attribution**: Expandable sections showing document sources
- **Confidence Scoring**: Visual indicators of response confidence
- **Message History**: Persistent chat history during session

### 📊 **System Monitoring**
- **Configuration Display**: Real-time system configuration
- **File Status**: Processed vs pending file counts
- **Performance Metrics**: System health and status indicators

## Running the Frontend

### Prerequisites
- Python 3.8+
- Virtual environment activated
- Backend dependencies installed

### Launch Instructions

1. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Frontend Dependencies** (if needed):
   ```bash
   pip install streamlit streamlit-chat
   ```

3. **Start the Application**:
   ```bash
   streamlit run frontend/app.py
   ```

4. **Access the Application**:
   Open your browser to `http://localhost:8501`

## Configuration

The frontend reads configuration from the backend system configuration. Ensure your `.env` file is properly configured:

```bash
# Required
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Optional
MILVUS_HOST=localhost
MILVUS_PORT=19530
DEBUG=false
```

### Prerequisites

Before running the frontend, ensure you have:
1. **Ollama running**: `ollama serve`
2. **Required model**: `ollama pull llama3.2:3b`
3. **Milvus running**: `docker-compose -f docker/docker-compose.yml up -d`

## Frontend Components

### Session State Management
The frontend uses Streamlit's session state to maintain:
- **RAG Orchestrator**: Main backend system instance
- **Document Pipeline**: File processing pipeline
- **Uploaded Files**: List of uploaded files
- **Processing Status**: Real-time file processing status
- **Chat History**: User and assistant messages
- **Current Query**: Active user input

### UI Components
- **Main Header**: Gradient-styled application title
- **Upload Section**: Styled file upload area with instructions
- **File Status Grid**: Three-column layout showing file information
- **Chat Interface**: Message bubbles with source attribution
- **Sidebar**: System configuration and management controls

### Styling
- **Custom CSS**: Professional gradient headers and styled components
- **Responsive Design**: Works on desktop and mobile devices
- **Status Indicators**: Color-coded success/warning/error states
- **Accessibility**: Proper contrast and readable fonts

## Integration with Backend

The frontend directly imports and uses backend modules:

```python
# Core system components
from src.core.config import get_config
from src.core.orchestration import RAGOrchestrator

# Document processing
from src.ingestion.pipeline import DocumentPipeline

# LLM providers
from src.llm.gemini_provider import GeminiProvider
from src.llm.ollama_provider import OllamaProvider
```

This tight integration ensures:
- **No API Layer**: Direct function calls for better performance
- **Real-time Updates**: Immediate feedback on system operations
- **Type Safety**: Full type checking and IDE support
- **Debugging**: Easier debugging with direct module access

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   - Ensure you're running from the project root directory
   - Check that `src/` is in your Python path
   - Verify all dependencies are installed

2. **Streamlit Not Starting**
   - Ensure Python environment is activated
   - Check for port conflicts (try `--server.port 8502`)
   - Verify Streamlit is installed

3. **File Processing Errors**
   - Check file format compatibility
   - Ensure sufficient disk space
   - Verify backend system is properly configured

### Debug Mode

Enable debug logging in the frontend:
```python
setup_logging(level="DEBUG")
```

### Performance Tips

1. **Large Files**: Process files in smaller batches
2. **Multiple Users**: Consider using session isolation
3. **Memory Usage**: Monitor memory usage for large document collections
4. **Network**: Ensure stable connection to LLM providers

## Development

### Adding New Features

1. **New UI Components**: Add to `frontend/app.py` following existing patterns
2. **Backend Integration**: Import required modules from `src/`
3. **State Management**: Use Streamlit session state for persistent data
4. **Styling**: Add custom CSS to the existing style block

### Testing

Test the frontend independently:
```bash
# Run the frontend
streamlit run frontend/app.py

# Test with sample documents
# Monitor console for any errors or warnings
```

## Security Considerations

- **File Upload**: Validates file types and sizes
- **Environment Variables**: Sensitive data stored in `.env`
- **Session State**: Data persists only during browser session
- **Input Validation**: User queries are sanitized before processing

## Deployment

For production deployment:
1. **Environment Variables**: Set all required environment variables
2. **SSL/TLS**: Enable HTTPS for production use
3. **Authentication**: Add user authentication if needed
4. **Monitoring**: Set up logging and monitoring
5. **Scaling**: Consider containerization for multiple users

## Support

For frontend-specific issues:
- Check the main project README for general setup
- Review backend documentation for system requirements
- Create issues for bugs or feature requests