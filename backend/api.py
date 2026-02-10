"""Production FastAPI backend for the Agentic RAG System."""

import sys
import os
from pathlib import Path
import logging
import tempfile
import traceback
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.core.config import get_config
    from src.core.orchestration import RAGOrchestrator
    from src.ingestion.pipeline import DocumentPipeline
    from src.llm.ollama_provider import OllamaProvider
    from src.core.error_handling import setup_logging
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    logger.error(f"Project root: {project_root}")
    logger.error(f"Python path: {sys.path}")
    raise

# Set up logging
setup_logging(level="INFO")

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG System API",
    description="Production API for the Agentic RAG System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system components
rag_orchestrator: Optional[RAGOrchestrator] = None
document_pipeline: Optional[DocumentPipeline] = None

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, Any]]] = []

class QueryResponse(BaseModel):
    question: str
    response: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_type: str
    retrieval_method: str
    chunks_used: int
    status: str = "success"

class UploadResponse(BaseModel):
    results: List[Dict[str, Any]]

class SystemStatus(BaseModel):
    status: str
    components: Dict[str, Any]
    storage_info: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_orchestrator, document_pipeline
    
    try:
        logger.info("🚀 Starting Agentic RAG System...")
        
        # Get configuration
        config = get_config()
        logger.info(f"Configuration loaded: LLM={config.llm.provider}, Model={config.llm.model}")
        
        # Initialize LLM provider
        logger.info("Initializing LLM provider...")
        llm_provider = OllamaProvider()
        logger.info(" LLM provider initialized")
        
        # Initialize document pipeline
        logger.info("Initializing document pipeline...")
        document_pipeline = DocumentPipeline()
        logger.info(" Document pipeline initialized")
        
        # Initialize RAG orchestrator
        logger.info("Initializing RAG orchestrator...")
        rag_orchestrator = RAGOrchestrator(llm_provider, document_pipeline)
        logger.info(" RAG orchestrator initialized")
        
        logger.info("🎉 Agentic RAG System initialized successfully!")
        
    except Exception as e:
        logger.error(f" Failed to initialize RAG system: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"System initialization failed: {e}")

@app.get("/")
async def root():
    """Root endpoint with system info."""
    return {
        "name": "Agentic RAG System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        system_ready = rag_orchestrator is not None and document_pipeline is not None
        
        return {
            "status": "healthy" if system_ready else "initializing",
            "timestamp": "2024-01-01T00:00:00Z",
            "components": {
                "rag_orchestrator": rag_orchestrator is not None,
                "document_pipeline": document_pipeline is not None
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/config")
async def get_config_info():
    """Get current system configuration."""
    try:
        config = get_config()
        return {
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
            "embedding_model": config.embedding.model_name,
            "milvus_host": config.milvus.host,
            "milvus_port": config.milvus.port,
            "milvus_collection": config.milvus.collection_name,
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
            "api_key_configured": False
        }
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {e}")

@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    if not document_pipeline:
        raise HTTPException(status_code=503, detail="Document pipeline not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    
    for uploaded_file in files:
        try:
            logger.info(f"Processing file: {uploaded_file.filename}")
            
            # Validate file type
            allowed_extensions = {'.pdf', '.docx', '.pptx', '.csv', '.xlsx', '.xls', '.txt', '.md'}
            file_extension = Path(uploaded_file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                results.append({
                    "filename": uploaded_file.filename,
                    "status": "error",
                    "message": f"Unsupported file type: {file_extension}"
                })
                continue
            
            # Save file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                content = await uploaded_file.read()
                tmp_file.write(content)
                temp_path = tmp_file.name
            
            # Process the file
            result = document_pipeline.process_document(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if result["success"]:
                results.append({
                    "filename": uploaded_file.filename,
                    "status": "success",
                    "chunks_processed": result.get('chunk_count', 0),
                    "message": f"Successfully processed {result.get('chunk_count', 0)} chunks"
                })
                logger.info(f" Successfully processed {uploaded_file.filename}")
            else:
                results.append({
                    "filename": uploaded_file.filename,
                    "status": "error",
                    "message": result.get("error", "Unknown error")
                })
                logger.error(f" Failed to process {uploaded_file.filename}: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.filename}: {e}")
            results.append({
                "filename": uploaded_file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return UploadResponse(results=results)

@app.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    """Query the RAG system with a natural language question."""
    if not rag_orchestrator:
        raise HTTPException(status_code=503, detail="RAG orchestrator not initialized")
    
    try:
        question = query_request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing query: {question}")
        
        # Generate response using RAG orchestrator
        response = rag_orchestrator.query(
            user_query=question,
            conversation_history=query_request.conversation_history
        )
        
        return QueryResponse(
            question=question,
            response=response.get('response', 'No response generated'),
            sources=response.get('sources', []),
            confidence=response.get('confidence', 0.0),
            query_type=response.get('query_type', 'unknown'),
            retrieval_method=response.get('retrieval_method', 'unknown'),
            chunks_used=response.get('chunks_used', 0)
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status and statistics."""
    try:
        # Get storage info
        storage_info = {"status": "unknown", "total_entities": 0}
        if document_pipeline and hasattr(document_pipeline, 'vector_store'):
            storage_info = document_pipeline.vector_store.get_collection_info()
        
        return SystemStatus(
            status="operational",
            components={
                "rag_orchestrator": rag_orchestrator is not None,
                "document_pipeline": document_pipeline is not None,
                "vector_store": storage_info.get("status") == "connected"
            },
            storage_info=storage_info
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return SystemStatus(
            status="error",
            components={
                "rag_orchestrator": rag_orchestrator is not None,
                "document_pipeline": document_pipeline is not None,
                "vector_store": False
            },
            storage_info={"status": "error", "error": str(e)}
        )

@app.delete("/clear")
async def clear_system():
    """Clear all processed documents and reset the system."""
    try:
        if document_pipeline and hasattr(document_pipeline, 'vector_store'):
            success = document_pipeline.vector_store.clear_collection()
            if success:
                return {"message": "System cleared successfully", "status": "success"}
            else:
                raise HTTPException(status_code=500, detail="Failed to clear system")
        else:
            return {"message": "No storage to clear", "status": "success"}
        
    except Exception as e:
        logger.error(f"Error clearing system: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing system: {e}")

@app.get("/debug/collection-info")
async def debug_collection_info():
    """Debug endpoint to check collection status."""
    try:
        if document_pipeline and hasattr(document_pipeline, 'vector_store'):
            collection_info = document_pipeline.vector_store.get_collection_info()
            return collection_info
        return {"error": "Vector store not available"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/search-test")
async def debug_search_test(query: str = "API styles"):
    """Debug endpoint to test search directly."""
    try:
        from src.llm.embeddings import get_embeddings
        
        # Generate query embedding
        query_embedding = get_embeddings([query])[0]
        
        # Search vector store directly
        results = document_pipeline.vector_store.search(
            query_vector=query_embedding,
            top_k=10
        )
        
        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "score": r.score,
                    "text_preview": r.chunk.text[:300],
                    "source": r.chunk.metadata.get("source_file", "unknown"),
                    "chunk_type": r.chunk.metadata.get("chunk_type", "unknown")
                }
                for r in results
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.delete("/debug/clear-collection")
async def debug_clear_collection():
    """Clear the entire collection for testing."""
    try:
        success = document_pipeline.vector_store.clear_collection()
        return {"success": success, "message": "Collection cleared"}
    except Exception as e:
        return {"error": str(e)}
    
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "type": type(exc).__name__
        }
    )

def main():
    """Run the FastAPI server."""
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )

if __name__ == "__main__":
    main()