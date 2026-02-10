"""Document ingestion pipeline for the Agentic RAG System."""

import logging
from typing import Dict, Any, List, Optional, Union
import os
import tempfile
from pathlib import Path

from ..core.error_handling import error_handler, DocumentProcessingError
from ..ingestion.parsers.docling_parser import docling_parser, ParsedDocument
from ..ingestion.parsers.tabular_parser import tabular_parser, TabularDocument
from ..ingestion.chunking import smart_chunker, Chunk
from ..llm.embeddings import get_embeddings
from ..retrieval.vector_store import milvus_vector_store

logger = logging.getLogger(__name__)

class DocumentPipeline:
    """Complete document ingestion pipeline."""
    
    def __init__(self):
        self.docling_parser = docling_parser
        self.tabular_parser = tabular_parser
        self.chunker = smart_chunker
        self.vector_store = milvus_vector_store
    
    @error_handler("ingestion_pipeline")
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document through the entire ingestion pipeline."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentProcessingError(
                f"File not found: {file_path}",
                "ingestion_pipeline",
                {"file_path": str(file_path)}
            )
        
        file_extension = file_path.suffix.lower()
        
        try:
            # Step 1: Parse document
            parsed_document = self._parse_document(file_path, file_extension)
            
            # Step 2: Chunk document
            chunks = self._chunk_document(parsed_document, file_path, file_extension)
            
            # Step 3: Generate embeddings
            chunks_with_embeddings = self._generate_embeddings(chunks)
            
            # Step 4: Store in vector database
            success = self._store_chunks(chunks_with_embeddings)
            
            return {
                "success": success,
                "chunk_count": len(chunks_with_embeddings),
                "file_path": str(file_path),
                "file_type": file_extension
            }
            
        except Exception as e:
            logger.error(f"Document processing failed for {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": str(file_path),
                "chunk_count": 0
            }
    
    def _parse_document(self, file_path: Path, file_extension: str) -> Union[ParsedDocument, TabularDocument]:
        """Parse document based on file type with robust error handling."""
        
        # Docling-supported formats
        if file_extension in ['.pdf', '.docx', '.pptx', '.html', '.xlsx', '.xls', '.png', '.jpg', '.jpeg']:
            try:
                return self.docling_parser.parse_file(str(file_path))
            except Exception as e:
                logger.error(f"Docling parsing failed for {file_path}: {e}")
                # For PDFs, we can try a fallback approach
                if file_extension == '.pdf':
                    logger.info("Attempting fallback PDF parsing")
                    return self._parse_pdf_simple(file_path)
                else:
                    raise
        
        # Custom parser for CSV (Docling doesn't support CSV)
        elif file_extension == '.csv':
            return self.tabular_parser.parse_file(str(file_path))
        
        # Plain text files
        elif file_extension in ['.txt', '.md']:
            return self._parse_text(file_path)
        
        else:
            raise DocumentProcessingError(
                f"Unsupported file type: {file_extension}",
                "ingestion_pipeline",
                {"file_path": str(file_path), "extension": file_extension}
            )

    def _parse_pdf_simple(self, file_path: Path) -> ParsedDocument:
        """Simple PDF parsing as ultimate fallback."""
        try:
            import pypdf
            
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                content = ""
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
            
            return ParsedDocument(
                content=content,
                metadata={"title": file_path.stem, "file_type": "pdf"},
                sections=[],
                tables=[],
                figures=[]
            )
        except Exception as e:
            raise DocumentProcessingError(
                f"All PDF parsing methods failed: {e}",
                "ingestion_pipeline",
                {"file_path": str(file_path)}
            )
        
    def _chunk_document(
        self, 
        parsed_document: Union[ParsedDocument, TabularDocument], 
        file_path: Path, 
        file_extension: str
    ) -> List[Chunk]:
        """Chunk parsed document."""
        source_file = file_path.name
        file_type = file_extension[1:]  # Remove the dot
        
        if isinstance(parsed_document, ParsedDocument):
            return self.chunker.chunk_structured_document(
                parsed_document, source_file, file_type
            )
        elif isinstance(parsed_document, TabularDocument):
            return self.chunker.chunk_tabular_document(
                parsed_document, source_file, file_type
            )
        else:
            raise DocumentProcessingError(
                f"Unknown document type: {type(parsed_document)}",
                "ingestion_pipeline",
                {"file_path": str(file_path), "document_type": type(parsed_document).__name__}
            )
    
    def _generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for chunks with optimized batching."""
        if not chunks:
            return chunks
        
        # Process in smaller batches for better performance
        batch_size = 16  # Smaller batch size for faster processing
        all_embeddings = []
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]
            
            # Generate embeddings for batch
            batch_embeddings = get_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Log progress for large documents
            if len(chunks) > 50:
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        # Assign embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk.embedding = all_embeddings[i]
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    def _store_chunks(self, chunks: List[Chunk]) -> bool:
        """Store chunks in vector database."""
        if not chunks:
            logger.warning("No chunks to store")
            return True
        
        try:
            result = self.vector_store.insert(chunks)
            if result["inserted_count"] > 0:
                logger.info(f"Successfully stored {result['inserted_count']} chunks in vector database")
            return result["inserted_count"] > 0
        except Exception as e:
            logger.error(f"Failed to store chunks in vector database: {e}")
            return False
    
    @error_handler("ingestion_pipeline")
    def process_multiple_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple documents."""
        results = []
        total_chunks = 0
        successful_files = 0
        
        for file_path in file_paths:
            result = self.process_document(file_path)
            results.append(result)
            
            if result["success"]:
                successful_files += 1
                total_chunks += result["chunk_count"]
        
        return {
            "total_files": len(file_paths),
            "successful_files": successful_files,
            "failed_files": len(file_paths) - successful_files,
            "total_chunks": total_chunks,
            "results": results
        }
    
    @error_handler("ingestion_pipeline")
    def delete_document(self, source_file: str) -> bool:
        """Delete all chunks from a specific source file."""
        try:
            success = self.vector_store.delete_by_source(source_file)
            if success:
                logger.info(f"Deleted all chunks for source file: {source_file}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete document {source_file}: {e}")
            return False
    
    @error_handler("ingestion_pipeline")
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        try:
            stats = self.vector_store.get_collection_stats()
            return {
                "vector_store_stats": stats,
                "status": "Connected" if self.vector_store.connected else "Disconnected"
            }
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {e}")
            return {
                "vector_store_stats": {"error": str(e)},
                "status": "Error"
            }

# Global pipeline instance
document_pipeline = DocumentPipeline()