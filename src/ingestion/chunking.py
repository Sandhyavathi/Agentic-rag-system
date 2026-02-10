"""Smart document chunking for the Agentic RAG System."""

import logging
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..core.config import config
from ..core.error_handling import error_handler, DocumentProcessingError

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Document chunk with metadata."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class SmartChunker:
    """Intelligent document chunking with context awareness."""
    
    def __init__(self):
        self.chunk_size = config.chunking.chunk_size
        self.chunk_overlap = config.chunking.chunk_overlap
        self.separators = config.chunking.separators
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
    
    @error_handler("chunking")
    def chunk_structured_document(
        self, 
        parsed_document, 
        source_file: str,
        file_type: str
    ) -> List[Chunk]:
        """Chunk a structured document (from Docling parser)."""
        chunks = []
        chunk_id_counter = 0
        
        logger.info(f"Starting chunking for {source_file} (type: {file_type})")
        logger.info(f"Document content length: {len(parsed_document.content)} chars")
        
        # 1. Chunk main content if available and substantial
        if parsed_document.content and len(parsed_document.content.strip()) > 50:
            logger.info("Chunking main content")
            main_chunks = self._chunk_text(
                parsed_document.content, 
                source_file, 
                file_type, 
                chunk_id_counter,
                chunk_type="text",
                section_heading="Main Content"
            )
            chunks.extend(main_chunks)
            chunk_id_counter += len(main_chunks)
            logger.info(f"Generated {len(main_chunks)} main content chunks")
        
        # 2. If no main content, try to chunk sections individually
        elif hasattr(parsed_document, 'sections') and parsed_document.sections:
            logger.info("No main content, chunking sections individually")
            for section in parsed_document.sections:
                section_chunks = self._chunk_section(
                    section, source_file, file_type, chunk_id_counter
                )
                chunks.extend(section_chunks)
                chunk_id_counter += len(section_chunks)
        
        # 3. Add table chunks if available
        if hasattr(parsed_document, 'tables') and parsed_document.tables:
            logger.info(f"Adding {len(parsed_document.tables)} table chunks")
            for table in parsed_document.tables:
                table_chunks = self._chunk_table(
                    table, source_file, file_type, chunk_id_counter
                )
                chunks.extend(table_chunks)
                chunk_id_counter += len(table_chunks)
        
        # 4. Add figure chunks if available
        if hasattr(parsed_document, 'figures') and parsed_document.figures:
            logger.info(f"Adding {len(parsed_document.figures)} figure chunks")
            for figure in parsed_document.figures:
                figure_chunks = self._chunk_figure(
                    figure, source_file, file_type, chunk_id_counter
                )
                chunks.extend(figure_chunks)
                chunk_id_counter += len(figure_chunks)
        
        # 5. Add metadata summary chunk
        if parsed_document.metadata:
            metadata_chunk = self._create_metadata_chunk(
                parsed_document.metadata, source_file, file_type, chunk_id_counter
            )
            chunks.append(metadata_chunk)
        
        # 6. If still no chunks, create a fallback chunk
        if not chunks:
            logger.warning("No chunks generated, creating fallback chunk")
            fallback_content = f"Document: {source_file}\nType: {file_type}\nContent could not be extracted properly."
            fallback_chunk = Chunk(
                id=f"{source_file}_fallback_0",
                text=fallback_content,
                metadata={
                    "source_file": source_file,
                    "file_type": file_type,
                    "chunk_type": "fallback",
                    "chunk_index": 0,
                    "text_length": len(fallback_content)
                }
            )
            chunks.append(fallback_chunk)
        
        # 7. Ensure all chunks have proper metadata
        for chunk in chunks:
            if not chunk.metadata.get('source_file'):
                chunk.metadata['source_file'] = source_file
            if not chunk.metadata.get('file_type'):
                chunk.metadata['file_type'] = file_type
            if not chunk.metadata.get('chunk_type'):
                chunk.metadata['chunk_type'] = 'text'
        
        logger.info(f"Chunking complete: {len(chunks)} total chunks for {source_file}")
        return chunks

    @error_handler("chunking")
    def chunk_tabular_document(
        self, 
        tabular_document, 
        source_file: str,
        file_type: str
    ) -> List[Chunk]:
        """Chunk a tabular document (from tabular parser)."""
        chunks = []
        chunk_id_counter = 0
        
        # 1. Add column summary chunk
        column_summary = self._create_column_summary_chunk(
            tabular_document.metadata, source_file, file_type, chunk_id_counter
        )
        chunks.append(column_summary)
        chunk_id_counter += 1
        
        # 2. Add statistical summary chunk (if numeric data exists)
        if any("numeric_statistics" in meta.values() for meta in tabular_document.metadata.values()):
            stats_summary = self._create_statistical_summary_chunk(
                tabular_document.metadata, source_file, file_type, chunk_id_counter
            )
            chunks.append(stats_summary)
            chunk_id_counter += 1
        
        # 3. Add sample rows chunks
        for sheet_name, df in tabular_document.dataframes.items():
            sample_chunks = self._create_sample_rows_chunks(
                df, sheet_name, source_file, file_type, chunk_id_counter
            )
            chunks.extend(sample_chunks)
            chunk_id_counter += len(sample_chunks)
        
        # 4. Add data quality summary chunk
        quality_summary = self._create_quality_summary_chunk(
            tabular_document.metadata, source_file, file_type, chunk_id_counter
        )
        chunks.append(quality_summary)
        
        logger.info(f"Created {len(chunks)} chunks from tabular document {source_file}")
        return chunks
    
    def _chunk_section(
        self, 
        section: Dict[str, Any], 
        source_file: str, 
        file_type: str,
        chunk_id_start: int
    ) -> List[Chunk]:
        """Chunk a document section."""
        section_text = section.get('content', '')
        section_heading = section.get('heading', 'Untitled Section')
        section_level = section.get('level', 1)
        
        if not section_text.strip():
            return []
        
        # Add section heading to context
        context_text = f"## {section_heading}\n\n{section_text}"
        
        return self._chunk_text(
            context_text,
            source_file,
            file_type,
            chunk_id_start,
            chunk_type="text",
            section_heading=section_heading,
            section_level=section_level
        )
    
    def _chunk_table(
        self, 
        table: Dict[str, Any], 
        source_file: str, 
        file_type: str,
        chunk_id_start: int
    ) -> List[Chunk]:
        """Chunk a table."""
        table_content = table.get('markdown', '')
        table_index = table.get('index', 0)
        page = table.get('page', None)
        
        if not table_content.strip():
            return []
        
        # Create table context
        table_context = f"Table {table_index + 1}:\n{table_content}"
        
        return self._chunk_text(
            table_context,
            source_file,
            file_type,
            chunk_id_start,
            chunk_type="table",
            table_index=table_index,
            page=page
        )
    
    def _chunk_figure(
        self, 
        figure: Dict[str, Any], 
        source_file: str, 
        file_type: str,
        chunk_id_start: int
    ) -> List[Chunk]:
        """Chunk a figure."""
        caption = figure.get('caption', '')
        content = figure.get('content', '')
        figure_index = figure.get('index', 0)
        page = figure.get('page', None)
        
        if not caption.strip() and not content.strip():
            return []
        
        # Create figure context
        figure_context = f"Figure {figure_index + 1}: {caption}\n\n{content}"
        
        return self._chunk_text(
            figure_context,
            source_file,
            file_type,
            chunk_id_start,
            chunk_type="figure",
            figure_index=figure_index,
            page=page
        )
    
    def _chunk_text(
        self, 
        text: str, 
        source_file: str, 
        file_type: str,
        chunk_id_start: int,
        chunk_type: str = "text",
        **additional_metadata
    ) -> List[Chunk]:
        """Chunk text using the configured text splitter."""
        if not text or not text.strip():
            return []
        
        # Split text into chunks
        split_docs = self.text_splitter.create_documents([text])
        
        chunks = []
        for i, doc in enumerate(split_docs):
            chunk_id = f"{source_file}_{chunk_type}_{chunk_id_start + i}"
            
            # Ensure all required metadata fields are present
            metadata = {
                "source_file": source_file,  # Critical: ensure this is always set
                "file_type": file_type,
                "chunk_type": chunk_type,
                "chunk_index": chunk_id_start + i,
                "chunk_size": len(doc.page_content),
                "text_length": len(doc.page_content),
                **additional_metadata
            }
            
            chunk = Chunk(
                id=chunk_id,
                text=doc.page_content,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_metadata_chunk(
        self, 
        metadata: Dict[str, Any], 
        source_file: str, 
        file_type: str,
        chunk_id: int
    ) -> Chunk:
        """Create a metadata summary chunk."""
        metadata_text = f"Document Metadata for {source_file}:\n"
        metadata_text += "=" * 50 + "\n\n"
        
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                metadata_text += f"{key}: {json.dumps(value, indent=2)}\n\n"
            else:
                metadata_text += f"{key}: {value}\n\n"
        
        return Chunk(
            id=f"{source_file}_metadata_{chunk_id}",
            text=metadata_text,
            metadata={
                "source_file": source_file,
                "file_type": file_type,
                "chunk_type": "metadata",
                "chunk_index": chunk_id,
                "text_length": len(metadata_text)
            }
        )
    
    def _create_column_summary_chunk(
        self, 
        metadata: Dict[str, Any], 
        source_file: str, 
        file_type: str,
        chunk_id: int
    ) -> Chunk:
        """Create a column summary chunk for tabular data."""
        summary_text = f"Column Summary for {source_file}:\n"
        summary_text += "=" * 50 + "\n\n"
        
        for sheet_name, sheet_meta in metadata.items():
            summary_text += f"Sheet: {sheet_name}\n"
            summary_text += f"  Rows: {sheet_meta.get('rows', 0)}\n"
            summary_text += f"  Columns: {sheet_meta.get('columns', 0)}\n"
            summary_text += f"  Column Names: {', '.join(sheet_meta.get('column_names', []))}\n"
            summary_text += "\n"
        
        return Chunk(
            id=f"{source_file}_column_summary_{chunk_id}",
            text=summary_text,
            metadata={
                "source_file": source_file,
                "file_type": file_type,
                "chunk_type": "column_summary",
                "chunk_index": chunk_id,
                "text_length": len(summary_text)
            }
        )
    
    def _create_statistical_summary_chunk(
        self, 
        metadata: Dict[str, Any], 
        source_file: str, 
        file_type: str,
        chunk_id: int
    ) -> Chunk:
        """Create a statistical summary chunk for tabular data."""
        summary_text = f"Statistical Summary for {source_file}:\n"
        summary_text += "=" * 50 + "\n\n"
        
        for sheet_name, sheet_meta in metadata.items():
            if "numeric_statistics" in sheet_meta:
                summary_text += f"Sheet: {sheet_name}\n"
                for col, stats in sheet_meta["numeric_statistics"].items():
                    summary_text += f"  Column: {col}\n"
                    for stat_name, stat_value in stats.items():
                        if stat_value is not None:
                            summary_text += f"    {stat_name}: {stat_value}\n"
                    summary_text += "\n"
                summary_text += "\n"
        
        return Chunk(
            id=f"{source_file}_stats_summary_{chunk_id}",
            text=summary_text,
            metadata={
                "source_file": source_file,
                "file_type": file_type,
                "chunk_type": "statistical_summary",
                "chunk_index": chunk_id,
                "text_length": len(summary_text)
            }
        )
    
    def _create_sample_rows_chunks(
        self, 
        df: Any, 
        sheet_name: str, 
        source_file: str, 
        file_type: str,
        chunk_id_start: int
    ) -> List[Chunk]:
        """Create sample rows chunks for tabular data."""
        # Convert dataframe to string representation
        sample_text = df.to_string(index=False)
        
        # Split into chunks if too large
        if len(sample_text) > self.chunk_size * 2:
            # Split by rows
            rows = sample_text.split('\n')
            chunks = []
            current_chunk = ""
            current_rows = []
            
            for row in rows:
                if len(current_chunk + row + '\n') > self.chunk_size:
                    # Create chunk from current_rows
                    chunk_text = '\n'.join(current_rows)
                    chunk = Chunk(
                        id=f"{source_file}_sample_{sheet_name}_{chunk_id_start + len(chunks)}",
                        text=chunk_text,
                        metadata={
                            "source_file": source_file,
                            "file_type": file_type,
                            "chunk_type": "sample_rows",
                            "sheet_name": sheet_name,
                            "chunk_index": chunk_id_start + len(chunks),
                            "text_length": len(chunk_text)
                        }
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk
                    current_rows = [row]
                    current_chunk = row + '\n'
                else:
                    current_rows.append(row)
                    current_chunk += row + '\n'
            
            # Add final chunk
            if current_rows:
                chunk_text = '\n'.join(current_rows)
                chunk = Chunk(
                    id=f"{source_file}_sample_{sheet_name}_{chunk_id_start + len(chunks)}",
                    text=chunk_text,
                    metadata={
                        "source_file": source_file,
                        "file_type": file_type,
                        "chunk_type": "sample_rows",
                        "sheet_name": sheet_name,
                        "chunk_index": chunk_id_start + len(chunks),
                        "text_length": len(chunk_text)
                    }
                )
                chunks.append(chunk)
            
            return chunks
        else:
            # Single chunk
            return [Chunk(
                id=f"{source_file}_sample_{sheet_name}_{chunk_id_start}",
                text=sample_text,
                metadata={
                    "source_file": source_file,
                    "file_type": file_type,
                    "chunk_type": "sample_rows",
                    "sheet_name": sheet_name,
                    "chunk_index": chunk_id_start,
                    "text_length": len(sample_text)
                }
            )]

# Global chunker instance
smart_chunker = SmartChunker()