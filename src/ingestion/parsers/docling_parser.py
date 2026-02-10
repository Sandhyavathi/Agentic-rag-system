"""Docling-based document parser for PDF, DOCX, and PPTX files."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import asyncio
from dataclasses import dataclass

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PipelineOptions

from ...core.error_handling import error_handler, DocumentProcessingError

logger = logging.getLogger(__name__)

@dataclass
class ParsedDocument:
    """Result of document parsing."""
    content: str
    metadata: Dict[str, Any]
    sections: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]

class DoclingParser:
    """Parser for structured documents using Docling."""
    
    def __init__(self):
        self.converter = DocumentConverter()
    
    @error_handler("docling_parser")
    # In src/ingestion/parsers/docling_parser.py

    def parse_file(self, file_path: str) -> ParsedDocument:
        """Parse a document file using Docling with robust content extraction."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentProcessingError(
                f"File not found: {file_path}",
                "docling_parser",
                {"file_path": str(file_path)}
            )
        
        try:
            logger.info(f"Starting Docling parsing for: {file_path}")
            
            # Convert document
            result = self.converter.convert(file_path)
            document = result.document
            
            logger.info(f"Docling conversion completed for: {file_path}")
            
            # Extract content - try multiple methods for robustness
            content = ""
            
            # Method 1: export_to_markdown (preferred)
            try:
                if hasattr(document, 'export_to_markdown'):
                    content = document.export_to_markdown()
                    logger.info(f"Extracted content via export_to_markdown: {len(content)} chars")
            except Exception as e:
                logger.warning(f"export_to_markdown failed: {e}")
            
            # Method 2: export_to_text (fallback)
            if not content and hasattr(document, 'export_to_text'):
                try:
                    content = document.export_to_text()
                    logger.info(f"Extracted content via export_to_text: {len(content)} chars")
                except Exception as e:
                    logger.warning(f"export_to_text failed: {e}")
            
            # Method 3: manual extraction from pages (ultimate fallback)
            if not content:
                logger.warning("Using manual extraction from pages")
                page_contents = []
                if hasattr(document, 'pages'):
                    for page in document.pages:
                        if hasattr(page, 'items'):
                            for item in page.items:
                                if hasattr(item, 'text') and item.text:
                                    page_contents.append(item.text)
                content = "\n".join(page_contents)
                logger.info(f"Manually extracted content: {len(content)} chars")
            
            # If still no content, this is an error
            if not content or len(content.strip()) < 10:
                raise DocumentProcessingError(
                    f"No meaningful text content could be extracted from {file_path}",
                    "docling_parser",
                    {"file_path": str(file_path), "content_length": len(content)}
                )
            
            # Extract metadata and structure
            metadata = self._extract_metadata(document)
            sections = self._extract_sections(document)
            tables = self._extract_tables(document)
            figures = self._extract_figures(document)
            
            logger.info(f"Parsing complete: {len(content)} chars, {len(sections)} sections, {len(tables)} tables")
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                sections=sections,
                tables=tables,
                figures=figures
            )
            
        except Exception as e:
            logger.error(f"Docling parsing failed for {file_path}: {e}")
            # Try fallback parsing for PDFs
            if file_path.suffix.lower() == '.pdf':
                return self._parse_pdf_fallback(file_path)
            else:
                raise DocumentProcessingError(
                    f"Failed to parse document: {e}",
                    "docling_parser",
                    {"file_path": str(file_path)}
                )

    def _parse_pdf_fallback(self, file_path: Path) -> ParsedDocument:
        """Fallback PDF parsing using PyPDF when Docling fails."""
        try:
            logger.warning(f"Using fallback PDF parsing for {file_path}")
            import pypdf
            
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text_content = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            if not text_content or len(text_content.strip()) < 10:
                raise DocumentProcessingError(
                    f"Fallback PDF parsing extracted no content from {file_path}",
                    "docling_parser",
                    {"file_path": str(file_path)}
                )
            
            metadata = {
                "title": file_path.stem,
                "page_count": len(pdf_reader.pages),
                "file_type": "pdf",
                "parser": "pypdf_fallback"
            }
            
            sections = [{
                "index": 0,
                "heading": "Full Document",
                "content": text_content,
                "level": 1
            }]
            
            logger.info(f"Fallback parsing successful: {len(text_content)} chars")
            
            return ParsedDocument(
                content=text_content,
                metadata=metadata,
                sections=sections,
                tables=[],
                figures=[]
            )
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Both Docling and fallback PDF parsing failed: {e}",
                "docling_parser",
                {"file_path": str(file_path)}
            )
        
    def _get_input_format(self, file_path: Path) -> InputFormat:
        """Determine input format based on file extension."""
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return InputFormat.PDF
        elif extension == '.docx':
            return InputFormat.DOCX
        elif extension == '.pptx':
            return InputFormat.PPTX
        elif extension == '.html':
            return InputFormat.HTML
        elif extension in ['.xlsx', '.xls']:
            return InputFormat.XLSX  # Docling supports Excel!
        elif extension in ['.png', '.jpg', '.jpeg']:
            return InputFormat.IMAGE
        else:
            raise DocumentProcessingError(
                f"Unsupported file format: {extension}",
                "docling_parser",
                {"file_path": str(file_path), "extension": extension}
            )
    
    def _extract_metadata(self, document) -> Dict[str, Any]:
        """Extract metadata from parsed document."""
        metadata = {
            "title": getattr(document, 'title', None),
            "author": getattr(document, 'author', None),
            "creation_date": getattr(document, 'creation_date', None),
            "modification_date": getattr(document, 'modification_date', None),
            "page_count": len(document.pages) if hasattr(document, 'pages') else 0,
            "file_type": "structured_document"
        }
        
        # Clean up metadata
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _extract_sections(self, document) -> List[Dict[str, Any]]:
        """Extract sections and headings from document."""
        sections = []
        
        # Try to extract from pages structure (Docling's actual structure)
        if hasattr(document, 'pages'):
            for page_idx, page in enumerate(document.pages):
                if hasattr(page, 'items'):
                    for item_idx, item in enumerate(page.items):
                        if hasattr(item, 'type') and item.type in ['heading', 'title']:
                            sections.append({
                                "index": len(sections),
                                "heading": getattr(item, 'text', ''),
                                "content": getattr(item, 'text', ''),
                                "level": getattr(item, 'level', 1),
                                "page": page_idx + 1,
                                "item_index": item_idx
                            })
        
        # Fallback: try original structure
        elif hasattr(document, 'sections'):
            for i, section in enumerate(document.sections):
                sections.append({
                    "index": i,
                    "heading": getattr(section, 'heading', ''),
                    "content": getattr(section, 'content', ''),
                    "level": getattr(section, 'level', 1),
                    "page_range": getattr(section, 'page_range', None)
                })
        
        return sections
    
    def _extract_tables(self, document) -> List[Dict]:
        """Extract and structure tables."""
        tables = []
        
        # Extract from pages structure (Docling's actual structure)
        if hasattr(document, 'pages'):
            for page_idx, page in enumerate(document.pages):
                if hasattr(page, 'items'):
                    for item in page.items:
                        if hasattr(item, 'type') and item.type == 'table':
                            table_data = {
                                'index': len(tables),
                                'content': getattr(item, 'text', ''),
                                'markdown': getattr(item, 'markdown', ''),
                                'page': page_idx + 1,
                                'bbox': getattr(item, 'bbox', None),
                                'caption': getattr(item, 'caption', '')
                            }
                            tables.append(table_data)
        
        # Fallback: try original structure
        elif hasattr(document, 'tables'):
            for i, table in enumerate(document.tables):
                table_data = {
                    'index': i,
                    'markdown': getattr(table, 'markdown', ''),
                    'page': getattr(table, 'page', None),
                    'caption': getattr(table, 'caption', ''),
                }
                
                # Try to export as DataFrame for structured data
                try:
                    if hasattr(table, 'export_to_dataframe'):
                        table_data['data'] = table.export_to_dataframe()
                    table_data['text_representation'] = self._table_to_text(table)
                except:
                    table_data['content'] = getattr(table, 'content', '')
                
                tables.append(table_data)
        
        return tables
    
    def _table_to_text(self, table) -> str:
        """Convert table to text representation."""
        try:
            if hasattr(table, 'export_to_markdown'):
                return table.export_to_markdown()
            elif hasattr(table, 'markdown'):
                return table.markdown
            else:
                return str(table)
        except:
            return str(table)
    
    def _extract_figures(self, document) -> List[Dict[str, Any]]:
        """Extract figures from document."""
        figures = []
        
        # Extract from pages structure (Docling's actual structure)
        if hasattr(document, 'pages'):
            for page_idx, page in enumerate(document.pages):
                if hasattr(page, 'items'):
                    for item in page.items:
                        if hasattr(item, 'type') and item.type == 'image':
                            figures.append({
                                "index": len(figures),
                                "caption": getattr(item, 'text', ''),
                                "content": getattr(item, 'text', ''),
                                "page": page_idx + 1,
                                "bbox": getattr(item, 'bbox', None)
                            })
        
        # Fallback: try original structure
        elif hasattr(document, 'figures'):
            for i, figure in enumerate(document.figures):
                figures.append({
                    "index": i,
                    "caption": getattr(figure, 'caption', ''),
                    "content": getattr(figure, 'content', ''),
                    "bbox": getattr(figure, 'bbox', None),
                    "page": getattr(figure, 'page', None)
                })
        
        return figures
    
    def _parse_text(self, file_path: Path) -> ParsedDocument:
        """Parse plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "title": file_path.stem,
                "file_type": "text",
                "file_size": file_path.stat().st_size
            }
            
            sections = [{
                "index": 0,
                "heading": "Main Content",
                "content": content,
                "level": 1,
                "page_range": None
            }]
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                sections=sections,
                tables=[],
                figures=[]
            )
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to parse text file: {e}",
                "docling_parser",
                {"file_path": str(file_path)}
            )

# Global parser instance
docling_parser = DoclingParser()