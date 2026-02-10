"""Production-ready Milvus vector store implementation."""

import logging
import threading
import time
from typing import List, Dict, Any, Optional

from pymilvus import MilvusClient, DataType
from pymilvus.exceptions import (
    MilvusException, ConnectError, MilvusUnavailableException,
    CollectionNotExistException, DataNotMatchException, ParamError
)

from ..core.config import config
from ..core.state_management import Chunk, SearchResult
from ..core.error_handling import error_handler, VectorStoreError

logger = logging.getLogger(__name__)

class MockMilvusClient:
    """Mock Milvus client for development when Milvus is unavailable."""
    
    def __init__(self):
        self.connected = False
        logger.warning("MockMilvusClient initialized - vector operations will fail")
    
    def list_collections(self):
        raise VectorStoreError(
            "Mock Milvus client - no actual connection available",
            "mock_client",
            {"operation": "list_collections"}
        )
    
    def has_collection(self, collection_name):
        return False
    
    def create_collection(self, **kwargs):
        raise VectorStoreError(
            "Mock Milvus client - cannot create collections",
            "mock_client",
            {"operation": "create_collection"}
        )
    
    def insert(self, **kwargs):
        raise VectorStoreError(
            "Mock Milvus client - cannot insert data",
            "mock_client",
            {"operation": "insert"}
        )
    
    def search(self, **kwargs):
        raise VectorStoreError(
            "Mock Milvus client - cannot perform searches",
            "mock_client",
            {"operation": "search"}
        )
    
    def get_collection_stats(self, **kwargs):
        raise VectorStoreError(
            "Mock Milvus client - cannot get collection stats",
            "mock_client",
            {"operation": "get_collection_stats"}
        )
    
    def drop_collection(self, **kwargs):
        raise VectorStoreError(
            "Mock Milvus client - cannot drop collections",
            "mock_client",
            {"operation": "drop_collection"}
        )
    
    def delete(self, **kwargs):
        raise VectorStoreError(
            "Mock Milvus client - cannot delete data",
            "mock_client",
            {"operation": "delete"}
        )

class MilvusConnectionManager:
    """Thread-safe Milvus connection manager with retry logic."""
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_client(cls, uri=None, token=None):
        """Get singleton MilvusClient instance with connection retry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if uri is None:
                        uri = f"http://{config.milvus.host}:{config.milvus.port}"
                    if token is None:
                        token = "root:Milvus"
                    
                    # Try to connect to Milvus server first
                    for attempt in range(5):
                        try:
                            cls._instance = MilvusClient(uri=uri, token=token, timeout=30)
                            # Verify connection
                            cls._instance.list_collections()
                            logger.info(f"Connected to Milvus server at {uri}")
                            return cls._instance
                        except (ConnectError, MilvusException) as e:
                            wait_time = 2 ** attempt
                            logger.warning(f"Milvus server connection attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                    
                    # If server connection fails, fall back to Milvus Lite for development
                    logger.warning("Milvus server unavailable, falling back to Milvus Lite for development")
                    try:
                        cls._instance = MilvusClient(uri="sqlite://./milvus_dev.db")
                        logger.info("Connected to Milvus Lite (development mode)")
                    except Exception as e:
                        logger.error(f"Failed to connect to Milvus Lite: {e}")
                        # For development, create a mock connection that raises helpful errors
                        logger.warning("Creating mock Milvus connection for development - vector operations will fail")
                        cls._instance = MockMilvusClient()
                        logger.info("Using mock Milvus connection (development mode)")
        return cls._instance

class MilvusVectorStore:
    """Production-ready Milvus vector store with proper schema and error handling."""
    
    def __init__(self):
        self.client = None
        self.collection_name = config.milvus.collection_name
        self.embedding_dim = config.milvus.embedding_dim
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Milvus connection and ensure collection exists."""
        try:
            self.client = MilvusConnectionManager.get_client()
            
            # Create collection if it doesn't exist
            if not self.client.has_collection(self.collection_name):
                self._create_collection()
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Milvus connection: {e}")
            raise VectorStoreError(
                f"Failed to initialize Milvus: {e}",
                "vector_store_init",
                {"collection_name": self.collection_name}
            )
    
    def _create_collection(self):
        """Create Milvus collection with optimized schema for RAG."""
        try:
            logger.info(f"Creating collection: {self.collection_name} with {self.embedding_dim} dimensions")
            
            # Create schema using MilvusClient API (recommended since v2.3.7)
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
                description="Document chunks with embeddings for RAG"
            )
            
            # Primary key: chunk ID
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=512,
                is_primary=True
            )
            
            # Chunk text content
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            
            # Dense embeddings
            schema.add_field(
                field_name="embedding",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim
            )
            
            # Source document
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                max_length=512
            )
            
            # Chunk index within document
            schema.add_field(
                field_name="chunk_index",
                datatype=DataType.INT64
            )
            
            # JSON metadata for flexible attributes
            schema.add_field(
                field_name="metadata",
                datatype=DataType.JSON
            )
            
            # Create optimized indexes
            index_params = self.client.prepare_index_params()
            
            # HNSW index for embeddings (optimal for < 100M vectors)
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200}
            )
            
            # Scalar indexes for filtering
            index_params.add_index(field_name="source", index_type="AUTOINDEX")
            index_params.add_index(field_name="chunk_index", index_type="AUTOINDEX")
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            
            logger.info(f"Successfully created collection: {self.collection_name}")
            
        except MilvusException as e:
            logger.error(f"Failed to create collection: {e}")
            raise VectorStoreError(
                f"Failed to create collection: {e}",
                "collection_creation",
                {"collection_name": self.collection_name, "embedding_dim": self.embedding_dim}
            )
    
    @error_handler("vector_store")
    def insert(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Insert chunks into Milvus with proper batching."""
        if not chunks:
            logger.warning("No chunks provided for insertion")
            return {"inserted_count": 0, "insert_time": 0.0}
        
        try:
            # Prepare data for insertion
            data = []
            for chunk in chunks:
                if not chunk.embedding:
                    raise VectorStoreError(
                        f"Chunk {chunk.id} missing embedding",
                        "data_preparation",
                        {"chunk_id": chunk.id}
                    )
                
                if len(chunk.embedding) != self.embedding_dim:
                    raise VectorStoreError(
                        f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(chunk.embedding)}",
                        "dimension_mismatch",
                        {"chunk_id": chunk.id, "expected": self.embedding_dim, "actual": len(chunk.embedding)}
                    )
                
                # Ensure text length doesn't exceed VARCHAR limit (bytes, not characters)
                chunk_text = chunk.text
                if len(chunk_text.encode('utf-8')) > 65535:
                    chunk_text = chunk_text[:65000] + "..."
                    logger.warning(f"Truncated text for chunk {chunk.id} due to length limit")
                
                chunk_data = {
                    "id": chunk.id,
                    "text": chunk_text,
                    "embedding": chunk.embedding,
                    "source": chunk.metadata.get("source_file", "unknown"),
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "metadata": chunk.metadata
                }
                data.append(chunk_data)
            
            # Batch insertion with smaller batch size for better performance
            batch_size = 500
            total_inserted = 0
            start_time = time.time()
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                result = self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
                total_inserted += len(batch)
                logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch)} chunks")
            
            insert_time = time.time() - start_time
            logger.info(f"Successfully inserted {total_inserted} chunks in {insert_time:.3f}s")
            
            return {
                "inserted_count": total_inserted,
                "insert_time": insert_time
            }
            
        except MilvusException as e:
            logger.error(f"Failed to insert chunks: {e}")
            raise VectorStoreError(
                f"Failed to insert chunks: {e}",
                "insertion",
                {"chunk_count": len(chunks), "error_code": getattr(e, 'code', None)}
            )
    
    @error_handler("vector_store")
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar chunks using HNSW index."""
        try:
            logger.info(f"Starting vector search: query_dim={len(query_vector)}, top_k={top_k}")
            
            # Validate query vector
            if len(query_vector) != self.embedding_dim:
                raise VectorStoreError(
                    f"Query vector dimension mismatch: expected {self.embedding_dim}, got {len(query_vector)}",
                    "query_dimension_mismatch",
                    {"expected": self.embedding_dim, "actual": len(query_vector)}
                )
            
            # Build search parameters (ef must be >= top_k)
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": max(64, top_k * 4)}
            }
            
            # Build filter expression
            filter_expr = self._build_filter_expression(filters)
            
            # Build search parameters correctly for MilvusClient
            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [query_vector],
                "anns_field": "embedding",
                "search_params": search_params,
                "limit": top_k,
                "output_fields": ["id", "text", "source", "chunk_index", "metadata"]
            }
            
            # Add filter expression only if it exists
            if filter_expr:
                search_kwargs["expr"] = filter_expr
            
            # Perform search
            start_time = time.time()
            
            logger.info(f"Executing Milvus search with params: {search_kwargs}")
            results = self.client.search(**search_kwargs)
            search_time = time.time() - start_time
            
            logger.info(f"Milvus returned {len(results[0]) if results and results[0] else 0} raw results")
            
            # Check if we got results
            if not results or not results[0]:
                logger.warning("No search results returned from Milvus")
                return []
            
            # Convert to SearchResult objects
            search_results = []
            for hit in results[0]:
                # Extract entity data - fields are nested inside 'entity'
                entity = hit.get("entity", {})
                
                chunk = Chunk(
                    id=hit.get("id", entity.get("id", "")),
                    text=entity.get("text", ""),
                    metadata={
                        "source_file": entity.get("source", ""),
                        "chunk_index": entity.get("chunk_index", 0),
                        **(entity.get("metadata", {}))
                    }
                )
                
                # IMPORTANT: Convert COSINE distance to similarity score
                # Milvus COSINE distance: 0 = identical, 1 = completely different
                # We convert to similarity: 1 = identical, 0 = completely different
                cosine_distance = hit["distance"]
                similarity_score = max(0.0, 1.0 - cosine_distance)  # Ensure non-negative
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=similarity_score,  # Now higher is better
                    metadata={
                        "search_time": search_time,
                        "search_type": "vector",
                        "original_distance": cosine_distance
                    }
                )
                search_results.append(search_result)
            
            # Log scores for debugging
            if search_results:
                scores = [r.score for r in search_results]
                logger.info(f"Similarity scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")
            
            logger.info(f"Vector search completed: {len(search_results)} results in {search_time:.3f}s")
            return search_results
            
        except (MilvusException, CollectionNotExistException) as e:
            logger.error(f"Vector search failed: {e}")
            raise VectorStoreError(
                f"Vector search failed: {e}",
                "search",
                {"top_k": top_k, "filters": filters, "error_code": getattr(e, 'code', None)}
            )
        except Exception as e:
            logger.error(f"Unexpected error in vector search: {e}", exc_info=True)
            raise VectorStoreError(
                f"Vector search failed: {e}",
                "search",
                {"top_k": top_k, "filters": filters}
            )
    
    def _build_filter_expression(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """Build Milvus filter expression with proper escaping."""
        if not filters:
            return None
        
        expressions = []
        
        for key, value in filters.items():
            if key in ["source_file", "chunk_type"]:  # Direct metadata fields
                if isinstance(value, str):
                    # Escape quotes in string values
                    escaped_value = value.replace('"', '\\"')
                    expressions.append(f'metadata["{key}"] == "{escaped_value}"')
                elif isinstance(value, (int, float)):
                    expressions.append(f'metadata["{key}"] == {value}')
                elif isinstance(value, list):
                    list_conditions = []
                    for item in value:
                        if isinstance(item, str):
                            escaped_item = item.replace('"', '\\"')
                            list_conditions.append(f'metadata["{key}"] == "{escaped_item}"')
                        else:
                            list_conditions.append(f'metadata["{key}"] == {item}')
                    expressions.append(f"({' or '.join(list_conditions)})")
            
            elif key == "source":  # Direct field
                if isinstance(value, str):
                    escaped_value = value.replace('"', '\\"')
                    expressions.append(f'source == "{escaped_value}"')
                elif isinstance(value, list):
                    list_conditions = []
                    for v in value:
                        if isinstance(v, str):
                            escaped_v = v.replace('"', '\\"')
                            list_conditions.append(f'source == "{escaped_v}"')
                    if list_conditions:
                        expressions.append(f"({' or '.join(list_conditions)})")
        
        return " and ".join(expressions) if expressions else None
    
    @error_handler("vector_store")
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and status."""
        try:
            if not self.client.has_collection(self.collection_name):
                return {
                    "total_entities": 0,
                    "status": "not_exists",
                    "collection_name": self.collection_name,
                    "storage_type": "milvus"
                }
            
            # Get collection statistics
            stats = self.client.get_collection_stats(self.collection_name)
            total_entities = stats.get("row_count", 0)
            
            return {
                "total_entities": int(total_entities),
                "status": "connected",
                "collection_name": self.collection_name,
                "dimension": self.embedding_dim,
                "storage_type": "milvus",
                "index_type": "HNSW"
            }
            
        except MilvusException as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "total_entities": 0,
                "status": "error",
                "error": str(e),
                "storage_type": "milvus"
            }
    
    @error_handler("vector_store")
    def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info(f"Dropped collection: {self.collection_name}")
            
            # Recreate the collection
            self._create_collection()
            logger.info(f"Collection {self.collection_name} cleared and recreated")
            return True
            
        except MilvusException as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    @error_handler("vector_store")
    def delete_by_source(self, source_file: str) -> bool:
        """Delete all chunks from a specific source file."""
        try:
            # Use delete with filter expression
            escaped_source = source_file.replace('"', '\\"')
            filter_expr = f'source == "{escaped_source}"'
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
            
            logger.info(f"Deleted chunks from source: {source_file}")
            return True
            
        except MilvusException as e:
            logger.error(f"Failed to delete by source {source_file}: {e}")
            return False
    
    def connect(self) -> bool:
        """Check if connected to Milvus."""
        try:
            if self.client:
                self.client.list_collections()
                return True
            return False
        except:
            return False
    
    def disconnect(self):
        """Disconnect from Milvus (no-op for MilvusClient)."""
        # MilvusClient handles connection lifecycle automatically
        pass

# Global vector store instance
milvus_vector_store = MilvusVectorStore()