"""Hybrid search implementation combining semantic and keyword search."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import time
from rank_bm25 import BM25Okapi

from ..core.config import config
from ..core.state_management import Chunk, SearchResult
from ..core.error_handling import error_handler, RetrievalError
from ..retrieval.vector_store import milvus_vector_store

logger = logging.getLogger(__name__)

class HybridSearch:
    """Hybrid search combining semantic and keyword search with RRF ranking."""
    
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.chunk_index_map = {}
    
    @error_handler("hybrid_search")
    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform hybrid search with RRF (Reciprocal Rank Fusion) ranking."""
        try:
            # Get filtered chunks from Milvus
            vector_results = milvus_vector_store.search(
                query_vector=query_vector,
                top_k=top_k * 2,  # Get more results for hybrid ranking
                filters=filters
            )
            
            # If we have few results from vector search, try to get more
            if len(vector_results) < top_k:
                additional_results = milvus_vector_store.search(
                    query_vector=query_vector,
                    top_k=top_k * 4,
                    filters=filters
                )
                vector_results = additional_results
            
            # Perform keyword search using BM25
            keyword_results = self._bm25_search(query, top_k * 2, filters)
            
            # If we have vector results but no keyword results, return vector results
            if vector_results and not keyword_results:
                logger.info(f"Hybrid search: using vector results only ({len(vector_results)} results)")
                return vector_results[:top_k]
            
            # If we have both, combine using RRF
            if vector_results and keyword_results:
                combined_results = self._reciprocal_rank_fusion(
                    vector_results, 
                    keyword_results, 
                    top_k
                )
                logger.info(f"Hybrid search completed: {len(combined_results)} results")
                return combined_results
            
            # If we have no results, return empty list
            logger.warning("Hybrid search: no results found")
            return []
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RetrievalError(
                f"Hybrid search failed: {e}",
                "hybrid_search",
                {"query": query, "top_k": top_k}
            )
    
    def _bm25_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform BM25 keyword search."""
        try:
            # Initialize BM25 if not already done
            if self.bm25 is None:
                self._initialize_bm25()
            
            if not self.bm25:
                # Fallback to vector search only
                return []
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Create SearchResult objects
            keyword_results = []
            for i, score in enumerate(scores):
                if score > 0:  # Only include chunks with non-zero scores
                    chunk = self.chunks[i]
                    
                    # Apply filters if provided
                    if filters and not self._matches_filters(chunk.metadata, filters):
                        continue
                    
                    search_result = SearchResult(
                        chunk=chunk,
                        score=float(score),
                        metadata={
                            "search_type": "keyword",
                            "bm25_score": float(score)
                        }
                    )
                    keyword_results.append(search_result)
            
            # Sort by score and take top_k
            keyword_results.sort(key=lambda x: x.score, reverse=True)
            return keyword_results[:top_k]
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _initialize_bm25(self):
        """Initialize BM25 index from Milvus data."""
        try:
            # Get all chunks from Milvus for BM25 indexing
            # Note: In production, you might want to cache this or use a separate keyword index
            all_chunks = self._get_all_chunks_for_bm25()
            
            if not all_chunks:
                logger.warning("No chunks found for BM25 indexing")
                return
            
            # Prepare documents for BM25
            documents = []
            self.chunks = []
            self.chunk_index_map = {}
            
            for i, chunk in enumerate(all_chunks):
                # Tokenize text
                tokens = chunk.text.lower().split()
                documents.append(tokens)
                self.chunks.append(chunk)
                self.chunk_index_map[chunk.id] = i
            
            # Initialize BM25
            self.bm25 = BM25Okapi(documents)
            
            logger.info(f"Initialized BM25 with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.bm25 = None
    
    def _get_all_chunks_for_bm25(self) -> List[Chunk]:
        """Get all chunks from Milvus for BM25 indexing."""
        try:
            # This is a simplified approach - in production you might want:
            # 1. A separate keyword index
            # 2. Periodic updates to the BM25 index
            # 3. Caching of the BM25 index
            
            # For now, we'll return empty list to avoid performance issues
            # In a real implementation, you'd want to implement proper keyword indexing
            logger.warning("BM25 indexing not implemented - using vector search only")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get chunks for BM25: {e}")
            return []
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if chunk metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            chunk_value = metadata[key]
            
            if isinstance(value, str):
                if chunk_value != value:
                    return False
            elif isinstance(value, (int, float)):
                if chunk_value != value:
                    return False
            elif isinstance(value, list):
                if chunk_value not in value:
                    return False
        
        return True
    
    def _reciprocal_rank_fusion(
        self, 
        vector_results: List[SearchResult], 
        keyword_results: List[SearchResult], 
        top_k: int
    ) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        try:
            # Create a combined score dictionary
            combined_scores = {}
            
            # Add vector search scores
            for rank, result in enumerate(vector_results):
                doc_id = result.chunk.id
                rrf_score = 1.0 / (50 + rank + 1)  # RRF formula with k=50
                combined_scores[doc_id] = rrf_score
            
            # Add keyword search scores
            for rank, result in enumerate(keyword_results):
                doc_id = result.chunk.id
                rrf_score = 1.0 / (50 + rank + 1)  # RRF formula with k=50
                if doc_id in combined_scores:
                    combined_scores[doc_id] += rrf_score
                else:
                    combined_scores[doc_id] = rrf_score
            
            # Sort by combined score
            sorted_docs = sorted(
                combined_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Convert back to SearchResult objects
            final_results = []
            for doc_id, combined_score in sorted_docs[:top_k]:
                # Find the original SearchResult
                original_result = None
                
                # Look in vector results first
                for result in vector_results:
                    if result.chunk.id == doc_id:
                        original_result = result
                        break
                
                # If not found, look in keyword results
                if original_result is None:
                    for result in keyword_results:
                        if result.chunk.id == doc_id:
                            original_result = result
                            break
                
                if original_result:
                    # Create new SearchResult with combined score
                    final_result = SearchResult(
                        chunk=original_result.chunk,
                        score=combined_score,
                        metadata={
                            "search_type": "hybrid_rrf",
                            "original_score": original_result.score,
                            "combined_score": combined_score
                        }
                    )
                    final_results.append(final_result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"RRF fusion failed: {e}")
            # Fallback to vector search results
            return vector_results[:top_k]
    
    @error_handler("hybrid_search")
    def update_index(self, new_chunks: List[Chunk]):
        """Update the hybrid search index with new chunks."""
        try:
            # In a production system, you'd want to:
            # 1. Update the BM25 index incrementally
            # 2. Update any keyword indexes
            # 3. Invalidate caches as needed
            
            # For now, we'll just log the update
            logger.info(f"Index update requested for {len(new_chunks)} chunks")
            
            # Reinitialize BM25 to include new chunks
            self.bm25 = None
            self._initialize_bm25()
            
        except Exception as e:
            logger.error(f"Index update failed: {e}")

# Global hybrid search instance
hybrid_search = HybridSearch()