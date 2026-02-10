"""Retrieval agents for the Agentic RAG System."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
import time

from ..core.state_management import AgentState, RetrievalStrategy, AgentName, SearchResult
from ..core.error_handling import error_handler, RetrievalError
from ..core.config import config
from ..retrieval.vector_store import milvus_vector_store
from ..retrieval.hybrid_search import HybridSearch
from ..llm.embeddings import get_embeddings
from ..core.state_management import add_agent_trace

logger = logging.getLogger(__name__)

class RetrievalAgent:
    """Base retrieval agent."""
    
    def __init__(self):
        self.hybrid_search = HybridSearch()
    
    def _get_query_vector(self, query: str) -> List[float]:
        """Get embedding vector for query."""
        return get_embeddings([query])[0]
    

    def _filter_relevant_chunks(self, chunks: List[SearchResult]) -> List[SearchResult]:
        """Filter chunks based on relevance score."""
        if not chunks:
            return []
        
        # Log all scores for debugging
        scores = [chunk.score for chunk in chunks]
        logger.info(f"All similarity scores: {[f'{s:.3f}' for s in scores]}")
        
        # Use a very low threshold to ensure we don't filter out good content
        threshold = 0.1
        
        # For debugging, let's also check the actual text content
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"Chunk {i} (score={chunk.score:.3f}): {chunk.chunk.text[:200]}...")
        
        filtered_chunks = [chunk for chunk in chunks if chunk.score >= threshold]
        
        logger.info(f"After filtering with threshold {threshold}: {len(filtered_chunks)} chunks remain")
        
        # If we filtered out everything, return the top 3 chunks anyway
        if not filtered_chunks and chunks:
            logger.warning("All chunks filtered out, returning top 3 chunks")
            filtered_chunks = chunks[:3]
        
        return filtered_chunks

class SimpleRetrievalAgent(RetrievalAgent):
    """Simple retrieval agent using only vector search."""
    
    @error_handler("retrieval_agent")
    def retrieve(self, state: AgentState) -> AgentState:
        """Perform simple vector retrieval."""
        user_query = state["user_query"]
        query_analysis = state["query_analysis"]
        
        logger.info(f"Simple retrieval for query: {user_query}")
        
        try:
            # Get query vector
            query_vector = self._get_query_vector(user_query)
            
            # Build filters from query analysis
            filters = query_analysis.requires_filters if query_analysis else None
            
            # Perform vector search
            start_time = time.time()
            results = milvus_vector_store.search(
                query_vector=query_vector,
                top_k=config.retrieval.top_k,
                filters=filters
            )
            search_time = time.time() - start_time
            
            # Debug logging
            logger.info(f"Raw search results: {len(results)} chunks found")
            for i, result in enumerate(results[:3]):  # Log first 3 results
                logger.info(f"Result {i}: score={result.score:.3f}, text_preview='{result.chunk.text[:100]}...'")
            
            # Filter results
            relevant_results = self._filter_relevant_chunks(results)
            
            logger.info(f"After filtering with threshold {config.retrieval.confidence_threshold}: {len(relevant_results)} chunks remain")
            
            # Update state
            state["retrieved_chunks"] = [result.chunk for result in relevant_results]
            state["retrieval_metadata"] = {
                "method": RetrievalStrategy.SIMPLE,
                "num_chunks": len(relevant_results),
                "search_queries": [user_query],
                "execution_time": search_time
            }
            
            # Add agent trace
            from ..core.state_management import add_agent_trace
            state = add_agent_trace(
                state=state,
                agent=AgentName.RETRIEVAL_AGENT,
                decision=f"Simple retrieval: {len(relevant_results)} chunks found",
                reasoning=f"Used vector search with {config.retrieval.top_k} results",
                input_state={"query": user_query, "filters": filters},
                output_state={
                    "retrieved_chunks_count": len(relevant_results),
                    "execution_time": search_time
                }
            )
            
            logger.info(f"Simple retrieval completed: {len(relevant_results)} chunks in {search_time:.3f}s")
            return state
            
        except Exception as e:
            logger.error(f"Simple retrieval failed: {e}")
            raise RetrievalError(
                f"Simple retrieval failed: {e}",
                "retrieval_agent",
                {"query": user_query}
            )

class HybridRetrievalAgent(RetrievalAgent):
    """Hybrid retrieval agent using semantic + keyword search."""
    
    @error_handler("retrieval_agent")
    def retrieve(self, state: AgentState) -> AgentState:
        """Perform hybrid retrieval."""
        user_query = state["user_query"]
        query_analysis = state["query_analysis"]
        
        logger.info(f"Hybrid retrieval for query: {user_query}")
        
        try:
            # Get query vector
            query_vector = self._get_query_vector(user_query)
            
            # Build filters
            filters = query_analysis.requires_filters if query_analysis else None
            
            # Perform hybrid search
            start_time = time.time()
            results = self.hybrid_search.search(
                query=user_query,
                query_vector=query_vector,
                top_k=config.retrieval.top_k,
                filters=filters
            )
            search_time = time.time() - start_time
            
            # Filter results
            relevant_results = self._filter_relevant_chunks(results)
            
            # Update state
            state["retrieved_chunks"] = [result.chunk for result in relevant_results]
            state["retrieval_metadata"] = {
                "method": RetrievalStrategy.HYBRID,
                "num_chunks": len(relevant_results),
                "search_queries": [user_query],
                "execution_time": search_time
            }
            
            # Add agent trace
            state = add_agent_trace(
                state=state,
                agent=AgentName.RETRIEVAL_AGENT,
                decision=f"Hybrid retrieval: {len(relevant_results)} chunks found",
                reasoning=f"Combined semantic and keyword search",
                input_state={"query": user_query, "filters": filters},
                output_state={
                    "retrieved_chunks_count": len(relevant_results),
                    "execution_time": search_time
                }
            )
            
            logger.info(f"Hybrid retrieval completed: {len(relevant_results)} chunks in {search_time:.3f}s")
            return state
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise RetrievalError(
                f"Hybrid retrieval failed: {e}",
                "retrieval_agent",
                {"query": user_query}
            )

class ParallelRetrievalAgent(RetrievalAgent):
    """Parallel retrieval agent for comparison queries."""
    
    @error_handler("retrieval_agent")
    def retrieve(self, state: AgentState) -> AgentState:
        """Perform parallel retrieval for comparison queries."""
        user_query = state["user_query"]
        query_analysis = state["query_analysis"]
        
        logger.info(f"Parallel retrieval for query: {user_query}")
        
        if not query_analysis or not query_analysis.entities:
            logger.warning("No entities found for parallel retrieval, falling back to simple retrieval")
            return SimpleRetrievalAgent().retrieve(state)
        
        try:
            # Extract entities for comparison
            entities = query_analysis.entities
            logger.info(f"Parallel retrieval for entities: {entities}")
            
            all_results = []
            search_queries = []
            total_time = 0
            
            # Search for each entity separately
            for entity in entities:
                query_vector = self._get_query_vector(entity)
                filters = query_analysis.requires_filters if query_analysis else None
                
                start_time = time.time()
                results = milvus_vector_store.search(
                    query_vector=query_vector,
                    top_k=config.retrieval.top_k,
                    filters=filters
                )
                search_time = time.time() - start_time
                total_time += search_time
                
                # Add entity label to results
                for result in results:
                    result.chunk.metadata["entity"] = entity
                
                all_results.extend(results)
                search_queries.append(entity)
            
            # Filter and deduplicate results
            relevant_results = self._filter_relevant_chunks(all_results)
            
            # Update state
            state["retrieved_chunks"] = [result.chunk for result in relevant_results]
            state["retrieval_metadata"] = {
                "method": RetrievalStrategy.PARALLEL,
                "num_chunks": len(relevant_results),
                "search_queries": search_queries,
                "execution_time": total_time
            }
            
            # Add agent trace
            state = add_agent_trace(
                state=state,
                agent=AgentName.RETRIEVAL_AGENT,
                decision=f"Parallel retrieval: {len(relevant_results)} chunks from {len(entities)} entities",
                reasoning=f"Searched for entities: {entities}",
                input_state={"query": user_query, "entities": entities},
                output_state={
                    "retrieved_chunks_count": len(relevant_results),
                    "entities_searched": len(entities),
                    "execution_time": total_time
                }
            )
            
            logger.info(f"Parallel retrieval completed: {len(relevant_results)} chunks in {total_time:.3f}s")
            return state
            
        except Exception as e:
            logger.error(f"Parallel retrieval failed: {e}")
            raise RetrievalError(
                f"Parallel retrieval failed: {e}",
                "retrieval_agent",
                {"query": user_query, "entities": entities}
            )

class MultiHopRetrievalAgent(RetrievalAgent):
    """Multi-hop retrieval agent for complex reasoning."""
    
    @error_handler("retrieval_agent")
    def retrieve(self, state: AgentState) -> AgentState:
        """Perform multi-hop retrieval."""
        user_query = state["user_query"]
        query_analysis = state["query_analysis"]
        
        logger.info(f"Multi-hop retrieval for query: {user_query}")
        
        try:
            # Break query into sub-queries
            sub_queries = self._generate_sub_queries(user_query, query_analysis)
            logger.info(f"Generated {len(sub_queries)} sub-queries: {sub_queries}")
            
            all_results = []
            search_queries = []
            total_time = 0
            
            # Retrieve for each sub-query iteratively
            for i, sub_query in enumerate(sub_queries):
                query_vector = self._get_query_vector(sub_query)
                filters = query_analysis.requires_filters if query_analysis else None
                
                start_time = time.time()
                results = milvus_vector_store.search(
                    query_vector=query_vector,
                    top_k=config.retrieval.top_k,
                    filters=filters
                )
                search_time = time.time() - start_time
                total_time += search_time
                
                # Add sub-query context
                for result in results:
                    result.chunk.metadata["sub_query"] = sub_query
                    result.chunk.metadata["hop_number"] = i + 1
                
                all_results.extend(results)
                search_queries.append(sub_query)
                
                # Use results to refine next query (simplified approach)
                if i < len(sub_queries) - 1:
                    # In a full implementation, we would use the results to refine the next query
                    # For now, we just continue with the pre-generated sub-queries
                    pass
            
            # Filter and combine results
            relevant_results = self._filter_relevant_chunks(all_results)
            
            # Update state
            state["retrieved_chunks"] = [result.chunk for result in relevant_results]
            state["retrieval_metadata"] = {
                "method": RetrievalStrategy.MULTI_HOP,
                "num_chunks": len(relevant_results),
                "search_queries": search_queries,
                "execution_time": total_time
            }
            
            # Add agent trace
            state = add_agent_trace(
                state=state,
                agent=AgentName.RETRIEVAL_AGENT,
                decision=f"Multi-hop retrieval: {len(relevant_results)} chunks from {len(sub_queries)} hops",
                reasoning=f"Sub-queries: {sub_queries}",
                input_state={"query": user_query, "sub_queries": sub_queries},
                output_state={
                    "retrieved_chunks_count": len(relevant_results),
                    "hops_completed": len(sub_queries),
                    "execution_time": total_time
                }
            )
            
            logger.info(f"Multi-hop retrieval completed: {len(relevant_results)} chunks in {total_time:.3f}s")
            return state
            
        except Exception as e:
            logger.error(f"Multi-hop retrieval failed: {e}")
            raise RetrievalError(
                f"Multi-hop retrieval failed: {e}",
                "retrieval_agent",
                {"query": user_query, "sub_queries": sub_queries}
            )
    
    def _generate_sub_queries(self, query: str, query_analysis) -> List[str]:
        """Generate sub-queries for multi-hop retrieval."""
        if not query_analysis or not query_analysis.entities:
            # Simple fallback: split query by common conjunctions
            sub_queries = []
            conjunctions = ["and", "or", "but", "because", "since", "while"]
            
            for conj in conjunctions:
                if conj in query.lower():
                    parts = query.split(conj)
                    sub_queries.extend([part.strip() for part in parts if part.strip()])
                    break
            
            if not sub_queries:
                sub_queries = [query]
            
            return sub_queries[:3]  # Limit to 3 hops maximum
        
        # Use entities to generate sub-queries
        entities = query_analysis.entities
        if len(entities) >= 2:
            return [f"{query} related to {entity}" for entity in entities[:2]]
        else:
            return [query]

# Global agent instances
simple_retrieval_agent = SimpleRetrievalAgent()
hybrid_retrieval_agent = HybridRetrievalAgent()
parallel_retrieval_agent = ParallelRetrievalAgent()
multi_hop_retrieval_agent = MultiHopRetrievalAgent()