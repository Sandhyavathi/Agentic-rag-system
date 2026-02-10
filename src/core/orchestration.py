"""RAG Orchestration for the Agentic RAG System."""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from ..core.config import get_config
from ..core.state_management import create_initial_state, get_state_summary, AgentState
from ..core.error_handling import error_handler, GenerationError
from ..agents.query_analyzer import query_analyzer_agent
from ..agents.routing_agent import routing_agent
from ..agents.retrieval_agents import (
    simple_retrieval_agent,
    hybrid_retrieval_agent,
    parallel_retrieval_agent,
    multi_hop_retrieval_agent
)
from ..agents.grading_agent import grading_agent
from ..agents.generation_agent import generation_agent
from ..agents.validation_agent import validation_agent
from ..llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """Main orchestrator for the RAG system without LangGraph dependency."""
    
    def __init__(self, llm_provider: BaseLLM, document_pipeline=None):
        self.llm_provider = llm_provider
        self.document_pipeline = document_pipeline
        self.config = get_config()
    
    @error_handler("orchestrator")
    def query(self, user_query: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user query through the agentic RAG pipeline."""
        try:
            # Create initial state
            state = create_initial_state(user_query, conversation_history or [])
            
            # Step 1: Analyze query
            logger.info("Step 1: Analyzing query")
            state = query_analyzer_agent.analyze_query(state)
            
            # Step 2: Route to appropriate retrieval strategy
            logger.info("Step 2: Routing query")
            state = routing_agent.route_query(state)
            
            # Step 3: Retrieve relevant chunks
            logger.info("Step 3: Retrieving chunks")
            state = self._execute_retrieval(state)
            
            # Step 4: Grade chunks for relevance
            logger.info("Step 4: Grading chunks")
            state = grading_agent.grade_chunks(state)
            
            # Step 5: Generate answer
            logger.info("Step 5: Generating answer")
            state = generation_agent.generate_answer(state)
            
            # Step 6: Validate answer
            logger.info("Step 6: Validating answer")
            state = validation_agent.validate_answer(state)
            
            # Format response
            response = self._format_response(state)
            
            logger.info("RAG orchestration completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"RAG orchestration failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your query. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _execute_retrieval(self, state: AgentState) -> AgentState:
        """Execute the appropriate retrieval strategy."""
        routing_decision = state["routing_decision"]
        
        if routing_decision == "simple":
            return simple_retrieval_agent.retrieve(state)
        elif routing_decision == "hybrid":
            return hybrid_retrieval_agent.retrieve(state)
        elif routing_decision == "parallel":
            return parallel_retrieval_agent.retrieve(state)
        elif routing_decision == "multi_hop":
            return multi_hop_retrieval_agent.retrieve(state)
        else:
            # Default to hybrid
            return hybrid_retrieval_agent.retrieve(state)
    
    def _format_response(self, state: AgentState) -> Dict[str, Any]:
        """Format the final response."""
        response = {
            "response": state["generated_answer"],
            "sources": [
                {
                    "source_file": citation.source,
                    "page": citation.page,
                    "section": citation.section
                }
                for citation in state["citations"]
            ],
            "confidence": self._calculate_confidence_score(state),
            "query_type": state["query_analysis"].query_type.value if state["query_analysis"] else "unknown",
            "retrieval_method": state["retrieval_metadata"]["method"].value if state["retrieval_metadata"] else "unknown",
            "chunks_used": len(state["graded_chunks"])
        }
        
        # Add validation info if available
        if state["validation_result"]:
            response["validation"] = {
                "passes_checks": state["validation_result"].passes_checks,
                "issues": state["validation_result"].issues
            }
        
        return response
    
    def _calculate_confidence_score(self, state: AgentState) -> float:
        """Calculate overall confidence score."""
        confidence_map = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.4
        }
        
        base_confidence = confidence_map.get(state["confidence"], 0.5)
        
        # Adjust based on validation
        if state["validation_result"]:
            if not state["validation_result"].passes_checks:
                base_confidence *= 0.7
        
        # Adjust based on number of relevant chunks
        chunk_count = len(state["graded_chunks"])
        if chunk_count >= 3:
            base_confidence = min(base_confidence * 1.1, 1.0)
        elif chunk_count == 0:
            base_confidence = 0.1
        
        return round(base_confidence, 2)

# Global orchestrator instance (will be initialized by the API)
rag_orchestrator = None

def create_rag_orchestrator(llm_provider: BaseLLM, document_pipeline=None) -> RAGOrchestrator:
    """Create and return a RAG orchestrator instance."""
    global rag_orchestrator
    rag_orchestrator = RAGOrchestrator(llm_provider, document_pipeline)
    return rag_orchestrator