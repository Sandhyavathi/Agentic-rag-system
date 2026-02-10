"""Routing agent for the Agentic RAG System."""

import logging
from typing import Dict, Any, Optional

from ..core.state_management import AgentState, RetrievalStrategy, AgentName, QueryType
from ..core.error_handling import error_handler, GenerationError

logger = logging.getLogger(__name__)

class RoutingAgent:
    """Agent that routes queries to appropriate retrieval strategies."""
    
    def __init__(self):
        pass
    
    @error_handler("routing_agent")
    def route_query(self, state: AgentState) -> AgentState:
        """Route the query to appropriate retrieval strategy."""
        query_analysis = state["query_analysis"]
        
        if query_analysis is None:
            raise GenerationError(
                "Cannot route query: no query analysis available",
                "routing_agent"
            )
        
        query_type = query_analysis.query_type
        complexity = query_analysis.complexity
        entities = query_analysis.entities
        
        logger.info(f"Routing query: type={query_type}, complexity={complexity}")
        
        # Decision logic for routing
        routing_decision = self._determine_routing_strategy(
            query_type, complexity, entities
        )
        
        # Update state
        state["routing_decision"] = routing_decision
        
        # Add agent trace
        from ..core.state_management import add_agent_trace
        state = add_agent_trace(
            state=state,
            agent=AgentName.ROUTING_AGENT,
            decision=f"Routing to {routing_decision} strategy",
            reasoning=f"Query type: {query_type}, Complexity: {complexity}, Entities: {len(entities)}",
            input_state={"query_analysis": query_analysis},
            output_state={"routing_decision": routing_decision}
        )
        
        logger.info(f"Query routed to {routing_decision} strategy")
        return state
    
    def _determine_routing_strategy(
        self, 
        query_type: QueryType, 
        complexity: int, 
        entities: list
    ) -> RetrievalStrategy:
        """Determine the appropriate retrieval strategy."""
        
        # Simple factual queries with low complexity
        if query_type == QueryType.FACTUAL and complexity <= 2:
            return RetrievalStrategy.SIMPLE
        
        # Comparison queries always use parallel retrieval
        elif query_type == QueryType.COMPARISON:
            return RetrievalStrategy.PARALLEL
        
        # Multi-hop queries use multi-hop retrieval
        elif query_type == QueryType.MULTI_HOP or complexity >= 4:
            return RetrievalStrategy.MULTI_HOP
        
        # Summary queries and medium complexity factual queries use hybrid
        elif query_type == QueryType.SUMMARY or complexity == 3:
            return RetrievalStrategy.HYBRID
        
        # Default to hybrid for other cases
        else:
            return RetrievalStrategy.HYBRID

# Global agent instance
routing_agent = RoutingAgent()