"""Query analyzer agent for the Agentic RAG System."""

import logging
import json
from typing import Dict, Any, List, Optional

from ..core.state_management import AgentState, QueryType, QueryAnalysis, AgentName
from ..core.error_handling import error_handler, RetrievalError
from ..llm.base_llm import BaseLLM
from ..llm.ollama_provider import OllamaProvider
from ..llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class QueryAnalyzerAgent:
    """Analyzes user queries to determine type and complexity using LLM."""
    
    def __init__(self):
        self.llm_provider = self._get_llm_provider()
    
    def _get_llm_provider(self) -> BaseLLM:
        """Get the configured LLM provider."""
        try:
            from ..core.config import get_config
            config = get_config()
            
            if config.llm.provider != "ollama":
                raise ValueError(f"Invalid LLM provider: {config.llm.provider}. Only Ollama is supported.")
            
            return OllamaProvider()
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise RetrievalError(f"Failed to initialize LLM provider: {e}", "query_analysis")
    
    @error_handler("query_analysis")
    def analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user query and update state with analysis."""
        user_query = state["user_query"]
        
        if not user_query or not user_query.strip():
            raise RetrievalError(
                "Empty or invalid query provided",
                "query_analysis",
                {"query": user_query}
            )
        
        try:
            # Use LLM to analyze the query
            prompt = PromptTemplates.format_query_analysis(user_query)
            
            # Convert prompt to string format for LLM
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
            
            response = self.llm_provider.generate_structured(
                prompt_text,
                schema={
                    "query_type": "string",
                    "complexity": "integer",
                    "entities": "array",
                    "requires_filters": "object",
                    "reasoning": "string"
                }
            )
            
            # Parse the response with error handling
            query_type_str = response["query_type"].lower().strip()
            
            # Handle compound query types
            if "|" in query_type_str:
                query_type_str = query_type_str.split("|")[0].strip()
            
            # Map variations to standard types
            query_type_mapping = {
                "factual": QueryType.FACTUAL,
                "fact": QueryType.FACTUAL,
                "comparison": QueryType.COMPARISON,
                "compare": QueryType.COMPARISON,
                "summary": QueryType.SUMMARY,
                "summarize": QueryType.SUMMARY,
                "multi_hop": QueryType.MULTI_HOP,
                "multi-hop": QueryType.MULTI_HOP,
                "complex": QueryType.MULTI_HOP,
                "unclear": QueryType.UNCLEAR
            }
            
            query_type = query_type_mapping.get(query_type_str, QueryType.FACTUAL)
            
            # Parse the response
            complexity = int(response["complexity"])
            entities = response["entities"]
            requires_filters = response["requires_filters"]
            reasoning = response["reasoning"]
            
            # Create analysis result
            analysis = QueryAnalysis(
                query_type=query_type,
                complexity=complexity,
                entities=entities,
                requires_filters=requires_filters,
                reasoning=reasoning
            )
            
            # Update state
            state["query_analysis"] = analysis
            
            # Add agent trace
            from ..core.state_management import add_agent_trace
            state = add_agent_trace(
                state=state,
                agent=AgentName.QUERY_ANALYZER,
                decision=f"Query type: {query_type}, Complexity: {complexity}",
                reasoning=reasoning,
                input_state={"query": user_query},
                output_state={"query_analysis": analysis}
            )
            
            logger.info(f"Query analyzed: type={query_type}, complexity={complexity}, entities={entities}")
            
            return state
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback to simple analysis
            return self._fallback_analysis(state, user_query)
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._fallback_analysis(state, user_query)
    
    def _fallback_analysis(self, state: AgentState, user_query: str) -> AgentState:
        """Fallback analysis using simple heuristics."""
        try:
            query_lower = user_query.lower().strip()
            
            # Enhanced entity extraction for technical terms
            entities = []
            
            # Look for technical terms and product names
            tech_terms = ["milvus", "api", "styles", "client", "orm", "database", "vector"]
            entities.extend([word for word in user_query.split() if word.lower() in tech_terms])
            
            # Look for capitalized words (likely proper nouns)
            entities.extend([word for word in user_query.split() if word[0].isupper() and len(word) > 2])
            
            # Remove duplicates
            entities = list(set(entities))
            
            # Enhanced query type classification
            if any(keyword in query_lower for keyword in ["what are", "types", "kinds", "styles"]):
                if any(keyword in query_lower for keyword in ["two", "2", "both"]):
                    query_type = QueryType.FACTUAL
                    complexity = 2  # Slightly higher for enumeration questions
                else:
                    query_type = QueryType.FACTUAL
                    complexity = 1
            elif any(keyword in query_lower for keyword in ["compare", "versus", "vs", "difference"]):
                query_type = QueryType.COMPARISON
                complexity = 3
            elif any(keyword in query_lower for keyword in ["summarize", "summary", "overview"]):
                query_type = QueryType.SUMMARY
                complexity = 2
            elif any(keyword in query_lower for keyword in ["how does", "why does", "relationship"]):
                query_type = QueryType.MULTI_HOP
                complexity = 4
            else:
                query_type = QueryType.FACTUAL
                complexity = 1
            
            # Enhanced filters for technical queries
            filters = {}
            if "api" in query_lower:
                filters["technical"] = True
            if any(term in query_lower for term in ["milvus", "database", "vector"]):
                filters["product_specific"] = True
            
            reasoning = f"Enhanced fallback analysis: {query_type.value} query with complexity {complexity}, entities: {entities}"
            
            analysis = QueryAnalysis(
                query_type=query_type,
                complexity=complexity,
                entities=entities,
                requires_filters=filters,
                reasoning=reasoning
            )
            
            state["query_analysis"] = analysis
            return state
            
        except Exception as e:
            logger.error(f"Enhanced fallback analysis failed: {e}")
            raise RetrievalError(f"Query analysis failed completely: {e}", "query_analysis")

# Global agent instance
query_analyzer_agent = QueryAnalyzerAgent()