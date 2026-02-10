"""State management for the Agentic RAG System."""

from typing import TypedDict, List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

class QueryType(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    MULTI_HOP = "multi_hop"
    UNCLEAR = "unclear"

class RetrievalStrategy(str, Enum):
    SIMPLE = "simple"
    HYBRID = "hybrid"
    PARALLEL = "parallel"
    MULTI_HOP = "multi_hop"

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class AgentName(str, Enum):
    QUERY_ANALYZER = "query_analyzer"
    ROUTING_AGENT = "routing_agent"
    RETRIEVAL_AGENT = "retrieval_agent"
    GRADING_AGENT = "grading_agent"
    GENERATION_AGENT = "generation_agent"
    VALIDATION_AGENT = "validation_agent"

@dataclass
class Message:
    """Chat message structure."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Chunk:
    """Document chunk structure."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    """Search result with chunk and score."""
    chunk: Chunk
    score: float  # Similarity score between 0.0 and 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class GradedChunk:
    """Graded chunk with relevance score."""
    chunk: Chunk
    score: float  # 0.0 to 1.0
    reasoning: str

@dataclass
class Citation:
    """Citation for answer generation."""
    source: str
    page: Optional[int] = None
    section: Optional[str] = None

@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    query_type: QueryType
    complexity: int  # 1-5 scale
    entities: List[str]
    requires_filters: Dict[str, Any]
    reasoning: str

@dataclass
class RetrievalMetadata:
    """Metadata about retrieval process."""
    method: RetrievalStrategy
    num_chunks: int
    search_queries: List[str]
    execution_time: float

@dataclass
class ValidationResult:
    """Result of answer validation."""
    passes_checks: bool
    issues: List[str]
    needs_retry: bool
    reasoning: str

@dataclass
class AgentTrace:
    """Trace of agent decision making."""
    agent: AgentName
    decision: str
    reasoning: str
    timestamp: datetime
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]

class AgentState(TypedDict):
    """Complete state for agents."""
    
    # Input
    user_query: str
    conversation_history: List[Message]
    
    # Query Analysis
    query_analysis: Optional[QueryAnalysis]
    
    # Routing
    routing_decision: Optional[RetrievalStrategy]
    
    # Retrieval
    retrieved_chunks: List[Chunk]
    retrieval_metadata: Optional[RetrievalMetadata]
    
    # Grading
    graded_chunks: List[GradedChunk]
    relevance_scores: List[float]
    confidence: Optional[ConfidenceLevel]
    
    # Generation
    generated_answer: str
    citations: List[Citation]
    
    # Validation
    validation_result: Optional[ValidationResult]
    
    # Loop Control
    iteration_count: int
    max_iterations: int
    
    # Agent Decisions
    agent_trace: List[AgentTrace]
    
    # Error Handling
    errors: List[str]
    retry_reasons: List[str]

def create_initial_state(user_query: str, conversation_history: List[Message] = None) -> AgentState:
    """Create initial agent state."""
    return AgentState(
        user_query=user_query,
        conversation_history=conversation_history or [],
        query_analysis=None,
        routing_decision=None,
        retrieved_chunks=[],
        retrieval_metadata=None,
        graded_chunks=[],
        relevance_scores=[],
        confidence=None,
        generated_answer="",
        citations=[],
        validation_result=None,
        iteration_count=0,
        max_iterations=2,
        agent_trace=[],
        errors=[],
        retry_reasons=[]
    )

def add_agent_trace(
    state: AgentState, 
    agent: AgentName, 
    decision: str, 
    reasoning: str,
    input_state: Dict[str, Any],
    output_state: Dict[str, Any]
) -> AgentState:
    """Add agent trace to state."""
    trace = AgentTrace(
        agent=agent,
        decision=decision,
        reasoning=reasoning,
        timestamp=datetime.now(),
        input_state=input_state,
        output_state=output_state
    )
    state["agent_trace"].append(trace)
    return state

def get_state_summary(state: AgentState) -> Dict[str, Any]:
    """Get summary of current state for debugging."""
    return {
        "user_query": state["user_query"],
        "query_type": state["query_analysis"].query_type.value if state["query_analysis"] else None,
        "routing_decision": state["routing_decision"].value if state["routing_decision"] else None,
        "num_retrieved_chunks": len(state["retrieved_chunks"]),
        "confidence": state["confidence"].value if state["confidence"] else None,
        "iteration_count": state["iteration_count"],
        "has_errors": len(state["errors"]) > 0,
        "agent_trace_count": len(state["agent_trace"])
    }

def increment_iteration(state: AgentState) -> AgentState:
    """Increment the iteration count."""
    state["iteration_count"] += 1
    return state
