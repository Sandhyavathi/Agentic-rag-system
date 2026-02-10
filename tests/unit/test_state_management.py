"""Unit tests for state management module."""

import unittest
from datetime import datetime
from src.core.state_management import (
    AgentState, create_initial_state, add_agent_trace, 
    QueryType, RetrievalStrategy, ConfidenceLevel
)

class TestStateManagement(unittest.TestCase):
    """Test state management functionality."""
    
    def test_create_initial_state(self):
        """Test creating initial agent state."""
        user_query = "What is the capital of France?"
        conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        state = create_initial_state(user_query, conversation_history)
        
        self.assertEqual(state["user_query"], user_query)
        self.assertEqual(state["conversation_history"], conversation_history)
        self.assertEqual(state["iteration_count"], 0)
        self.assertEqual(state["max_iterations"], 2)
        self.assertEqual(len(state["agent_trace"]), 0)
    
    def test_add_agent_trace(self):
        """Test adding agent trace to state."""
        state = create_initial_state("Test query")
        
        trace_state = add_agent_trace(
            state=state,
            agent="query_analyzer",
            decision="factual query",
            reasoning="Query seeks specific information",
            input_state={"query": "Test query"},
            output_state={"query_type": "factual"}
        )
        
        self.assertEqual(len(trace_state["agent_trace"]), 1)
        trace = trace_state["agent_trace"][0]
        self.assertEqual(trace.agent, "query_analyzer")
        self.assertEqual(trace.decision, "factual query")
        self.assertEqual(trace.reasoning, "Query seeks specific information")
        self.assertIsInstance(trace.timestamp, datetime)
    
    def test_state_summary(self):
        """Test getting state summary."""
        from src.core.state_management import get_state_summary
        
        state = create_initial_state("Test query")
        state["query_analysis"] = type('QueryAnalysis', (), {
            'query_type': QueryType.FACTUAL
        })()
        state["routing_decision"] = RetrievalStrategy.SIMPLE
        state["retrieved_chunks"] = [1, 2, 3]  # Mock chunks
        state["confidence"] = ConfidenceLevel.HIGH
        state["iteration_count"] = 1
        state["agent_trace"] = [1, 2]  # Mock traces
        
        summary = get_state_summary(state)
        
        self.assertEqual(summary["user_query"], "Test query")
        self.assertEqual(summary["query_type"], QueryType.FACTUAL)
        self.assertEqual(summary["routing_decision"], RetrievalStrategy.SIMPLE)
        self.assertEqual(summary["num_retrieved_chunks"], 3)
        self.assertEqual(summary["confidence"], ConfidenceLevel.HIGH)
        self.assertEqual(summary["iteration_count"], 1)
        self.assertEqual(summary["agent_trace_count"], 2)

if __name__ == '__main__':
    unittest.main()