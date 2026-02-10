"""Grading agent for the Agentic RAG System."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from langchain_core.prompts import PromptTemplate

from ..core.state_management import AgentState, GradedChunk, ConfidenceLevel, AgentName, Chunk
from ..core.error_handling import error_handler, GenerationError
from ..llm.ollama_provider import OllamaProvider
from ..core.config import config

logger = logging.getLogger(__name__)

class GradingAgent:
    """Agent that grades retrieved chunks for relevance."""
    
    def __init__(self):
        self.llm = self._get_llm()
        
        self.prompt_template = PromptTemplate(
            template="""
You are an expert relevance grader. Grade the relevance of each retrieved chunk to the user's query.

Query: {query}

Conversation History:
{conversation_history}

Instructions:
1. For each chunk, assess its relevance to the query on a scale of 0.0 to 1.0
2. Provide a brief reasoning for your score
3. Focus on how well the chunk addresses the specific query
4. Consider the context from conversation history

Scoring Guidelines:
- 0.9-1.0: Highly relevant, directly answers the query
- 0.7-0.8: Very relevant, contains key information
- 0.5-0.6: Moderately relevant, contains some useful information
- 0.3-0.4: Slightly relevant, tangential information
- 0.0-0.2: Not relevant, unrelated content

Chunks to grade:
{chunks}

Please provide your grading in the following format for each chunk:
Chunk [number]: score: [0.0-1.0], reasoning: [your reasoning]

Example:
Chunk 1: score: 0.8, reasoning: Contains key information about the topic
Chunk 2: score: 0.3, reasoning: Only tangential information

Grading:
""",
            input_variables=["query", "conversation_history", "chunks"]
        )
    
    def _get_llm(self):
        """Get the configured LLM."""
        return OllamaProvider()
    
    @error_handler("grading_agent")
    def grade_chunks(self, state: AgentState) -> AgentState:
        """Grade retrieved chunks and update state."""
        user_query = state["user_query"]
        conversation_history = state["conversation_history"]
        retrieved_chunks = state["retrieved_chunks"]
        
        if not retrieved_chunks:
            logger.warning("No chunks to grade")
            state["graded_chunks"] = []
            state["relevance_scores"] = []
            state["confidence"] = ConfidenceLevel.LOW
            return state
        
        logger.info(f"Grading {len(retrieved_chunks)} chunks for query: {user_query}")
        
        try:
            # Format conversation history
            history_text = self._format_conversation_history(conversation_history)
            
            # Format chunks for grading
            chunks_text = self._format_chunks_for_grading(retrieved_chunks)
            
            # Create prompt
            prompt = self.prompt_template.format(
                query=user_query,
                conversation_history=history_text,
                chunks=chunks_text
            )
            
            # Get LLM response
            response = self.llm.generate(prompt)
            response_text = response.content
            
            # Parse response
            try:
                graded_chunks = self._parse_grading_response(response_text, retrieved_chunks)
            except OutputParserException as e:
                # Fallback grading if structured output fails
                graded_chunks = self._fallback_grade_chunks(user_query, retrieved_chunks)
            
            # Calculate relevance scores and confidence
            relevance_scores = [chunk.score for chunk in graded_chunks]
            confidence = self._calculate_confidence(graded_chunks)
            
            # Update state
            state["graded_chunks"] = graded_chunks
            state["relevance_scores"] = relevance_scores
            state["confidence"] = confidence
            
            # Add agent trace
            from ..core.state_management import add_agent_trace
            state = add_agent_trace(
                state=state,
                agent=AgentName.GRADING_AGENT,
                decision=f"Graded {len(graded_chunks)} chunks, confidence: {confidence}",
                reasoning=f"Average relevance score: {sum(relevance_scores)/len(relevance_scores):.2f}",
                input_state={
                    "query": user_query,
                    "chunk_count": len(retrieved_chunks)
                },
                output_state={
                    "graded_chunks_count": len(graded_chunks),
                    "average_score": sum(relevance_scores)/len(relevance_scores) if relevance_scores else 0,
                    "confidence": confidence
                }
            )
            
            logger.info(f"Grading completed: {len(graded_chunks)} chunks, confidence: {confidence}")
            return state
            
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            raise GenerationError(
                f"Failed to grade chunks: {e}",
                "grading_agent",
                {"query": user_query, "chunk_count": len(retrieved_chunks)}
            )
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "No previous conversation."
        
        formatted_history = []
        for message in history[-3:]:  # Use last 3 messages
            role = message.get("role", "unknown")
            content = message.get("content", "")
            formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)
    
    def _format_chunks_for_grading(self, chunks: List[Any]) -> str:
        """Format chunks for the grading prompt."""
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted_chunk = f"Chunk {i+1}:\n"
            formatted_chunk += f"Content: {chunk.text[:500]}{'...' if len(chunk.text) > 500 else ''}\n"
            formatted_chunk += f"Metadata: {chunk.metadata}\n"
            formatted_chunk += "---\n"
            formatted_chunks.append(formatted_chunk)
        
        return "\n".join(formatted_chunks)
    
    # In grading_agent.py - update the _parse_grading_response method

    def _parse_grading_response(self, response_text: str, original_chunks: List[Any]) -> List[GradedChunk]:
        """Parse the LLM response into graded chunks with robust parsing."""
        graded_chunks = []
        
        try:
            # Split response by lines and look for scoring patterns
            lines = response_text.strip().split('\n')
            
            current_chunk_index = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for various score patterns
                score_patterns = [
                    r'chunk\s*(\d+).*?score[:\s]*(\d*\.?\d+)',
                    r'(\d+)[:\.\)\s]+.*?score[:\s]*(\d*\.?\d+)',
                    r'score[:\s]*(\d*\.?\d+)',
                    r'(\d*\.?\d+)/1\.0',
                    r'(\d*\.?\d+)\s*out\s*of\s*1'
                ]
                
                import re
                score_found = False
                
                for pattern in score_patterns:
                    match = re.search(pattern, line.lower())
                    if match:
                        try:
                            # Extract score (might be in different groups)
                            groups = match.groups()
                            if len(groups) >= 2:
                                score = float(groups[-1])  # Last group is usually the score
                            else:
                                score = float(groups[0])
                            
                            # Ensure score is in valid range
                            score = max(0.0, min(1.0, score))
                            
                            # Create graded chunk if we have a valid index
                            if current_chunk_index < len(original_chunks):
                                graded_chunk = GradedChunk(
                                    chunk=original_chunks[current_chunk_index],
                                    score=score,
                                    reasoning=f"LLM graded with score {score:.2f}"
                                )
                                graded_chunks.append(graded_chunk)
                                current_chunk_index += 1
                                score_found = True
                                break
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Failed to parse score from line '{line}': {e}")
                            continue
                
                # If no score pattern found but this looks like a chunk line, assign default score
                if not score_found and any(word in line.lower() for word in ['chunk', 'relevant', 'score']):
                    if current_chunk_index < len(original_chunks):
                        # Assign a medium score as fallback
                        graded_chunk = GradedChunk(
                            chunk=original_chunks[current_chunk_index],
                            score=0.5,
                            reasoning="Default score assigned due to parsing issues"
                        )
                        graded_chunks.append(graded_chunk)
                        current_chunk_index += 1
            
            # If we didn't get enough graded chunks, fill the rest with fallback scoring
            while len(graded_chunks) < len(original_chunks):
                remaining_chunk = original_chunks[len(graded_chunks)]
                fallback_score = self._calculate_fallback_score(remaining_chunk, "")
                
                graded_chunk = GradedChunk(
                    chunk=remaining_chunk,
                    score=fallback_score,
                    reasoning="Fallback scoring due to parsing failure"
                )
                graded_chunks.append(graded_chunk)
            
            logger.info(f"Successfully parsed {len(graded_chunks)} graded chunks")
            return graded_chunks
            
        except Exception as e:
            logger.warning(f"Grading response parsing failed: {e}")
            return self._fallback_grade_chunks("", original_chunks)

    def _calculate_fallback_score(self, chunk: Any, query: str) -> float:
        """Calculate fallback score for a chunk."""
        # Use the vector similarity score if available
        if hasattr(chunk, 'metadata') and 'similarity_score' in chunk.metadata:
            return chunk.metadata['similarity_score']
        
        # Simple keyword matching fallback
        if query:
            query_words = set(query.lower().split())
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            score = min(overlap / max(len(query_words), 1) * 2, 1.0)  # Boost the score
        else:
            score = 0.6  # Default reasonable score
        
        # Boost score for structured content
        if 'chunk_type' in chunk.metadata:
            chunk_type = chunk.metadata['chunk_type']
            if chunk_type in ['table', 'figure', 'metadata']:
                score = min(score + 0.2, 1.0)
        
        return score
    
    def _fallback_grade_chunks(self, query: str, chunks: List[Any]) -> List[GradedChunk]:
        """Fallback grading when structured parsing fails."""
        logger.warning("Using fallback grading")
        
        graded_chunks = []
        for i, chunk in enumerate(chunks):
            # Simple keyword matching for fallback
            query_words = set(query.lower().split())
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            score = min(overlap / max(len(query_words), 1), 1.0)
            
            # Boost score if metadata contains relevant information
            if any(key in chunk.metadata for key in ["section", "table", "figure"]):
                score = min(score + 0.2, 1.0)
            
            graded_chunk = GradedChunk(
                chunk=chunk,
                score=score,
                reasoning=f"Fallback scoring based on keyword overlap"
            )
            graded_chunks.append(graded_chunk)
        
        return graded_chunks
    
    def _calculate_confidence(self, graded_chunks: List[GradedChunk]) -> ConfidenceLevel:
        """Calculate overall confidence based on graded chunks."""
        if not graded_chunks:
            return ConfidenceLevel.LOW
        
        scores = [chunk.score for chunk in graded_chunks]
        avg_score = sum(scores) / len(scores)
        high_quality_count = sum(1 for score in scores if score >= 0.7)
        
        if avg_score >= 0.8 and high_quality_count >= 3:
            return ConfidenceLevel.HIGH
        elif avg_score >= 0.6 and high_quality_count >= 2:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

# Global agent instance
grading_agent = GradingAgent()