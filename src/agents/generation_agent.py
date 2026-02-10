"""Generation agent for the Agentic RAG System."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..core.state_management import AgentState, Citation, AgentName
from ..core.error_handling import error_handler, GenerationError
from ..core.config import config

logger = logging.getLogger(__name__)

class GenerationAgent:
    """Agent that generates answers from graded chunks."""
    
    def __init__(self):
        self.llm_provider = self._get_llm_provider()
        
        self.prompt_template = PromptTemplate(
            template="""
You are an expert answer generator. Generate a comprehensive answer based on the provided document chunks.

Query: {query}

Conversation History:
{conversation_history}

Relevant Document Chunks:
{document_chunks}

Instructions:
1. Use ONLY the information provided in the document chunks
2. Synthesize the information into a coherent answer
3. Include citations for all information used
4. If the information is insufficient, clearly state this
5. Format the answer clearly and concisely
6. Use markdown formatting for better readability

Citation Format:
- Use [Source: filename, Page: X] format for citations
- Include page numbers when available
- Include section information when relevant

Answer:
""",
            input_variables=["query", "conversation_history", "document_chunks"]
        )
    
    def _get_llm_provider(self):
        """Get the configured LLM provider."""
        if config.llm.provider != "ollama":
            raise ValueError(f"Invalid LLM provider: {config.llm.provider}. Only Ollama is supported.")
        
        from ..llm.ollama_provider import OllamaProvider
        return OllamaProvider()
    
    @error_handler("generation_agent")
    def generate_answer(self, state: AgentState) -> AgentState:
        """Generate answer from graded chunks."""
        user_query = state["user_query"]
        conversation_history = state["conversation_history"]
        graded_chunks = state["graded_chunks"]
        confidence = state["confidence"]
        
        logger.info(f"Generating answer for query: {user_query} (confidence: {confidence})")
        
        try:
            # Check if we have sufficient information
            if not graded_chunks:
                state["generated_answer"] = (
                    "I apologize, but I don't have sufficient information to answer your query "
                    "based on the documents available. Please try asking a different question "
                    "or upload more relevant documents."
                )
                state["citations"] = []
                
                # Add agent trace
                from ..core.state_management import add_agent_trace
                state = add_agent_trace(
                    state=state,
                    agent=AgentName.GENERATION_AGENT,
                    decision="Insufficient information - no relevant chunks found",
                    reasoning="No relevant chunks found for the query",
                    input_state={
                        "query": user_query,
                        "graded_chunks_count": 0,
                        "confidence": confidence
                    },
                    output_state={
                        "answer_length": len(state["generated_answer"]),
                        "citations_count": 0
                    }
                )
                
                return state
            
            # Format document chunks for the prompt
            document_chunks_text = self._format_document_chunks(graded_chunks)
            
            # Format conversation history
            history_text = self._format_conversation_history(conversation_history)
            
            # Create prompt
            prompt = self.prompt_template.format(
                query=user_query,
                conversation_history=history_text,
                document_chunks=document_chunks_text
            )
            
            # Generate answer using our custom LLM provider
            response = self.llm_provider.generate(prompt)
            generated_answer = response.content
            
            # Extract citations from the answer
            citations = self._extract_citations(generated_answer, graded_chunks)
            
            # Update state
            state["generated_answer"] = generated_answer
            state["citations"] = citations
            
            # Add agent trace
            from ..core.state_management import add_agent_trace
            state = add_agent_trace(
                state=state,
                agent=AgentName.GENERATION_AGENT,
                decision=f"Generated answer with {len(citations)} citations",
                reasoning=f"Used {len(graded_chunks)} relevant chunks",
                input_state={
                    "query": user_query,
                    "graded_chunks_count": len(graded_chunks),
                    "confidence": confidence
                },
                output_state={
                    "answer_length": len(generated_answer),
                    "citations_count": len(citations)
                }
            )
            
            logger.info(f"Answer generated: {len(generated_answer)} characters, {len(citations)} citations")
            return state
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise GenerationError(
                f"Failed to generate answer: {e}",
                "generation_agent",
                {"query": user_query, "graded_chunks_count": len(graded_chunks)}
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
    
    def _format_document_chunks(self, graded_chunks: List[Any]) -> str:
        """Format document chunks for the generation prompt."""
        formatted_chunks = []
        for i, graded_chunk in enumerate(graded_chunks):
            chunk = graded_chunk.chunk
            score = graded_chunk.score
            
            formatted_chunk = f"Chunk {i+1} (Relevance: {score:.2f}):\n"
            formatted_chunk += f"Content: {chunk.text}\n"
            
            # Add metadata information
            metadata = chunk.metadata
            if "source_file" in metadata:
                formatted_chunk += f"Source: {metadata['source_file']}\n"
            if "page" in metadata:
                formatted_chunk += f"Page: {metadata['page']}\n"
            if "section_heading" in metadata:
                formatted_chunk += f"Section: {metadata['section_heading']}\n"
            if "chunk_type" in metadata:
                formatted_chunk += f"Type: {metadata['chunk_type']}\n"
            
            formatted_chunk += "---\n"
            formatted_chunks.append(formatted_chunk)
        
        return "\n".join(formatted_chunks)
    
    def _extract_citations(self, answer: str, graded_chunks: List[Any]) -> List[Citation]:
        """Extract citations from the generated answer."""
        citations = []
        used_sources = set()
        
        for graded_chunk in graded_chunks:
            chunk = graded_chunk.chunk
            metadata = chunk.metadata
            
            # Check if this source is referenced in the answer
            source_name = metadata.get("source_file", "Unknown")
            if source_name in answer and source_name not in used_sources:
                citation = Citation(
                    source=source_name,
                    page=metadata.get("page"),
                    section=metadata.get("section_heading")
                )
                citations.append(citation)
                used_sources.add(source_name)
        
        return citations

# Global agent instance
generation_agent = GenerationAgent()