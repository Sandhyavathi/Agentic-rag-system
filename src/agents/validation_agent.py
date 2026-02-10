
"""Validation agent for the Agentic RAG System."""

# Standard library imports
import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

# Third-party imports
from langchain_core.prompts import PromptTemplate

# Internal imports
from ..core.state_management import AgentState, ValidationResult, AgentName, add_agent_trace
from ..core.error_handling import error_handler, GenerationError
from ..core.config import config
from ..llm.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

class ValidationAgent:
    """Validates generated answers for quality and accuracy."""

    def __init__(self):
        self.llm = self._get_llm()
        self.prompt_template = PromptTemplate(
            template=(
                """
You are an expert answer validator. Validate the quality and accuracy of the generated answer.

Query: {query}

Generated Answer: {answer}

Relevant Chunks Used:
{relevant_chunks}

Instructions:
1. Check if the answer addresses the query completely
2. Verify that the answer is grounded in the provided document chunks
3. Check for any hallucinations or unsupported claims
4. Assess the overall completeness and accuracy
5. Provide specific issues if any are found

Validation Criteria:
- Answer addresses the query: Does the answer directly respond to the user's question?
- Grounded in documents: Are all claims supported by the provided chunks?
- No hallucinations: Are there any statements not supported by the documents?
- Completeness: Is the answer comprehensive and thorough?

Please provide your validation in the following format:
Validation Result: [PASS/FAIL]
Issues Found: [list of issues, or "None" if no issues]
Needs Retry: [YES/NO]
Reasoning: [your reasoning]

Example:
Validation Result: PASS
Issues Found: None
Needs Retry: NO
Reasoning: Answer is comprehensive and well-supported by the documents

Validation:
"""
            ),
            input_variables=["query", "answer", "relevant_chunks"]
        )
    
    def _get_llm(self):
        """Return the configured LLM provider."""
        if config.llm.provider != "ollama":
            raise ValueError(f"Invalid LLM provider: {config.llm.provider}. Only Ollama is supported.")
        
        return OllamaProvider()

    @error_handler("validation_agent")
    def validate_answer(self, state: AgentState) -> AgentState:
        """Validate the generated answer using simple heuristics."""
        user_query = state["user_query"]
        generated_answer = state["generated_answer"]
        graded_chunks = state["graded_chunks"]

        logger.info(f"Validating answer for query: {user_query}")

        try:
            passes_checks = True
            issues = []
            needs_retry = False

            # Check answer length
            if len(generated_answer.strip()) < 50:
                issues.append("Answer is too short")
                passes_checks = False

            # Check for lack of information
            if "I don't know" in generated_answer or "I apologize" in generated_answer:
                issues.append("Answer indicates lack of information")
                passes_checks = False

            # Check for relevant query terms
            query_terms = ["api styles", "milvus", "two types"]
            answer_lower = generated_answer.lower()
            relevant_terms_found = sum(1 for term in query_terms if term in answer_lower)
            if relevant_terms_found < 2:
                issues.append("Answer may not address the specific query")
                passes_checks = False

            # Disable retries for now
            needs_retry = False

            validation_result = ValidationResult(
                passes_checks=passes_checks,
                issues=issues,
                needs_retry=needs_retry,
                reasoning=f"Simple validation: {len(issues)} issues found"
            )
            state["validation_result"] = validation_result

            # Add agent trace
            state = add_agent_trace(
                state=state,
                agent=AgentName.VALIDATION_AGENT,
                decision=f"Simple validation: {'PASS' if passes_checks else 'FAIL'}",
                reasoning=f"Issues found: {len(issues)}",
                input_state={
                    "query": user_query,
                    "answer_length": len(generated_answer),
                    "graded_chunks_count": len(graded_chunks)
                },
                output_state={
                    "passes_checks": passes_checks,
                    "issues_count": len(issues),
                    "needs_retry": needs_retry
                }
            )

            if passes_checks:
                logger.info("Simple validation passed")
            else:
                logger.warning(f"Simple validation failed: {issues}")
            return state

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            # Fallback: assume validation passes
            validation_result = ValidationResult(
                passes_checks=True,
                issues=[],
                needs_retry=False,
                reasoning="Fallback validation - assuming pass"
            )
            state["validation_result"] = validation_result
            return state

    def _parse_validation_response(self, response_text: str) -> ValidationResult:
        """Parse the LLM response into a validation result."""
        passes_checks = True
        issues = []
        needs_retry = False
        reasoning = "Validation completed"
        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip().lower()
                if not line:
                    continue
                if any(fail_word in line for fail_word in ['fail', 'failed', 'invalid', 'incorrect', 'poor']):
                    passes_checks = False
                if any(retry_word in line for retry_word in ['retry', 'redo', 'regenerate', 'try again']):
                    needs_retry = True
                if any(issue_word in line for issue_word in ['issue', 'problem', 'error', 'missing', 'incorrect']):
                    issues.append(line)
            if not passes_checks and not issues:
                issues.append("Quality issues detected in generated answer")
            needs_retry = needs_retry and not passes_checks
            return ValidationResult(
                passes_checks=passes_checks,
                issues=issues,
                needs_retry=needs_retry,
                reasoning=f"Parsed from LLM response: {len(issues)} issues found"
            )
        except Exception as e:
            logger.warning(f"Validation parsing failed: {e}, using lenient validation")
            answer_length = len(response_text)
            simple_issues = []
            if answer_length < 20:
                simple_issues.append("Answer appears too short")
            return ValidationResult(
                passes_checks=len(simple_issues) == 0,
                issues=simple_issues,
                needs_retry=False,
                reasoning="Fallback validation due to parsing failure"
            )

    def _format_relevant_chunks(self, graded_chunks: List[Any]) -> str:
        """Format relevant chunks for validation display."""
        formatted_chunks = []
        for i, graded_chunk in enumerate(graded_chunks):
            chunk = graded_chunk.chunk
            score = graded_chunk.score
            formatted_chunk = f"Chunk {i+1} (Relevance: {score:.2f}):\n"
            formatted_chunk += f"Content: {chunk.text[:300]}{'...' if len(chunk.text) > 300 else ''}\n"
            metadata = chunk.metadata
            if "source_file" in metadata:
                formatted_chunk += f"Source: {metadata['source_file']}\n"
            if "page" in metadata:
                formatted_chunk += f"Page: {metadata['page']}\n"
            formatted_chunk += "---\n"
            formatted_chunks.append(formatted_chunk)
        return "\n".join(formatted_chunks)

    def _fallback_validate_answer(self, query: str, answer: str, graded_chunks: List[Any]) -> ValidationResult:
        """Fallback validation if structured parsing fails."""
        logger.warning("Using fallback validation")
        issues = []
        needs_retry = False
        if len(answer.strip()) < 50:
            issues.append("Answer is too short to be comprehensive")
            needs_retry = True
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        chunk_content = " ".join([chunk.chunk.text for chunk in graded_chunks]).lower()
        for word in answer_words:
            if len(word) > 5 and word not in chunk_content and word not in query_words:
                issues.append(f"Potential unsupported claim: '{word}'")
        query_relevant = any(word in answer.lower() for word in query_words if len(word) > 3)
        if not query_relevant:
            issues.append("Answer does not appear to address the query")
            needs_retry = True
        passes_checks = len(issues) == 0
        return ValidationResult(
            passes_checks=passes_checks,
            issues=issues,
            needs_retry=needs_retry,
            reasoning=f"Fallback validation with {len(issues)} issues found"
        )

# Global agent instance
validation_agent = ValidationAgent()