"""Centralized prompt templates for all LLM operations."""

from typing import Dict, Any, List


class PromptTemplates:
    """Centralized prompt templates for consistent LLM interactions."""
    
    # Query Analysis Prompts
    QUERY_ANALYSIS_SYSTEM = """You are an expert query analyzer for a Retrieval-Augmented Generation system.
Analyze the user's query and provide structured information about:
1. Query type (factual, comparison, summary, multi-hop, or unclear)
2. Complexity level (1-5, where 1 is simple fact lookup, 5 is complex reasoning)
3. Key entities mentioned
4. Required filters or constraints
5. Reasoning for your analysis

Return your response as JSON with the following structure:
{
    "query_type": "factual|comparison|summary|multi-hop|unclear",
    "complexity": 1-5,
    "entities": ["entity1", "entity2", ...],
    "requires_filters": {"filter_name": "filter_value", ...},
    "reasoning": "Your detailed reasoning here"
}"""

    QUERY_ANALYSIS_USER = """Analyze this query:
"{query}"

Consider the following:
- What type of information is being requested?
- How complex is this query?
- What entities or topics are mentioned?
- Are there any implicit constraints or filters?
- What would be the best retrieval strategy for this query?"""

    # Chunk Grading Prompts
    CHUNK_GRADING_SYSTEM = """You are an expert document chunk grader for a Retrieval-Augmented Generation system.
Evaluate the relevance and quality of a document chunk for answering a specific query.

Grade each chunk on a scale of 0.0 to 1.0 based on:
1. Relevance to the query (0.0 = completely irrelevant, 1.0 = perfectly relevant)
2. Information density (0.0 = no useful information, 1.0 = highly informative)
3. Factual vs opinion content (prefer factual information)
4. Source credibility indicators

Return your response as JSON with the following structure:
{
    "score": 0.0-1.0,
    "reasoning": "Detailed explanation for your score",
    "key_points": ["point1", "point2", ...],
    "confidence": "high|medium|low"
}"""

    CHUNK_GRADING_USER = """Query: {query}
Chunk: {chunk_text}

Evaluate this chunk's relevance and quality for answering the query above."""

    # Answer Generation Prompts
    ANSWER_GENERATION_SYSTEM = """You are an expert answer generator for a Retrieval-Augmented Generation system.
Generate a comprehensive, well-structured answer based on the provided document chunks.

Requirements:
1. Use ONLY the information provided in the chunks
2. Do not hallucinate or add external knowledge
3. Cite sources properly using chunk IDs
4. Structure your answer clearly
5. If insufficient information is available, clearly state this
6. Be concise but thorough

Return your response as JSON with the following structure:
{
    "answer": "Your comprehensive answer here",
    "citations": [{"chunk_id": "id", "reason": "why this chunk was used"}],
    "confidence": "high|medium|low",
    "insufficient_info": true|false
}"""

    ANSWER_GENERATION_USER = """Query: {query}
Conversation History: {conversation_history}
Relevant Chunks: {chunk_summaries}

Generate a comprehensive answer based on the information above."""

    # Validation Prompts
    VALIDATION_SYSTEM = """You are an expert answer validator for a Retrieval-Augmented Generation system.
Validate the quality and accuracy of a generated answer.

Check for:
1. Answer completeness (does it fully address the query?)
2. Factual accuracy (are the claims supported by the chunks?)
3. Citation accuracy (do citations match the claims?)
4. Hallucination detection (are there unsupported claims?)
5. Overall quality assessment

Return your response as JSON with the following structure:
{
    "query_addressed": true|false,
    "factual_accuracy": "high|medium|low",
    "citation_accuracy": "high|medium|low",
    "hallucinations": ["unsupported claim1", "unsupported claim2", ...],
    "overall_quality": "high|medium|low",
    "retry_needed": true|false,
    "reasoning": "Detailed explanation for your assessment"
}"""

    VALIDATION_USER = """Query: {query}
Generated Answer: {answer}
Citations: {citations}
Source Chunks: {chunk_summaries}

Validate the quality and accuracy of this answer."""

    # Query Refinement Prompts
    QUERY_REFINEMENT_SYSTEM = """You are an expert query refiner for a Retrieval-Augmented Generation system.
Refine the user's query to improve retrieval quality.

Consider:
1. Adding relevant synonyms or related terms
2. Clarifying ambiguous terms
3. Breaking complex queries into sub-queries
4. Adding context that might help retrieval

Return your response as JSON with the following structure:
{
    "refined_query": "Your refined query here",
    "synonyms": ["synonym1", "synonym2", ...],
    "sub_queries": ["sub_query1", "sub_query2", ...],
    "reasoning": "Explanation for your refinements"
}"""

    QUERY_REFINEMENT_USER = """Original Query: {query}
Previous Retrieval Results: {retrieval_results}
Validation Feedback: {validation_feedback}

Refine this query to improve retrieval quality."""

    # Multi-hop Query Decomposition
    MULTI_HOP_DECOMPOSITION_SYSTEM = """You are an expert at decomposing complex queries into simpler sub-queries for multi-hop reasoning.

Given a complex query, break it down into 2-4 simpler sub-queries that can be answered sequentially.
Each sub-query should:
1. Be answerable with document retrieval
2. Build towards answering the overall query
3. Be logically connected to the next sub-query

Return your response as JSON with the following structure:
{
    "sub_queries": [
        {"query": "sub_query_1", "purpose": "what this sub-query aims to find"},
        {"query": "sub_query_2", "purpose": "what this sub-query aims to find"},
        ...
    ],
    "reasoning": "Explanation of how these sub-queries work together"
}"""

    MULTI_HOP_DECOMPOSITION_USER = """Complex Query: {query}

Decompose this query into simpler sub-queries for multi-hop reasoning."""

    # Entity Extraction for Comparison Queries
    ENTITY_EXTRACTION_SYSTEM = """You are an expert at extracting entities for comparison queries.

Given a comparison query, identify all entities that need to be compared and the comparison criteria.

Return your response as JSON with the following structure:
{
    "entities": [
        {"name": "entity1", "type": "company|product|person|etc"},
        {"name": "entity2", "type": "company|product|person|etc"},
        ...
    ],
    "comparison_criteria": ["criterion1", "criterion2", ...],
    "reasoning": "Explanation of entity identification"
}"""

    ENTITY_EXTRACTION_USER = """Comparison Query: {query}

Extract all entities that need to be compared and the comparison criteria."""

    @classmethod
    def format_query_analysis(cls, query: str) -> List[Dict[str, str]]:
        """Format query analysis prompt."""
        return [
            {"role": "system", "content": cls.QUERY_ANALYSIS_SYSTEM},
            {"role": "user", "content": cls.QUERY_ANALYSIS_USER.format(query=query)}
        ]
    
    @classmethod
    def format_chunk_grading(cls, query: str, chunk_text: str) -> List[Dict[str, str]]:
        """Format chunk grading prompt."""
        return [
            {"role": "system", "content": cls.CHUNK_GRADING_SYSTEM},
            {"role": "user", "content": cls.CHUNK_GRADING_USER.format(query=query, chunk_text=chunk_text)}
        ]
    
    @classmethod
    def format_answer_generation(cls, query: str, conversation_history: List[Dict[str, str]], 
                               chunk_summaries: List[str]) -> List[Dict[str, str]]:
        """Format answer generation prompt."""
        return [
            {"role": "system", "content": cls.ANSWER_GENERATION_SYSTEM},
            {"role": "user", "content": cls.ANSWER_GENERATION_USER.format(
                query=query,
                conversation_history=conversation_history,
                chunk_summaries=chunk_summaries
            )}
        ]
    
    @classmethod
    def format_validation(cls, query: str, answer: str, citations: List[Dict[str, Any]], 
                         chunk_summaries: List[str]) -> List[Dict[str, str]]:
        """Format validation prompt."""
        return [
            {"role": "system", "content": cls.VALIDATION_SYSTEM},
            {"role": "user", "content": cls.VALIDATION_USER.format(
                query=query,
                answer=answer,
                citations=citations,
                chunk_summaries=chunk_summaries
            )}
        ]
    
    @classmethod
    def format_query_refinement(cls, query: str, retrieval_results: List[str], 
                              validation_feedback: str) -> List[Dict[str, str]]:
        """Format query refinement prompt."""
        return [
            {"role": "system", "content": cls.QUERY_REFINEMENT_SYSTEM},
            {"role": "user", "content": cls.QUERY_REFINEMENT_USER.format(
                query=query,
                retrieval_results=retrieval_results,
                validation_feedback=validation_feedback
            )}
        ]
    
    @classmethod
    def format_multi_hop_decomposition(cls, query: str) -> List[Dict[str, str]]:
        """Format multi-hop decomposition prompt."""
        return [
            {"role": "system", "content": cls.MULTI_HOP_DECOMPOSITION_SYSTEM},
            {"role": "user", "content": cls.MULTI_HOP_DECOMPOSITION_USER.format(query=query)}
        ]
    
    @classmethod
    def format_entity_extraction(cls, query: str) -> List[Dict[str, str]]:
        """Format entity extraction prompt."""
        return [
            {"role": "system", "content": cls.ENTITY_EXTRACTION_SYSTEM},
            {"role": "user", "content": cls.ENTITY_EXTRACTION_USER.format(query=query)}
        ]