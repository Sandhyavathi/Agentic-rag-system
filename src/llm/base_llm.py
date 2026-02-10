"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured output according to schema."""
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[LLMResponse]:
        """Stream text generation."""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        pass