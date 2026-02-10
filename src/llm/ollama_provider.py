"""Ollama LLM provider implementation."""

import os
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
import ollama

from src.core.config import get_config
from src.core.error_handling import handle_llm_errors
from src.llm.base_llm import BaseLLM, LLMResponse


@dataclass
class OllamaConfig:
    """Ollama-specific configuration."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.1
    max_tokens: int = 2000
    top_k: int = 5
    top_p: float = 0.9


class OllamaProvider(BaseLLM):
    """Ollama LLM provider."""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize Ollama provider."""
        self.config = config or self._load_config()
        self._setup_client()
    
    def _load_config(self) -> OllamaConfig:
        """Load configuration from environment or config file."""
        config = get_config()
        return OllamaConfig(
            base_url=config.llm.ollama_base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            top_k=config.llm.top_k,
            top_p=0.9  # Default value
        )
    
    def _setup_client(self):
        """Setup Ollama client."""
        # Ollama client is stateless, just validate connection
        try:
            # Test connection
            ollama.list()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.config.base_url}: {str(e)}")
    
    @handle_llm_errors
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from prompt."""
        try:
            response = ollama.generate(
                model=self.config.model,
                prompt=prompt,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": self.config.max_tokens
                }
            )
            
            return LLMResponse(
                content=response["response"],
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
                },
                model=self.config.model
            )
            
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")
    
    @handle_llm_errors
    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured output according to schema."""
        # Create a prompt that asks for JSON output
        json_schema = str(schema)  # Convert dict to string representation
        structured_prompt = f"""{prompt}

Please provide your response as valid JSON according to the following schema:
{json_schema}

Return ONLY the JSON object, nothing else."""
        
        response = self.generate(structured_prompt)
        
        try:
            import json
            # Try to parse the response as JSON
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse structured response as JSON: {str(e)}")
    
    @handle_llm_errors
    def stream(self, prompt: str, **kwargs) -> Iterator[LLMResponse]:
        """Stream text generation."""
        try:
            response_stream = ollama.generate(
                model=self.config.model,
                prompt=prompt,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": self.config.max_tokens
                },
                stream=True
            )
            
            for chunk in response_stream:
                yield LLMResponse(
                    content=chunk["response"],
                    usage={
                        "prompt_tokens": chunk.get("prompt_eval_count", 0),
                        "completion_tokens": chunk.get("eval_count", 0),
                        "total_tokens": chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0)
                    },
                    model=self.config.model
                )
                        
        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {str(e)}")
    
    @handle_llm_errors
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts using Ollama's embedding model."""
        try:
            results = []
            for text in texts:
                # Preprocess text
                text = text.replace("\n", " ")
                
                # Get embedding
                response = ollama.embeddings(
                    model=self.config.model,
                    prompt=text
                )
                
                results.append(response["embedding"])
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Ollama embedding generation failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "provider": "ollama",
            "model": self.config.model,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            models = ollama.list()
            return models["models"]
        except Exception as e:
            raise RuntimeError(f"Failed to list Ollama models: {str(e)}")
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            ollama.pull(model_name)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to pull model {model_name}: {str(e)}")