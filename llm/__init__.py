# llm/__init__.py
"""LLM client module for GlassOps Knowledge Pipeline."""

from .client import GenerationError, LLMClient

__all__ = ["LLMClient", "GenerationError"]
