# knowledge/embeddings/__init__.py
# Expose embedding APIs

from .gemini_embedding import GeminiEmbedding, EmbeddingError, TASK_DOCUMENT, TASK_QUERY
from .router_embedding import get_embeddings_for_docs

__all__ = [
    "GeminiEmbedding",
    "EmbeddingError",
    "TASK_DOCUMENT",
    "TASK_QUERY",
    "get_embeddings_for_docs",
]
