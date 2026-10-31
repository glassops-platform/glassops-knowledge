# knowledge/rag/__init__.py
# Expose RAG query engine

from .query_engine import QueryError, query_index

__all__ = ["query_index", "QueryError"]
