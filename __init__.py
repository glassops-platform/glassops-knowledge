# knowledge/__init__.py
# Expose the main pipeline entrypoint and the submodule APIs.
#
# The repository root is this package. pytest imports a root __init__.py as a plain top-level
# module with no parent package, and relative imports fail there. The tests register the root
# as `knowledge` in tests/conftest.py, so the exports are only needed when a package is set.
if __package__:
    from .main import run_pipeline
    from .embeddings import get_embeddings_for_docs
    from .ingestion import discover_and_chunk_docs, build_or_update_index
    from .drift import detect_drift
    from .rag import query_index
    from .utils import hash_file, batch_items

    __all__ = [
        "run_pipeline",
        "get_embeddings_for_docs",
        "discover_and_chunk_docs",
        "build_or_update_index",
        "detect_drift",
        "query_index",
        "hash_file",
        "batch_items",
    ]
