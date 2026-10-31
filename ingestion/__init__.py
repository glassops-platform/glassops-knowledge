# knowledge/ingestion/__init__.py
# Expose ingestion APIs

from .federated_loader import discover_and_chunk_docs, DocumentError
from .index_builder import build_or_update_index, IndexBuildError
from .manifest import ManifestError, build_manifest, check_manifest, read_manifest, write_manifest

__all__ = [
    "discover_and_chunk_docs",
    "DocumentError",
    "build_or_update_index",
    "IndexBuildError",
    "ManifestError",
    "build_manifest",
    "check_manifest",
    "read_manifest",
    "write_manifest",
]
