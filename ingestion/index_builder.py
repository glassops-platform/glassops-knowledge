# ingestion/index_builder.py
# Builds or updates the Chroma vector store with embeddings

from typing import Dict, List, Optional, Tuple

from knowledge.settings import (
    EmbeddingSettings,
    collection_name,
    embedding_settings,
    load_config,
    persist_dir,
)


class IndexBuildError(Exception):
    """The vector store could not be written, or would mix incompatible vectors."""


def _open_client(path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def _existing_collection(client, name):
    try:
        return client.get_collection(name=name)
    except Exception:  # Chroma raises NotFoundError or ValueError depending on version
        return None


def build_or_update_index(
    embeddings: List[Tuple[Dict, List[float]]],
    fresh: bool = False,
    cfg: Optional[Dict] = None,
    settings: Optional[EmbeddingSettings] = None,
    client=None,
) -> int:
    """
    embeddings: list of tuples (doc_dict, embedding_vector)
    fresh: drop the collection first and build it from nothing
    Returns the number of chunks in the collection. Raises IndexBuildError on any failure.
    """
    cfg = cfg if cfg is not None else load_config()
    settings = settings or embedding_settings(cfg)
    name = collection_name(cfg)

    if not embeddings:
        raise IndexBuildError("no documents to index. Check the docs root and federated_doc_paths.")

    for doc, emb in embeddings:
        if len(emb) != settings.dimensions:
            raise IndexBuildError(
                f"{doc['path']} has a {len(emb)}-dimension vector, config expects {settings.dimensions}"
            )

    try:
        if client is None:
            path = persist_dir(cfg)
            print(f"[INFO] Using ChromaDB at {path}")
            client = _open_client(path)

        existing = _existing_collection(client, name)
        if existing is not None and not fresh:
            meta = existing.metadata or {}
            if meta.get("embedding_model") != settings.model or meta.get("dimensions") != settings.dimensions:
                raise IndexBuildError(
                    f"the existing index was built with {meta.get('embedding_model', 'an unrecorded model')} "
                    f"at {meta.get('dimensions', 'unrecorded')} dimensions. Config names {settings.model} at "
                    f"{settings.dimensions}. Vectors from different models cannot share an index. "
                    "Rebuild with --index --fresh."
                )
        if existing is not None and fresh:
            client.delete_collection(name=name)

        collection = client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": settings.model,
                "dimensions": settings.dimensions,
            },
        )

        ids, documents, metadatas, vectors = [], [], [], []
        for doc, emb in embeddings:
            ids.append(doc["path"])
            documents.append(doc["content"])
            meta = {"path": doc["path"], "hash": doc["hash"]}
            for k, v in doc.items():
                if k not in ("content", "path", "hash") and isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)
            vectors.append(emb)

        collection.upsert(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)

        # Drop chunks whose source changed or was removed, so the index matches the manifest.
        stored = set(collection.get(include=[])["ids"])
        stale = sorted(stored - set(ids))
        if stale:
            collection.delete(ids=stale)
            print(f"[INFO] Removed {len(stale)} stale chunks.")

        count = collection.count()
    except IndexBuildError:
        raise
    except Exception as e:
        raise IndexBuildError(f"writing the vector store failed: {e}") from e

    if count != len(ids):
        raise IndexBuildError(f"the collection holds {count} chunks after the upsert, expected {len(ids)}")
    print(f"[SUCCESS] Indexed {len(ids)} chunks in ChromaDB.")
    return count
