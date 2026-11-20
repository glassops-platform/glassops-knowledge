# embeddings/router_embedding.py
"""
Embeds documents in batches with the configured model.

There is one model and no fallback. The old primary and fallback modules called the same
retired model, so the fallback was never a real one. A failure raises EmbeddingError and the
run stops.
"""

from typing import Dict, List, Optional, Tuple

from knowledge.settings import embedding_settings, load_config
from .gemini_embedding import TASK_DOCUMENT, GeminiEmbedding


def get_embeddings_for_docs(
    docs: List[Dict],
    batch_size: Optional[int] = None,
    embedder: Optional[GeminiEmbedding] = None,
    task_type: str = TASK_DOCUMENT,
) -> List[Tuple[Dict, List[float]]]:
    if embedder is None:
        embedder = GeminiEmbedding(embedding_settings(load_config()))
    size = batch_size or embedder.settings.batch_size
    embeddings: List[Tuple[Dict, List[float]]] = []

    for i in range(0, len(docs), size):
        print(f"  Processed {i}/{len(docs)}...", end="\r")
        batch = docs[i:i + size]
        vectors = embedder.get_embeddings([d["content"] for d in batch], task_type=task_type)
        embeddings.extend(zip(batch, vectors))
    print(f"  Processed {len(docs)}/{len(docs)}... Done.")
    return embeddings
