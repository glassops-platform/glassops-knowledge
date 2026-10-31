# rag/query_engine.py
"""
Answers a question from the index: embed the query, retrieve chunks, generate an answer.

Model ids come from config. Every failure raises QueryError so the CLI exits non-zero.
"""

from typing import Dict, Optional

from knowledge.settings import (
    PACKAGE_ROOT,
    collection_name,
    embedding_settings,
    generation_settings,
    load_config,
    persist_dir,
)

DEFAULT_SYSTEM_CONTEXT = "You are an expert for the GlassOps platform."


class QueryError(Exception):
    """The query could not be answered."""


def _open_collection(cfg: Dict, client=None):
    name = collection_name(cfg)
    if client is None:
        path = persist_dir(cfg)
        if not path.exists():
            raise QueryError(f"index not found at {path}. Build it with --index first.")
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    try:
        return client.get_collection(name=name)
    except Exception as e:
        raise QueryError(f'collection "{name}" not found in the index. Build it with --index first.') from e


def _inject_triggers(cfg: Dict, query: str, chunks, sources) -> None:
    for keyword, rel_path in cfg.get("retrieval_triggers", {}).items():
        if keyword.lower() not in query.lower():
            continue
        abs_path = PACKAGE_ROOT / rel_path
        if abs_path.exists() and str(abs_path) not in sources:
            content = abs_path.read_text(encoding="utf-8")
            chunks.insert(0, f"--- START SYSTEM REPORT ({rel_path}) ---\n{content}\n--- END SYSTEM REPORT ---\n")
            sources.insert(0, str(abs_path))
            print(f"[INFO] Trigger '{keyword}' detected. Injected {rel_path}.")


def query_index(query: str, n_results: int = 5, cfg: Optional[Dict] = None,
                embedder=None, llm=None, chroma_client=None) -> str:
    """Returns the answer text followed by its sources. Raises QueryError on failure."""
    from knowledge.embeddings import TASK_QUERY, EmbeddingError, GeminiEmbedding
    from knowledge.llm import GenerationError, LLMClient

    cfg = cfg if cfg is not None else load_config()
    emb_settings = embedding_settings(cfg)
    gen_settings = generation_settings(cfg)

    collection = _open_collection(cfg, chroma_client)
    meta = collection.metadata or {}
    if meta.get("embedding_model") != emb_settings.model or meta.get("dimensions") != emb_settings.dimensions:
        raise QueryError(
            f"the index was built with {meta.get('embedding_model', 'an unrecorded model')}, "
            f"config names {emb_settings.model} at {emb_settings.dimensions} dimensions. "
            "Rebuild with --index --fresh."
        )

    try:
        embedder = embedder or GeminiEmbedding(emb_settings)
        query_vector = embedder.get_embeddings([query], task_type=TASK_QUERY)[0]
    except EmbeddingError as e:
        raise QueryError(f"could not embed the query: {e}") from e

    results = collection.query(query_embeddings=[query_vector], n_results=n_results)
    chunks = list(results["documents"][0])
    sources = list(results["ids"][0])
    _inject_triggers(cfg, query, chunks, sources)

    if not chunks:
        raise QueryError("no relevant chunks found in the index")

    system_context = cfg.get("system_context") or DEFAULT_SYSTEM_CONTEXT
    context_text = "\n\n---\n\n".join(chunks)
    # Gemma models on the Gemini API take no system instruction, so it goes in the prompt.
    prompt = f"""{system_context}

Answer the user's question based strictly on the provided context.
If the answer is not in the context, say you don't know.

Context:
{context_text}

Question:
{query}

Answer:"""

    try:
        llm = llm or LLMClient(model=gen_settings.query_model, settings=gen_settings)
        answer = llm.generate(prompt)
    except GenerationError as e:
        raise QueryError(f"could not generate an answer: {e}") from e
    return f"{answer}\n\nSources:\n- " + "\n- ".join(sources)
