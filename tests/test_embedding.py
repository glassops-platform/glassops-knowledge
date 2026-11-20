import pytest
from google.genai import errors

from knowledge.embeddings import EmbeddingError, GeminiEmbedding, TASK_QUERY, get_embeddings_for_docs
from knowledge.settings import embedding_settings


def make(cfg, fakes, script, **kwargs):
    models = fakes.FakeModels(embed=script)
    emb = GeminiEmbedding(embedding_settings(cfg), client=fakes.client(models),
                          sleep=lambda s: None, **kwargs)
    return emb, models


def test_missing_api_key_raises(cfg):
    with pytest.raises(EmbeddingError, match="GOOGLE_API_KEY"):
        GeminiEmbedding(embedding_settings(cfg))


def test_router_with_no_key_raises_instead_of_fallback_vectors(cfg, monkeypatch):
    monkeypatch.setattr("knowledge.embeddings.router_embedding.load_config", lambda: cfg)
    with pytest.raises(EmbeddingError):
        get_embeddings_for_docs([{"content": "x"}])


def test_success_uses_config_model_and_dimensions(cfg, fakes):
    emb, models = make(cfg, fakes, [fakes.echo_embedder(8)])
    vectors = emb.get_embeddings(["a", "bb"])
    assert len(vectors) == 2 and all(len(v) == 8 for v in vectors)
    call = models.embed_calls[0]
    assert call["model"] == "test-embed"
    assert call["config"].output_dimensionality == 8
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_query_task_type_passed(cfg, fakes):
    emb, models = make(cfg, fakes, [fakes.echo_embedder(8)])
    emb.get_embeddings(["q"], task_type=TASK_QUERY)
    assert models.embed_calls[0]["config"].task_type == "RETRIEVAL_QUERY"


def test_not_found_model_raises_without_retry(cfg, fakes):
    err = errors.ClientError(404, {"error": {"code": 404, "message": "model not found", "status": "NOT_FOUND"}})
    emb, models = make(cfg, fakes, [err])
    with pytest.raises(EmbeddingError, match="failed"):
        emb.get_embeddings(["a"])
    assert len(models.embed_calls) == 1


def test_rate_limit_retries_then_succeeds(cfg, fakes):
    err = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    emb, models = make(cfg, fakes, [err, fakes.echo_embedder(8)])
    assert len(emb.get_embeddings(["a"])) == 1
    assert len(models.embed_calls) == 2


def test_rate_limit_exhausts_retries_and_raises(cfg, fakes):
    err = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    emb, models = make(cfg, fakes, [err, err, err], max_retries=2)
    with pytest.raises(EmbeddingError):
        emb.get_embeddings(["a"])
    assert len(models.embed_calls) == 3


def test_count_mismatch_raises(cfg, fakes):
    emb, _ = make(cfg, fakes, [fakes.embed_response([[0.1] * 8])])
    with pytest.raises(EmbeddingError, match="1 embeddings for 2 inputs"):
        emb.get_embeddings(["a", "b"])


def test_wrong_dimensions_raise(cfg, fakes):
    emb, _ = make(cfg, fakes, [fakes.embed_response([[0.1] * 3072])])
    with pytest.raises(EmbeddingError, match="3072 dimensions"):
        emb.get_embeddings(["a"])


def test_router_batches_and_stops_on_first_failure(cfg, fakes):
    err = errors.ServerError(500, {"error": {"code": 500, "message": "boom", "status": "INTERNAL"}})
    emb, models = make(cfg, fakes, [fakes.echo_embedder(8), err], max_retries=0)
    docs = [{"content": c} for c in ("a", "b", "c", "d")]
    with pytest.raises(EmbeddingError):
        get_embeddings_for_docs(docs, embedder=emb)
    assert [len(c["contents"]) for c in models.embed_calls] == [2, 2]


def test_router_pairs_docs_with_vectors(cfg, fakes):
    emb, _ = make(cfg, fakes, [fakes.echo_embedder(8), fakes.echo_embedder(8)])
    docs = [{"content": c} for c in ("a", "bb", "ccc")]
    pairs = get_embeddings_for_docs(docs, embedder=emb)
    assert [d for d, _ in pairs] == docs
    assert all(len(v) == 8 for _, v in pairs)
