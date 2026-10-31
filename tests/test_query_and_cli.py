import json

import chromadb
import pytest
from chromadb.config import Settings
from google.genai import errors

import knowledge.main as cli
from knowledge.embeddings import GeminiEmbedding
from knowledge.ingestion import build_manifest, build_or_update_index, discover_and_chunk_docs, write_manifest
from knowledge.llm import LLMClient
from knowledge.rag import QueryError, query_index
from knowledge.settings import embedding_settings, generation_settings, persist_dir


def built_index(cfg, docs_root, tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "q"), settings=Settings(anonymized_telemetry=False))
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    build_or_update_index([(d, [0.5] * 8) for d in docs], cfg=cfg, client=client)
    return client


# Query engine


def test_query_missing_index_raises(cfg):
    with pytest.raises(QueryError, match="index not found"):
        query_index("q", cfg=cfg)


def test_query_index_from_another_model_raises(cfg, tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "q"), settings=Settings(anonymized_telemetry=False))
    client.get_or_create_collection("test_collection")
    with pytest.raises(QueryError, match="Rebuild"):
        query_index("q", cfg=cfg, chroma_client=client)


def test_query_uses_query_model_and_query_task(cfg, fakes, docs_root, tmp_path):
    client = built_index(cfg, docs_root, tmp_path)
    em = fakes.FakeModels(embed=[fakes.echo_embedder(8)])
    gm = fakes.FakeModels(generate=[fakes.text_response("The answer.")])
    embedder = GeminiEmbedding(embedding_settings(cfg), client=fakes.client(em))
    llm = LLMClient(model="test-query-model", settings=generation_settings(cfg), client=fakes.client(gm))
    out = query_index("what is A", n_results=2, cfg=cfg, embedder=embedder, llm=llm, chroma_client=client)
    assert out.startswith("The answer.")
    assert "Sources:" in out
    assert em.embed_calls[0]["config"].task_type == "RETRIEVAL_QUERY"
    assert gm.generate_calls[0]["model"] == "test-query-model"


def test_query_generation_failure_raises(cfg, fakes, docs_root, tmp_path):
    client = built_index(cfg, docs_root, tmp_path)
    err = errors.ClientError(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
    embedder = GeminiEmbedding(embedding_settings(cfg), client=fakes.client(fakes.FakeModels(embed=[fakes.echo_embedder(8)])))
    llm = LLMClient(settings=generation_settings(cfg), client=fakes.client(fakes.FakeModels(generate=[err])))
    with pytest.raises(QueryError, match="could not generate"):
        query_index("q", cfg=cfg, embedder=embedder, llm=llm, chroma_client=client)


def test_query_embedding_failure_raises(cfg, fakes, docs_root, tmp_path):
    client = built_index(cfg, docs_root, tmp_path)
    err = errors.ClientError(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
    embedder = GeminiEmbedding(embedding_settings(cfg), client=fakes.client(fakes.FakeModels(embed=[err])))
    with pytest.raises(QueryError, match="could not embed"):
        query_index("q", cfg=cfg, embedder=embedder, llm=object(), chroma_client=client)


# CLI exit codes


@pytest.fixture
def use_cfg(cfg, monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    return cfg


def test_check_index_passes_on_match(use_cfg, docs_root):
    s = embedding_settings(use_cfg)
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    write_manifest(cli.manifest_path(use_cfg), build_manifest(docs, s.model, s.dimensions))
    assert cli.run_pipeline(["--check-index", "--root", str(docs_root)]) == 0


def test_check_index_fails_on_stale_docs(use_cfg, docs_root):
    s = embedding_settings(use_cfg)
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    write_manifest(cli.manifest_path(use_cfg), build_manifest(docs, s.model, s.dimensions))
    (docs_root / "docs" / "a.md").write_text("# changed\n", encoding="utf-8")
    assert cli.run_pipeline(["--check-index", "--root", str(docs_root)]) == 1


def test_check_index_fails_without_manifest(use_cfg, docs_root):
    assert cli.run_pipeline(["--check-index", "--root", str(docs_root)]) == 1


def test_check_index_needs_no_key_or_client(use_cfg, docs_root, monkeypatch):
    # The autouse guard makes any google.genai.Client construction fail the test.
    s = embedding_settings(use_cfg)
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    write_manifest(cli.manifest_path(use_cfg), build_manifest(docs, s.model, s.dimensions))
    monkeypatch.setenv("GOOGLE_API_KEY", "would-be-real")
    assert cli.run_pipeline(["--check-index", "--root", str(docs_root)]) == 0


def test_index_without_key_exits_non_zero_and_writes_nothing(use_cfg, docs_root):
    assert cli.run_pipeline(["--index", "--root", str(docs_root)]) == 1
    assert not cli.manifest_path(use_cfg).exists()
    assert not persist_dir(use_cfg).exists()


def test_index_success_writes_manifest_that_checks_clean(use_cfg, docs_root, fakes, monkeypatch):
    s = embedding_settings(use_cfg)
    embedder = GeminiEmbedding(s, client=fakes.client(fakes.FakeModels(embed=[fakes.echo_embedder(8)] * 5)))
    import knowledge.embeddings.router_embedding as router
    real = router.get_embeddings_for_docs
    monkeypatch.setattr(router, "get_embeddings_for_docs",
                        lambda docs, batch_size=None: real(docs, batch_size, embedder=embedder))
    import sys
    monkeypatch.setattr(sys.modules["knowledge.drift.detect_drift"], "detect_drift", lambda *a, **k: [])
    assert cli.run_pipeline(["--index", "--root", str(docs_root)]) == 0
    manifest = json.loads(cli.manifest_path(use_cfg).read_text(encoding="utf-8"))
    assert manifest["embedding_model"] == "test-embed" and manifest["chunk_count"] == 4
    assert cli.run_pipeline(["--check-index", "--root", str(docs_root)]) == 0


def test_query_failure_exits_non_zero(use_cfg):
    assert cli.run_pipeline(["--query", "anything"]) == 1


def test_generate_failure_exits_non_zero(use_cfg, tmp_path):
    root = tmp_path / "repo"
    (root / "packages" / "p").mkdir(parents=True)
    (root / "packages" / "p" / "x.py").write_text("x = 1\n", encoding="utf-8")
    assert cli.run_pipeline(["--generate", "--root", str(root), "-p", "packages/**/*.py"]) == 1


def test_bad_config_exits_non_zero(monkeypatch):
    def broken():
        raise cli.ConfigError("bad config")
    monkeypatch.setattr(cli, "load_config", broken)
    assert cli.run_pipeline(["--check-index"]) == 1


def test_index_dry_run_makes_no_call_and_writes_nothing(use_cfg, docs_root, capsys):
    assert cli.run_pipeline(["--index", "--dry-run", "--root", str(docs_root)]) == 0
    assert "4 chunks, 2 embedding requests" in capsys.readouterr().out
    assert not cli.manifest_path(use_cfg).exists()
