import json

import chromadb
import pytest
from chromadb.config import Settings

from knowledge.ingestion import (
    DocumentError,
    IndexBuildError,
    ManifestError,
    build_manifest,
    build_or_update_index,
    check_manifest,
    discover_and_chunk_docs,
    read_manifest,
    write_manifest,
)
from knowledge.settings import embedding_settings


def pairs(docs, dims=8):
    return [(d, [0.1 * (i + 1)] * dims) for i, d in enumerate(docs)]


def chroma(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))


# Loader


def test_loader_ids_are_relative_posix(docs_root):
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    ids = [d["path"] for d in docs]
    assert ids == ["docs/a.md#chunk-0", "docs/a.md#chunk-1", "docs/a.md#chunk-2", "docs/sub/b.md#chunk-0"]


def test_frontmatter_hash_does_not_overwrite_chunk_hash(docs_root):
    docs = {d["path"]: d for d in discover_and_chunk_docs(str(docs_root), ["docs/"])}
    b = docs["docs/sub/b.md#chunk-0"]
    assert b["title"] == "B"
    assert b["hash"] != "frontmatter-hash"
    assert len(b["hash"]) == 64


def test_crlf_and_lf_give_the_same_hashes(docs_root, tmp_path):
    lf = discover_and_chunk_docs(str(docs_root), ["docs/"])
    crlf_root = tmp_path / "crlf"
    for src in (docs_root / "docs").rglob("*.md"):
        dst = crlf_root / src.relative_to(docs_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes().replace(b"\n", b"\r\n"))
    crlf = discover_and_chunk_docs(str(crlf_root), ["docs/"])
    assert [(d["path"], d["hash"]) for d in lf] == [(d["path"], d["hash"]) for d in crlf]


def test_unreadable_doc_raises(docs_root):
    (docs_root / "docs" / "bad.md").write_bytes(b"\xff\xfe\xfa not utf-8")
    with pytest.raises(DocumentError, match="bad.md"):
        discover_and_chunk_docs(str(docs_root), ["docs/"])


# Index builder


def test_upsert_error_raises(cfg, docs_root):
    class BrokenCollection:
        metadata = {}

        def upsert(self, **kwargs):
            raise RuntimeError("disk full")

    class BrokenClient:
        def get_collection(self, name):
            raise ValueError("missing")

        def get_or_create_collection(self, name, metadata):
            return BrokenCollection()

    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    with pytest.raises(IndexBuildError, match="disk full"):
        build_or_update_index(pairs(docs), cfg=cfg, client=BrokenClient())


def test_empty_input_raises(cfg, tmp_path):
    with pytest.raises(IndexBuildError, match="no documents"):
        build_or_update_index([], cfg=cfg, client=chroma(tmp_path))


def test_wrong_vector_size_raises(cfg, docs_root, tmp_path):
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    with pytest.raises(IndexBuildError, match="768-dimension"):
        build_or_update_index(pairs(docs, dims=768), cfg=cfg, client=chroma(tmp_path))


def test_build_records_model_and_removes_stale_chunks(cfg, docs_root, tmp_path):
    client = chroma(tmp_path)
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    assert build_or_update_index(pairs(docs), cfg=cfg, client=client) == 4
    col = client.get_collection("test_collection")
    assert col.metadata["embedding_model"] == "test-embed"
    assert col.metadata["dimensions"] == 8

    (docs_root / "docs" / "sub" / "b.md").unlink()
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    assert build_or_update_index(pairs(docs), cfg=cfg, client=client) == 3
    assert "docs/sub/b.md#chunk-0" not in client.get_collection("test_collection").get()["ids"]


def test_existing_index_from_another_model_is_refused(cfg, docs_root, tmp_path):
    client = chroma(tmp_path)
    client.get_or_create_collection("test_collection", metadata={"hnsw:space": "cosine"})
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    with pytest.raises(IndexBuildError, match="--fresh"):
        build_or_update_index(pairs(docs), cfg=cfg, client=client)
    assert build_or_update_index(pairs(docs), cfg=cfg, client=client, fresh=True) == 4


# Manifest


def test_manifest_matches_unchanged_docs(cfg, docs_root, tmp_path):
    s = embedding_settings(cfg)
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    path = tmp_path / "m.json"
    write_manifest(path, build_manifest(docs, s.model, s.dimensions, "abc123"))
    m = read_manifest(path)
    assert m["source_commit"] == "abc123"
    assert check_manifest(m, docs, s.model, s.dimensions) == []


def test_manifest_detects_changed_added_removed(cfg, docs_root):
    s = embedding_settings(cfg)
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    m = build_manifest(docs, s.model, s.dimensions)
    (docs_root / "docs" / "a.md").write_text("# A\n\n## One\nchanged\n", encoding="utf-8")
    (docs_root / "docs" / "new.md").write_text("# New\n", encoding="utf-8")
    problems = check_manifest(m, discover_and_chunk_docs(str(docs_root), ["docs/"]), s.model, s.dimensions)
    text = "\n".join(problems)
    assert "1 added, 1 removed, 1 changed" in text
    assert "added: docs/new.md#chunk-0" in text
    assert "removed: docs/a.md#chunk-2" in text
    assert "changed: docs/a.md#chunk-1" in text


def test_manifest_detects_model_or_dimension_change(cfg, docs_root):
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    m = build_manifest(docs, "old-model", 768)
    text = "\n".join(check_manifest(m, docs, "test-embed", 8))
    assert "old-model" in text and "768 dimensions" in text


def test_manifest_detects_hand_edit(cfg, docs_root):
    docs = discover_and_chunk_docs(str(docs_root), ["docs/"])
    m = build_manifest(docs, "test-embed", 8)
    m["chunks"][0]["hash"] = "0" * 64
    assert any("edited by hand" in p for p in check_manifest(m, docs, "test-embed", 8))


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        read_manifest(tmp_path / "none.json")


def test_malformed_manifest_raises(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ManifestError, match="missing"):
        read_manifest(p)
