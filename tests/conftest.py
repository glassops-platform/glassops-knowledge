import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Chroma must not send telemetry from a test run.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# The package code imports itself as `knowledge`. In this repository the package
# sits at the repository root, so register the root as the `knowledge` package
# when it is not already importable (for example in a monorepo checkout).
REPO_ROOT = Path(__file__).resolve().parent.parent

if importlib.util.find_spec("knowledge") is None:
    spec = importlib.util.spec_from_file_location(
        "knowledge",
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["knowledge"] = module
    spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def no_live_google_calls(monkeypatch):
    """No test may reach Google. A real client would fail the test at construction."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to build a real google.genai.Client")

    from google import genai

    monkeypatch.setattr(genai, "Client", refuse)


class FakeModels:
    """Stands in for client.models. Each call pops the next scripted result."""

    def __init__(self, embed=None, generate=None):
        self.embed_script = list(embed or [])
        self.generate_script = list(generate or [])
        self.embed_calls = []
        self.generate_calls = []

    @staticmethod
    def _next(script):
        if not script:
            raise AssertionError("unexpected extra call")
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def embed_content(self, model, contents, config=None):
        self.embed_calls.append({"model": model, "contents": contents, "config": config})
        item = self._next(self.embed_script)
        if callable(item):
            return item(model, contents, config)
        return item

    def generate_content(self, model, contents, config=None):
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        return self._next(self.generate_script)


def embed_response(vectors):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])


def echo_embedder(dims):
    """A scripted result that returns one deterministic vector per input."""

    def respond(model, contents, config):
        vectors = []
        for i, text in enumerate(contents):
            base = (len(text) % 7 + 1) / 10.0
            vectors.append([base + (j % 3) * 0.01 for j in range(dims)])
        return embed_response(vectors)

    return respond


def text_response(text, finish_reason="STOP"):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish_reason)])


@pytest.fixture
def fakes():
    return SimpleNamespace(
        FakeModels=FakeModels,
        embed_response=embed_response,
        echo_embedder=echo_embedder,
        text_response=text_response,
        client=lambda models: SimpleNamespace(models=models),
    )


@pytest.fixture
def cfg(tmp_path):
    """A full config whose index and manifest live under tmp_path. Ids here are test values."""
    return {
        "models": {
            "embedding": {"id": "test-embed", "dimensions": 8, "batch_size": 2,
                          "rpm_limit": 1000, "tpm_limit": 10_000_000},
            "generation": {"docs": "test-docs-model", "query": "test-query-model",
                           "max_output_tokens": 256, "rpm_limit": 1000, "tpm_limit": 10_000_000},
        },
        "vector_store": {"persist_dir": str(tmp_path / "index_store"), "collection": "test_collection"},
        "index": {"manifest": str(tmp_path / "manifest.json")},
        "federated_doc_paths": ["docs/"],
    }


@pytest.fixture
def docs_root(tmp_path):
    root = tmp_path / "corpus"
    (root / "docs" / "sub").mkdir(parents=True)
    # Written as bytes so the files hold LF endings on every OS.
    (root / "docs" / "a.md").write_bytes(b"# A\n\n## One\nfirst\n\n## Two\nsecond\n")
    (root / "docs" / "sub" / "b.md").write_bytes(b"---\ntitle: B\nhash: frontmatter-hash\n---\n# B\nbody\n")
    return root
