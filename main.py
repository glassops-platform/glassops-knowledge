# main.py
"""
GlassOps Knowledge Pipeline
- Discover documentation under a docs root
- Compute embeddings with the configured model
- Build or update the vector store and write the index manifest
- Check the manifest against the docs, with no model call
- Detect duplicate content
- RAG query
- Documentation generation from source code

Every failure exits non-zero.
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent


def _ensure_package_importable() -> None:
    """
    The modules import each other as `knowledge.*`. In this repository the package is the
    repository root, so register the root under that name when it is not importable already.
    """
    if importlib.util.find_spec("knowledge") is not None:
        return
    spec = importlib.util.spec_from_file_location(
        "knowledge", PACKAGE_ROOT / "__init__.py", submodule_search_locations=[str(PACKAGE_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["knowledge"] = module
    spec.loader.exec_module(module)


if __name__ == "__main__":
    _ensure_package_importable()

from knowledge.settings import (  # noqa: E402
    ConfigError,
    embedding_settings,
    load_config,
    manifest_path,
)

DEFAULT_GENERATE_PATTERNS = [
    "packages/**/*.go",
    "packages/**/*.py",
    "packages/**/*.ts",
    "packages/**/*.js",
    "packages/**/*.mjs",
    "packages/**/*.tsx",
    "packages/**/*.jsx",
    "packages/**/*.yml",
    "packages/**/*.yaml",
    "packages/**/*.json",
    "packages/**/*.tf",
    "packages/**/*.cls",
    "packages/**/*.trigger",
    "packages/**/Dockerfile",
    "!**/node_modules/**",
    "!**/dist/**",
    "!**/venv/**",
    "!**/__pycache__/**",
]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(PACKAGE_ROOT / ".env")
    load_dotenv(Path.cwd() / ".env")


def _git_head(root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def run_generate(root: Path, patterns: List[str]) -> int:
    """Run documentation generation. Returns the exit code."""
    from knowledge.generation import Generator

    print("[INFO] Starting documentation generation...")
    failures = Generator(str(root)).run(patterns)
    if failures:
        print(f"[ERROR] {failures} file(s) failed to generate.")
        return 1
    return 0


def run_index(root: Path, fresh: bool, cfg: dict, dry_run: bool = False) -> None:
    """Embed the docs, write the vector store and the manifest. Raises on any failure."""
    from knowledge.drift.detect_drift import detect_drift
    from knowledge.embeddings.router_embedding import get_embeddings_for_docs
    from knowledge.ingestion.federated_loader import discover_and_chunk_docs
    from knowledge.ingestion.index_builder import build_or_update_index
    from knowledge.ingestion.manifest import build_manifest, write_manifest

    settings = embedding_settings(cfg)
    print(f"[INFO] Discovering docs under {root}...")
    docs = discover_and_chunk_docs(str(root), cfg.get("federated_doc_paths"))
    batches = -(-len(docs) // settings.batch_size)
    print(f"[INFO] {len(docs)} chunks, {batches} embedding requests to {settings.model} "
          f"at {settings.dimensions} dimensions.")
    if dry_run:
        print("[INFO] Dry run: no embedding request sent, nothing written.")
        return

    embeddings = get_embeddings_for_docs(docs, batch_size=settings.batch_size)
    build_or_update_index(embeddings, fresh=fresh, cfg=cfg, settings=settings)

    manifest = build_manifest(docs, settings.model, settings.dimensions, _git_head(root))
    path = manifest_path(cfg)
    write_manifest(path, manifest)
    print(f"[INFO] Wrote the index manifest to {path} (digest {manifest['digest'][:12]}).")

    detect_drift(embeddings, threshold=cfg.get("drift_threshold", 0.85))


def run_check_index(root: Path, cfg: dict) -> int:
    """Compare the manifest with the docs and config. Makes no model call. Returns the exit code."""
    from knowledge.ingestion.federated_loader import discover_and_chunk_docs
    from knowledge.ingestion.manifest import check_manifest, read_manifest

    settings = embedding_settings(cfg)
    manifest = read_manifest(manifest_path(cfg))
    docs = discover_and_chunk_docs(str(root), cfg.get("federated_doc_paths"))
    problems = check_manifest(manifest, docs, settings.model, settings.dimensions)
    if problems:
        print("[ERROR] The index manifest does not match:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"[OK] The index manifest matches {len(docs)} chunks (digest {manifest['digest'][:12]}).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GlassOps Knowledge Pipeline")
    parser.add_argument("--query", "-q", type=str, help="Run a RAG query against the knowledge base")
    parser.add_argument("query_pos", nargs="*", help="Positional query string (joined by space)")
    parser.add_argument("--index", "-i", action="store_true", help="Embed the docs and update the index")
    parser.add_argument("--fresh", action="store_true", help="With --index, drop the index and build it from nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --index, count chunks and embedding requests without calling the API")
    parser.add_argument("--check-index", action="store_true",
                        help="Check the index manifest against the docs by hash. Makes no model call")
    parser.add_argument("--root", type=str, default=".",
                        help="Docs root to index, check or generate from (default: the current directory)")
    parser.add_argument("--generate", "-g", action="store_true", help="Generate documentation from source code")
    parser.add_argument("--pattern", "-p", type=str, action="append", dest="patterns",
                        help="Glob pattern(s) for --generate (can be specified multiple times)")
    return parser


def run_pipeline(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()

    from knowledge.embeddings import EmbeddingError
    from knowledge.ingestion import DocumentError, IndexBuildError, ManifestError
    from knowledge.llm import GenerationError
    from knowledge.rag import QueryError

    try:
        cfg = load_config()
        if args.check_index:
            return run_check_index(root, cfg)

        _load_env()
        if args.generate:
            return run_generate(root, args.patterns or DEFAULT_GENERATE_PATTERNS)

        final_query = args.query or (" ".join(args.query_pos) if args.query_pos else None)
        print("Starting GlassOps Knowledge Pipeline...")

        if args.index and args.dry_run:
            run_index(root, args.fresh, cfg, dry_run=True)
            return 0
        if args.index or not final_query:
            run_index(root, args.fresh, cfg)

        if final_query or not args.index:
            from knowledge.rag.query_engine import query_index

            query = final_query or "List all ADRs related to workflow layer"
            print(f"Query: {query}")
            print(f"\nRAG Response:\n{query_index(query, cfg=cfg)}\n")
        else:
            print("Re-indexing complete. Use --query '...' to ask questions.")
    except (ConfigError, DocumentError, EmbeddingError, IndexBuildError, ManifestError,
            GenerationError, QueryError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print("GlassOps Knowledge Pipeline finished!")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
