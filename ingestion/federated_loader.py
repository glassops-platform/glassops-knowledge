# ingestion/federated_loader.py
# Scans a docs root for Markdown, chunks it and hashes each chunk

import glob
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from knowledge.settings import load_config

DEFAULT_DOC_PATHS = ["docs/", "packages/**/adr", "packages/**/docs"]
IGNORED_DIRS = {"node_modules", "venv", ".venv", ".git", "__pycache__", "dist", "site-packages"}
# Fields the loader sets. Frontmatter keys with these names must not overwrite them.
SYSTEM_FIELDS = ("path", "source_file", "content", "hash")


class DocumentError(Exception):
    """A document matched the doc paths but could not be read."""


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_by_header(text: str, pattern: str) -> List[str]:
    chunks = []
    last_pos = 0
    for match in re.finditer(pattern, text):
        pos = match.start()
        if pos > last_pos:
            chunks.append(text[last_pos:pos].strip())
        last_pos = pos
    chunks.append(text[last_pos:].strip())
    return [c for c in chunks if c]


def chunk_markdown(content: str) -> List[str]:
    """Split on level 2 headers, or level 1 headers when there are none."""
    chunks = _split_by_header(content, r"(?m)^##\s+")
    if len(chunks) <= 1:
        chunks = _split_by_header(content, r"(?m)^#\s+")
    return chunks


def _find_doc_files(root: Path, base_paths: List[str]) -> List[Path]:
    found = set()
    for base in base_paths:
        pattern = f"{base.rstrip('/')}/**/*.md"
        for match in glob.glob(str(root / pattern), recursive=True):
            path = Path(match).resolve()
            if not path.is_file():
                continue
            if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
                continue
            found.add(path)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def discover_and_chunk_docs(root_dir: str = ".", doc_paths: Optional[List[str]] = None) -> List[Dict]:
    """
    Returns a list of dicts:
    { "path": "<relative/file.md>#chunk-<n>", "source_file": "<relative/file.md>",
      "content": <chunk text>, "hash": <sha256 of the chunk> }

    Paths are relative to root_dir and use forward slashes, and line endings are normalised to LF,
    so a Windows build and a Linux CI check produce the same ids and hashes.
    """
    root = Path(root_dir).resolve()
    if doc_paths is None:
        doc_paths = load_config().get("federated_doc_paths") or DEFAULT_DOC_PATHS

    docs: List[Dict] = []
    files = _find_doc_files(root, doc_paths)
    print(f"[INFO] Found {len(files)} documents under {root}.")

    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as e:
            raise DocumentError(f"could not read {rel}: {e}") from e
        content = content.replace("\r\n", "\n")
        if not content.strip():
            continue

        metadata: Dict = {}
        match = re.search(r"\A---\n(.*?)\n---", content, re.DOTALL)
        if match:
            try:
                parsed = yaml.safe_load(match.group(1))
                if isinstance(parsed, dict):
                    metadata = parsed
                content = content[match.end():].strip()
            except yaml.YAMLError as e:
                print(f"[WARNING] Failed to parse frontmatter for {rel}: {e}")

        for i, chunk_text in enumerate(chunk_markdown(content)):
            record = {k: v for k, v in metadata.items() if k not in SYSTEM_FIELDS}
            record.update({
                "path": f"{rel}#chunk-{i}",
                "source_file": rel,
                "content": chunk_text,
                "hash": hash_content(chunk_text),
            })
            docs.append(record)

    return docs
