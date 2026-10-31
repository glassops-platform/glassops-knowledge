# ingestion/manifest.py
"""
The index manifest: a committed record of what the local index was built from.

The Chroma index itself is not committed. The manifest records the embedding model, the vector
size and the id and SHA-256 of every chunk. CI recomputes the chunk hashes from the docs and
compares them with the manifest. That check needs no API key and makes no model call.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

MANIFEST_VERSION = 1


class ManifestError(Exception):
    """The manifest is missing, unreadable or malformed."""


def chunk_entries(docs: List[Dict]) -> List[Dict[str, str]]:
    return sorted(({"id": d["path"], "hash": d["hash"]} for d in docs), key=lambda e: e["id"])


def digest(entries: List[Dict[str, str]]) -> str:
    h = hashlib.sha256()
    for e in entries:
        h.update(f"{e['id']}\t{e['hash']}\n".encode("utf-8"))
    return h.hexdigest()


def build_manifest(docs: List[Dict], model: str, dimensions: int, source_commit: Optional[str] = None) -> Dict:
    entries = chunk_entries(docs)
    manifest = {
        "version": MANIFEST_VERSION,
        "embedding_model": model,
        "dimensions": dimensions,
        "chunk_count": len(entries),
        "digest": digest(entries),
        "chunks": entries,
    }
    if source_commit:
        manifest["source_commit"] = source_commit
    return manifest


def write_manifest(path: Path, manifest: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_manifest(path: Path) -> Dict:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ManifestError(f"index manifest not found: {path}. Build the index first.") from e
    except (OSError, json.JSONDecodeError) as e:
        raise ManifestError(f"could not read index manifest {path}: {e}") from e
    for key in ("version", "embedding_model", "dimensions", "digest", "chunks"):
        if key not in manifest:
            raise ManifestError(f'index manifest {path} is missing "{key}"')
    if manifest["version"] != MANIFEST_VERSION:
        raise ManifestError(f"index manifest version {manifest['version']} is not supported")
    return manifest


def check_manifest(manifest: Dict, docs: List[Dict], model: str, dimensions: int) -> List[str]:
    """Compare a manifest with the current docs and config. Returns problems, empty when it matches."""
    problems: List[str] = []
    recorded = manifest["chunks"]
    if digest(recorded) != manifest["digest"]:
        problems.append("the manifest digest does not match its own chunk list (edited by hand?)")
    if manifest["embedding_model"] != model:
        problems.append(
            f"the index was built with {manifest['embedding_model']}, config names {model}. Rebuild the index."
        )
    if manifest["dimensions"] != dimensions:
        problems.append(
            f"the index has {manifest['dimensions']} dimensions, config names {dimensions}. Rebuild the index."
        )

    current = chunk_entries(docs)
    if digest(current) != manifest["digest"]:
        old = {e["id"]: e["hash"] for e in recorded}
        new = {e["id"]: e["hash"] for e in current}
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        changed = sorted(i for i in set(old) & set(new) if old[i] != new[i])
        summary = f"the docs changed since the index was built: {len(added)} added, {len(removed)} removed, {len(changed)} changed chunks"
        problems.append(summary)
        for label, ids in (("added", added), ("removed", removed), ("changed", changed)):
            for i in ids[:20]:
                problems.append(f"  {label}: {i}")
            if len(ids) > 20:
                problems.append(f"  ... and {len(ids) - 20} more {label}")
    return problems
