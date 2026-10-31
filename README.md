# GlassOps Knowledge Package

Imported from glassops-platform/glassops at e48af7314cbdae46ad3e97ab5e7ca7fb0923bfe1.

This package manages the Retrieval-Augmented Generation (RAG) pipeline for GlassOps. It handles document discovery, embedding generation, vector indexing, an index check by hash, and documentation generation.

## Structure

- **ingestion/**: Finds and chunks Markdown files under a docs root, builds the Chroma index and writes the index manifest.
- **embeddings/**: Generates vector embeddings through the Google Gen AI SDK (`google-genai`).
- **llm/**: The generation client, with retries and rate limiting.
- **generation/**: Generates documentation from source code with the prompts in `config/prompts.yml`.
- **drift/**: Reports duplicate content across the indexed docs.
- **rag/**: Answers questions from the index.

## Models

Every model id is set in `config/config.json` under `models`. No module hard-codes one.

| Setting | Default | Notes |
|---|---|---|
| `models.embedding.id` | `gemini-embedding-001` | `gemini-embedding-2` also works |
| `models.embedding.dimensions` | `768` | Required. `gemini-embedding-001` returns 3072 when no size is sent |
| `models.generation.docs` | `gemma-4-31b-it` | Documentation generation |
| `models.generation.query` | `gemma-4-26b-a4b-it` | RAG answers |

Vectors from different models, or of different sizes, cannot share an index. After changing the embedding model or its dimensions, rebuild with `--index --fresh`.

## Failures

Every failure exits non-zero: a missing API key, an embedding or generation error, a vector of the wrong size, a failed index write and a missing prompts file. There is no fallback vector. A failed embedding stops the run.

## Usage

### 1. Setup

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt    # Windows: venv\Scripts\pip
```

Put your API key in a `.env` file in this folder or in the folder you run from:

```env
GOOGLE_API_KEY=your_key_here
```

### 2. Build the index

The index is built from a docs root, usually a checkout of the GlassOps repository. Paths under the root come from `federated_doc_paths` in the config.

```bash
# Count chunks and embedding requests first. This makes no API call.
python main.py --index --dry-run --root ../glassops

# Build from nothing
python main.py --index --fresh --root ../glassops
```

The Chroma index is written to `glassops_index/` (not committed). The build also writes `index/manifest.json`, which records the model, the vector size, the docs commit and the hash of every chunk. Commit the manifest.

Build from a clean checkout. Untracked files in the docs root end up in the manifest and the CI check then fails.

### 3. Check the index

```bash
python main.py --check-index --root ../glassops
```

This compares the manifest with the docs by hash and makes no model call. It exits 1 when the docs, the model or the dimensions changed since the index was built. CI runs it against the commit recorded in the manifest, then reports whether the index is behind the latest docs.

### 4. Query

```bash
python main.py --query "What is the update policy for ADRs?"
```

### 5. Generate documentation

```bash
python main.py --generate --root ../glassops -p "packages/**/*.py"
```

## Running the tests

From the repository root:

```bash
pytest
```

The tests replace every Google client with a fake and make no network call.
