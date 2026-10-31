# embeddings/gemini_embedding.py
"""
Embeddings through the Google Gen AI SDK (google-genai).

The model id and output size come from config. Any failure raises EmbeddingError. There is no
fallback vector: a failed embedding stops the run instead of writing noise into the index.
"""

import os
import time
from typing import Callable, List, Optional

from knowledge.settings import EmbeddingSettings
from knowledge.utils.rate_limit import RateLimiter, estimate_tokens
from knowledge.utils.retry import is_retryable

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class EmbeddingError(Exception):
    """An embedding could not be produced. The caller must stop, not substitute a vector."""


def _api_key() -> str:
    return os.getenv("GOOGLE_API_KEY", "").strip().strip("'\"")


class GeminiEmbedding:
    def __init__(
        self,
        settings: EmbeddingSettings,
        client=None,
        max_retries: int = 4,
        backoffs: Optional[List[float]] = None,
        sleep: Callable[[float], None] = time.sleep,
        limiter: Optional[RateLimiter] = None,
    ):
        self.settings = settings
        self.max_retries = max_retries
        self.backoffs = backoffs if backoffs is not None else [15, 30, 60, 120]
        self._sleep = sleep
        self.limiter = limiter or RateLimiter(settings.rpm_limit, settings.tpm_limit, sleep=sleep)
        if client is None:
            key = _api_key()
            if not key:
                raise EmbeddingError("GOOGLE_API_KEY is not set, so no embeddings can be made")
            from google import genai

            client = genai.Client(api_key=key)
        self.client = client

    def get_embeddings(self, texts: List[str], task_type: str = TASK_DOCUMENT) -> List[List[float]]:
        """Embed a batch of texts. Returns one vector per text, each of the configured size."""
        if not texts:
            return []
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.settings.dimensions,
        )
        self.limiter.acquire(sum(estimate_tokens(t) for t in texts))

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.settings.model,
                    contents=texts,
                    config=config,
                )
                break
            except Exception as e:  # the SDK raises several types, all handled the same way
                last_error = e
                if is_retryable(e) and attempt < self.max_retries:
                    wait = self.backoffs[min(attempt, len(self.backoffs) - 1)]
                    print(f"[WARNING] Embedding request failed ({str(e)[:60]}). Retrying in {wait}s...")
                    self._sleep(wait)
                    continue
                raise EmbeddingError(f"embedding request to {self.settings.model} failed: {e}") from e
        else:  # pragma: no cover - the loop always breaks or raises
            raise EmbeddingError(f"embedding request failed: {last_error}")

        embeddings = getattr(response, "embeddings", None) or []
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"{self.settings.model} returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        vectors = []
        for i, emb in enumerate(embeddings):
            values = list(getattr(emb, "values", None) or [])
            if len(values) != self.settings.dimensions:
                raise EmbeddingError(
                    f"embedding {i} from {self.settings.model} has {len(values)} dimensions, "
                    f"config expects {self.settings.dimensions}"
                )
            vectors.append([float(v) for v in values])
        return vectors
