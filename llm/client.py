# llm/client.py
"""
Shared LLM client for the GlassOps Knowledge Pipeline.
Handles API interaction, retry logic and rate limiting.

The model id comes from config. Every failure raises GenerationError, so a caller cannot
mistake a failed call for an empty answer.
"""

import os
import time
from typing import Callable, List, Optional

from knowledge.settings import GenerationSettings, generation_settings, load_config
from knowledge.utils.rate_limit import RateLimiter, estimate_tokens
from knowledge.utils.retry import is_retryable


class GenerationError(Exception):
    """A generation call failed or returned no text."""


class LLMClient:
    """
    A reusable client for Google generative models through the google-genai SDK.
    Retries transient errors (429, 5xx) and paces requests under the configured limits.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        settings: Optional[GenerationSettings] = None,
        client=None,
        sleep: Callable[[float], None] = time.sleep,
        backoffs: Optional[List[float]] = None,
    ):
        self.settings = settings or generation_settings(load_config())
        self.model = model or self.settings.docs_model
        self._sleep = sleep
        self.backoffs = backoffs if backoffs is not None else [15, 30, 60, 120, 180]
        self.limiter = RateLimiter(self.settings.rpm_limit, self.settings.tpm_limit, sleep=sleep)
        if client is None:
            api_key = os.getenv("GOOGLE_API_KEY", "").strip().strip("'\"")
            if not api_key:
                raise GenerationError("GOOGLE_API_KEY is not set, so no text can be generated")
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client

    def generate(
        self,
        prompt: str,
        max_retries: int = 5,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """Generate text for a prompt. Returns the text or raises GenerationError."""
        from google.genai import types

        self.limiter.acquire(estimate_tokens(prompt) + 100)
        config = types.GenerateContentConfig(
            temperature=self.settings.temperature if temperature is None else temperature,
            max_output_tokens=max_output_tokens or self.settings.max_output_tokens,
        )

        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
            except Exception as e:
                if is_retryable(e) and attempt < max_retries:
                    wait = self.backoffs[min(attempt, len(self.backoffs) - 1)]
                    print(f"[WARNING] Retryable error ({str(e)[:50]}). Retrying in {wait}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    self._sleep(wait)
                    continue
                raise GenerationError(f"{self.model} call failed: {e}") from e

            text = getattr(response, "text", None)
            if text:
                return text
            candidates = getattr(response, "candidates", None) or []
            reason = getattr(candidates[0], "finish_reason", "unknown") if candidates else "no candidates"
            raise GenerationError(f"{self.model} returned no text (finish reason: {reason})")

        raise GenerationError(f"{self.model} call failed after {max_retries} retries")  # pragma: no cover
