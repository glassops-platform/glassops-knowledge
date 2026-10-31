# utils/retry.py
"""Decides which Google API errors are worth retrying."""

try:
    from google.genai import errors as genai_errors
except ImportError:  # pragma: no cover
    genai_errors = None

RETRYABLE_CODES = {429, 500, 503, 504}


def is_retryable(exc: Exception) -> bool:
    if genai_errors is not None and isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None) in RETRYABLE_CODES
    text = str(exc).lower()
    return any(marker in text for marker in ("429", "503", "resource_exhausted", "overloaded", "unavailable"))
