# utils/rate_limit.py
"""
A sliding-window limiter for requests per minute and tokens per minute.

The clock and sleep functions are injectable so tests run without waiting.
"""

import time
from typing import Callable, List, Tuple


def estimate_tokens(text: str) -> int:
    """Rough token estimate at four characters per token."""
    return max(1, len(text) // 4)


class RateLimiter:
    def __init__(
        self,
        rpm_limit: int,
        tpm_limit: int,
        window_seconds: float = 65.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._history: List[Tuple[float, int]] = []

    def _prune(self, now: float) -> None:
        self._history = [(t, n) for (t, n) in self._history if now - t < self.window]

    def acquire(self, tokens: int) -> None:
        """Block until one request of the given token size fits in the window, then record it."""
        # A single request larger than the whole budget can never fit. Let it through alone
        # once the window is empty, so the API answers it instead of this loop waiting forever.
        tokens = min(tokens, self.tpm_limit)
        while True:
            now = self._clock()
            self._prune(now)
            used = sum(n for _, n in self._history)
            if len(self._history) < self.rpm_limit and used + tokens <= self.tpm_limit:
                self._history.append((now, tokens))
                return
            oldest = self._history[0][0]
            wait = (oldest + self.window) - now
            print(f"[THROTTLE] Waiting {max(wait, 0):.1f}s for the rate limit window...")
            self._sleep(max(wait, 0) + 1)
