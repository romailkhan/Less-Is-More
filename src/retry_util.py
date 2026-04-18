"""Retry wrapper for LLM calls that may fail on transient network errors."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

MAX_RETRIES = 5
BACKOFF_BASE = 10  # seconds


def retry_on_connection_error(fn: Callable[..., T], *args, **kwargs) -> T:
    """Call *fn* and retry up to MAX_RETRIES times on connection / DNS errors."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            is_transient = any(
                k in msg
                for k in ("connection error", "name resolution", "connect error", "timed out", "timeout")
            )
            if not is_transient:
                raise
            last_exc = exc
            wait = BACKOFF_BASE * attempt
            print(f"[retry {attempt}/{MAX_RETRIES}] {type(exc).__name__}: {exc!s:.120} — retrying in {wait}s")
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]
