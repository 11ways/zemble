"""Shared HTTP plumbing for remote embedding providers: batching, retries, error surfacing."""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

MAX_TRIES = 5
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 120.0

#: Characters per token used to keep a batch under a provider's token ceiling without
#: paying for a real tokenizer. Deliberately pessimistic: code tokenizes worse than prose.
CHARS_PER_TOKEN = 3.0


class EmbeddingRequestError(RuntimeError):
    """A provider refused the request; the message carries the provider's own wording."""


def _sleep(seconds: float) -> None:
    """Pause between retries; the seam tests replace so a backoff costs no wall time."""
    time.sleep(seconds)


def _extract_message(body: bytes) -> str:
    """Pull a human message out of a provider error body, falling back to the raw text."""
    text = body.decode("utf-8", errors="replace").strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if isinstance(detail, str):
            return detail
        error = parsed.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if isinstance(parsed.get("message"), str):
            return str(parsed["message"])
    return text


def _retry_after_seconds(headers: Any) -> float | None:
    """Read a Retry-After header as seconds, ignoring unparseable values."""
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    """POST a JSON body and return the decoded JSON response.

    429 and 5xx are retried with exponential backoff (honouring ``Retry-After``); every
    other 4xx is raised immediately with the provider's own message.

    :param url: The full endpoint URL.
    :param payload: The JSON request body.
    :param headers: Extra request headers; ``Content-Type`` is added automatically.
    :return: The decoded response body.
    :raises EmbeddingRequestError: If the request is refused or every attempt fails.
    """
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **headers}
    last_error: str = "no attempt was made"

    for attempt in range(1, MAX_TRIES + 1):
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")  # noqa: S310
        retry_after: float | None = None
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
                decoded: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return decoded
        except urllib.error.HTTPError as exc:
            message = _extract_message(exc.read())
            if exc.code != 429 and exc.code < 500:
                raise EmbeddingRequestError(f"{url} returned {exc.code}: {message}") from None
            retry_after = _retry_after_seconds(exc.headers)
            last_error = f"{url} returned {exc.code}: {message}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{url} failed: {exc}"

        if attempt == MAX_TRIES:
            break
        delay = (
            retry_after
            if retry_after is not None
            else min(BASE_BACKOFF_SECONDS * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
        )
        delay *= 1.0 + random.random() * 0.25  # noqa: S311 - jitter, not cryptography
        logger.warning(
            "Embedding request failed (attempt %d/%d), retrying in %.1fs: %s", attempt, MAX_TRIES, delay, last_error
        )
        _sleep(delay)

    raise EmbeddingRequestError(f"Embedding request gave up after {MAX_TRIES} attempts. Last error: {last_error}")


def batched(texts: list[str], max_texts: int, max_chars: int) -> list[tuple[int, list[str]]]:
    """Split texts into batches under both a count and an estimated-token ceiling.

    A single text longer than ``max_chars`` still gets its own batch; the provider
    truncates it rather than the client silently dropping it.

    :param texts: The texts to split.
    :param max_texts: Maximum texts per batch.
    :param max_chars: Maximum total characters per batch.
    :return: ``(offset, batch)`` pairs, in input order.
    """
    batches: list[tuple[int, list[str]]] = []
    start = 0
    current: list[str] = []
    current_chars = 0
    for index, text in enumerate(texts):
        length = len(text)
        if current and (len(current) >= max_texts or current_chars + length > max_chars):
            batches.append((start, current))
            start = index
            current = []
            current_chars = 0
        current.append(text)
        current_chars += length
    if current:
        batches.append((start, current))
    return batches
