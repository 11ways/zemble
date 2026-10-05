"""What the embedding server and its clients agree on: routes, settings and the vector encoding.

Both sides import this module and nothing else of each other, so a client process never loads
the server and the server never routes its own provider calls back through a client.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import numpy as np

from zemble.embedding.base import EmbeddingMatrix
from zemble.embedding.cache import FLUSH_EVERY

#: Env var naming the embedding server a client uses, e.g. ``https://embed.example:8765``.
SERVER_ENV = "ZEMBLE_EMBED_SERVER"
#: Env var holding the API key a client presents to that server.
SERVER_KEY_ENV = "ZEMBLE_EMBED_SERVER_KEY"

#: The HTTP header the API key travels in, as a bearer token.
AUTH_HEADER = "Authorization"

#: Liveness, answered without a key.
HEALTH_ROUTE = "/v1/health"
#: What the server holds and has done since it started.
STATUS_ROUTE = "/v1/status"
#: The identity of one embedder spec: model id, width, family, fusion bonus.
DESCRIBE_ROUTE = "/v1/describe"
#: Document vectors: served from the cache, bought on a miss, then stored.
DOCUMENTS_ROUTE = "/v1/documents"
#: Query vectors: always bought, never stored.
QUERIES_ROUTE = "/v1/queries"
#: Which text hashes the cache already serves, without sending or buying any text.
COVERED_ROUTE = "/v1/covered"
#: Vectors another cache already bought, stored where missing.
STORE_ROUTE = "/v1/store"
#: Reranker scores for one query and its passages.
RERANK_ROUTE = "/v1/rerank"

#: Texts one documents request carries: the server's own flush boundary, so one failed request
#: loses at most one provider slice.
DOCUMENTS_PER_REQUEST = FLUSH_EVERY
#: Hashes one covered request carries.
DIGESTS_PER_REQUEST = 10_000
#: Rows one store request carries.
ROWS_PER_REQUEST = 4096
#: The largest request body the server reads.
MAX_BODY_BYTES = 64 * 1024 * 1024

#: The HTTP status the server answers with when the provider behind it refused or failed for
#: good. Not a 5xx: the server already retried, so a client retrying it again pays nothing but time.
PROVIDER_FAILED_STATUS = 424


@dataclass(frozen=True)
class ServerSettings:
    """Where a client's embedding server is and the key it presents."""

    url: str
    key: str


def server_settings() -> ServerSettings | None:
    """Return the configured embedding server, or None when this process embeds directly.

    :return: The settings, or None when ``ZEMBLE_EMBED_SERVER`` is unset.
    :raises ValueError: If a server is named without a key.
    """
    url = os.environ.get(SERVER_ENV, "").strip().rstrip("/")
    if not url:
        return None
    key = os.environ.get(SERVER_KEY_ENV, "").strip()
    if not key:
        raise ValueError(f"{SERVER_ENV} is set but {SERVER_KEY_ENV} is not; the embedding server needs a key")
    return ServerSettings(url=url, key=key)


def encode_vectors(vectors: EmbeddingMatrix) -> str:
    """Encode a float32 matrix as base64 of its little-endian bytes."""
    return base64.b64encode(np.ascontiguousarray(vectors, dtype="<f4").tobytes()).decode("ascii")


def decode_vectors(encoded: str, rows: int, dimensions: int) -> EmbeddingMatrix:
    """Decode what :func:`encode_vectors` wrote back into a ``(rows, dimensions)`` float32 matrix.

    :param encoded: The base64 payload.
    :param rows: How many vectors it must hold.
    :param dimensions: Their width.
    :return: The matrix.
    :raises ValueError: If the payload does not hold exactly that many values.
    """
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    if len(raw) != rows * dimensions * 4:
        raise ValueError(f"expected {rows}x{dimensions} float32 values, got {len(raw)} bytes")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(rows, dimensions)


__all__ = [
    "AUTH_HEADER",
    "COVERED_ROUTE",
    "DESCRIBE_ROUTE",
    "DIGESTS_PER_REQUEST",
    "DOCUMENTS_PER_REQUEST",
    "DOCUMENTS_ROUTE",
    "HEALTH_ROUTE",
    "MAX_BODY_BYTES",
    "PROVIDER_FAILED_STATUS",
    "QUERIES_ROUTE",
    "RERANK_ROUTE",
    "ROWS_PER_REQUEST",
    "SERVER_ENV",
    "SERVER_KEY_ENV",
    "STATUS_ROUTE",
    "STORE_ROUTE",
    "ServerSettings",
    "decode_vectors",
    "encode_vectors",
    "server_settings",
]
