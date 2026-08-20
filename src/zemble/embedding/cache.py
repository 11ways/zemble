"""Content-hash embedding cache: a paid vector is paid for exactly once."""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
from pathlib import Path

import numpy as np

from zemble.embedding.base import Embedder, EmbeddingMatrix, normalize_rows

logger = logging.getLogger(__name__)

_SLUG_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    text_sha256 TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (text_sha256, dims)
)
"""


def cache_root() -> Path:
    """Return the directory holding the per-family sqlite files."""
    from zemble.cache import resolve_cache_folder

    return resolve_cache_folder() / "embeddings"


def family_slug(family: str) -> str:
    """Turn an embedder family (scheme plus model, without dimensions) into a filename."""
    slug = _SLUG_UNSAFE.sub("-", family).strip("-").lower()
    if len(slug) > 80:
        slug = f"{slug[:60]}-{hashlib.sha256(family.encode('utf-8')).hexdigest()[:12]}"
    return slug or "embedder"


def text_hash(text: str) -> str:
    """Return the sha256 hex digest of a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """A sqlite-backed store of vectors keyed by (text hash, dimensions).

    One connection per process, WAL mode, and no cross-family mixing: the file is
    chosen by embedder family so a Matryoshka slice can never come from a different model.
    """

    def __init__(self, family: str, directory: Path | None = None) -> None:
        """Open (creating if needed) the cache file for an embedder family.

        :param family: Scheme plus model, e.g. ``voyage:voyage-code-4``. Dimensions are NOT part of it.
        :param directory: Override for the cache directory; defaults to the zemble cache folder.
        """
        self.family = family
        root = directory if directory is not None else cache_root()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{family_slug(family)}.sqlite"
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._connection.close()

    def get(self, digest: str, dims: int) -> np.ndarray | None:
        """Return the cached vector for a text hash at a width, slicing a wider one when possible.

        :param digest: The sha256 of the text.
        :param dims: The requested width.
        :return: A float32 vector of length ``dims``, or None on a miss.
        """
        with self._lock:
            row = self._connection.execute(
                "SELECT vec FROM embeddings WHERE text_sha256 = ? AND dims = ?", (digest, dims)
            ).fetchone()
            if row is not None:
                return np.frombuffer(row[0], dtype=np.float32)
            # Matryoshka fallback: a wider vector from the same family truncates to a
            # usable narrower one. The slice is not stored - it is derivable, and storing
            # it would double the file for no gain.
            wider = self._connection.execute(
                "SELECT vec FROM embeddings WHERE text_sha256 = ? AND dims > ? ORDER BY dims ASC LIMIT 1",
                (digest, dims),
            ).fetchone()
        if wider is None:
            return None
        sliced = np.frombuffer(wider[0], dtype=np.float32)[:dims]
        return normalize_rows(sliced.reshape(1, -1))[0]

    def put_many(self, rows: list[tuple[str, int, np.ndarray]]) -> None:
        """Store vectors, ignoring any key another process wrote first.

        :param rows: ``(text hash, dims, vector)`` triples.
        """
        if not rows:
            return
        payload = [(digest, dims, np.asarray(vector, dtype=np.float32).tobytes()) for digest, dims, vector in rows]
        with self._lock:
            self._connection.executemany(
                "INSERT OR REPLACE INTO embeddings (text_sha256, dims, vec) VALUES (?, ?, ?)", payload
            )
            self._connection.commit()


class CachingEmbedder:
    """Wraps any embedder so identical document text is embedded at most once, ever.

    Only documents are cached. Queries are one-off, and for an asymmetric provider a
    query vector must never be served where a document vector was asked for, so
    :meth:`embed_queries` goes straight through.
    """

    def __init__(self, inner: Embedder, family: str, directory: Path | None = None) -> None:
        """Initialise the wrapper.

        :param inner: The embedder to call on a cache miss.
        :param family: Cache family key (scheme plus model, no dimensions).
        :param directory: Override for the cache directory.
        """
        self.inner = inner
        self.cache = EmbeddingCache(family, directory)

    @property
    def model_id(self) -> str:
        """The wrapped embedder's normalized spec string; caching is invisible to the index."""
        return self.inner.model_id

    @property
    def dimensions(self) -> int:
        """The wrapped embedder's vector width."""
        return self.inner.dimensions

    def embed_documents(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed documents, calling the provider only for texts not already stored.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        dims = self.dimensions
        if not texts:
            return np.empty((0, dims), dtype=np.float32)

        digests = [text_hash(text) for text in texts]
        result = np.zeros((len(texts), dims), dtype=np.float32)
        missing_positions: list[int] = []
        # Duplicate texts inside one call share a single provider slot.
        first_position: dict[str, int] = {}
        duplicates: list[tuple[int, int]] = []

        for position, digest in enumerate(digests):
            cached = self.cache.get(digest, dims)
            if cached is not None:
                result[position] = cached
                continue
            seen = first_position.get(digest)
            if seen is not None:
                duplicates.append((position, seen))
                continue
            first_position[digest] = position
            missing_positions.append(position)

        if missing_positions:
            fresh = self.inner.embed_documents([texts[position] for position in missing_positions])
            store: list[tuple[str, int, np.ndarray]] = []
            for row, position in enumerate(missing_positions):
                result[position] = fresh[row]
                store.append((digests[position], dims, fresh[row]))
            self.cache.put_many(store)

        for position, source in duplicates:
            result[position] = result[source]
        return result

    def embed_queries(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed queries without touching the cache.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        return self.inner.embed_queries(texts)
