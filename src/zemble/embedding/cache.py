"""Content-hash embedding cache: a paid vector is paid for exactly once."""

from __future__ import annotations

import fcntl
import hashlib
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from zemble.embedding.base import (
    Embedder,
    EmbeddingMatrix,
    declared_dimensions,
    is_remote,
    normalize_rows,
    semantic_weight_bonus,
)

_SLUG_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")

#: Misses are handed to the provider in slices of this many texts, and every slice is
#: written to sqlite before the next one is asked for. A cold workspace index is a single
#: ``embed_documents`` call of tens of thousands of texts and half an hour of paid requests;
#: without a flush boundary one failure at the end throws away every vector already bought.
FLUSH_EVERY = 512

#: The WAL is truncated back to this size whenever a checkpoint empties it. Without a limit
#: it keeps the high-water mark of the largest write burst forever (720 MB measured).
JOURNAL_SIZE_LIMIT_BYTES = 64 * 1024 * 1024

#: Native SQLite lock waiting is bounded; provider work never holds a database transaction.
BUSY_TIMEOUT_SECONDS = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    text_sha256 TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (text_sha256, dims)
);
CREATE TABLE IF NOT EXISTS used (
    text_sha256 TEXT PRIMARY KEY,
    day INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS cache_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

#: `cache_meta` key: the day the `used` stamps began. A vector stored before it has no stamp,
#: and counts as used that day, never as unused forever: an unknown age fails closed.
STAMPS_SINCE_KEY = "used_stamps_since"


def today() -> int:
    """Return the current day number (days since the epoch, UTC), the unit of a use stamp."""
    return int(time.time() // 86400)


def cache_root() -> Path:
    """Return the directory holding the per-family sqlite files."""
    from zemble.cache import resolve_cache_folder

    return resolve_cache_folder() / "embeddings"


def cache_file(family: str, directory: Path) -> Path:
    """Return the sqlite file an embedder family's vectors live in."""
    return directory / f"{family_slug(family)}.sqlite"


def family_slug(family: str) -> str:
    """Turn an embedder family (scheme plus model, without dimensions) into a filename."""
    slug = _SLUG_UNSAFE.sub("-", family).strip("-").lower()
    if len(slug) > 80:
        slug = f"{slug[:60]}-{hashlib.sha256(family.encode('utf-8')).hexdigest()[:12]}"
    return slug or "embedder"


def text_hash(text: str) -> str:
    """Return the sha256 hex digest of a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def _transaction(connection: sqlite3.Connection, *, immediate: bool = True) -> Iterator[None]:
    """Acquire the writer before reading, commit one short batch, and roll back any failed batch."""
    connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def connect_cache(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) one family's cache file: WAL mode, a bounded WAL, the stamp tables."""
    connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=BUSY_TIMEOUT_SECONDS)
    try:
        # A journal-mode transition on a brand-new family must not race another opener.
        # This lock covers initialization only, never embedding or normal cache reads.
        with path.with_suffix(path.suffix + ".init.lock").open("a+b") as initialization:
            fcntl.flock(initialization.fileno(), fcntl.LOCK_EX)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(f"PRAGMA journal_size_limit={JOURNAL_SIZE_LIMIT_BYTES}")
            connection.executescript(_SCHEMA)
            if stamps_since(connection) is None:
                with _transaction(connection):
                    connection.execute(
                        "INSERT OR IGNORE INTO cache_meta (key, value) VALUES (?, ?)",
                        (STAMPS_SINCE_KEY, str(today())),
                    )
        return connection
    except BaseException:
        connection.close()
        raise


def stamps_since(connection: sqlite3.Connection) -> int | None:
    """Return the day use stamps began in this file, or None before any zemble that writes them opened it."""
    try:
        row = connection.execute("SELECT value FROM cache_meta WHERE key = ?", (STAMPS_SINCE_KEY,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row[0]) if row is not None else None


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
        self.path = cache_file(family, root)
        self._lock = threading.Lock()
        self._connection = connect_cache(self.path)

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

    def covered(self, digests: list[str], dims: int) -> set[str]:
        """Return which of these text hashes already have a usable vector at this width.

        A wider vector counts: :meth:`get` slices it. One pass over the primary key instead
        of two queries per text, because a pre-flight asks this about every chunk in a tree.

        :param digests: The text hashes to look up.
        :param dims: The requested width.
        :return: The subset of ``digests`` that would be a cache hit.
        """
        if not digests:
            return set()
        with self._lock:
            # Group temporary writes, then release their transaction BEFORE reading the main DB.
            # Otherwise this lookup pins a WAL snapshot until a later paid embedding tries to write.
            with _transaction(self._connection, immediate=False):
                self._connection.execute("CREATE TEMP TABLE IF NOT EXISTS wanted (digest TEXT PRIMARY KEY)")
                self._connection.execute("DELETE FROM wanted")
                self._connection.executemany(
                    "INSERT OR IGNORE INTO wanted (digest) VALUES (?)", [(digest,) for digest in digests]
                )
            rows = self._connection.execute(
                "SELECT w.digest FROM wanted w JOIN embeddings e ON e.text_sha256 = w.digest WHERE e.dims >= ?",
                (dims,),
            ).fetchall()
        return {row[0] for row in rows}

    def stored_dimensions(self, limit: int = 2) -> list[int]:
        """Return up to `limit` distinct vector widths this family's cache file already holds.

        A model whose width only a provider request could tell has still been bought at SOME
        width by every earlier build, and this file is the one place to read it without asking
        anybody. Two is enough to tell an unambiguous file from a mixed one.

        :param limit: How many distinct widths to look for.
        :return: The widths, ascending.
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT DISTINCT dims FROM embeddings ORDER BY dims LIMIT ?", (limit,)
            ).fetchall()
        return [int(row[0]) for row in rows]

    def usable_width(self, declared: int | None) -> int | None:
        """Return the width a build's cache reads would use, without asking a provider anything.

        A model whose width only a probe could tell has still been bought at SOME width by every
        earlier build, and this file records it. Only an unambiguous file answers: with two
        widths stored, a build could read either, and over-billing beats inventing a hit.

        :param declared: The width the embedder already declares, when it declares one.
        :return: The width to look up, or None when nothing can be looked up honestly.
        """
        if declared is not None:
            return declared
        widths = self.stored_dimensions()
        return widths[0] if len(widths) == 1 else None

    def pending(self, texts: list[str], dims: int | None) -> list[str]:
        """Return the texts a buyer reading this file would really have to send.

        THE one answer to what a cached buy costs, so a report and a build cannot disagree about
        it: duplicates collapse, because :meth:`CachingEmbedder.embed_documents` gives every copy
        of one text a single provider slot, and a stored vector was bought already.

        :param texts: The texts a build is about to embed.
        :param dims: The width a stored vector has to serve; None names no usable width, so
            nothing counts as stored - never a discount nobody can prove.
        :return: Those still to be bought, first occurrence first.
        """
        digests = [text_hash(text) for text in texts]
        covered = self.covered(digests, dims) if dims is not None else set()
        pending: list[str] = []
        seen: set[str] = set()
        for text, digest in zip(texts, digests, strict=True):
            if digest in covered or digest in seen:
                continue
            seen.add(digest)
            pending.append(text)
        return pending

    def put_many(self, rows: list[tuple[str, int, np.ndarray]]) -> None:
        """Store vectors, ignoring any key another process wrote first, and stamp them as used today.

        :param rows: ``(text hash, dims, vector)`` triples.
        """
        if not rows:
            return
        payload = [(digest, dims, np.asarray(vector, dtype=np.float32).tobytes()) for digest, dims, vector in rows]
        with self._lock, _transaction(self._connection):
            self._connection.executemany(
                "INSERT OR REPLACE INTO embeddings (text_sha256, dims, vec) VALUES (?, ?, ?)", payload
            )
            self._stamp({digest for digest, _dims, _vector in rows})

    def touch(self, digests: Iterable[str]) -> None:
        """Stamp served vectors as used today, so a garbage collection keeps them through its grace period.

        :param digests: The text hashes a caller was served from this file.
        """
        distinct = set(digests)
        if not distinct:
            return
        with self._lock, _transaction(self._connection):
            self._stamp(distinct)

    def _stamp(self, digests: set[str]) -> None:
        """Write today's stamp for these hashes; a stamp already at today is left unwritten."""
        day = today()
        self._connection.executemany(
            "INSERT INTO used (text_sha256, day) VALUES (?, ?) "
            "ON CONFLICT(text_sha256) DO UPDATE SET day = excluded.day WHERE day < excluded.day",
            [(digest, day) for digest in digests],
        )


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

    @property
    def is_remote(self) -> bool:
        """Whether the wrapped embedder is a paid one; caching does not change who pays."""
        return is_remote(self.inner)

    @property
    def declared_dimensions(self) -> int | None:
        """The wrapped embedder's width, when it is known without a request."""
        return declared_dimensions(self.inner)

    @property
    def known_dimensions(self) -> int | None:
        """The width to judge stored vectors by where no provider may be asked, or None for none."""
        return self.cache.usable_width(self.declared_dimensions)

    @property
    def semantic_weight_bonus(self) -> float:
        """The wrapped embedder's fusion bonus; caching does not change how good its vectors are."""
        return semantic_weight_bonus(self.inner)

    def pending_documents(self, texts: list[str]) -> list[str]:
        """Return the distinct texts this cache would still have to buy, at the width a build reads.

        This is the cache's half of the bill guard - it answers WHAT would be bought, and
        :func:`zemble.embedding.pricing.require_affordable_bill` decides whether it may be.
        Reading that width costs a provider probe on a model that declares none, which the seam
        calling this is about to make anyway; :meth:`pending_documents_unprobed` is the same
        answer for a caller that may not.

        :param texts: The texts a build is about to embed.
        :return: Those with no usable vector stored, first occurrence first.
        """
        return self.cache.pending(texts, self.dimensions)

    def pending_documents_unprobed(self, texts: list[str]) -> list[str]:
        """Return the same answer for a caller that may not ask the provider anything.

        A pre-flight report contacts nobody, not even to learn a vector width, so the width comes
        from what this family's file already holds. Where that is not unambiguous nothing counts
        as stored, which over-bills rather than inventing a hit - but duplicates still collapse,
        because one text having one provider slot is a fact about the buyer, not about a width.

        :param texts: The texts a build would embed.
        :return: Those with no usable vector stored, first occurrence first.
        """
        return self.cache.pending(texts, self.known_dimensions)

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

        served: list[str] = []
        for position, digest in enumerate(digests):
            cached = self.cache.get(digest, dims)
            if cached is not None:
                result[position] = cached
                served.append(digest)
                continue
            seen = first_position.get(digest)
            if seen is not None:
                duplicates.append((position, seen))
                continue
            first_position[digest] = position
            missing_positions.append(position)

        for start in range(0, len(missing_positions), FLUSH_EVERY):
            slice_positions = missing_positions[start : start + FLUSH_EVERY]
            fresh = self.inner.embed_documents([texts[position] for position in slice_positions])
            store: list[tuple[str, int, np.ndarray]] = []
            for row, position in enumerate(slice_positions):
                result[position] = fresh[row]
                store.append((digests[position], dims, fresh[row]))
            self.cache.put_many(store)

        for position, source in duplicates:
            result[position] = result[source]
        self.cache.touch(served)
        return result

    def embed_queries(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed queries without touching the cache.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        return self.inner.embed_queries(texts)
