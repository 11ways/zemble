"""Content-hash embedding cache: a paid vector is paid for exactly once."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
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
from zemble.embedding.keys import cache_key, legacy_key

logger = logging.getLogger(__name__)

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

#: Days a use stamp stays fresh: one younger than this is never rewritten, so a vector in daily use
#: costs one stamp write a week rather than one a day, and may show a stamp up to this minus one
#: days old. A garbage collection's grace period must be longer (`zemble.embedding.gc`).
STAMP_REFRESH_DAYS = 7

#: Stamps one write transaction of a lookup or report writes, so a buyer waits on the writer for
#: milliseconds, not for a whole batch (measured in docs/embed-server.md).
STAMPS_PER_TRANSACTION = 250

#: The upsert every stamping path shares: a new hash gets today, an existing stamp only once it is stale.
_STAMP_UPSERT_CONFLICT = (
    f"ON CONFLICT(text_sha256) DO UPDATE SET day = excluded.day WHERE day <= excluded.day - {STAMP_REFRESH_DAYS}"
)

#: `cache_meta` keys of the vector count a file keeps of itself. TRACKED is moved by triggers on
#: every row added or removed since they were installed; BASE is COUNT(*) minus TRACKED, read
#: once from one snapshot. That difference never changes after the triggers exist, so the stored
#: count is TRACKED plus BASE, exact, and a file without BASE is still being counted.
VECTORS_TRACKED_KEY = "vectors_tracked"
VECTORS_BASE_KEY = "vectors_base"

_COUNT_INSERT_TRIGGER = "embeddings_count_insert"

# AIDEV-NOTE: the insert trigger runs BEFORE the row lands and counts only a key that is new and
# would pass NOT NULL, so INSERT OR IGNORE and INSERT OR REPLACE (whose replaced row fires no delete
# trigger unless recursive_triggers is on, which nothing here turns on) both count right, and so
# does any older zemble writing this file: triggers live in the file, not in this code.
_COUNT_TRIGGERS = (
    f"""CREATE TRIGGER IF NOT EXISTS {_COUNT_INSERT_TRIGGER} BEFORE INSERT ON embeddings
    WHEN NEW.text_sha256 IS NOT NULL AND NEW.dims IS NOT NULL AND NEW.vec IS NOT NULL
    AND NOT EXISTS (SELECT 1 FROM embeddings WHERE text_sha256 = NEW.text_sha256 AND dims = NEW.dims)
    BEGIN UPDATE cache_meta SET value = CAST(value AS INTEGER) + 1 WHERE key = '{VECTORS_TRACKED_KEY}'; END""",
    f"""CREATE TRIGGER IF NOT EXISTS embeddings_count_delete AFTER DELETE ON embeddings
    BEGIN UPDATE cache_meta SET value = CAST(value AS INTEGER) - 1 WHERE key = '{VECTORS_TRACKED_KEY}'; END""",
)


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


def uncovered_texts(texts: list[str], digests: list[str], covered: set[str]) -> list[str]:
    """Return the texts a caching buyer would really send: uncovered ones, each distinct text once.

    THE one answer to what a cached buy costs, shared by the local cache and the embedding
    server's client, because the buyer gives every copy of one text a single provider slot.

    :param texts: The texts a build is about to embed.
    :param digests: Their cache keys (`zemble.embedding.keys.cache_key`), in the same order.
    :param covered: The keys a stored vector already serves.
    :return: Those still to be bought, first occurrence first.
    """
    pending: list[str] = []
    seen: set[str] = set()
    for text, digest in zip(texts, digests, strict=True):
        if digest in covered or digest in seen:
            continue
        seen.add(digest)
        pending.append(text)
    return pending


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
            _install_count(connection)
        return connection
    except BaseException:
        connection.close()
        raise


def rekey_rows(connection: sqlite3.Connection, pairs: Iterable[tuple[str, str]]) -> list[str]:
    """Copy the vectors held under each pair's old key to its new key, where the new key holds none; stamp the copies.

    Runs inside the caller's write transaction and buys nothing. Each pair is two key-index probes, and the
    vector row is read only when its copy is due, so a pair already migrated costs no row read and no scan.

    :param pairs: ``(old key, new key)`` pairs.
    :return: The new keys that gained a vector.
    """
    copied: list[str] = []
    for old, new in pairs:
        if old == new:
            continue
        # The plan seeks the old key's index entry, runs the constant NOT EXISTS once, and reads the row only then.
        cursor = connection.execute(
            "INSERT OR IGNORE INTO embeddings (text_sha256, dims, vec) SELECT ?2, dims, vec FROM embeddings "
            "WHERE NOT EXISTS (SELECT 1 FROM embeddings WHERE text_sha256 = ?2) AND text_sha256 = ?1",
            (old, new),
        )
        if cursor.rowcount > 0:
            copied.append(new)
    day = today()
    connection.executemany(
        "INSERT INTO used (text_sha256, day) VALUES (?, ?) " + _STAMP_UPSERT_CONFLICT, [(new, day) for new in copied]
    )
    return copied


def read_rows(path: Path, batch_size: int = 4096) -> Iterator[list[tuple[str, int, bytes]]]:
    """Yield every vector of a cache file as ``(text hash, dims, raw float32 bytes)``, a batch at a time.

    The file is opened read-only, so a cache from anywhere (another machine's, a backup) can be
    read without being converted, and each batch is its own short read keyed on the rowid, so a
    long upload never pins a WAL snapshot a concurrent buyer's write would have to wait behind.

    :param path: The sqlite file.
    :param batch_size: Rows per batch.
    :yield: One batch of rows.
    :ytype: list[tuple[str, int, bytes]]
    """
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=BUSY_TIMEOUT_SECONDS)
    try:
        last = 0
        while True:
            found = connection.execute(
                "SELECT rowid, text_sha256, dims, vec FROM embeddings WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last, batch_size),
            ).fetchall()
            if not found:
                break
            last = int(found[-1][0])
            yield [(str(row[1]), int(row[2]), bytes(row[3])) for row in found]
    finally:
        connection.close()


def stamps_since(connection: sqlite3.Connection) -> int | None:
    """Return the day use stamps began in this file, or None before any zemble that writes them opened it."""
    try:
        row = connection.execute("SELECT value FROM cache_meta WHERE key = ?", (STAMPS_SINCE_KEY,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row[0]) if row is not None else None


def _install_count(connection: sqlite3.Connection) -> None:
    """Give a file its kept vector count: the tracking row and its triggers in one transaction."""
    installed = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?", (_COUNT_INSERT_TRIGGER,)
    ).fetchone()
    if installed is not None:
        return
    with _transaction(connection):
        connection.execute("INSERT OR IGNORE INTO cache_meta (key, value) VALUES (?, '0')", (VECTORS_TRACKED_KEY,))
        for trigger in _COUNT_TRIGGERS:
            connection.execute(trigger)
        if connection.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone() is None:
            connection.execute("INSERT OR IGNORE INTO cache_meta (key, value) VALUES (?, '0')", (VECTORS_BASE_KEY,))


def _kept_count(connection: sqlite3.Connection) -> int | None:
    """Return the count a file keeps of its vectors, or None while its first count has not been recorded."""
    rows = dict(
        connection.execute(
            "SELECT key, value FROM cache_meta WHERE key IN (?, ?)", (VECTORS_TRACKED_KEY, VECTORS_BASE_KEY)
        ).fetchall()
    )
    if VECTORS_TRACKED_KEY not in rows or VECTORS_BASE_KEY not in rows:
        return None
    return int(rows[VECTORS_TRACKED_KEY]) + int(rows[VECTORS_BASE_KEY])


def settle_count(path: Path) -> None:
    """Record what a file's kept count starts from, scanning it once on a connection of its own.

    The scan and the tracked change are read in one snapshot, and their difference is the same in
    every snapshot, so a writer between the scan and the record, or a second settler, cannot make it
    wrong. WAL readers block no writer; the WAL only cannot be checkpointed past the scan meanwhile.
    """
    connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=BUSY_TIMEOUT_SECONDS)
    try:
        with _transaction(connection, immediate=False):
            counted = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            tracked = connection.execute(
                "SELECT value FROM cache_meta WHERE key = ?", (VECTORS_TRACKED_KEY,)
            ).fetchone()
        if tracked is None:
            return
        with _transaction(connection):
            connection.execute(
                "INSERT OR IGNORE INTO cache_meta (key, value) VALUES (?, ?)",
                (VECTORS_BASE_KEY, str(counted - int(tracked[0]))),
            )
    finally:
        connection.close()


_settling: dict[Path, threading.Thread] = {}
_settling_lock = threading.Lock()


def _settle_in_background(path: Path) -> None:
    """Start the one-time first count of a file on a thread, unless one already runs in this process."""

    def settle() -> None:
        try:
            settle_count(path)
        except sqlite3.Error:
            logger.warning("counting the vectors of %s failed; the next status retries", path, exc_info=True)

    with _settling_lock:
        running = _settling.get(path)
        if running is not None and running.is_alive():
            return
        thread = threading.Thread(target=settle, name=f"zemble-count-{path.name}", daemon=True)
        _settling[path] = thread
        thread.start()


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
            self._load_wanted(digests)
            # EXISTS pins the probe direction: as a JOIN, sqlite chose to scan the whole embeddings
            # key for every request, 1.6 s per call on a server holding millions of vectors.
            rows = self._connection.execute(
                "SELECT w.digest FROM wanted w WHERE EXISTS "
                "(SELECT 1 FROM embeddings e WHERE e.text_sha256 = w.digest AND e.dims >= ?)",
                (dims,),
            ).fetchall()
        return {row[0] for row in rows}

    def _load_wanted(self, digests: Iterable[str]) -> None:
        """Replace the temp table `wanted` with these digests; the caller holds the lock."""
        # Group temporary writes, then release their transaction BEFORE reading the main DB.
        # Otherwise a lookup pins a WAL snapshot until a later paid embedding tries to write.
        with _transaction(self._connection, immediate=False):
            self._connection.execute("CREATE TEMP TABLE IF NOT EXISTS wanted (digest TEXT PRIMARY KEY)")
            self._connection.execute("DELETE FROM wanted")
            self._connection.executemany(
                "INSERT OR IGNORE INTO wanted (digest) VALUES (?)", [(digest,) for digest in digests]
            )

    def _held_wanted(self) -> list[tuple[str, int]]:
        """Return each `wanted` digest this file holds at any width, with 1 when its stamp is still fresh."""
        return self._connection.execute(
            "SELECT w.digest, EXISTS (SELECT 1 FROM used u WHERE u.text_sha256 = w.digest AND u.day > ?) "
            "FROM wanted w WHERE EXISTS (SELECT 1 FROM embeddings e WHERE e.text_sha256 = w.digest)",
            (today() - STAMP_REFRESH_DAYS,),
        ).fetchall()

    def _stamp_stale(self, digests: list[str]) -> None:
        """Stamp today held digests a read found without a fresh stamp, a few hundred per write transaction.

        Nothing is written, and the writer never taken, when the read found every stamp fresh. Between
        two small transactions a buyer's `put_many` gets the writer. The upsert re-checks that each is
        held, so a vector deleted since the read gets no stamp.
        """
        day = today()
        for start in range(0, len(digests), STAMPS_PER_TRANSACTION):
            with _transaction(self._connection):
                self._connection.executemany(
                    "INSERT INTO used (text_sha256, day) SELECT ?1, ?2 "
                    "WHERE EXISTS (SELECT 1 FROM embeddings WHERE text_sha256 = ?1) " + _STAMP_UPSERT_CONFLICT,
                    [(digest, day) for digest in digests[start : start + STAMPS_PER_TRANSACTION]],
                )

    def stamp_held(self, digests: Iterable[str]) -> int:
        """Stamp today the vectors this file holds for these text hashes, at any width, and count them.

        :param digests: Text hashes a client's saved indexes reference; unknown ones are ignored.
        :return: How many of them this file holds.
        """
        with self._lock:
            self._load_wanted(digests)
            hits = self._held_wanted()
            self._stamp_stale([digest for digest, stamped_today in hits if not stamped_today])
        return len(hits)

    def covered_keys(self, keys: list[str], legacy: list[str | None], dims: int) -> set[str]:
        """Return which keys have a usable vector at this width, under the key or, not migrated yet, its legacy key.

        :param keys: The cache keys to look up.
        :param legacy: Each key's legacy key (`zemble.embedding.keys.legacy_key`), None where there is none.
        :param dims: The requested width.
        :return: The covered subset of `keys`.
        """
        hits = self.covered(keys, dims)
        unmigrated = [(old, key) for key, old in zip(keys, legacy, strict=True) if old is not None and key not in hits]
        if unmigrated:
            held = self.covered([old for old, _key in unmigrated], dims)
            hits.update(key for old, key in unmigrated if old in held)
        return hits

    def covered_texts(self, texts: Sequence[str], dims: int) -> set[str]:
        """Return the cache keys of these texts that have a usable vector at this width (:meth:`covered_keys`)."""
        return self.covered_keys([cache_key(text) for text in texts], [legacy_key(text) for text in texts], dims)

    def rekey(self, pairs: Iterable[tuple[str, str]]) -> int:
        """Copy each old key's vectors to its new key where that holds none, a few hundred per transaction.

        :param pairs: ``(old key, new key)`` pairs; an old key this file never stored is ignored.
        :return: How many new keys gained a vector.
        """
        pending = list(pairs)
        copied = 0
        for start in range(0, len(pending), STAMPS_PER_TRANSACTION):
            with self._lock, _transaction(self._connection):
                moved = rekey_rows(self._connection, pending[start : start + STAMPS_PER_TRANSACTION])
            copied += len(moved)
        return copied

    def adopt(self, pairs: list[tuple[str, str]], dims: int) -> dict[str, np.ndarray]:
        """Return, by new key, the vectors held under these pairs' old keys at this width, copied to the new keys.

        :param pairs: ``(old key, new key)`` pairs of texts no vector is stored for under the new key.
        :param dims: The requested width.
        :return: New key to vector, for every old key held.
        """
        found: dict[str, np.ndarray] = {}
        moved: list[tuple[str, str]] = []
        for old, new in pairs:
            if new in found:
                continue
            vector = self.get(old, dims)
            if vector is not None:
                found[new] = vector
                moved.append((old, new))
        if moved:
            self.rekey(moved)
        return found

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

        Read through :func:`uncovered_texts`, so a report and a build cannot disagree about it:
        duplicates collapse, and a stored vector was bought already.

        :param texts: The texts a build is about to embed.
        :param dims: The width a stored vector has to serve; None names no usable width, so
            nothing counts as stored - never a discount nobody can prove.
        :return: Those still to be bought, first occurrence first.
        """
        digests = [cache_key(text) for text in texts]
        legacy = [legacy_key(text) for text in texts]
        covered = self.covered_keys(digests, legacy, dims) if dims is not None else set()
        return uncovered_texts(texts, digests, covered)

    def count(self) -> int:
        """Return how many vectors this file stores, at every width, scanning it here if it was never counted."""
        with self._lock:
            kept = _kept_count(self._connection)
        if kept is not None:
            return kept
        settle_count(self.path)
        with self._lock:
            kept = _kept_count(self._connection)
        if kept is None:
            raise sqlite3.DatabaseError(f"{self.path} keeps no vector count after counting it")
        return kept

    def stored_count(self) -> int | None:
        """Return how many vectors this file stores without scanning it, or None while its first count runs.

        A None starts that count in the background; it runs once per file, ever.
        """
        with self._lock:
            kept = _kept_count(self._connection)
        if kept is None:
            _settle_in_background(self.path)
        return kept

    def put_missing(self, rows: list[tuple[str, int, bytes]]) -> int:
        """Store vectors another cache already bought, keeping any this file holds, and stamp them.

        :param rows: ``(text hash, dims, raw float32 bytes)`` triples.
        :return: How many rows were new to this file.
        """
        if not rows:
            return 0
        with self._lock, _transaction(self._connection):
            # The cursor counts the rows inserted; `total_changes` would add the count triggers' updates.
            added = self._connection.executemany(
                "INSERT OR IGNORE INTO embeddings (text_sha256, dims, vec) VALUES (?, ?, ?)", rows
            ).rowcount
            self._stamp({digest for digest, _dims, _vector in rows})
        return added

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
        if distinct:
            self.stamp_held(distinct)

    def _stamp(self, digests: set[str]) -> None:
        """Stamp these hashes inside the caller's write transaction: a new one today, a fresh one left unwritten."""
        day = today()
        self._connection.executemany(
            "INSERT INTO used (text_sha256, day) VALUES (?, ?) " + _STAMP_UPSERT_CONFLICT,
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
    def family(self) -> str:
        """The cache family key (scheme plus model, no dimensions) the price table is keyed by."""
        return self.cache.family

    @property
    def store_location(self) -> str:
        """Where the vectors this embedder reuses are kept, for a report."""
        return str(self.cache.path)

    def stored_keys(self, texts: Sequence[str]) -> set[str]:
        """Return the cache keys of these texts that already have a usable vector, asking no provider anything.

        :param texts: The texts to look up.
        :return: The covered keys; empty when no width can be read without a provider probe.
        """
        width = self.known_dimensions
        return self.cache.covered_texts(texts, width) if width is not None else set()

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

        digests = [cache_key(text) for text in texts]
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

        missing_positions = self._adopt_legacy(texts, digests, missing_positions, result)

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

    def _adopt_legacy(
        self, texts: list[str], digests: list[str], positions: list[int], result: EmbeddingMatrix
    ) -> list[int]:
        """Fill the rows a text first bought before keys left its path out from that purchase; return the rest."""
        adopted = self.cache.adopt(
            [(old, digests[position]) for position in positions if (old := legacy_key(texts[position]))],
            result.shape[1],
        )
        for position in positions:
            if digests[position] in adopted:
                result[position] = adopted[digests[position]]
        return [position for position in positions if digests[position] not in adopted]

    def embed_queries(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed queries without touching the cache.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        return self.inner.embed_queries(texts)
