"""The per-root unit index a focused scan reads: every file's unit hashes and bodies, keyed by content.

A focused run (`zemble dupes --focus`) compares a handful of files against the whole workspace,
so it needs every OTHER file's units without parsing them again. Statement windows dominate a
workspace's units (1.76M units, 57k bodies on the javaweb workspace, 2026-10-09) and pickled they
cost 1.4 GB and 17 s to load, so a row keeps only what the comparison reads: both stream hashes of
every unit, and the whole-body units (with their text) that logic mode embeds and checks. A file
whose hashes meet a focus hash is parsed again for its full units.

AIDEV-NOTE: a row is valid for exactly one file content AND one extraction: the signature folds in
the extraction options, the source of every module that decides a unit, and the grammar package
versions, so an edited extractor or a grammar upgrade can never serve units it would not produce.
"""

from __future__ import annotations

import functools
import hashlib
import json
import pickle
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from zemble.dedup.model import Unit
from zemble.embedding.cache import today

#: The index file inside the root's cache folder.
INDEX_FILE_NAME = "dupes-units.sqlite"
#: The folder, beside the index, holding the local mirror of the body vectors logic mode reads.
VECTOR_FOLDER_NAME = "dupes-vectors"
#: Bytes one unit's hashes take in a row: the exact and the alpha-renamed digest, 16 bytes each.
HASH_BYTES = 32
#: Packages whose version decides what a grammar parses a file into.
_GRAMMAR_PACKAGES = ("tree-sitter", "semble-grammars")
#: Days the rows of an extraction nobody ran (other options, older code) are kept.
UNUSED_SIGNATURE_DAYS = 7
#: Seconds a writer waits for another process's transaction.
_BUSY_TIMEOUT_SECONDS = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT NOT NULL,
    signature TEXT NOT NULL,
    digest TEXT NOT NULL,
    failed INTEGER NOT NULL,
    hashes BLOB NOT NULL,
    bodies BLOB NOT NULL,
    PRIMARY KEY (path, signature)
);
CREATE TABLE IF NOT EXISTS signatures (signature TEXT PRIMARY KEY, day INTEGER NOT NULL);
"""


@dataclass(frozen=True, slots=True)
class FileRow:
    """What the index holds for one file: its content digest, unit hashes and whole-body units."""

    path: str
    digest: str
    failed: bool
    #: `HASH_BYTES` per unit, in extraction order: exact digest then alpha-renamed digest.
    hashes: bytes = b""
    bodies: tuple[Unit, ...] = ()

    @property
    def unit_count(self) -> int:
        """How many units (bodies and windows) the file produced."""
        return len(self.hashes) // HASH_BYTES

    @classmethod
    def of(cls, path: str, digest: str, units: Sequence[Unit] | None) -> FileRow:
        """Build the row of one extracted file; None units mean the extraction failed."""
        if units is None:
            return cls(path=path, digest=digest, failed=True)
        hashes = b"".join(bytes.fromhex(unit.exact_hash) + bytes.fromhex(unit.renamed_hash) for unit in units)
        return cls(
            path=path,
            digest=digest,
            failed=False,
            hashes=hashes,
            bodies=tuple(unit for unit in units if unit.is_body),
        )


@functools.cache
def _code_fingerprint() -> str:
    """Hash the dedup and language modules plus the grammar package versions, once per process."""
    package = Path(__file__).resolve().parent.parent
    sources = sorted(
        [*(package / "dedup").rglob("*.py"), *(package / "languages").rglob("*.py")],
        key=lambda path: path.as_posix(),
    )
    digest = hashlib.blake2b(digest_size=16)
    for source in sources:
        digest.update(source.relative_to(package).as_posix().encode())
        digest.update(source.read_bytes())
    for name in _GRAMMAR_PACKAGES:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "absent"
        digest.update(f"{name}={version}".encode())
    return digest.hexdigest()


def extraction_signature(extract_options: Mapping[str, object]) -> str:
    """The key part naming one extraction: its options plus the code that runs it.

    :param extract_options: The keyword arguments `extract_units` is called with.
    :return: A short hex digest.
    """
    payload = json.dumps(dict(extract_options), sort_keys=True) + _code_fingerprint()
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def content_digest(data: bytes) -> str:
    """The digest a row is valid for."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def root_cache_folder(root: Path) -> Path:
    """The cache folder of a root: the one its search index and graph live in, gone with the root."""
    from zemble.cache import cache_key, resolve_cache_folder

    return resolve_cache_folder() / cache_key(str(root))


class UnitIndex:
    """The sqlite file of one root's rows, for one extraction signature."""

    def __init__(self, root: Path, signature: str) -> None:
        """Open (creating if needed) the index of a root."""
        folder = root_cache_folder(root)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / INDEX_FILE_NAME
        self.signature = signature
        self._connection = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_SECONDS)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO signatures (signature, day) VALUES (?, ?)", (signature, today())
            )

    def close(self) -> None:
        """Close the connection."""
        self._connection.close()

    def load(self, wanted: Mapping[str, str]) -> dict[str, FileRow]:
        """Return the rows whose path and content digest both match.

        :param wanted: Content digest by root-relative path.
        :return: The matching rows by path; a changed or unknown file is simply absent.
        """
        found: dict[str, FileRow] = {}
        cursor = self._connection.execute(
            "SELECT path, digest, failed, hashes, bodies FROM files WHERE signature = ?", (self.signature,)
        )
        for path, digest, failed, hashes, bodies in cursor:
            if wanted.get(path) != digest:
                continue
            found[path] = FileRow(
                path=path, digest=digest, failed=bool(failed), hashes=hashes, bodies=tuple(pickle.loads(bodies))
            )
        return found

    def store(self, rows: Iterable[FileRow]) -> None:
        """Write rows, replacing each path's previous one."""
        payload = [
            (row.path, self.signature, row.digest, int(row.failed), row.hashes, pickle.dumps(list(row.bodies)))
            for row in rows
        ]
        if not payload:
            return
        with self._connection:
            self._connection.executemany(
                "INSERT OR REPLACE INTO files (path, signature, digest, failed, hashes, bodies)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                payload,
            )

    def retain(self, paths: Iterable[str]) -> None:
        """Drop this signature's rows of every file a whole-root walk no longer finds, and unused extractions."""
        with self._connection:
            self._connection.execute("CREATE TEMP TABLE IF NOT EXISTS walked (path TEXT PRIMARY KEY)")
            self._connection.execute("DELETE FROM walked")
            self._connection.executemany("INSERT OR IGNORE INTO walked (path) VALUES (?)", ((path,) for path in paths))
            self._connection.execute(
                "DELETE FROM files WHERE signature = ? AND path NOT IN (SELECT path FROM walked)", (self.signature,)
            )
            stale = today() - UNUSED_SIGNATURE_DAYS
            self._connection.execute(
                "DELETE FROM files WHERE signature IN (SELECT signature FROM signatures WHERE day < ?)", (stale,)
            )
            self._connection.execute("DELETE FROM signatures WHERE day < ?", (stale,))


__all__ = [
    "HASH_BYTES",
    "INDEX_FILE_NAME",
    "VECTOR_FOLDER_NAME",
    "FileRow",
    "UnitIndex",
    "content_digest",
    "extraction_signature",
    "root_cache_folder",
]
