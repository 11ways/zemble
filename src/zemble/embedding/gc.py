"""Garbage collection of the shared embedding cache: keep what an index references or what was used lately."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from zemble.chunking.capsule import embedding_text
from zemble.embedding.cache import connect_cache, family_slug, stamps_since, text_hash, today
from zemble.embedding.registry import cached_family
from zemble.index.chunk_store import load_chunks
from zemble.openfiles import holders

#: Days a vector stays after its last use even when no saved index references it. It covers
#: what only ever lived in memory: benchmark variants, `dupes --kind logic` bodies, a build
#: whose index is not saved yet.
DEFAULT_GRACE_DAYS = 14

#: Rough per-row cost of the key, the row header and the `used` stamp beside the vector bytes.
_ROW_OVERHEAD_BYTES = 100


@dataclass
class EmbeddingSweep:
    """What a collection found, or did, in one embedder family's cache file."""

    path: Path
    rows: int = 0
    referenced: int = 0
    swept: int = 0
    swept_bytes: int = 0
    #: Free pages and WAL bytes a VACUUM and a truncating checkpoint return, swept rows aside.
    slack_bytes: int = 0
    size_before: int = 0
    size_after: int | None = None
    stamps_since: int | None = None
    #: The newest day of last use that is still swept.
    cutoff_day: int = 0
    indexes: int = 0
    refused: str | None = None
    holders: list[int] = field(default_factory=list)


def day_text(day: int) -> str:
    """Render a use-stamp day number as an ISO date."""
    return (date(1970, 1, 1) + timedelta(days=day)).isoformat()


def file_size(path: Path) -> int:
    """Return the bytes a sqlite file and its WAL take, zero for what is missing."""
    total = 0
    for candidate in (path, path.with_name(path.name + "-wal")):
        try:
            total += candidate.stat().st_size
        except FileNotFoundError:
            pass
    return total


def indexes_by_family(cache_folder: Path) -> dict[str, list[Path]]:
    """Group every saved index folder by the slug of the embedding-cache file its embedder reads.

    :return: Cache-file slug to index folders; an index whose embedder is never cached is absent.
    """
    grouped: dict[str, list[Path]] = {}
    for metadata_path in sorted(cache_folder.glob("*/index*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        embedder = metadata.get("embedder") if isinstance(metadata, dict) else None
        family = cached_family(embedder) if isinstance(embedder, str) else None
        if family is not None:
            grouped.setdefault(family_slug(family), []).append(metadata_path.parent)
    return grouped


def referenced_digests(index_folders: Iterable[Path]) -> set[str]:
    """Return the cache key of every chunk any of these indexes embedded.

    The key is recomputed from the stored chunk exactly as a build computed it
    (`embedding_text`, then `text_hash`), so the two can never disagree.

    :raises OSError, ValueError: If an index cannot be read; a collection then refuses, because a
        reference it cannot see would be swept.
    """
    digests: set[str] = set()
    for folder in index_folders:
        for chunk in load_chunks(folder / "chunks"):
            digests.add(text_hash(embedding_text(chunk)))
    return digests


def collect_embeddings(cache_folder: Path, *, grace_days: int, dry_run: bool) -> list[EmbeddingSweep]:
    """Sweep every embedding-cache file of the vectors no index references and nobody used lately, then VACUUM.

    A file no index uses at all is left alone here: deleting it whole is `zemble clear orphans`'s
    job. A file another process holds open is refused, because VACUUM needs it to itself and a
    process mid-build would buy back what was swept.

    :param cache_folder: The zemble cache folder.
    :param grace_days: Keep an unreferenced vector used (stored or served) within this many days; 0 keeps none.
    :param dry_run: Report what would go, writing nothing.
    :return: One report per cache file, in name order.
    """
    directory = cache_folder / "embeddings"
    users = indexes_by_family(cache_folder)
    reports: list[EmbeddingSweep] = []
    for path in sorted(directory.glob("*.sqlite")) if directory.is_dir() else []:
        report = EmbeddingSweep(path=path, size_before=file_size(path), cutoff_day=today() - grace_days)
        reports.append(report)
        folders = users.get(path.stem, [])
        report.indexes = len(folders)
        if not folders:
            report.refused = "no index uses this embedder; `zemble clear orphans` removes the whole file"
            continue
        found = holders(path, exclude_self=True)
        if found is None:
            report.refused = "cannot tell which processes have it open (no /proc)"
            continue
        report.holders = found
        if found and not dry_run:
            report.refused = f"held open by pid {', '.join(map(str, found))}; stop them (`zemble daemon stop`) first"
            continue
        try:
            marked = referenced_digests(folders)
        except (OSError, ValueError) as exc:
            report.refused = f"an index of this embedder is unreadable ({exc}); nothing swept"
            continue
        _sweep_file(report, marked, dry_run=dry_run)
    return reports


def _sweep_file(report: EmbeddingSweep, marked: set[str], *, dry_run: bool) -> None:
    """Count, and unless dry, delete and VACUUM, the rows of one file that are neither marked nor recent."""
    if dry_run:
        connection = sqlite3.connect(f"{report.path.resolve().as_uri()}?mode=ro", uri=True)
    else:
        connection = connect_cache(report.path)
    try:
        has_stamps = _has_table(connection, "used")
        # A vector stored before stamps existed counts as used the day they began, never as
        # unused forever: an age nobody recorded fails closed.
        report.stamps_since = stamps_since(connection) if has_stamps else None
        unstamped_day = report.stamps_since if report.stamps_since is not None else today()
        connection.execute("CREATE TEMP TABLE keep (digest TEXT PRIMARY KEY) WITHOUT ROWID")
        connection.executemany("INSERT OR IGNORE INTO keep (digest) VALUES (?)", ((digest,) for digest in marked))
        last_used = "COALESCE(u.day, :unstamped)" if has_stamps else ":unstamped"
        stamp_join = "LEFT JOIN used u ON u.text_sha256 = e.text_sha256" if has_stamps else ""
        connection.execute(
            f"""
            CREATE TEMP TABLE doomed AS
            SELECT e.rowid AS rid, length(e.vec) AS size FROM embeddings e
            LEFT JOIN keep k ON k.digest = e.text_sha256
            {stamp_join}
            WHERE k.digest IS NULL AND {last_used} <= :cutoff
            """,  # noqa: S608 - both fragments are fixed strings chosen above
            {"unstamped": unstamped_day, "cutoff": report.cutoff_day},
        )
        report.rows = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        report.referenced = connection.execute(
            "SELECT COUNT(*) FROM embeddings e JOIN keep k ON k.digest = e.text_sha256"
        ).fetchone()[0]
        count, size = connection.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM doomed").fetchone()
        report.swept = count
        report.swept_bytes = size + count * _ROW_OVERHEAD_BYTES
        pages = connection.execute("PRAGMA freelist_count").fetchone()[0]
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        wal = report.path.with_name(report.path.name + "-wal")
        report.slack_bytes = pages * page_size + (wal.stat().st_size if wal.exists() else 0)
        if dry_run:
            return
        connection.execute("DELETE FROM embeddings WHERE rowid IN (SELECT rid FROM doomed)")
        connection.execute("DELETE FROM used WHERE text_sha256 NOT IN (SELECT text_sha256 FROM embeddings)")
        connection.commit()
        connection.execute("DROP TABLE doomed")
        connection.execute("DROP TABLE keep")
        connection.execute("VACUUM")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    report.size_after = file_size(report.path)


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    """Return whether the main database of a connection has a table."""
    row = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone()
    return row is not None
