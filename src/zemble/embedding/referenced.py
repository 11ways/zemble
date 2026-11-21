"""Tell the embedding server which of its vectors this machine's saved indexes still reference.

An index that is only ever loaded never asks the server for its vectors again, so without this
report a vector in daily use ages on the server like a dead one and its `gc` sweeps it. Stamps
have day granularity, so each family is reported at most once a day per server.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path

import numpy as np
import numpy.typing as npt

from zemble.embedding.cache import today
from zemble.embedding.gc import index_digests, saved_indexes
from zemble.embedding.http import EmbeddingRequestError
from zemble.embedding.registry import cached_family
from zemble.embedding.served import ServerClient
from zemble.embedding.wire import REFERENCED_DIGESTS_PER_REQUEST

logger = logging.getLogger(__name__)

#: The file in the client cache folder recording, per server and family, the last day reported.
REPORTED_FILE = "embed-server-referenced.json"
#: The skip reason of a family whose report already went out today.
REPORTED_TODAY = "reported today already"
#: The lock one report on this host holds, so the daemon's loop and a hand-run report never overlap.
REPORT_LOCK_FILE = "embed-server-referenced.lock"


class ReportRunning(RuntimeError):
    """Another referenced report holds this host's report lock; this one skips."""


@dataclass
class FamilyReport:
    """What one family's report sent, or why it sent nothing."""

    family: str
    indexes: int
    sent: int = 0
    held: int = 0
    unreadable: int = 0
    #: Why nothing was sent today, or None when the report ran.
    skipped: str | None = None


def report_referenced(cache_folder: Path, client: ServerClient, *, force: bool = False) -> list[FamilyReport]:
    """Send the server the text hashes of every paid family's saved indexes, each family once a day.

    A day is recorded only for a family the server answered for, a refusal that holds all day
    included (a family it does not serve, a server without the route); anything else is retried
    on the next call.

    :param cache_folder: The zemble cache folder.
    :param client: The configured server.
    :param force: Report a family again although it was reported today.
    :return: One report per family, in name order.
    :raises EmbeddingRequestError: If the server is unreachable or refuses in a way a day does not settle.
    :raises ReportRunning: If another report on this host is running.
    """
    cache_folder.mkdir(parents=True, exist_ok=True)
    with (cache_folder / REPORT_LOCK_FILE).open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ReportRunning(
                "another referenced report is running on this host (the daemon or `zemble embed-server report`)"
            ) from None
        return _report_locked(cache_folder, client, force=force)


def _report_locked(cache_folder: Path, client: ServerClient, *, force: bool) -> list[FamilyReport]:
    """Run the report under this host's report lock; each family's day is saved as soon as it is done."""
    families: dict[str, tuple[str, list[Path]]] = {}
    for spec, folder in saved_indexes(cache_folder):
        family = cached_family(spec)
        if family is not None:
            families.setdefault(family, (spec, []))[1].append(folder)
    state_path = cache_folder / REPORTED_FILE
    state = _read_state(state_path)
    days = state.setdefault(client.settings.url, {})
    day = today()
    reports: list[FamilyReport] = []
    old_server = False
    for family, (spec, folders) in sorted(families.items()):
        report = FamilyReport(family=family, indexes=len(folders))
        reports.append(report)
        if not force and days.get(family) == day:
            report.skipped = REPORTED_TODAY
            continue
        if old_server:
            report.skipped = "the server predates this report; retried tomorrow"
        else:
            try:
                for batch in distinct_batches(folders, REFERENCED_DIGESTS_PER_REQUEST, report):
                    report.held += client.referenced(spec, batch)
                    report.sent += len(batch)
            except EmbeddingRequestError as exc:
                if exc.status == HTTPStatus.NOT_FOUND:
                    logger.info(
                        "%s has no referenced-vectors route (an older zemble); retried tomorrow", client.settings.url
                    )
                    old_server = True
                    report.skipped = "the server predates this report; retried tomorrow"
                elif exc.status == HTTPStatus.FORBIDDEN:
                    report.skipped = f"refused: {exc}"
                else:
                    raise
        days[family] = day
        _write_state(state_path, state)
    return reports


def distinct_batches(folders: Iterable[Path], size: int, report: FamilyReport) -> Iterator[list[str]]:
    """Yield the text hashes of these indexes in batches of `size`, a hash an earlier index held never again.

    The roots and lanes of one workspace share most of their chunks, so the hashes already sent are
    kept, as 32 raw bytes each in one sorted array rather than as a set of strings. A hash repeated
    inside one index may go twice; the server's stamp is idempotent. An index that cannot be read
    is counted in `report.unreadable` and passed over.
    """
    sent: npt.NDArray[np.bytes_] = np.empty(0, dtype="S32")
    pending: list[str] = []
    for folder in folders:
        fresh: list[npt.NDArray[np.bytes_]] = []
        chunk: list[str] = []
        try:
            for digest in index_digests(folder):
                chunk.append(digest)
                if len(chunk) == size:
                    pending.extend(_unsent(chunk, sent, fresh))
                    chunk = []
                    while len(pending) >= size:
                        yield pending[:size]
                        del pending[:size]
            pending.extend(_unsent(chunk, sent, fresh))
        except (OSError, ValueError) as exc:
            logger.warning("skipping unreadable index %s in the referenced report: %s", folder, exc)
            report.unreadable += 1
        if fresh:
            # The index's new keys are unique and absent from `sent`, so one insert keeps it sorted.
            new = np.unique(np.concatenate(fresh))
            sent = np.insert(sent, np.searchsorted(sent, new), new)
        while len(pending) >= size:
            yield pending[:size]
            del pending[:size]
    if pending:
        yield pending


def _unsent(digests: list[str], sent: npt.NDArray[np.bytes_], fresh: list[npt.NDArray[np.bytes_]]) -> list[str]:
    """Return the digests of one slice not in `sent` nor repeated in it, in order, and add their keys to `fresh`."""
    if not digests:
        return []
    keys, first = np.unique(np.frombuffer(bytes.fromhex("".join(digests)), dtype="S32"), return_index=True)
    if len(sent):
        position = np.minimum(np.searchsorted(sent, keys), len(sent) - 1)
        unseen = sent[position] != keys
        keys, first = keys[unseen], first[unseen]
    fresh.append(keys)
    return [digests[index] for index in sorted(first)]


def _read_state(path: Path) -> dict[str, dict[str, int]]:
    """Read the reported days; a missing or unreadable file means nothing was reported."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {
        url: {family: day for family, day in days.items() if isinstance(day, int)}
        for url, days in loaded.items()
        if isinstance(days, dict)
    }


def _write_state(path: Path, state: dict[str, dict[str, int]]) -> None:
    """Replace the reported days in one rename, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    staging.replace(path)


__all__ = [
    "REPORTED_FILE",
    "REPORTED_TODAY",
    "REPORT_LOCK_FILE",
    "FamilyReport",
    "ReportRunning",
    "distinct_batches",
    "report_referenced",
]
