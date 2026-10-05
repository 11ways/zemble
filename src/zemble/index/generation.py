"""Index generations: a build writes beside the live stores, then publishes them file by file."""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import orjson

from zemble.chunking.capsule import CapsuleOptions
from zemble.embedding.base import Embedder
from zemble.index.types import CACHE_FORMAT_VERSION, FileManifestEntry, PersistencePath
from zemble.types import ContentType

#: The lock a publish holds exclusively and a load holds shared, in the variant folder itself.
LOCK_NAME = "index.lock"
#: Prefix of a staging folder, `.staging-<variant>-<pid>-<random>`, beside the variant it replaces.
STAGING_PREFIX = ".staging-"


def staging_for(final: Path) -> Path:
    """Create a private folder beside *final* to write its next generation into.

    It lives in the same folder as *final* so publishing is a rename, never a copy; the pid in its
    name lets `zemble clear orphans` tell a crashed build's leftovers from a running one's.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{STAGING_PREFIX}{final.name}-{os.getpid()}-", dir=final.parent))


@contextmanager
def _locked(final: Path, kind: int) -> Iterator[None]:
    """Hold the variant's publish lock; a folder that cannot hold a lock file is read unlocked."""
    try:
        handle = open(final / LOCK_NAME, "a")  # noqa: SIM115 - closed below
    except OSError:
        yield
        return
    try:
        fcntl.flock(handle.fileno(), kind)
        yield
    finally:
        handle.close()


@contextmanager
def reading(final: Path) -> Iterator[None]:
    """Hold off a publish while the stores of *final* are opened, so one load never mixes two generations."""
    with _locked(final, fcntl.LOCK_SH):
        yield


def publish(staging: Path, final: Path) -> None:
    """Move a staged generation into *final*, metadata last, and remove the staging folder.

    Each file replaces its predecessor by rename, so a process that has the previous generation
    mapped keeps reading exactly that; a loader opening *final* meanwhile waits on the lock.
    """
    final.mkdir(parents=True, exist_ok=True)
    metadata = PersistencePath.from_path(staging).metadata
    with _locked(final, fcntl.LOCK_EX):
        for source in sorted(path for path in staging.rglob("*") if path.is_file() and path != metadata):
            target = final / source.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
        os.replace(metadata, PersistencePath.from_path(final).metadata)
    shutil.rmtree(staging, ignore_errors=True)


def write_metadata(
    directory: Path,
    *,
    root: Path | None,
    embedder: Embedder,
    content: Sequence[ContentType],
    capsules: CapsuleOptions,
    exclude: Sequence[str],
    manifest: dict[str, FileManifestEntry],
) -> None:
    """Write the metadata that makes a staged generation a complete, loadable index."""
    from zemble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS  # avoid circular import at module level

    metadata = {
        "root_path": None if root is None else str(root),
        "time": datetime.now().timestamp(),
        "embedder": embedder.model_id,
        "dimensions": embedder.dimensions,
        "content_type": [content_type.value for content_type in content],
        "chunk_size": _DESIRED_CHUNK_LENGTH_CHARS,
        "cache_version": CACHE_FORMAT_VERSION,
        "capsules": capsules.key,
        "exclude": list(exclude),
        "files": manifest,
    }
    PersistencePath.from_path(directory).metadata.write_bytes(orjson.dumps(metadata))
