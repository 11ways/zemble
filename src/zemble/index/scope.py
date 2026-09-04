"""Refuse an accidental aggregation root, or one whose sheer volume is runaway work.

Both refusals happen before a single file is parsed: the expensive half of a build is
chunking, and a tree big enough to be refused is a tree big enough for that to take minutes.
Neither of them is about money: what a build costs is decided once the uncached set is known,
by the budget guard in :mod:`zemble.embedding.pricing`.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from zemble.embedding.base import Embedder
from zemble.embedding.pricing import CONFIRM_ENV, confirmed, embedder_family, remedies
from zemble.index.file_walker import _DEFAULT_IGNORED_DIRS, walk_entries
from zemble.index.files import MAX_FILE_BYTES, get_extensions
from zemble.refusal import Refused
from zemble.types import ContentType
from zemble.workspace import HOME_CONFIG_RELATIVE_PATH

#: An undeclared directory containing this many repositories is probably a parent of workspaces.
MAX_UNDECLARED_REPOSITORIES = 8

#: How many of the root's immediate children a refusal names, largest first.
BREAKDOWN_LIMIT = 8

#: Names the ceiling, in megabytes, on the source volume one build may chunk.
WORK_LIMIT_ENV = "ZEMBLE_INDEX_WORK_LIMIT_MB"

#: Where runaway WORK begins: ~180 MB of source, the figure the old local-lane ceiling
#: encoded as 50M tokens. A real multi-repo workspace (javaweb: 64 MB of code) is normal
#: work; a tree carrying thirteen copies of itself is not.
DEFAULT_WORK_LIMIT_BYTES = 180_000_000

_IGNORED_DIRECTORY_NAMES = frozenset(pattern.removesuffix("/") for pattern in _DEFAULT_IGNORED_DIRS)


class ScopeRefused(Refused):
    """A deterministic refusal to build an index over a root, decided before anything is parsed."""

    DEFAULT_KNOB = CONFIRM_ENV


class BroadRootRefused(ScopeRefused):
    """A local root spans too many repositories to index without an explicit declaration."""


class OversizedRootRefused(ScopeRefused):
    """A root holds more source than one build may chunk, so nothing was parsed."""


@dataclass(frozen=True)
class DirectoryWeight:
    """What one immediate child of the root contributes to a build."""

    name: str
    files: int
    bytes: int


@dataclass(frozen=True)
class TreeEstimate:
    """What a full build over a root would chunk, measured from the walk alone."""

    root: Path
    files: int
    bytes: int
    children: tuple[DirectoryWeight, ...]

    def breakdown(self, limit: int = BREAKDOWN_LIMIT) -> str:
        """Render the fattest children, largest first, with their share of the whole."""
        lines = []
        for child in self.children[:limit]:
            share = f"{child.bytes * 100 / self.bytes:.0f}%" if self.bytes else "0%"
            lines.append(f"  {child.name + '/':<24} {child.files:>7} files  {megabytes(child.bytes)}  (~{share})")
        return "\n".join(lines)


def megabytes(size: int) -> str:
    """Render a byte count in MB, the one unit every work number is read in.

    :param size: A byte count.
    :return: The count in megabytes.
    """
    return f"{size / 1_000_000:.1f} MB"


def work_limit_bytes() -> int:
    """Return the byte ceiling one build may chunk; 0 or less disables the guard.

    The environment names the ceiling in MEGABYTES, because that is the unit a human reads a
    tree in; nonsense falls back to the default rather than crashing a build.

    :return: The ceiling in bytes.
    """
    raw = os.environ.get(WORK_LIMIT_ENV, "").strip()
    if not raw:
        return DEFAULT_WORK_LIMIT_BYTES
    try:
        return int(raw) * 1_000_000
    except ValueError:
        return DEFAULT_WORK_LIMIT_BYTES


def exceeds_work_limit(size: int) -> bool:
    """Return whether chunking this many bytes of source would be refused right now.

    :param size: The byte volume a build would chunk.
    :return: Whether the work guard refuses it.
    """
    limit = work_limit_bytes()
    return limit > 0 and not confirmed() and size > limit


def _nested_repository_count(root: Path, limit: int) -> int:
    """Count nearest nested Git roots, stopping before walking their contents or exceeding `limit`."""
    count = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                items = list(entries)
        except OSError:
            continue
        if any(entry.name == ".git" and not entry.is_symlink() for entry in items):
            count += 1
            if count >= limit:
                return count
            continue
        children = [entry for entry in items if entry.is_dir(follow_symlinks=False)]
        pending.extend(
            Path(entry.path)
            for entry in children
            if entry.name not in _IGNORED_DIRECTORY_NAMES and not entry.is_symlink()
        )
    return count


def require_declared_scope(root: str | Path) -> None:
    """Refuse a broad non-workspace root before its files are chunked or embedded.

    A Git root is already an explicit project boundary. A multi-repository workspace declares
    itself with `.zemble/home.toml`; smaller ad-hoc trees remain valid. `--yes` confirms both
    this scope and the paid-embedding budget through the existing confirmation variable.

    :param root: Local directory about to be indexed.
    :raises BroadRootRefused: If the directory appears to aggregate unrelated workspaces.
    """
    resolved = Path(root).expanduser().resolve()
    if (resolved / ".git").exists() or (resolved / HOME_CONFIG_RELATIVE_PATH).is_file() or confirmed():
        return
    repositories = _nested_repository_count(resolved, MAX_UNDECLARED_REPOSITORIES)
    if repositories < MAX_UNDECLARED_REPOSITORIES:
        return
    raise BroadRootRefused(
        f"Refusing to index {resolved}: found at least {repositories} nested Git repositories, but the root "
        f"does not declare a Zemble workspace. Search a narrower project root, add {HOME_CONFIG_RELATIVE_PATH}, "
        f"or pass --yes (or set {CONFIRM_ENV}=1) to index the broad root deliberately."
    )


def estimate_tree(
    root: Path,
    content: Sequence[ContentType] = (ContentType.CODE,),
    exclude: Sequence[str] = (),
    previous_manifest: dict[str, object] | None = None,
) -> TreeEstimate:
    """Measure what a build would chunk, from the walker alone: no file is opened or parsed.

    The walk is the build's own walk - the same default ignores, .gitignore, .zembleignore and
    1 MB file cap - so a path the build would skip is never counted. Files a previous index
    already covers unchanged are left out too, because a build would reuse them without
    embedding a thing.

    :param root: The resolved directory a build would index.
    :param content: The content types a build would index.
    :param exclude: Extra gitignore-style patterns this build was told to skip.
    :param previous_manifest: A previous build's manifest, whose unchanged files cost nothing.
    :return: The file count, byte count and per-child breakdown.
    """
    extensions = get_extensions(tuple(content))
    files = 0
    total = 0
    child_files: Counter[str] = Counter()
    child_bytes: Counter[str] = Counter()
    for walked in walk_entries(root, extensions, ignore=list(exclude)):
        size = walked.stat.st_size
        if size > MAX_FILE_BYTES:
            continue
        previous = previous_manifest.get(walked.relative_path) if previous_manifest is not None else None
        if previous is not None and getattr(previous, "mtime_ns", None) == walked.stat.st_mtime_ns:
            continue
        head, separator, _rest = walked.relative_path.partition("/")
        child = head if separator else "."
        files += 1
        total += size
        child_files[child] += 1
        child_bytes[child] += size
    children = tuple(
        DirectoryWeight(name=name, files=child_files[name], bytes=size)
        for name, size in sorted(child_bytes.items(), key=lambda item: (-item[1], item[0]))
    )
    return TreeEstimate(root=root, files=files, bytes=total, children=children)


def require_affordable_scope(
    root: str | Path,
    embedder: Embedder,
    content: Sequence[ContentType] = (ContentType.CODE,),
    exclude: Sequence[str] = (),
    previous_manifest: dict[str, object] | None = None,
) -> TreeEstimate:
    """Refuse a build whose walk alone is more source than one build may chunk.

    This guards EVERY embedder, local ones included, because it is about WORK: the minutes a
    parse costs, which nobody's cache and nobody's price list can shorten. It says nothing
    about a bill and must never print one - what a build costs is decided from the uncached
    set, by the budget guard in the caching embedder.

    :param root: The directory about to be indexed.
    :param embedder: The embedder the build resolved, named in the refusal.
    :param content: The content types the build will index.
    :param exclude: Extra gitignore-style patterns this build was told to skip.
    :param previous_manifest: The manifest the build itself will reuse from, whose unchanged
        files are not work; None means a build that reuses nothing.
    :return: The estimate, so a caller may log what it just approved.
    :raises OversizedRootRefused: If the walk exceeds the work limit and nothing confirmed it.
    """
    # AIDEV-NOTE: file bytes are a LOWER bound on what is embedded - a capsule adds a header to
    # every chunk, measured at +21% over this repo and +52% over the small fixture tree. That is
    # honest for WORK, which is what this guard measures, and it is void for MONEY: a lower bound
    # on bytes says nothing about a bill once the content-addressed cache has already paid for
    # most of the chunks those bytes produce. Pricing these bytes is what refused `home` for days.
    resolved = Path(root).expanduser().resolve()
    if confirmed() or work_limit_bytes() <= 0:
        return TreeEstimate(root=resolved, files=0, bytes=0, children=())
    estimate = estimate_tree(resolved, content, exclude, previous_manifest)
    if not exceeds_work_limit(estimate.bytes):
        return estimate
    family = embedder_family(embedder)
    raise OversizedRootRefused(
        f"Refusing to index {resolved} with {family or 'the configured embedder'}: "
        f"{estimate.files:,} files, {megabytes(estimate.bytes)} of source exceeds the "
        f"{megabytes(work_limit_bytes())} this build may chunk. Nothing was parsed or embedded.\n"
        f"{estimate.breakdown()}\n"
        f"{remedies(resolved, WORK_LIMIT_ENV)}",
        WORK_LIMIT_ENV,
    )


__all__ = [
    "BREAKDOWN_LIMIT",
    "BroadRootRefused",
    "DEFAULT_WORK_LIMIT_BYTES",
    "DirectoryWeight",
    "MAX_UNDECLARED_REPOSITORIES",
    "OversizedRootRefused",
    "ScopeRefused",
    "TreeEstimate",
    "WORK_LIMIT_ENV",
    "estimate_tree",
    "exceeds_work_limit",
    "megabytes",
    "require_affordable_scope",
    "require_declared_scope",
    "work_limit_bytes",
]
