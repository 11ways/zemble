"""Refuse an accidental aggregation root, or one whose sheer volume is runaway work.

Both refusals happen before a single file is parsed: the expensive half of a build is
chunking, and a tree big enough to be refused is a tree big enough for that to take minutes.
Neither of them is about money: what a build costs is decided once the uncached set is known,
by the budget guard in :mod:`zemble.embedding.pricing`.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from zemble.embedding.base import Embedder
from zemble.embedding.pricing import CONFIRM_ENV, confirmed, embedder_family, remedies
from zemble.envknob import env_int
from zemble.index.file_walker import _DEFAULT_IGNORED_DIRS, ignored_prefix, walk_entries
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
    """What one build would chunk: the whole walk, or only the paths a change set named."""

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
    tree in; nonsense falls back to the default, loudly, rather than crashing a build.

    :return: The ceiling in bytes.
    """
    return env_int(WORK_LIMIT_ENV, DEFAULT_WORK_LIMIT_BYTES // 1_000_000) * 1_000_000


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
    itself with `.zemble/home.toml`; smaller ad-hoc trees remain valid. `--yes` confirms this
    scope, the work ceiling and the spending budget alike: one confirmation lifts every guard.

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


def _still_work(indexed_path: str, size: int, mtime_ns: int, previous_manifest: dict[str, object] | None) -> bool:
    """Return whether a file is work a build would really redo, rather than reuse or skip."""
    if size > MAX_FILE_BYTES:
        return False
    previous = previous_manifest.get(indexed_path) if previous_manifest is not None else None
    return not (previous is not None and getattr(previous, "mtime_ns", None) == mtime_ns)


def _weigh(root: Path, measured: Iterable[tuple[str, int]]) -> TreeEstimate:
    """Fold (indexed path, size) pairs into the file count, byte count and per-child breakdown."""
    files = 0
    total = 0
    child_files: Counter[str] = Counter()
    child_bytes: Counter[str] = Counter()
    for indexed_path, size in measured:
        head, separator, _rest = indexed_path.partition("/")
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

    def _walked() -> Iterator[tuple[str, int]]:
        for walked in walk_entries(root, extensions, ignore=list(exclude)):
            if _still_work(walked.relative_path, walked.stat.st_size, walked.stat.st_mtime_ns, previous_manifest):
                yield walked.relative_path, walked.stat.st_size

    return _weigh(root, _walked())


def changed_indexed_paths(
    root: Path,
    changed: Iterable[Path],
    content: Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
    exclude: Sequence[str] = (),
) -> dict[str, Path]:
    """Map every named path a build would really consider to the path its chunks are stored under.

    THE one home of "which of these named paths count": a path outside the root, carrying an
    extension no content type covers, or under an ignored prefix is dropped, because those are
    exactly the paths a build from a known change set refuses to plan. The guard and the plan
    read it, so neither can measure a set the other does not.

    :param root: The resolved directory the index covers.
    :param changed: The paths that were added, edited or removed.
    :param content: The content types the build indexes.
    :param display_root: The root chunk paths are stored relative to; None means `root`.
    :param exclude: The gitignore-style patterns the index was built with.
    :return: Indexed path to filesystem path, in the order the caller named them.
    """
    extensions = {extension.lower() for extension in get_extensions(tuple(content))}
    root_for_paths = display_root if display_root is not None else root
    ignore = list(exclude)
    selected: dict[str, Path] = {}
    for candidate in changed:
        try:
            indexed_path = candidate.relative_to(root_for_paths).as_posix()
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            continue
        if candidate.suffix.lower() not in extensions or ignored_prefix(root, relative, ignore) is not None:
            continue
        selected[indexed_path] = candidate
    return selected


def estimate_changed_paths(
    root: Path,
    changed: Iterable[Path],
    content: Sequence[ContentType] = (ContentType.CODE,),
    exclude: Sequence[str] = (),
    previous_manifest: dict[str, object] | None = None,
    display_root: Path | None = None,
) -> TreeEstimate:
    """Measure what a build from a KNOWN change set would chunk, without walking the tree.

    The watcher lane exists precisely to avoid a walk, and it chunks the paths it was handed,
    not every file whose modification time has drifted since the manifest. Walking here cost a
    full stat pass on every coalesced file event AND refused a one-file rebuild for drift the
    build would have reused - which, since a refused rebuild swaps nothing, left the manifest
    where it was and refused every later event too.

    :param root: The resolved directory the index covers.
    :param changed: The paths a watcher saw move.
    :param content: The content types the build indexes.
    :param exclude: The gitignore-style patterns the index was built with.
    :param previous_manifest: The manifest this build reuses from; a named path whose
        modification time did not move is reused, so it is not work.
    :param display_root: The root chunk paths are stored relative to; None means `root`.
    :return: The file count, byte count and per-child breakdown of the named paths.
    """
    named = changed_indexed_paths(root, changed, content, display_root, exclude)

    def _named() -> Iterator[tuple[str, int]]:
        for indexed_path, candidate in named.items():
            try:
                stat = candidate.stat()
            except OSError:
                continue  # A path that is gone is a removal, and removing chunks costs no parse.
            if _still_work(indexed_path, stat.st_size, stat.st_mtime_ns, previous_manifest):
                yield indexed_path, stat.st_size

    return _weigh(root, _named())


def measure_work(
    root: Path,
    content: Sequence[ContentType] = (ContentType.CODE,),
    exclude: Sequence[str] = (),
    previous_manifest: dict[str, object] | None = None,
    changed: Iterable[Path] | None = None,
    display_root: Path | None = None,
) -> TreeEstimate:
    """Measure the source one build would chunk, the way that build is going to find it.

    :param root: The resolved directory a build would index.
    :param content: The content types a build would index.
    :param exclude: Extra gitignore-style patterns this build was told to skip.
    :param previous_manifest: The manifest the build reuses from, whose unchanged files cost nothing.
    :param changed: The exact paths a watcher named, when the build plans from them; None walks
        the tree, which is what a build with no change set does.
    :param display_root: The root chunk paths are stored relative to; None means `root`.
    :return: The file count, byte count and per-child breakdown.
    """
    if changed is None:
        return estimate_tree(root, content, exclude, previous_manifest)
    return estimate_changed_paths(root, changed, content, exclude, previous_manifest, display_root)


def work_refusal(estimate: TreeEstimate) -> tuple[str, str] | None:
    """Decide a work refusal once, returning its wording and the knob that named the ceiling.

    THE one home of the work verdict, the way ``_bill_refusal`` is the home of the money one.
    The pre-flight report and the guard read the same sentence off the same measurement, so a
    report can never call a build allowed that the guard refuses.

    :param estimate: What the build would chunk.
    :return: The reason and the environment variable that raises the ceiling, or None.
    """
    if not exceeds_work_limit(estimate.bytes):
        return None
    return (
        f"{estimate.files:,} files, {megabytes(estimate.bytes)} of source exceeds the "
        f"{megabytes(work_limit_bytes())} this build may chunk. Nothing was parsed or embedded.",
        WORK_LIMIT_ENV,
    )


def require_affordable_scope(
    root: str | Path,
    embedder: Embedder,
    content: Sequence[ContentType] = (ContentType.CODE,),
    exclude: Sequence[str] = (),
    previous_manifest: dict[str, object] | None = None,
    changed: Iterable[Path] | None = None,
    display_root: Path | None = None,
) -> TreeEstimate:
    """Refuse a build that would chunk more source than one build may.

    This guards EVERY embedder, local ones included, because it is about WORK: the minutes a
    parse costs, which nobody's cache and nobody's price list can shorten. It says nothing
    about a bill and must never print one - what a build costs is decided from the uncached
    set, by :func:`zemble.embedding.pricing.require_affordable_bill`.

    :param root: The directory about to be indexed.
    :param embedder: The embedder the build resolved, named in the refusal.
    :param content: The content types the build will index.
    :param exclude: Extra gitignore-style patterns this build was told to skip.
    :param previous_manifest: The manifest the build itself will reuse from, whose unchanged
        files are not work; None means a build that reuses nothing.
    :param changed: The exact paths the build will plan from, when it has a change set; None
        measures the walk, which is what a build without one does.
    :param display_root: The root chunk paths are stored relative to; None means `root`.
    :return: The estimate, so a caller may log what it just approved.
    :raises OversizedRootRefused: If the measurement exceeds the work limit and nothing confirmed it.
    """
    # AIDEV-NOTE: file bytes are a LOWER bound on what is embedded - a capsule adds a header to
    # every chunk, measured at +21% over this repo and +52% over the small fixture tree. That is
    # honest for WORK, which is what this guard measures, and it is void for MONEY: a lower bound
    # on bytes says nothing about a bill once the content-addressed cache has already paid for
    # most of the chunks those bytes produce. Pricing these bytes is what refused `home` for days.
    resolved = Path(root).expanduser().resolve()
    # The named lane matches paths as strings, so it measures against the root the BUILD was
    # handed; the walk lane yields relative paths and is indifferent to how the root resolved.
    measured_root = Path(root) if changed is not None else resolved
    if confirmed() or work_limit_bytes() <= 0:
        return TreeEstimate(root=resolved, files=0, bytes=0, children=())
    estimate = measure_work(measured_root, content, exclude, previous_manifest, changed, display_root)
    refusal = work_refusal(estimate)
    if refusal is None:
        return estimate
    reason, knob = refusal
    family = embedder_family(embedder)
    raise OversizedRootRefused(
        f"Refusing to index {resolved} with {family or 'the configured embedder'}: {reason}\n"
        f"{estimate.breakdown()}\n"
        f"{remedies(resolved, knob)}",
        knob,
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
    "changed_indexed_paths",
    "estimate_changed_paths",
    "estimate_tree",
    "exceeds_work_limit",
    "measure_work",
    "megabytes",
    "require_affordable_scope",
    "require_declared_scope",
    "work_limit_bytes",
    "work_refusal",
]
