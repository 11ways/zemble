"""Refuse an accidental aggregation root, or an unaffordable one, before a build starts.

Both refusals happen before a single file is parsed: the expensive half of a build is
chunking, and a tree big enough to be refused is a tree big enough for that to take minutes.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from zemble.chunking.capsule import CapsuleOptions
from zemble.embedding.base import Embedder, is_remote
from zemble.embedding.pricing import (
    CONFIRM_ENV,
    ESTIMATE_CHARS_PER_TOKEN,
    budget_tokens,
    confirmed,
    embedder_family,
    exceeds_budget,
    format_cost,
    price_per_million,
    remedies,
)
from zemble.index.file_walker import _DEFAULT_IGNORED_DIRS, walk_entries
from zemble.index.files import MAX_FILE_BYTES, get_extensions
from zemble.types import ContentType
from zemble.workspace import HOME_CONFIG_RELATIVE_PATH

#: An undeclared directory containing this many repositories is probably a parent of workspaces.
MAX_UNDECLARED_REPOSITORIES = 8

#: How many of the root's immediate children a refusal names, largest first.
BREAKDOWN_LIMIT = 8

_IGNORED_DIRECTORY_NAMES = frozenset(pattern.removesuffix("/") for pattern in _DEFAULT_IGNORED_DIRS)


class ScopeRefused(RuntimeError):
    """A deterministic refusal to build an index over a root, decided before anything is parsed.

    One base class so every surface - the CLI, the MCP tools and the daemon wire - can tell a
    refusal, which is the same answer in every process, from a failure, which may not be.
    """


class BroadRootRefused(ScopeRefused):
    """A local root spans too many repositories to index without an explicit declaration."""


class OversizedRootRefused(ScopeRefused):
    """A root holds more indexable text than the token budget allows, so nothing was parsed."""


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

    @property
    def tokens(self) -> int:
        """The token count the bytes are worth, at the one estimate ratio the budget guard uses."""
        return math.ceil(self.bytes / ESTIMATE_CHARS_PER_TOKEN)

    def breakdown(self, limit: int = BREAKDOWN_LIMIT) -> str:
        """Render the fattest children, largest first, with their share of the whole."""
        lines = []
        for child in self.children[:limit]:
            share = f"{child.bytes * 100 / self.bytes:.0f}%" if self.bytes else "0%"
            lines.append(f"  {child.name + '/':<24} {child.files:>7} files  {_megabytes(child.bytes)}  (~{share})")
        return "\n".join(lines)


def _megabytes(size: int) -> str:
    """Render a byte count in MB, the unit a refusal is read in."""
    return f"{size / 1_000_000:.1f} MB"


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
    capsules: CapsuleOptions | None = None,
) -> TreeEstimate:
    """Refuse a build whose walk alone already exceeds the token budget, before anything is parsed.

    This guards EVERY embedder, local ones included: the budget is about the work a build does,
    not only about a bill, and the post-chunk guard in the caching embedder never sees a local
    lane at all. The estimate is deliberately taken from file bytes rather than capsule text -
    the exact number costs the very parse this refusal exists to avoid.

    :param root: The directory about to be indexed.
    :param embedder: The embedder the build resolved, used to price the estimate.
    :param content: The content types the build will index.
    :param exclude: Extra gitignore-style patterns this build was told to skip.
    :param capsules: The capsule configuration, used to find the previous index's manifest.
    :return: The estimate, so a caller may log what it just approved.
    :raises OversizedRootRefused: If the estimate exceeds the budget and nothing confirmed it.
    """
    # AIDEV-NOTE: file bytes are a LOWER bound on what is embedded - a capsule adds a header to
    # every chunk, measured at +21% over this repo and +52% over the small fixture tree - so this
    # guard refuses only what is certainly over budget, and the post-chunk guard in the caching
    # embedder stays as the exact second line of defence for the lanes that cost money.
    from zemble.cache import load_manifest_for_incremental

    resolved = Path(root).expanduser().resolve()
    remote = is_remote(embedder)
    if confirmed() or budget_tokens(remote) <= 0:
        return TreeEstimate(root=resolved, files=0, bytes=0, children=())
    manifest = load_manifest_for_incremental(str(resolved), embedder.model_id, content, capsules, exclude)
    estimate = estimate_tree(resolved, content, exclude, manifest)
    if not exceeds_budget(estimate.tokens, remote):
        return estimate
    family = embedder_family(embedder)
    price = price_per_million(family) if remote else None
    cost = f" (~{format_cost(estimate.tokens, price)})" if price is not None else ""
    raise OversizedRootRefused(
        f"Refusing to index {resolved} with {family or 'the configured embedder'}: "
        f"{estimate.files:,} files, {_megabytes(estimate.bytes)}, ~{estimate.tokens:,} estimated tokens{cost} "
        f"exceeds the budget of {budget_tokens(remote):,} tokens. Nothing was parsed or embedded.\n"
        f"{estimate.breakdown()}\n"
        f"{remedies(resolved)}"
    )


__all__ = [
    "BREAKDOWN_LIMIT",
    "BroadRootRefused",
    "DirectoryWeight",
    "MAX_UNDECLARED_REPOSITORIES",
    "OversizedRootRefused",
    "ScopeRefused",
    "TreeEstimate",
    "estimate_tree",
    "require_affordable_scope",
    "require_declared_scope",
]
