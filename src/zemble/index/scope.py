"""Refuse accidental aggregation roots before a semantic index build starts."""

from __future__ import annotations

import os
from pathlib import Path

from zemble.embedding.pricing import CONFIRM_ENV, confirmed
from zemble.index.file_walker import _DEFAULT_IGNORED_DIRS
from zemble.workspace import HOME_CONFIG_RELATIVE_PATH

#: An undeclared directory containing this many repositories is probably a parent of workspaces.
MAX_UNDECLARED_REPOSITORIES = 8

_IGNORED_DIRECTORY_NAMES = frozenset(pattern.removesuffix("/") for pattern in _DEFAULT_IGNORED_DIRS)


class BroadRootRefused(RuntimeError):
    """A local root spans too many repositories to index without an explicit declaration."""


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


__all__ = ["BroadRootRefused", "MAX_UNDECLARED_REPOSITORIES", "require_declared_scope"]
