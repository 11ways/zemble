"""Filesystem watching for the daemon's loaded roots.

The ignore rules are the file walker's own: a watcher that disagreed with the
indexer about what counts as a source file would either rebuild on noise or miss
edits, and gitignore semantics are not worth a second implementation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from pathspec import GitIgnoreSpec

# AIDEV-NOTE: these are file_walker internals on purpose - the walker owns the ignore
# vocabulary, and the watcher must answer "would this file have been indexed?" the same way.
from zemble.index.file_walker import (
    _DEFAULT_IGNORED_DIRS,
    IgnoreSpec,
    _is_ignored,
    _load_ignore_for_dir,
    _prefilter,
    _prepare,
)

logger = logging.getLogger(__name__)

#: Milliseconds of quiet before a burst of changes is delivered as one set.
DEBOUNCE_MS = 500
#: How often the watcher checks its stop event while idle.
STEP_MS = 50


class IgnoreRules:
    """Answers whether a changed path is one the index would have walked."""

    def __init__(self, root: Path, extensions: Iterable[str]) -> None:
        """Build the rules for one root.

        :param root: The indexed root directory.
        :param extensions: Lower-case file suffixes (with the dot) that count as source.
        """
        self.root = root
        self.extensions = frozenset(extension.lower() for extension in extensions)
        base_spec = GitIgnoreSpec.from_lines(sorted(_DEFAULT_IGNORED_DIRS), backend="simple")
        base_patterns = _prepare(base_spec)
        self._base = IgnoreSpec(
            base=root,
            spec=base_spec,
            patterns=base_patterns,
            prefilter=_prefilter(base_patterns),
            base_offset=0,
        )

    def _specs_for(self, relative_directory: str) -> list[IgnoreSpec]:
        """Return the ignore specs applying inside a root-relative directory, root-first.

        The walker's own loader is used, so an edited .gitignore is picked up by its
        modification time exactly as it is during a build.
        """
        specs = [self._base]
        walked = ""
        parts = [part for part in relative_directory.split("/") if part]
        for depth in range(len(parts) + 1):
            directory = self.root.joinpath(*parts[:depth])
            loaded = _load_ignore_for_dir(str(directory))
            if loaded is not None:
                specs.append(
                    IgnoreSpec(
                        base=directory,
                        spec=loaded[0],
                        patterns=loaded[1],
                        prefilter=loaded[2],
                        base_offset=len(walked) + 1 if walked else 0,
                    )
                )
            if depth < len(parts):
                walked = f"{walked}/{parts[depth]}" if walked else parts[depth]
        return specs

    def matches(self, path: Path) -> bool:
        """Return whether a changed path is one the index cares about."""
        if path.suffix.lower() not in self.extensions:
            return False
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError:
            return False
        directory = relative.rsplit("/", 1)[0] if "/" in relative else ""
        ignored, _bypasses = _is_ignored(relative, False, self._specs_for(directory))
        return not ignored


class RootWatcher:
    """Watches one indexed root and reports coalesced sets of relevant changed paths."""

    def __init__(
        self,
        root: Path,
        rules: IgnoreRules,
        on_change: Callable[[set[Path]], Awaitable[None]],
    ) -> None:
        """Create a watcher.

        :param root: Directory to watch recursively.
        :param rules: The ignore rules deciding which events matter.
        :param on_change: Coroutine called with each coalesced set of changed paths.
        """
        self.root = root
        self.rules = rules
        self._on_change = on_change
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start watching in the background."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"zemble-watch:{self.root}")

    def stop(self) -> None:
        """Ask the watcher to stop; the task ends on its next step."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def _filter(self, _change: object, path: str) -> bool:
        """Watchfiles filter: keep only paths the index would have covered."""
        candidate = Path(path)
        return self.rules.matches(candidate)

    async def _run(self) -> None:
        """Feed coalesced change sets to the callback until stopped."""
        from watchfiles import awatch

        try:
            async for changes in awatch(
                self.root,
                watch_filter=self._filter,
                debounce=DEBOUNCE_MS,
                step=STEP_MS,
                stop_event=self._stop,
                recursive=True,
            ):
                paths = {Path(path) for _change, path in changes}
                if not paths:
                    continue
                try:
                    await self._on_change(paths)
                except Exception:
                    logger.exception("Rebuild after changes under %s failed", self.root)
        except asyncio.CancelledError:  # pragma: no cover - normal teardown
            raise
        except Exception:
            logger.exception("Watcher for %s stopped", self.root)
