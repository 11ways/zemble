"""Filesystem watching for the daemon's loaded roots.

The ignore rules are the file walker's own: a watcher that disagreed with the
indexer about what counts as a source file would either rebuild on noise or miss
edits, and gitignore semantics are not worth a second implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
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
#: The line a watch worker writes once every watch is registered.
READY_LINE = "ready"
#: Longest change-set line read from a worker: a branch switch can name every file of a workspace.
_LINE_LIMIT = 64 * 1024 * 1024


class IgnoreRules:
    """Answers whether a changed path is one the index would have walked."""

    def __init__(self, root: Path, extensions: Iterable[str], always: Callable[[Path], bool] | None = None) -> None:
        """Build the rules for one root.

        :param root: The indexed root directory.
        :param extensions: Lower-case file suffixes (with the dot) that count as source.
        :param always: Optional predicate for paths that matter whatever the ignore rules
            say. A graph facts file lives under `build/` or `.zemble/` by convention, both
            of which the walker skips, so without this the daemon would never see one change.
        """
        self.root = root
        self.always = always
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
        if self.always is not None and self.always(path):
            return True
        if path.suffix.lower() not in self.extensions:
            return False
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError:
            return False
        directory = relative.rsplit("/", 1)[0] if "/" in relative else ""
        ignored, _bypasses = _is_ignored(relative, False, self._specs_for(directory))
        return not ignored


@dataclass(frozen=True)
class WatchSpec:
    """What one watcher watches: a root, the suffixes that count as source, and whether graph facts files count."""

    root: Path
    extensions: tuple[str, ...]
    facts: bool = False

    def rules(self) -> IgnoreRules:
        """Build the ignore rules this spec describes."""
        if not self.facts:
            return IgnoreRules(self.root, self.extensions)
        from zemble.graph.facts import matches_facts_glob

        root = self.root
        return IgnoreRules(root, self.extensions, always=lambda path: matches_facts_glob(root, path))

    def to_json(self) -> str:
        """Serialise the spec for a watch worker's command line."""
        return json.dumps({"root": str(self.root), "extensions": sorted(self.extensions), "facts": self.facts})

    @classmethod
    def from_json(cls, text: str) -> WatchSpec:
        """Read a spec written by :meth:`to_json`."""
        raw = json.loads(text)
        return cls(Path(raw["root"]), tuple(raw["extensions"]), bool(raw["facts"]))


class RootWatcher:
    """Watches one indexed root and reports coalesced sets of relevant changed paths.

    AIDEV-NOTE: the watches live in a `zemble.daemon.watch_worker` process. watchfiles registers a
    recursive watch while holding the GIL, which on the zenit workspace froze the daemon's event
    loop and starved every thread (an index load among them) for 5 to 18 s per root.
    """

    def __init__(
        self,
        spec: WatchSpec,
        on_change: Callable[[set[Path]], Awaitable[None]],
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        """Create a watcher.

        :param spec: The root to watch recursively and the rules deciding which events matter.
        :param on_change: Coroutine called with each coalesced set of changed paths.
        :param on_ready: Called once every watch is registered; edits before that were not seen.
        """
        self.spec = spec
        self.root = spec.root
        self._on_change = on_change
        self._on_ready = on_ready
        self._stopping = False
        self._task: asyncio.Task[None] | None = None
        self._process: asyncio.subprocess.Process | None = None

    def start(self) -> None:
        """Start watching in the background."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"zemble-watch:{self.root}")

    def stop(self) -> None:
        """Stop the worker and the task reading it."""
        self._stopping = True
        if self._process is not None and self._process.returncode is None:
            self._process.kill()
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        """Feed the worker's change sets to the callback until stopped."""
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "zemble.daemon.watch_worker",
            self.spec.to_json(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            limit=_LINE_LIMIT,
        )
        self._process = process
        try:
            assert process.stdout is not None
            async for raw in process.stdout:
                line = raw.decode("utf-8").strip()
                if line == READY_LINE:
                    logger.info("watches ready under %s in %.1f s", self.root, time.monotonic() - started)
                    if self._on_ready is not None:
                        self._on_ready()
                    continue
                paths = {Path(path) for path in json.loads(line)}
                if not paths:
                    continue
                try:
                    await self._on_change(paths)
                except Exception:
                    logger.exception("Rebuild after changes under %s failed", self.root)
            code = await process.wait()
            if not self._stopping:
                logger.error("Watcher for %s stopped: its worker exited with %s", self.root, code)
        except asyncio.CancelledError:  # pragma: no cover - normal teardown
            raise
        except Exception:
            logger.exception("Watcher for %s stopped", self.root)
        finally:
            if process.returncode is None:
                process.kill()
