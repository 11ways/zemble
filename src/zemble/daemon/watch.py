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

from zemble.index.file_walker import duplicate_tree_prefix, ignored_prefix

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
    """Answers whether a changed path is one the index would have walked.

    AIDEV-NOTE: the answer is the file walker's own `ignored_prefix`, never a copy of its rules: a watcher
    that disagreed with the walk re-planned a lane on every write to its `.m2/` and run state (25,000
    re-plans in five days on one host), because only the walk knew the lane's mirrored root.
    """

    def __init__(
        self,
        root: Path,
        extensions: Iterable[str],
        always: Callable[[Path], bool] | None = None,
        mirrored_root: Path | None = None,
    ) -> None:
        """Build the rules for one root.

        :param root: The indexed root directory.
        :param extensions: Lower-case file suffixes (with the dot) that count as source.
        :param always: Optional predicate for paths that matter whatever the ignore rules
            say. A graph facts file lives under `build/` or `.zemble/` by convention, both
            of which the walker skips, so without this the daemon would never see one change.
        :param mirrored_root: The mirrored root the root's walk is given (`zemble.lane.mirrored_root`).
        """
        self.root = root
        self.always = always
        self.mirrored_root = mirrored_root
        self.extensions = frozenset(extension.lower() for extension in extensions)

    def matches(self, path: Path) -> bool:
        """Return whether a changed path is one the index cares about."""
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError:
            relative = None
        if relative is not None and duplicate_tree_prefix(self.root, relative) is not None:
            # A lane parked inside this root is its own root; its edits are never this root's.
            return False
        if self.always is not None and self.always(path):
            return True
        if path.suffix.lower() not in self.extensions or relative is None:
            return False
        return ignored_prefix(self.root, relative, mirrored_root=self.mirrored_root) is None


@dataclass(frozen=True)
class WatchSpec:
    """What one watcher watches: a root, the suffixes that count as source, whether graph facts files count."""

    root: Path
    extensions: tuple[str, ...]
    facts: bool = False
    #: The mirrored root the root's walk is given, for a lane root that is no checkout.
    mirrored_root: Path | None = None

    def rules(self) -> IgnoreRules:
        """Build the ignore rules this spec describes."""
        if not self.facts:
            return IgnoreRules(self.root, self.extensions, mirrored_root=self.mirrored_root)
        from zemble.graph.facts import matches_facts_glob

        root = self.root
        return IgnoreRules(
            root, self.extensions, always=lambda path: matches_facts_glob(root, path), mirrored_root=self.mirrored_root
        )

    def to_json(self) -> str:
        """Serialise the spec for a watch worker's command line."""
        mirrored = str(self.mirrored_root) if self.mirrored_root is not None else None
        return json.dumps(
            {"root": str(self.root), "extensions": sorted(self.extensions), "facts": self.facts, "mirrored": mirrored}
        )

    @classmethod
    def from_json(cls, text: str) -> WatchSpec:
        """Read a spec written by :meth:`to_json`."""
        raw = json.loads(text)
        mirrored = raw.get("mirrored")
        return cls(
            Path(raw["root"]), tuple(raw["extensions"]), bool(raw["facts"]), Path(mirrored) if mirrored else None
        )


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
