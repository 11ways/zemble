"""What code this process is actually running: version, source root, revision, staleness.

AIDEV-NOTE: zemble is installed as an editable uv tool, so every process is a snapshot of
the checkout taken when it started, and long-lived servers (the MCP stdio server, the warm
daemon) keep serving that snapshot for hours after a pull. Nothing in the wire protocols
used to say which snapshot, so a stale server was indistinguishable from a fresh one. This
module is the one place that answers "which code is this"; every surface reports from here.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zemble.version import __version__

#: Seconds a `git rev-parse` may take before the revision is reported as unknown.
GIT_TIMEOUT_SECONDS = 2.0

#: Upper bound on the source walk; src/zemble is ~100 files, a runaway root is not ours.
MAX_SOURCE_FILES = 2000


def _python_files(root: Path) -> tuple[Path, ...]:
    """Return the *.py files under a source root, capped so a surprising root cannot stall a request."""
    found: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        found.append(path)
        if len(found) >= MAX_SOURCE_FILES:
            break
    return tuple(found)


def _newest_mtime(files: tuple[Path, ...]) -> float:
    """Return the newest mtime among these files; a vanished file counts as no mtime."""
    newest = 0.0
    for path in files:
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        newest = max(newest, stamp)
    return newest


def git_revision(root: Path) -> str | None:
    """Return the short HEAD sha of the checkout holding this root, or None when there is none.

    Never raises: a missing git, a non-checkout, or a slow filesystem all mean "unknown".
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _is_editable(root: Path) -> bool:
    """Return whether this source root sits outside an installed package directory."""
    return not any(part in ("site-packages", "dist-packages") for part in root.parts)


@dataclass(frozen=True)
class RuntimeIdentity:
    """The code snapshot one process is running, captured when that process started."""

    zemble_version: str
    source_root: str
    editable: bool
    source_revision: str | None
    process_started_at: str
    pid: int
    source_mtime: float
    files: tuple[Path, ...] = field(default=(), repr=False, compare=False)

    @classmethod
    def capture(cls, root: Path | None = None) -> RuntimeIdentity:
        """Capture the identity of the code loaded from a source root (the zemble package by default)."""
        source_root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
        files = _python_files(source_root)
        return cls(
            zemble_version=__version__,
            source_root=str(source_root),
            editable=_is_editable(source_root),
            source_revision=git_revision(source_root),
            process_started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            pid=os.getpid(),
            source_mtime=_newest_mtime(files),
            files=files,
        )

    def current_revision(self) -> str | None:
        """Return the source root's HEAD sha right now, which a pull moves away from `source_revision`."""
        return git_revision(Path(self.source_root))

    def source_changed_since_start(self) -> bool:
        """Return whether the loaded source changed after this process started.

        Re-stats the file list captured at import; a file added since then is only seen
        through the revision, which is the case a pull produces anyway.
        """
        if _newest_mtime(self.files) > self.source_mtime:
            return True
        return self.current_revision() != self.source_revision

    def to_dict(self) -> dict[str, Any]:
        """Return the identity as a JSON-ready object; the cached file list stays internal."""
        return {
            "zemble_version": self.zemble_version,
            "source_root": self.source_root,
            "editable": self.editable,
            "source_revision": self.source_revision,
            "process_started_at": self.process_started_at,
            "pid": self.pid,
        }


_IDENTITY = RuntimeIdentity.capture()


def identity() -> RuntimeIdentity:
    """Return this process's runtime identity, captured once at import."""
    return _IDENTITY


def stale_note(subject: RuntimeIdentity | None = None) -> str | None:
    """Return the sentence to show a caller when this process serves stale source, else None."""
    current = subject or _IDENTITY
    if not current.source_changed_since_start():
        return None
    return (
        f"zemble source changed after this server started "
        f"(rev {current.source_revision or 'unknown'} -> {current.current_revision() or 'unknown'}); "
        "restart the MCP server to pick it up"
    )


def status_payload(subject: RuntimeIdentity | None = None) -> dict[str, Any]:
    """Return the identity, its staleness flag, and the human note every surface reports."""
    current = subject or _IDENTITY
    note = stale_note(current)
    return {"identity": current.to_dict(), "stale": note is not None, "note": note}


__all__ = [
    "GIT_TIMEOUT_SECONDS",
    "MAX_SOURCE_FILES",
    "RuntimeIdentity",
    "git_revision",
    "identity",
    "stale_note",
    "status_payload",
]
