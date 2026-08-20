"""Wire format, filesystem locations and error types for the zemble daemon.

Imported by both the client and the server, so it must stay free of index, model
and watcher imports: the client half runs inside every short-lived CLI process.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import orjson

#: Newline-delimited JSON: one request per line, one response per line.
LINE_SEPARATOR = b"\n"

DEFAULT_MAX_INDEXES = 4
DEFAULT_IDLE_MINUTES = 30
#: How long a client waits for a freshly spawned daemon to accept connections.
START_TIMEOUT_SECONDS = 10.0
#: How long a client waits for the socket itself, once it exists.
CONNECT_TIMEOUT_SECONDS = 5.0


class DaemonError(Exception):
    """Base class for every daemon-related failure a caller may fall back from."""


class DaemonUnavailable(DaemonError):
    """The daemon could not be reached, started, or is disabled."""


class CommandFailed(DaemonError):
    """The daemon answered, and the answer was an error."""


def _preferred_directory() -> Path:
    """Return the directory holding the socket, pidfile and lock file."""
    override = os.environ.get("ZEMBLE_DAEMON_DIR")
    if override:
        return Path(override)
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and Path(runtime_dir).is_dir():
        return Path(runtime_dir) / "zemble"
    from zemble.cache import resolve_cache_folder

    return resolve_cache_folder()


def runtime_directory(create: bool = False) -> Path:
    """Return (and optionally create, private to the user) the daemon's runtime directory."""
    directory = _preferred_directory()
    if create:
        directory.mkdir(parents=True, exist_ok=True)
        # One daemon per user: nobody else may reach the socket.
        os.chmod(directory, 0o700)
    return directory


def socket_path(create_dir: bool = False) -> Path:
    """Return the unix domain socket path the daemon listens on."""
    override = os.environ.get("ZEMBLE_DAEMON_SOCKET")
    if override:
        return Path(override)
    return runtime_directory(create_dir) / "daemon.sock"


def pid_path() -> Path:
    """Return the pidfile path, which sits beside the socket."""
    return socket_path().with_name(socket_path().name + ".pid")


def lock_path() -> Path:
    """Return the lock file guaranteeing a single daemon per socket."""
    return socket_path().with_name(socket_path().name + ".lock")


def log_path() -> Path:
    """Return the file a detached daemon's stdout and stderr are appended to."""
    from zemble.cache import resolve_cache_folder

    return resolve_cache_folder() / "daemon.log"


def encode(payload: dict[str, Any]) -> bytes:
    """Encode one protocol message as a JSON line."""
    return orjson.dumps(payload) + LINE_SEPARATOR


def decode(line: bytes) -> dict[str, Any]:
    """Decode one protocol line.

    :param line: One newline-terminated JSON message.
    :return: The decoded message.
    :raises CommandFailed: If the line is not a JSON object.
    """
    try:
        payload = orjson.loads(line)
    except orjson.JSONDecodeError as exc:
        raise CommandFailed(f"malformed message: {exc}") from exc
    if not isinstance(payload, dict):
        raise CommandFailed(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def read_pid() -> int | None:
    """Return the pid recorded in the pidfile, or None if it is missing or unreadable."""
    try:
        return int(pid_path().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def process_alive(pid: int) -> bool:
    """Return whether a process with this pid exists and is signalable by this user."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
