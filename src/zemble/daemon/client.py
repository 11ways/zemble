"""Thin client for the zemble daemon: connect, auto-start, one request, one response.

Every caller is expected to fall back to the in-process path when this module raises
`DaemonError`; the daemon is an accelerator, never a requirement.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import socket
import subprocess
import sys
import time
from itertools import count
from typing import Any

from zemble.daemon.protocol import (
    CONNECT_TIMEOUT_SECONDS,
    REVISION_FIELD,
    START_TIMEOUT_SECONDS,
    CommandFailed,
    CommandRefused,
    DaemonUnavailable,
    ErrorKind,
    decode,
    encode,
    error_kind,
    log_path,
    pid_path,
    process_alive,
    read_pid,
    socket_path,
)
from zemble.utils import is_git_url

logger = logging.getLogger(__name__)

_request_ids = count(1)
#: Set once this process has complained about a daemon running different code.
_warned_about_revision = False
#: Set once a process must never talk to a daemon: the daemon itself, or --no-daemon.
_disabled_reason: str | None = None


def absolutize_path(args: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``path`` argument against THIS process's cwd before it leaves the process.

    The daemon runs with cwd ``/`` and cannot know the caller's directory; a relative path sent
    as-is would make it index the filesystem root. Git URLs pass through untouched.
    """
    path = args.get("path")
    if isinstance(path, str) and path and not is_git_url(path):
        return {**args, "path": os.path.abspath(os.path.expanduser(path))}
    return args


def disable_for_this_process(reason: str) -> None:
    """Make every `call` in this process raise instead of contacting a daemon.

    Used by `--no-daemon` and by the daemon itself, which would otherwise deadlock
    calling back into its own socket through a shared code path.
    """
    global _disabled_reason
    _disabled_reason = reason


def disabled_reason() -> str | None:
    """Return why this process refuses to use a daemon, or None if it may."""
    if _disabled_reason is not None:
        return _disabled_reason
    if os.environ.get("ZEMBLE_DAEMON", "").strip() == "0":
        return "disabled by ZEMBLE_DAEMON=0"
    return None


def _connect(timeout: float = CONNECT_TIMEOUT_SECONDS) -> socket.socket:
    """Open a connection to the daemon socket.

    :param timeout: Seconds to wait for the connect itself.
    :return: The connected socket.
    :raises DaemonUnavailable: If nothing is listening.
    """
    path = socket_path()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(path))
    except OSError as exc:
        client.close()
        raise DaemonUnavailable(f"not running ({errno.errorcode.get(exc.errno, exc.strerror)})") from exc
    return client


def is_running() -> bool:
    """Return whether a daemon is accepting connections right now."""
    try:
        _connect(timeout=1.0).close()
    except DaemonUnavailable:
        return False
    return True


def clear_stale() -> bool:
    """Remove a socket and pidfile left behind by a dead daemon.

    :return: Whether anything was removed.
    """
    path = socket_path()
    if not path.exists() and not pid_path().exists():
        return False
    if is_running():
        return False
    pid = read_pid()
    if pid is not None and process_alive(pid):
        # A live process owns the pidfile but the socket does not answer: leave it alone
        # rather than pulling the socket out from under a daemon that is still starting.
        return False
    for stale in (path, pid_path()):
        with contextlib.suppress(OSError):
            stale.unlink()
    return True


def spawn(timeout: float = START_TIMEOUT_SECONDS) -> None:
    """Start a detached daemon and wait for it to accept connections.

    :param timeout: Seconds to wait for the socket to answer.
    :raises DaemonUnavailable: If the daemon did not come up in time.
    """
    clear_stale()
    log_file = log_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = open(log_file, "ab")  # handed to the child; closed here right after the spawn
    except OSError as exc:
        raise DaemonUnavailable(f"cannot open daemon log {log_file}: {exc}") from exc
    try:
        subprocess.Popen(
            [sys.executable, "-m", "zemble.daemon", "run"],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            start_new_session=True,
            cwd="/",
        )
    except OSError as exc:
        raise DaemonUnavailable(f"cannot start daemon: {exc}") from exc
    finally:
        handle.close()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running():
            return
        time.sleep(0.05)
    raise DaemonUnavailable(f"did not start within {timeout:.0f}s (see {log_file})")


def _warn_on_revision_mismatch(response: dict[str, Any]) -> None:
    """Warn once when the answering daemon runs a different checkout revision than this process.

    A mismatch is never a refusal: the daemon is an accelerator, and an older one still
    answers correctly for everything its snapshot already knew how to do.
    """
    global _warned_about_revision
    if _warned_about_revision:
        return
    from zemble.runtime.identity import identity

    theirs = response.get(REVISION_FIELD)
    mine = identity().source_revision
    if theirs is None or mine is None or theirs == mine:
        return
    _warned_about_revision = True
    logger.warning(
        "zemble daemon runs revision %s, this process runs %s; run `zemble daemon restart` to match them",
        theirs,
        mine,
    )


def call(cmd: str, args: dict[str, Any] | None = None, *, auto_start: bool = True, timeout: float | None = None) -> Any:
    """Send one command to the daemon and return its result.

    :param cmd: Command name, as registered in the server's command table.
    :param args: Command arguments.
    :param auto_start: Whether to spawn a daemon when none is listening.
    :param timeout: Seconds to wait for the response. None blocks until the daemon answers,
        which is what a first, index-building request needs.
    :return: The command's result.
    :raises DaemonUnavailable: If this process may not use a daemon, or none could be reached.
    :raises CommandRefused: If the daemon deliberately refused the command.
    :raises CommandFailed: If the daemon reported any other error for this command.
    """
    reason = disabled_reason()
    if reason is not None:
        raise DaemonUnavailable(reason)
    try:
        client = _connect()
    except DaemonUnavailable:
        if not auto_start:
            raise
        spawn()
        client = _connect()

    request_id = next(_request_ids)
    try:
        client.settimeout(timeout)
        with client, client.makefile("rwb") as stream:
            stream.write(encode({"id": request_id, "cmd": cmd, "args": absolutize_path(args or {})}))
            stream.flush()
            line = stream.readline()
    except (OSError, TimeoutError) as exc:
        raise DaemonUnavailable(f"connection lost: {exc}") from exc
    if not line:
        raise DaemonUnavailable("daemon closed the connection without answering")

    response = decode(line)
    _warn_on_revision_mismatch(response)
    if not response.get("ok"):
        message = str(response.get("error", "unknown daemon error"))
        if error_kind(response.get("kind")) is ErrorKind.REFUSED:
            raise CommandRefused(message)
        raise CommandFailed(message)
    return response.get("result")
