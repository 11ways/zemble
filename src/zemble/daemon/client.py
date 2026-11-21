"""Thin client for the zemble daemon: connect, auto-start, one request, one response.

A refusal is an answer to report. Anything else ends in an answer too: `answer` waits out a busy
daemon, starts one that is gone, and grants this process the work for a request the daemon cannot
answer (outgrown, unreachable or failing), never the other way round.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextvars import ContextVar
from itertools import count
from pathlib import Path
from typing import Any

from zemble.daemon.protocol import (
    ACCEPTS_BUSY_FIELD,
    ANSWER_GRACE_SECONDS,
    CONNECT_TIMEOUT_SECONDS,
    CONSTRUCTION_WAIT_SECONDS,
    DEADLINE_FIELD,
    DEFAULT_QUERY_WAIT_SECONDS,
    QUERY_WAIT_ENV,
    REVISION_FIELD,
    START_TIMEOUT_SECONDS,
    STOP_RELEASE_SECONDS,
    CommandBusy,
    CommandFailed,
    CommandOutgrown,
    CommandRefused,
    DaemonDisabled,
    DaemonError,
    DaemonUnavailable,
    ErrorKind,
    QueryTimedOut,
    decode,
    encode,
    error_kind,
    graph_job_in_flight,
    lock_path,
    log_path,
    pid_path,
    process_alive,
    read_pid,
    socket_path,
)
from zemble.utils import is_git_url

logger = logging.getLogger(__name__)

_request_ids = count(1)
#: Last observed response revision; the next call connects to the current socket again.
_daemon_revision: str | None = None
#: Set once a process must never talk to a daemon: the daemon itself, or --no-daemon.
_disabled_reason: str | None = None
#: Why the daemon could not answer the request running in this context, which this process then answers.
_answering_here: ContextVar[str | None] = ContextVar("zemble_answering_here", default=None)
#: When this process last asked whether the daemon runs older code than its own; None: not yet.
_replace_checked_at: float | None = None
_replace_lock = threading.Lock()

#: Seconds between two checks of a daemon whose revision differs from this client's.
REPLACE_CHECK_SECONDS = 60.0
#: How long `answer` first waits before asking a busy or failing daemon again, and the longest it waits.
RETRY_FIRST_SECONDS = 0.25
RETRY_LONGEST_SECONDS = 5.0
#: How often a daemon may be unreachable, and how often it may fail one request, before this process answers it.
UNAVAILABLE_TRIES = 3
FAILED_TRIES = 2


class AnswerHere(Exception):
    """The daemon cannot answer this request, so the process that asked answers it: see `answer`."""


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


def in_process() -> bool:
    """Allow local work for --no-daemon, inside the daemon itself, and for a request the daemon cannot answer."""
    return _disabled_reason in {"--no-daemon", "running inside the daemon"} or _answering_here.get() is not None


def answer_here(reason: str) -> None:
    """Grant the request running in this context the local work its daemon cannot do, for `reason`."""
    _answering_here.set(reason)


def granted() -> str | None:
    """Why the request running in this context answers here because its daemon could not, or None."""
    return _answering_here.get()


@contextlib.contextmanager
def request_scope() -> Iterator[None]:
    """Scope what `answer_here` grants to one request, which a server runs each tool call in."""
    token = _answering_here.set(None)
    try:
        yield
    finally:
        _answering_here.reset(token)


def daemon_revision() -> str | None:
    """Return the revision observed on the last daemon response."""
    return _daemon_revision


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


def lock_held() -> bool:
    """Return whether a daemon process holds the singleton lock: listening, starting, or still stopping.

    AIDEV-NOTE: a stopping daemon closes its socket first and releases the lock only after its read
    drain, so "not listening" is not "gone": a daemon started meanwhile finds the lock taken and exits.
    """
    path = lock_path()
    if not path.exists():
        return False
    try:
        handle = open(path, "a", encoding="utf-8")
    except OSError:
        return False
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def wait_released(timeout: float | None = None) -> bool:
    """Wait until no daemon holds the singleton lock; whether it was released within *timeout* (default the bound)."""
    deadline = time.monotonic() + (STOP_RELEASE_SECONDS if timeout is None else timeout)
    while lock_held():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def stop(pid: int | None = None) -> str | None:
    """Ask the running daemon to exit and wait, up to `STOP_RELEASE_SECONDS`, until it released its lock.

    :param pid: The daemon to stop; a daemon of another pid listening by now is left running.
    :return: None once no daemon holds the lock, else what holds it and why it is still held.
    """
    if is_running():
        try:
            reply = call("shutdown", {} if pid is None else {"pid": pid}, auto_start=False, timeout=10.0)
        except DaemonError as exc:
            return f"the daemon did not take the stop request: {exc}"
        if reply.get("stopping") is False:
            return f"the daemon listening now is pid {reply.get('pid')}, not pid {pid}; it was left running"
    if not wait_released():
        return (
            f"zemble daemon pid {read_pid() or 'unknown'} still holds {lock_path()} {STOP_RELEASE_SECONDS:.0f} s "
            f"after it was asked to stop; it is still stopping (see {log_path()})"
        )
    clear_stale()
    return None


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

    A daemon that holds the lock without listening is waited out first: one still starting comes
    up and is used, one still stopping releases the lock and is then replaced.

    :param timeout: Seconds to wait for the spawned daemon's socket to answer.
    :raises DaemonUnavailable: If the daemon did not come up in time, naming what holds the lock.
    """
    deadline = time.monotonic() + STOP_RELEASE_SECONDS
    while lock_held():
        if is_running():
            return
        if time.monotonic() >= deadline:
            raise DaemonUnavailable(
                f"zemble daemon pid {read_pid() or 'unknown'} holds {lock_path()} without listening "
                f"{STOP_RELEASE_SECONDS:.0f} s on (still stopping or starting; see {log_path()})"
            )
        time.sleep(0.1)
    clear_stale()
    log_file = log_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = open(log_file, "ab")  # handed to the child; closed here right after the spawn
    except OSError as exc:
        raise DaemonUnavailable(f"cannot open daemon log {log_file}: {exc}") from exc
    try:
        process = subprocess.Popen(
            # --log-file: the child logs through a rotating handler instead of the inherited
            # stderr, so the log below only ever collects crash output written before that.
            [sys.executable, "-m", "zemble.daemon", "run", "--log-file"],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            start_new_session=True,
            cwd="/",
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2"},
        )
    except OSError as exc:
        raise DaemonUnavailable(f"cannot start daemon: {exc}") from exc
    finally:
        handle.close()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running():
            return
        if process.poll() is not None and not lock_held():
            raise DaemonUnavailable(f"the daemon exited with {process.returncode} before listening (see {log_file})")
        time.sleep(0.05)
    raise DaemonUnavailable(f"did not start within {timeout:.0f}s (see {log_file})")


def older_daemon(status: dict[str, Any]) -> str | None:
    """Return why the daemon reporting *status* runs older code of this client's checkout, else None.

    Older means another revision of the same source root that is a strict ancestor of this
    client's: a daemon of another install, of a newer revision, or of an unknown one is never older.
    """
    from zemble.runtime.identity import identity, is_ancestor

    mine = identity()
    runtime = status.get("runtime") or {}
    theirs, root = runtime.get("source_revision"), runtime.get("source_root")
    if not mine.source_revision or not isinstance(theirs, str) or not isinstance(root, str):
        return None
    if Path(root).resolve() != Path(mine.source_root).resolve():
        return None
    if not is_ancestor(Path(mine.source_root), theirs, mine.source_revision):
        return None
    return (
        f"zemble daemon pid {status.get('pid')} runs revision {theirs}, older than this client's {mine.source_revision}"
    )


def in_flight(status: dict[str, Any]) -> list[str]:
    """Name the construction a daemon's status reports running or queued, which stopping it now discards.

    Pending watcher changes are not in it: a new daemon's first full rescan covers them.
    """
    busy = [f"index build of {entry.get('root')}" for entry in status.get("building", [])]
    busy += [f"rebuild of {entry.get('root')}" for entry in status.get("indexes", []) if entry.get("rebuilding")]
    jobs = (status.get("graph_construction") or {}).get("jobs", [])
    busy += [f"graph job of {job.get('root')}" for job in jobs if graph_job_in_flight(job.get("state"))]
    return busy


def _replace_older_daemon() -> None:
    """Replace a daemon running older code of this client's checkout before asking it, unless it is building.

    AIDEV-NOTE: an editable install's daemon keeps the code it started with until something replaces
    it. Asked once per process, then every REPLACE_CHECK_SECONDS while a response names another revision.
    """
    global _replace_checked_at
    from zemble.runtime.identity import identity

    mine = identity().source_revision
    with _replace_lock:
        now = time.monotonic()
        if _replace_checked_at is not None and (
            _daemon_revision == mine or now - _replace_checked_at < REPLACE_CHECK_SECONDS
        ):
            return
        _replace_checked_at = now
        try:
            if call("ping", auto_start=False, timeout=5.0).get("source_revision") == mine:
                return
            status = call("status", auto_start=False, timeout=10.0)
        except DaemonError:
            return
        reason = older_daemon(status)
        if reason is None:
            return
        busy = in_flight(status)
        if busy:
            logger.warning("%s; not replaced while it runs %s; continuing through it", reason, ", ".join(busy[:3]))
            return
        logger.warning("%s; replacing it", reason)
        problem = stop(pid=status.get("pid"))
        if problem is not None:
            logger.warning("could not replace the older daemon: %s; continuing through it", problem)
            return
        try:
            spawn()
        except DaemonUnavailable as exc:
            logger.warning("could not start the newer daemon: %s", exc)


def _observe_revision(response: dict[str, Any]) -> None:
    """Notice each daemon revision transition without routing work back into a stale client."""
    global _daemon_revision
    from zemble.runtime.identity import identity

    theirs = response.get(REVISION_FIELD)
    mine = identity().source_revision
    if not isinstance(theirs, str) or theirs == _daemon_revision:
        return
    previous = _daemon_revision
    _daemon_revision = theirs
    if previous is not None:
        logger.info("zemble daemon revision changed %s -> %s; continuing through the new daemon", previous, theirs)
    if mine is not None and theirs != mine:
        logger.warning("zemble daemon runs revision %s, client runs %s; continuing through the daemon", theirs, mine)


def _response_result(response: dict[str, Any]) -> Any:
    """Reject failed or empty replies rather than mistaking them for permission to answer locally."""
    if response.get("ok") is not True:
        message = str(response.get("error", "unknown daemon error"))
        kind = error_kind(response.get("kind"))
        if kind is ErrorKind.REFUSED:
            raise CommandRefused(message)
        if kind is ErrorKind.BUSY:
            raise CommandBusy(message)
        if kind is ErrorKind.OUTGROWN:
            raise CommandOutgrown(message)
        raise CommandFailed(message)
    result = response.get("result")
    if result is None:
        raise CommandFailed("daemon returned no result")
    return result


def call(
    cmd: str,
    args: dict[str, Any] | None = None,
    *,
    auto_start: bool = True,
    timeout: float | None = CONSTRUCTION_WAIT_SECONDS,
) -> Any:
    """Send one command to the daemon and return its result.

    :param cmd: Command name, as registered in the server's command table.
    :param args: Command arguments.
    :param auto_start: Whether to spawn a daemon when none is listening.
    :param timeout: How long the daemon may take, shared index or graph construction included; it
        travels as the request's deadline. The default waits out a whole construction, so a cold
        request blocks until it can answer instead of returning busy. None blocks the socket too.
    :return: The command's result.
    :raises DaemonUnavailable: If this process may not use a daemon, or none could be reached.
    :raises CommandRefused: If the daemon deliberately refused the command.
    :raises CommandFailed: If the daemon reports an error, including retryable CommandBusy replies.
    """
    reason = disabled_reason()
    if reason is not None:
        raise DaemonDisabled(reason)
    if auto_start:
        _replace_older_daemon()
    try:
        client = _connect()
    except DaemonUnavailable:
        if not auto_start:
            raise
        spawn()
        client = _connect()
    request_id = next(_request_ids)
    # AIDEV-NOTE: a request without a deadline gets the daemon's 25 s legacy default, which turned
    # every cold query into "busy, retry" while one shared construction ran. Construction is joined,
    # never repeated, so waiting for it costs the daemon nothing; the caller must say how long.
    wait = CONSTRUCTION_WAIT_SECONDS if timeout is None else min(timeout, CONSTRUCTION_WAIT_SECONDS)
    request = {
        "id": request_id,
        "cmd": cmd,
        "args": absolutize_path(args or {}),
        ACCEPTS_BUSY_FIELD: True,
        DEADLINE_FIELD: wait * 1000,
    }
    try:
        client.settimeout(None if timeout is None else timeout + min(timeout, ANSWER_GRACE_SECONDS))
        with client, client.makefile("rwb") as stream:
            stream.write(encode(request))
            stream.flush()
            line = stream.readline()
    except (OSError, TimeoutError) as exc:
        raise DaemonUnavailable(f"connection lost: {exc}") from exc
    if not line:
        raise DaemonUnavailable("daemon closed the connection without answering")

    response = decode(line)
    if response.get("id") != request_id:
        raise CommandFailed("daemon response has a mismatched request id")
    _observe_revision(response)
    return _response_result(response)


def answer(cmd: str, args: dict[str, Any] | None = None) -> Any:
    """Send one command until the daemon answers it.

    AIDEV-NOTE: zemble never fails a query for a reason another try fixes. Busy is waited out for
    up to `ZEMBLE_QUERY_WAIT_SECONDS` in all (a query used to wait 30+ minutes behind constructions),
    then reported as `QueryTimedOut`; an unreachable daemon is started again; only what the daemon
    cannot do at all - a read it outgrew, a daemon that will not come up, one that keeps failing the
    request, or a daemon this process may not use - becomes AnswerHere, and the caller answers in
    its own process. A refusal (scope, work ceiling, spending budget) is still raised: it is the answer.

    :raises CommandRefused: When the daemon refused the command on purpose.
    :raises AnswerHere: When only this process can answer the request.
    """
    delay = RETRY_FIRST_SECONDS
    tries = {DaemonUnavailable: 0, CommandFailed: 0}
    budget = query_wait_seconds()
    deadline = time.monotonic() + budget
    while True:
        remaining = deadline - time.monotonic()
        try:
            result = call(cmd, args, timeout=max(remaining, 1.0))
            if result is not None:
                return result
            raise CommandFailed("daemon returned no result")
        except CommandRefused:
            raise
        except (DaemonDisabled, CommandOutgrown) as exc:
            raise AnswerHere(str(exc)) from exc
        except CommandBusy as exc:
            if time.monotonic() + delay >= deadline:
                raise QueryTimedOut(
                    f"zemble {cmd}: the daemon is still busy after {budget:.0f} s ({exc}). Its work goes on; "
                    f"ask again shortly, `zemble daemon status` shows what it is doing, and {QUERY_WAIT_ENV} "
                    "sets how long a query waits."
                ) from exc
            logger.info("daemon busy with %s, asking again in %.2f s: %s", cmd, delay, exc)
        except (DaemonUnavailable, CommandFailed) as exc:
            _count_try(tries, exc, cmd)
        time.sleep(delay)
        delay = min(delay * 2, RETRY_LONGEST_SECONDS)


def query_wait_seconds() -> float:
    """Return how long one query may wait for the daemon in all, from `ZEMBLE_QUERY_WAIT_SECONDS`."""
    raw = os.environ.get(QUERY_WAIT_ENV)
    try:
        value = float(raw) if raw else DEFAULT_QUERY_WAIT_SECONDS
    except ValueError:
        logger.warning("Ignoring %s=%r: not a number of seconds", QUERY_WAIT_ENV, raw)
        value = DEFAULT_QUERY_WAIT_SECONDS
    return value if value > 0 else DEFAULT_QUERY_WAIT_SECONDS


def _count_try(tries: dict[type, int], error: DaemonUnavailable | CommandFailed, cmd: str) -> None:
    """Count one more unreachable or failed try at cmd, raising AnswerHere once its kind ran out of tries."""
    kind, limit = (
        (DaemonUnavailable, UNAVAILABLE_TRIES)
        if isinstance(error, DaemonUnavailable)
        else (CommandFailed, FAILED_TRIES)
    )
    tries[kind] += 1
    if tries[kind] >= limit:
        label = "unavailable" if kind is DaemonUnavailable else f"failed {cmd}"
        raise AnswerHere(f"daemon {label}: {error}") from error
