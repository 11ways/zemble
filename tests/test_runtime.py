"""Runtime identity: which code a process runs, and whether the checkout moved under it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from zemble.runtime import identity, status_payload
from zemble.runtime.identity import RuntimeIdentity, git_revision
from zemble.version import __version__


@pytest.fixture
def source_copy(tmp_path: Path) -> Path:
    """A throwaway copy of the zemble package, standing in for a loaded source root."""
    root = tmp_path / "zemble"
    root.mkdir()
    (root / "__init__.py").write_text('"""Stand-in package."""\n', encoding="utf-8")
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def test_identity_describes_this_process() -> None:
    """The captured identity names this interpreter's version, source root, pid and start time."""
    current = identity()
    assert current.zemble_version == __version__, "the version comes from version.py"
    assert Path(current.source_root).name == "zemble", "the source root is the loaded package directory"
    assert (Path(current.source_root) / "runtime" / "identity.py").is_file(), "and it is the one holding this module"
    assert current.pid > 0
    assert current.process_started_at.endswith("+00:00"), "the start time is ISO 8601 UTC"
    payload = current.to_dict()
    assert set(payload) == {
        "zemble_version",
        "source_root",
        "editable",
        "source_revision",
        "process_started_at",
        "pid",
    }, "to_dict is the wire shape; the cached file list stays internal"
    assert json.dumps(payload), "the identity is JSON-ready"


def test_editable_is_false_inside_site_packages(tmp_path: Path) -> None:
    """A package installed into site-packages is not an editable checkout."""
    installed = tmp_path / "lib" / "python3.14" / "site-packages" / "zemble"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("", encoding="utf-8")
    assert RuntimeIdentity.capture(installed).editable is False, "site-packages means installed"
    assert RuntimeIdentity.capture(tmp_path).editable is True, "anything else is treated as a checkout"


def test_revision_is_none_outside_a_checkout(source_copy: Path) -> None:
    """A directory that is not a git checkout reports no revision instead of raising."""
    assert git_revision(source_copy) is None, "no repository, no revision"
    assert RuntimeIdentity.capture(source_copy).source_revision is None


def test_stale_flips_when_the_loaded_source_is_touched(source_copy: Path) -> None:
    """Touching a *.py file under the captured source root makes the process stale."""
    captured = RuntimeIdentity.capture(source_copy)
    assert captured.source_changed_since_start() is False, "step 1: nothing moved yet"
    assert status_payload(captured)["stale"] is False, "step 2: and the payload agrees"

    (source_copy / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    stamp = captured.source_mtime + 10
    os.utime(source_copy / "module.py", (stamp, stamp))

    assert captured.source_changed_since_start() is True, "step 3: an edit under the root is a change"
    payload = status_payload(captured)
    assert payload["stale"] is True, "step 4: the payload reports it"
    assert "restart the MCP server" in payload["note"], "step 5: with an actionable note"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required to move a revision")
def test_stale_flips_when_the_revision_moves(source_copy: Path) -> None:
    """A commit under the source root makes an already-captured identity stale."""
    subprocess.run(["git", "init", "-q"], cwd=source_copy, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=source_copy, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=source_copy, check=True)
    subprocess.run(["git", "add", "-A"], cwd=source_copy, check=True)
    subprocess.run(["git", "commit", "-qm", "first"], cwd=source_copy, check=True)

    captured = RuntimeIdentity.capture(source_copy)
    assert captured.source_revision is not None, "a checkout has a revision"
    assert captured.source_changed_since_start() is False, "nothing moved yet"

    (source_copy / "module.py").write_text("VALUE = 3\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "second"], cwd=source_copy, check=True)
    assert captured.current_revision() != captured.source_revision, "HEAD moved"
    assert captured.source_changed_since_start() is True, "so the captured process is stale"


def test_status_cli_prints_identity_without_a_daemon(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`zemble status` reports this process and says when no daemon answers."""
    import argparse

    from zemble.runtime import cli as runtime_cli

    monkeypatch.setattr(runtime_cli, "_daemon_status", lambda: None)
    assert runtime_cli.run_status(argparse.Namespace(json=False)) == 0
    text = capsys.readouterr().out
    assert f"zemble {__version__}" in text, "the version is on the first line"
    assert identity().source_root in text, "and the source root that produced it"
    assert "daemon: not running" in text


def test_status_cli_json_carries_the_daemon_identity(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The machine-readable form nests both identities."""
    import argparse

    from zemble.runtime import cli as runtime_cli

    daemon: dict[str, Any] = {
        "pid": 4242,
        "uptime_seconds": 12.0,
        "runtime": {"source_revision": "deadbee", "process_started_at": "2026-01-01T00:00:00+00:00", "stale": True},
    }
    monkeypatch.setattr(runtime_cli, "_daemon_status", lambda: daemon)
    assert runtime_cli.run_status(argparse.Namespace(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["identity"]["zemble_version"] == __version__
    assert payload["daemon"] == {
        "running": True,
        "source_revision": "deadbee",
        "process_started_at": "2026-01-01T00:00:00+00:00",
        "stale": True,
    }, "the daemon's own identity travels with the answer"


def test_stale_warning_is_logged_once_and_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MCP staleness probe warns once per process and never re-probes within the interval."""
    from zemble.runtime import mcp as runtime_mcp

    monkeypatch.setattr(runtime_mcp, "_last_check_at", 0.0)
    monkeypatch.setattr(runtime_mcp, "_warned", False)
    probes: list[float] = []

    def _note() -> str | None:
        probes.append(1.0)
        return "zemble source changed after this server started (rev a -> b); restart the MCP server"

    monkeypatch.setattr(runtime_mcp, "stale_note", _note)
    assert runtime_mcp.warn_if_stale(now=100.0) is not None, "the first stale probe warns"
    assert runtime_mcp.warn_if_stale(now=100.5) is None, "and never warns twice in one process"
    assert len(probes) == 1, "the second call does not even probe"

    monkeypatch.setattr(runtime_mcp, "_warned", False)
    assert runtime_mcp.warn_if_stale(now=100.5) is None, "a probe inside the interval is skipped"
    assert len(probes) == 1, "the throttle spares the git call"
    assert runtime_mcp.warn_if_stale(now=100.0 + runtime_mcp.STALE_CHECK_INTERVAL_SECONDS + 1) is not None
    assert len(probes) == 2, "past the interval it probes again"


def test_stale_warning_is_silent_when_current(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process running current source logs nothing."""
    from zemble.runtime import mcp as runtime_mcp

    monkeypatch.setattr(runtime_mcp, "_last_check_at", 0.0)
    monkeypatch.setattr(runtime_mcp, "_warned", False)
    monkeypatch.setattr(runtime_mcp, "stale_note", lambda: None)
    assert runtime_mcp.warn_if_stale(now=10.0) is None, "nothing to say about a fresh server"
