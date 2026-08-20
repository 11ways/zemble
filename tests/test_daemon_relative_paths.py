"""A relative path must be resolved by the caller, never by the daemon (whose cwd is /)."""

from __future__ import annotations

import os

import pytest

from zemble.daemon.client import absolutize_path
from zemble.daemon.server import _root_of


def test_client_absolutizes_local_paths_and_leaves_urls_alone(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The client resolves against its own cwd; git URLs and absolute paths pass through."""
    monkeypatch.chdir(tmp_path)
    assert absolutize_path({"path": "."})["path"] == str(tmp_path), "step 1: '.' becomes the caller's cwd"
    assert absolutize_path({"path": "sub/dir"})["path"] == os.path.join(str(tmp_path), "sub", "dir"), (
        "step 2: a relative path is joined to the caller's cwd"
    )
    assert absolutize_path({"path": "/abs/x"})["path"] == "/abs/x", "step 3: absolute paths are unchanged"
    url = "https://github.com/MinishLab/semble"
    assert absolutize_path({"path": url})["path"] == url, "step 4: git URLs are not touched"
    assert absolutize_path({"query": "x"}) == {"query": "x"}, "step 5: requests without a path are unchanged"


def test_daemon_refuses_a_relative_path() -> None:
    """The daemon never resolves a relative path itself: that would index the filesystem root."""
    with pytest.raises(ValueError, match="absolute path"):
        _root_of({"path": "."})
    assert _root_of({"path": "/home/x"}) == "/home/x"
    assert _root_of({"path": "https://github.com/MinishLab/semble"}).startswith("https://")
