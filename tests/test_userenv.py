"""The per-user env file: parsed leniently, never overrides the real environment."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from zemble.userenv import ENV_FILE_VAR, load_user_env, parse_env_lines, user_env_path


def test_user_env_file_journey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Walk one env file through parsing, selection, precedence and the missing-file case."""
    env_file = tmp_path / "env"
    env_file.write_text(
        "# comment\n"
        "VOYAGE_API_KEY=pa-secret\n"
        "export ZEMBLE_EMBEDDER='voyage:voyage-4-lite@1024'\n"
        'ZEMBLE_RERANKER="voyage:rerank-2.5-lite"\n'
        "ZEMBLE_RERANK_ALPHA = 0.7\n"
        "not a valid line\n"
        "=novalue\n",
        encoding="utf-8",
    )
    # step 1: parsing is lenient about comments, export prefixes, quotes and spaces
    parsed = parse_env_lines(env_file.read_text().splitlines())
    assert parsed == {
        "VOYAGE_API_KEY": "pa-secret",
        "ZEMBLE_EMBEDDER": "voyage:voyage-4-lite@1024",
        "ZEMBLE_RERANKER": "voyage:rerank-2.5-lite",
        "ZEMBLE_RERANK_ALPHA": "0.7",
    }, "step 1: every well-formed line is read, malformed ones are skipped"
    # step 2: the explicit file variable selects the file
    monkeypatch.setenv(ENV_FILE_VAR, str(env_file))
    assert user_env_path() == env_file, "step 2: ZEMBLE_ENV_FILE wins over the XDG default"
    # step 3: values already in the environment are never overridden; patch.dict restores the
    # whole environment afterwards, including keys the loader adds
    with mock.patch.dict(os.environ, {"ZEMBLE_EMBEDDER": "model2vec:explicit"}, clear=False):
        for key in ("VOYAGE_API_KEY", "ZEMBLE_RERANKER", "ZEMBLE_RERANK_ALPHA"):
            os.environ.pop(key, None)
        applied = load_user_env(env_file)
        assert os.environ["ZEMBLE_EMBEDDER"] == "model2vec:explicit", "step 3: explicit env wins over the file"
        assert os.environ["VOYAGE_API_KEY"] == "pa-secret", "step 3: absent keys are filled from the file"
        assert set(applied) == {"VOYAGE_API_KEY", "ZEMBLE_RERANKER", "ZEMBLE_RERANK_ALPHA"}, (
            "step 3: only applied keys are reported"
        )
    assert os.environ.get("VOYAGE_API_KEY") != "pa-secret", "step 3: the fake key does not leak past the block"
    # step 4: a missing file is a silent no-op
    assert load_user_env(tmp_path / "absent") == {}, "step 4: no file, no change, no error"


def test_default_path_follows_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without ZEMBLE_ENV_FILE the file lives under XDG_CONFIG_HOME/zemble/env."""
    monkeypatch.delenv(ENV_FILE_VAR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert user_env_path() == tmp_path / "zemble" / "env"
