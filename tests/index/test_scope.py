"""Safety checks for semantic-index roots that accidentally span unrelated workspaces."""

from __future__ import annotations

from pathlib import Path

import pytest

from zemble.embedding.pricing import CONFIRM_ENV
from zemble.index import BroadRootRefused, ZembleIndex
from zemble.index.scope import MAX_UNDECLARED_REPOSITORIES, require_declared_scope


def _repositories(root: Path, count: int) -> None:
    """Plant `count` small Git roots under grouping directories."""
    for index in range(count):
        repository = root / f"group-{index % 3}" / f"repo-{index}"
        repository.mkdir(parents=True)
        marker = repository / ".git"
        if index % 2:
            marker.write_text("gitdir: /tmp/worktree\n", encoding="utf-8")
        else:
            marker.mkdir()
        (repository / "module.py").write_text(f"def value_{index}():\n    return {index}\n", encoding="utf-8")


def test_broad_root_refusal_journey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_embedder) -> None:
    """A parent of workspaces is refused, while explicit project boundaries remain indexable."""
    broad = tmp_path / "projects"
    _repositories(broad, MAX_UNDECLARED_REPOSITORIES)

    # 1. The bounded preflight refuses before the semantic build reaches the embedder.
    before = len(mock_embedder.document_calls)
    with pytest.raises(BroadRootRefused) as raised:
        ZembleIndex.from_path(broad, embedder=mock_embedder)
    message = str(raised.value)
    assert len(mock_embedder.document_calls) == before, "step 1: refusal happens before embedding"
    for fragment in (str(broad), "nested Git repositories", ".zemble/home.toml", "--yes"):
        assert fragment in message, f"step 1: the refusal names {fragment}"

    # 2. A smaller ad-hoc multi-repo tree is still a valid local search root.
    small = tmp_path / "small"
    _repositories(small, MAX_UNDECLARED_REPOSITORIES - 1)
    require_declared_scope(small)

    # 3. A declared workspace may deliberately span any number of repositories.
    (broad / ".zemble").mkdir()
    (broad / ".zemble" / "home.toml").write_text("order = []\n", encoding="utf-8")
    require_declared_scope(broad)

    # 4. The existing embedding confirmation is the explicit one-shot override.
    (broad / ".zemble" / "home.toml").unlink()
    monkeypatch.setenv(CONFIRM_ENV, "1")
    require_declared_scope(broad)

    # 5. A Git root is already an explicit boundary, even when it carries nested repositories.
    monkeypatch.delenv(CONFIRM_ENV)
    (broad / ".git").mkdir()
    require_declared_scope(broad)
