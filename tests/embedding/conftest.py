from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every embedding test off the real user cache directory."""
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(tmp_path / "zemble-cache"))
