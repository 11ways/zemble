"""The baseline file every report diffs against: one run's keyed entries, as a versioned JSON document.

`zemble dupes` and `zemble health` each own what an entry holds and how a diff reads it; this is
where the document itself is written, read and refused, so both refuse a stale one the same way.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

_Loaded = TypeVar("_Loaded")


def save_document(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a baseline document, creating parent directories."""
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def load_document(path: str | Path, version: int) -> dict[str, Any]:
    """Read a baseline document of one version.

    :raises ValueError: If the file cannot be read, is not JSON, or is not a version `version` document.
    """
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read baseline {source}: {error}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Baseline {source} is not valid JSON: {error}") from None
    if not isinstance(payload, dict) or payload.get("version") != version:
        raise ValueError(f"Baseline {source} is not a version {version} document")
    return payload


def load_for_diff(path: Path, relative: str, load: Callable[[Path], _Loaded]) -> tuple[_Loaded | None, list[str]]:
    """Load a workspace's fixed-path baseline for a surface that answers instead of failing.

    :param relative: How the note names the file, relative to the workspace.
    :return: The baseline, or None plus the note saying why there is none.
    """
    try:
        return load(path), []
    except ValueError as error:
        if path.is_file():
            return None, [f"baseline: {error}"]
        return None, [f"baseline: none at {relative}; pass save_baseline=true to write one"]


__all__ = ["load_document", "load_for_diff", "save_document"]
