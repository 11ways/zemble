"""Shared facts that identify a declared Zemble workspace."""

from pathlib import Path

#: Where a workspace declares its module map, relative to its root.
HOME_CONFIG_RELATIVE_PATH = ".zemble/home.toml"


def resolve_home_root(path: str | Path) -> Path:
    """Resolve a home query to the nearest ancestor declaring its module boundaries."""
    requested = Path(path).expanduser().resolve()
    for candidate in (requested, *requested.parents):
        # Present but malformed declarations still select this boundary; loading
        # them must refuse, never silently fall through to a different workspace.
        declaration = candidate / HOME_CONFIG_RELATIVE_PATH
        if declaration.exists() or declaration.is_symlink():
            return candidate
    return requested


__all__ = ["HOME_CONFIG_RELATIVE_PATH", "resolve_home_root"]
