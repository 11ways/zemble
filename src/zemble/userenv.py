"""Load per-user settings (API keys, default embedder/reranker) from one private env file.

The file is ``$ZEMBLE_ENV_FILE``, else ``$XDG_CONFIG_HOME/zemble/env``, else
``~/.config/zemble/env``; ``KEY=VALUE`` lines, ``#`` comments, optional ``export `` prefix
and surrounding quotes. Values never override variables already present in the process
environment, so an agent config or a shell export always wins over the file. This is what
lets a secret stay in a mode-600 file instead of an MCP config.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_FILE_VAR = "ZEMBLE_ENV_FILE"
_STATE_VAR = "_ZEMBLE_USER_ENV_LOADED"


def user_env_path() -> Path:
    """Return the env file location without checking that it exists."""
    explicit = os.environ.get(ENV_FILE_VAR)
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "zemble" / "env"


def parse_env_lines(lines: list[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines into a dict, skipping blanks, comments and malformed lines."""
    values: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_user_env(path: Path | None = None) -> dict[str, str]:
    """Apply the env file to ``os.environ`` for keys not already set; return what was applied.

    Idempotent per process: a second call is a no-op unless an explicit ``path`` is given.
    A missing file is not an error; an unreadable one logs once and is ignored.
    """
    if path is None:
        if os.environ.get(_STATE_VAR):
            return {}
        os.environ[_STATE_VAR] = "1"
        path = user_env_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)
        return {}
    applied: dict[str, str] = {}
    for key, value in parse_env_lines(text.splitlines()).items():
        if key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
