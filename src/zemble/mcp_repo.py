"""The default `repo` an MCP tool falls back to, fixed at server start.

Lives in its own module so every per-package MCP surface can import it without
touching `zemble.mcp`, which imports those surfaces to register their tools.

AIDEV-NOTE: resolved at import time, which is process start for a stdio server.
Claude Code launches an MCP server with its working directory set to the
session's project, so this is the workspace the caller means when `repo` is
omitted. The daemon never uses it: its cwd is "/" and it refuses relative paths.
"""

from __future__ import annotations

import os

DEFAULT_REPO: str = os.getcwd()


def resolve_repo(repo: str | None) -> str:
    """Return the repo the caller meant, falling back to the server's start directory."""
    return repo if repo else DEFAULT_REPO


def with_default_note(description: str) -> str:
    """Append the announced fallback to a repo parameter description."""
    return f"{description} Optional: when omitted, the server's start directory is used: {DEFAULT_REPO}"
