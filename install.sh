#!/usr/bin/env bash
# Reinstall zemble as an editable uv tool from this checkout and stop the stale daemon.
# AIDEV-NOTE: a running MCP server keeps its old code until the Claude Code session restarts.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

uv tool install --editable --reinstall ".[mcp]"

# The daemon would otherwise keep serving the previous code for up to 30 idle minutes.
zemble daemon stop || true

echo "Installed: $(command -v zemble) ($(zemble --version))"
