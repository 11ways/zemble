"""The `status` MCP tool, plus the throttled staleness warning the server logs.

AIDEV-NOTE: the warning is checked at most once a minute because `source_changed_since_start`
shells out to git; a per-request git call would tax every search to report a rare event.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ContentBlock

from zemble.runtime.identity import identity, stale_note, status_payload

logger = logging.getLogger(__name__)

#: Minimum seconds between two staleness probes in one process.
STALE_CHECK_INTERVAL_SECONDS = 60.0

_last_check_at = 0.0
_warned = False


def warn_if_stale(now: float | None = None) -> str | None:
    """Log one WARNING the first time this process is found to be serving stale source.

    :param now: Monotonic timestamp to judge the throttle by; defaults to the clock.
    :return: The note that was logged, or None when nothing was logged.
    """
    global _last_check_at, _warned
    if _warned:
        return None
    stamp = time.monotonic() if now is None else now
    if _last_check_at and stamp - _last_check_at < STALE_CHECK_INTERVAL_SECONDS:
        return None
    _last_check_at = stamp
    note = stale_note()
    if note is None:
        return None
    _warned = True
    logger.warning("%s", note)
    return note


class StaleAwareFastMCP(FastMCP):
    """A FastMCP that checks, at the start of every tool call, whether its source went stale.

    AIDEV-NOTE: subclassing rather than wrapping `server.call_tool` after construction:
    FastMCP binds that method into the low-level request handlers in `__init__`, so a later
    instance-attribute wrap would only be seen by direct callers, never by a real request.
    """

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Sequence[ContentBlock] | dict[str, Any]:
        """Warn once about stale source, then answer the tool call as FastMCP would."""
        warn_if_stale()
        return await super().call_tool(name, arguments)


def register_status_tool(server: FastMCP) -> None:
    """Register the runtime-identity tool on a FastMCP server."""

    @server.tool(structured_output=False)
    async def status() -> dict[str, Any]:
        """Report which zemble code this MCP server is running and whether it went stale.

        zemble is installed as an editable checkout, so a server keeps serving the snapshot
        it started with; `stale` means the checkout moved since and the server needs a restart.
        """
        return await asyncio.to_thread(status_payload, identity())
