"""The `status` MCP tool, the throttled staleness warning the server logs, and the missing-argument check.

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
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ContentBlock

from zemble.nib.parts import nib_text_only
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


def missing_arguments(server: FastMCP, name: str, arguments: dict[str, Any] | None) -> None:
    """Refuse a call that leaves out a required argument, naming each one and what it takes, from the tool's schema.

    AIDEV-NOTE: FastMCP would otherwise answer with pydantic's validation dump. A client that
    defers tool schemas (Claude Code loads them through ToolSearch) can send `{}` for a tool
    whose schema it never loaded, so an empty call says so. An unknown tool is left to FastMCP.

    :raises ToolError: When any argument the schema marks required is absent.
    """
    tool = server._tool_manager.get_tool(name)
    if tool is None:
        return
    given = arguments or {}
    properties = tool.parameters.get("properties", {})
    missing = [field for field in tool.parameters.get("required", []) if field not in given]
    if not missing:
        return
    lines = [f"{name} needs {_expected(field, properties.get(field, {}))}" for field in missing]
    if not given:
        lines.append("The call carried no arguments at all; load this tool's schema before calling it.")
    raise ToolError("\n".join(lines))


def _expected(field: str, schema: dict[str, Any]) -> str:
    """Describe one argument as its schema declares it: name, JSON type and description."""
    kind = schema.get("type")
    head = f"`{field}` ({kind})" if isinstance(kind, str) else f"`{field}`"
    description = schema.get("description")
    return f"{head}: {description}" if description else head


class StaleAwareFastMCP(FastMCP):
    """A FastMCP that checks, at the start of every tool call, whether its source went stale.

    AIDEV-NOTE: subclassing rather than wrapping `server.call_tool` after construction:
    FastMCP binds that method into the low-level request handlers in `__init__`, so a later
    instance-attribute wrap would only be seen by direct callers, never by a real request.
    """

    def before_call(self, name: str, arguments: dict[str, Any] | None) -> None:
        """Run what every tool call in every server variant does first: the staleness probe, then the argument check."""
        warn_if_stale()
        missing_arguments(self, name, arguments)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Sequence[ContentBlock] | dict[str, Any]:
        """Warn once about stale source and refuse missing arguments, then answer the tool call as FastMCP would."""
        from zemble.daemon.client import request_scope

        with request_scope():
            self.before_call(name, arguments)
            return await super().call_tool(name, arguments)


def register_status_tool(server: FastMCP) -> None:
    """Register the runtime-identity tool on a FastMCP server."""
    nib_text_only(server, "status")

    @server.tool(structured_output=False)
    async def status() -> dict[str, Any]:
        """Report which zemble code this MCP server is running and whether it went stale.

        zemble is installed as an editable checkout, so a server keeps serving the snapshot
        it started with; `stale` means the checkout moved since and the server needs a restart.
        """
        from zemble.daemon import client
        from zemble.daemon.protocol import DaemonError, failure_message

        payload = await asyncio.to_thread(status_payload, identity())
        daemon = None
        if not client.in_process():
            try:
                daemon = await asyncio.to_thread(client.call, "status", auto_start=False, timeout=5)
            except DaemonError as error:
                daemon = {"error": failure_message(error)}
        return {**payload, "daemon": daemon, "daemon_revision": client.daemon_revision()}
