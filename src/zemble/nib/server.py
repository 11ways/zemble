"""The FastMCP server `zemble nib-provider` runs: the default tools, answering with nib parts too."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mcp.types import CallToolResult, ContentBlock, TextContent

from zemble.nib.parts import NibAnswer, NibMode, structured, text_part
from zemble.runtime.mcp import StaleAwareFastMCP


class NibProviderFastMCP(NibMode, StaleAwareFastMCP):
    """A server whose tool results carry nib's `structuredContent` beside their unchanged text.

    AIDEV-NOTE: the tool's raw return is converted with the tool's own `convert_result`,
    exactly as `FastMCP.call_tool` does, so `content` is byte-identical to the default
    server's. A tool that returned no `NibAnswer` (a git URL repo) and was not declared
    text only gets no structured content at all, and nib then shows its text as prose:
    unknown answers fail closed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the server with no tool declared text only yet."""
        super().__init__(*args, **kwargs)
        self.text_only_tools = set()

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any] | CallToolResult:
        """Answer one tool call, adding nib's parts when the tool supplied an answer for nib."""
        from zemble.daemon.client import request_scope

        with request_scope():
            self.before_call(name, arguments)
            raw = await self._tool_manager.call_tool(name, arguments, context=self.get_context(), convert_result=False)
        tool = self._tool_manager.get_tool(name)
        if tool is None:
            raise RuntimeError(f"tool {name!r} ran but is no longer registered")
        if isinstance(raw, NibAnswer):
            plain, parts = raw.plain, raw.parts
        elif name in self.text_only_tools:
            plain, parts = raw, None
        else:
            return tool.fn_metadata.convert_result(raw)
        content = list(tool.fn_metadata.convert_result(plain))
        if parts is None:
            parts = [text_part("".join(block.text for block in content if isinstance(block, TextContent)))]
        return CallToolResult(content=content, structuredContent=structured(parts))


__all__ = ["NibProviderFastMCP"]
