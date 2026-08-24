from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

from zemble.version import __version__

_HOME = Path.home()

Action = Literal["created", "updated", "unchanged", "not-found", "removed", "error", "skipped"]
Mode = Literal["install", "uninstall"]


class IntegrationType(str, Enum):
    """Identifier for one of zemble's install/uninstall integrations."""

    MCP = "mcp"
    INSTRUCTIONS = "instructions"
    SUBAGENT = "subagent"


ZEMBLE_START = "<!-- ZEMBLE_START -->"
ZEMBLE_END = "<!-- ZEMBLE_END -->"


def zemble_pin() -> str:
    """Return the uvx --from specifier for the zemble MCP server.

    Version-pinned for normal installs (rerunning `zemble install` after an
    upgrade rewrites this pin to match). For an editable or local-directory
    install, pins to the local source path instead, so generated configs
    launch the checkout being developed rather than the released package.
    For a non-editable git install, pins to the exact installed commit, since
    that source may not correspond to any released PyPI version at all.
    """
    try:
        raw = importlib.metadata.distribution("zemble").read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            url = data.get("url", "")
            if "dir_info" in data and url.startswith("file://"):
                path = url2pathname(urlparse(url).path)
                return f"{path}[mcp]"
            vcs_info = data.get("vcs_info", {})
            if vcs_info.get("vcs") == "git" and vcs_info.get("commit_id"):
                return f"git+{url}@{vcs_info['commit_id']}#egg=zemble[mcp]"
    except Exception:
        pass
    return f"zemble[mcp]=={__version__}"


ZEMBLE_PIN = zemble_pin()

_STDIO_SERVER_CONFIG: dict[str, object] = {
    "command": "uvx",
    "args": ["--from", ZEMBLE_PIN, "zemble"],
    "type": "stdio",
}

_OPENCODE_SERVER_CONFIG: dict[str, object] = {
    "command": ["uvx", "--from", ZEMBLE_PIN, "zemble"],
    "type": "local",  # opencode uses "local"/"remote", not "stdio"
    "enabled": True,
}

_BARE_STDIO_SERVER_CONFIG: dict[str, object] = {  # Windsurf: command/args only, no "type"
    "command": "uvx",
    "args": ["--from", ZEMBLE_PIN, "zemble"],
}

_ZED_SERVER_CONFIG: dict[str, object] = {  # Zed: command/args only, no "source"
    "command": "uvx",
    "args": ["--from", ZEMBLE_PIN, "zemble"],
}

INSTRUCTIONS = f"""\
{ZEMBLE_START}
## Zemble Code Search

A `zemble` MCP server is available. Its tools:
- `mcp__zemble__search` — search a codebase with a natural-language or code query. Returns file path + exact line.
- `mcp__zemble__find_related` — code similar to a specific file and line (logic-duplication leads, parallel implementations).
- `mcp__zemble__graph_definition` / `graph_callers` / `graph_implementations` / `graph_overrides` / `graph_tests_of` / `graph_neighbors` — the Java symbol graph: where a symbol is declared, who calls it, its subtypes, every override of one method (`Type.member`), its tests, its one-hop neighbourhood.
- `mcp__zemble__outline` — a Java file's or type's declarations, signatures only (~200 tokens).
- `mcp__zemble__explain` — a budgeted evidence bundle for a query (primary chunks + enclosing-type outline + tests + callers, each labelled with why it is there).
- `mcp__zemble__signatures` — a symbol's signature plus its exactly-resolved call sites.
- `mcp__zemble__dupes` — clone classes (exact / alpha-renamed / logic) ranked by weight; a REPORT, never a gate.
- `mcp__zemble__home` — "does this already exist, and where should it live?": existing mechanisms, candidate home modules ranked with reasons, and a verdict. Run it BEFORE designing any mechanism; cite its answer.

`repo` is optional on every tool: it defaults to the directory the session started in (each tool's description announces the resolved path). Output cost is bounded and stated: `search` returns `top_k` (default 5) results of `max_snippet_lines` (default 10) lines; graph tools cap at `limit` (default 50) and always report `total`.

Pick the tool by the shape of the question:
- "Where is X declared / who calls X / what implements or overrides X / what tests X?" -> the `graph_*` tools. MANDATORY for Java symbol relationships; never grep for these.
- "What is the shape of this class?" -> `outline`, instead of reading the file.
- "How does mechanism Y work?" -> `explain`, instead of reading several files.
- "Does this already exist, and where should it live?" -> `home`, BEFORE designing any mechanism.
- "Where is the code that does Z?" -> `search`. Pass `content="docs"` for prose, `"config"` for config files, `"all"` for everything (CLI: `--content`). A workspace root containing several repos is one index; one call covers them all.
- A literal-string sweep feeding a shell pipeline (extract a capture group, count, build a table) -> Grep is fine; that is its niche. After any zemble hit, navigate directly to the file and line; never re-grep for the same content.

For CLI fallback or sub-agents without MCP access, use:

```bash
zemble search "authentication flow" ./my-project --max-snippet-lines 10
zemble search "deployment guide" ./my-project --content docs
zemble find-related src/auth.py 42 ./my-project
zemble graph callers ./my-project Type.method
zemble graph overridden-by ./my-project Type.method
zemble outline ./my-project package.Type
zemble explain ./my-project "what does X do" --budget 3000
zemble dupes ./my-project --kind renamed --limit 20
zemble home ./my-project "a per-user remembered UI preference stored in a cookie"
```

The index is built on first run and cached automatically. If `zemble` is not on `$PATH`, use `uvx --from "{ZEMBLE_PIN}" zemble`.
{ZEMBLE_END}
"""


@dataclass(frozen=True)
class McpConfig:
    """MCP integration config for one agent."""

    path: Path
    key: str
    entry: dict[str, object]
    format: Literal["json", "toml"] = "json"


@dataclass(frozen=True)
class WriteResult:
    """Result of a single file write operation."""

    path: Path
    action: Action


@dataclass(frozen=True)
class AgentTarget:
    """Configuration for a single coding agent integration target."""

    id: str
    display_name: str
    binary: str | None  # for shutil.which detection
    config_dir: Path | None  # directory existence check for detection
    mcp: McpConfig | None
    instructions_path: Path | None  # None = not supported for this agent
    subagent_path: Path | None = None  # global (user-level) sub-agent file; None = unsupported

    def resolved_mcp_path(self) -> Path | None:
        """Return the resolved MCP config path, or None if MCP is unsupported."""
        return self.mcp.path if self.mcp else None


def _opencode_mcp_path() -> Path:
    """Return the opencode config path, preferring .jsonc over .json."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) / "opencode" if xdg else _HOME / ".config" / "opencode"
    jsonc = base / "opencode.jsonc"
    json_ = base / "opencode.json"
    return jsonc if jsonc.exists() else (json_ if json_.exists() else jsonc)


def _vscode_mcp_path() -> Path:
    """Return the user-level VS Code mcp.json path for the current OS."""
    if sys.platform == "darwin":
        base = _HOME / "Library" / "Application Support" / "Code" / "User"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", _HOME)) / "Code" / "User"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", _HOME / ".config")) / "Code" / "User"
    return base / "mcp.json"


AGENTS: list[AgentTarget] = [
    AgentTarget(
        id="claude",
        display_name="Claude Code",
        binary="claude",
        config_dir=_HOME / ".claude",
        mcp=McpConfig(_HOME / ".claude.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".claude" / "CLAUDE.md",
        subagent_path=_HOME / ".claude" / "agents" / "zemble-search.md",
    ),
    AgentTarget(
        id="cursor",
        display_name="Cursor",
        binary="cursor",
        config_dir=_HOME / ".cursor",
        mcp=McpConfig(_HOME / ".cursor" / "mcp.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=None,  # Cursor instructions are project-local .mdc files
        subagent_path=_HOME / ".cursor" / "agents" / "zemble-search.md",
    ),
    AgentTarget(
        id="gemini",
        display_name="Gemini CLI",
        binary="gemini",
        config_dir=_HOME / ".gemini",
        mcp=McpConfig(_HOME / ".gemini" / "settings.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".gemini" / "GEMINI.md",
        subagent_path=_HOME / ".gemini" / "agents" / "zemble-search.md",
    ),
    AgentTarget(
        id="kiro",
        display_name="Kiro",
        binary="kiro",
        config_dir=_HOME / ".kiro",
        mcp=McpConfig(_HOME / ".kiro" / "settings" / "mcp.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".kiro" / "steering" / "zemble.md",
        subagent_path=_HOME / ".kiro" / "agents" / "zemble-search.md",
    ),
    AgentTarget(
        id="opencode",
        display_name="Opencode",
        binary="opencode",
        config_dir=_HOME / ".config" / "opencode",
        mcp=McpConfig(_opencode_mcp_path(), "mcp", _OPENCODE_SERVER_CONFIG),
        instructions_path=_HOME / ".config" / "opencode" / "AGENTS.md",
        subagent_path=_HOME / ".config" / "opencode" / "agents" / "zemble-search.md",
    ),
    AgentTarget(
        id="copilot",
        display_name="GitHub Copilot",
        binary=None,
        config_dir=_HOME / ".config" / "github-copilot",
        mcp=McpConfig(_HOME / ".copilot" / "mcp-config.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=None,
        subagent_path=_HOME / ".copilot" / "agents" / "zemble-search.agent.md",
    ),
    AgentTarget(
        id="codex",
        display_name="Codex",
        binary="codex",
        config_dir=_HOME / ".codex",
        mcp=McpConfig(_HOME / ".codex" / "config.toml", "mcp_servers", _STDIO_SERVER_CONFIG, format="toml"),
        instructions_path=_HOME / ".codex" / "AGENTS.md",
        subagent_path=_HOME / ".codex" / "agents" / "zemble-search.toml",
    ),
    AgentTarget(
        id="zcode",
        display_name="ZCode",
        binary=None,
        config_dir=_HOME / ".zcode",
        mcp=McpConfig(_HOME / ".zcode" / "cli" / "config.json", "mcp.servers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".zcode" / "AGENTS.md",
        subagent_path=_HOME / ".zcode" / "agents" / "zemble-search.md",
    ),
    AgentTarget(
        id="vscode",
        display_name="VS Code",
        binary="code",
        config_dir=None,
        mcp=McpConfig(_vscode_mcp_path(), "servers", _STDIO_SERVER_CONFIG),
        instructions_path=None,
    ),
    AgentTarget(
        id="windsurf",
        display_name="Windsurf",
        binary="windsurf",
        config_dir=_HOME / ".codeium" / "windsurf",
        mcp=McpConfig(_HOME / ".codeium" / "windsurf" / "mcp_config.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=None,
    ),
    AgentTarget(
        id="zed",
        display_name="Zed",
        binary="zed",
        config_dir=_HOME / ".config" / "zed",
        mcp=McpConfig(_HOME / ".config" / "zed" / "settings.json", "context_servers", _ZED_SERVER_CONFIG),
        instructions_path=None,
    ),
    AgentTarget(
        id="reasonix",
        display_name="Reasonix",
        binary="reasonix",
        config_dir=_HOME / ".config" / "reasonix",
        # ~/.reasonix/config.json is the legacy v0.x path still read by v1.x for backwards compat.
        # The v1.x canonical config is ~/.config/reasonix/config.toml ([[plugins]]), but the JSON
        # path requires no special TOML handling and works for new users who have never had v0.x.
        mcp=McpConfig(_HOME / ".reasonix" / "config.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".config" / "reasonix" / "REASONIX.md",
        subagent_path=_HOME / ".reasonix" / "skills" / "zemble-search.md",
    ),
    AgentTarget(
        id="pi",
        display_name="Pi",
        binary="pi",
        config_dir=_HOME / ".pi",
        mcp=McpConfig(_HOME / ".pi" / "agent" / "mcp.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=None,
        subagent_path=_HOME / ".pi" / "agents" / "zemble-search.md",
    ),
    AgentTarget(
        id="commandcode",
        display_name="Command Code",
        binary=None,
        config_dir=_HOME / ".commandcode",
        mcp=McpConfig(_HOME / ".commandcode" / "mcp.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".commandcode" / "AGENTS.md",
        subagent_path=_HOME / ".commandcode" / "agents" / "zemble-search.md",
    ),
    AgentTarget(
        id="antigravity",
        display_name="Antigravity",
        binary="agy",
        config_dir=_HOME / ".gemini" / "antigravity-cli",
        mcp=McpConfig(_HOME / ".gemini" / "config" / "mcp_config.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".gemini" / "GEMINI.md",
        subagent_path=_HOME / ".gemini" / "config" / "skills" / "zemble-search" / "SKILL.md",
    ),
]


def is_detected(agent: AgentTarget) -> bool:
    """Return True if the agent appears to be installed."""
    if agent.binary and shutil.which(agent.binary):
        return True
    return bool(agent.config_dir and agent.config_dir.exists())
