# Installation

## Recommended: `zemble install`

The interactive installer detects your installed agents and configures any combination of three integrations globally:

- **[MCP server](#mcp-server)**: exposes Zemble as a native tool your agent can call directly.
- **[AGENTS.md](#instructions-agentsmd--claudemd)**: adds a Zemble usage guide to the agent's config file (`CLAUDE.md`, `AGENTS.md`, etc.).
- **[Sub-agent](#sub-agent)**: installs a dedicated `zemble-search` sub-agent for harnesses that support it.

zemble is not published on PyPI. Install the CLI from a checkout with
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv tool install --editable "/path/to/zemble[mcp]"
zemble install
```

To undo:

```bash
zemble uninstall
```

Supported agents: Claude Code, Cursor, Gemini CLI, Kiro, OpenCode, GitHub Copilot, Codex, VS Code, Windsurf, Zed, Reasonix, Pi, Command Code, Antigravity, and ZCode.

> **Pi prerequisite:** Pi requires the MCP extension to be installed before zemble can connect. Run `pi install npm:pi-mcp-extension` once, then `zemble install`.

### Unattended install

For sandboxed or scripted environments, pass `--agent` to skip the interactive prompts:

```bash
zemble install --agent claude pi --type mcp subagent --yes
```

- `--agent` — one or more agent ids (see the list above; use the lowercase form, e.g. `claude`, `codex`, `pi`).
- `--type` — one or more of `mcp`, `instructions`, `subagent`, or `all` (default: all). Requires `--agent`.
- `-y`/`--yes` — skip the confirmation prompt. Requires `--agent` for a fully non-interactive run.

`zemble uninstall` accepts the same flags.

### Keeping installed configuration up to date

The MCP server config, `AGENTS.md`/`CLAUDE.md` instructions, and sub-agent files that `zemble install` writes all pin `uvx` to the exact zemble you have installed: the local source path for an editable or directory install (`/path/to/zemble[mcp]`), the exact commit for a non-editable git install, and the version otherwise, so agents keep calling the code the instructions were written for. After upgrading (`uv tool upgrade zemble` or `pip install --upgrade zemble`), rerun `zemble install`. This is idempotent and rewrites the pin (and anything else that changed) in place for every agent you select.

---

## Manual setup

### MCP server

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

<details>
<summary>Claude Code</summary>

```bash
claude mcp add zemble -s user -- uvx --from "zemble[mcp]" zemble
```

</details>

<details>
<summary>Cursor</summary>

Add to `~/.cursor/mcp.json` (or `.cursor/mcp.json` in your project):

```json
{
  "mcpServers": {
    "zemble": {
      "command": "uvx",
      "args": ["--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>Codex</summary>

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.zemble]
command = "uvx"
args = ["--from", "zemble[mcp]", "zemble"]
```

</details>

<details>
<summary>OpenCode</summary>

Add to `~/.config/opencode/opencode.jsonc`:

```json
{
  "mcp": {
    "zemble": {
      "type": "local",
      "command": ["uvx", "--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>VS Code</summary>

Add to `.vscode/mcp.json` in your project (or your user profile's `mcp.json`):

```json
{
  "servers": {
    "zemble": {
      "command": "uvx",
      "args": ["--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>GitHub Copilot CLI</summary>

Add to `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "zemble": {
      "command": "uvx",
      "args": ["--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>Windsurf</summary>

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "zemble": {
      "command": "uvx",
      "args": ["--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>Gemini CLI</summary>

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "zemble": {
      "command": "uvx",
      "args": ["--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>Kiro</summary>

Add to `~/.kiro/settings/mcp.json` (or `.kiro/settings/mcp.json` in your project):

```json
{
  "mcpServers": {
    "zemble": {
      "command": "uvx",
      "args": ["--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>Zed</summary>

Add to `~/.config/zed/settings.json` (or `.zed/settings.json` in your project):

```json
{
  "context_servers": {
    "zemble": {
      "source": "custom",
      "command": "uvx",
      "args": ["--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>Reasonix</summary>

Add to `~/.reasonix/config.json` (the backwards-compatible MCP config path read by all Reasonix versions):

```json
{
  "mcpServers": {
    "zemble": {
      "command": "uvx",
      "args": ["--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>Pi</summary>

First install the Pi MCP extension (one-time prerequisite):

```bash
pi install npm:pi-mcp-extension
```

Then add to `~/.pi/agent/mcp.json`:

```json
{
 "mcpServers": {
 "zemble": {
 "command": "uvx",
 "args": ["--from", "zemble[mcp]", "zemble"]
 }
 }
}
```

</details>

<details>
<summary>Antigravity</summary>

Add to `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "zemble": {
      "command": "uvx",
      "args": ["--from", "zemble[mcp]", "zemble"]
    }
  }
}
```

</details>

<details>
<summary>Command Code</summary>

Add to `~/.commandcode/mcp.json`:

```json
{
 "mcpServers": {
 "zemble": {
 "command": "uvx",
 "args": ["--from", "zemble[mcp]", "zemble"]
 }
 }
}
```

Or use the CLI:

```bash
cmd mcp add --scope user zemble -- uvx --from "zemble[mcp]" zemble
```

</details>

<details>
<summary>ZCode</summary>

Add to `~/.zcode/cli/config.json` under the nested `mcp.servers` key (or use Settings -> MCP Servers -> Full configuration mode):

```json
{
  "mcp": {
    "servers": {
      "zemble": {
        "command": "uvx",
        "args": ["--from", "zemble[mcp]", "zemble"],
        "type": "stdio"
      }
    }
  }
}
```

</details>

The MCP server indexes each requested content selection on first use and caches it separately. Searches default to code; append `--content docs`, `--content config`, or `--content all` to the server command to change that default. The `content` argument on an individual MCP search overrides it. For example, in Claude Code:

```bash
claude mcp add zemble -s user -- uvx --from "zemble[mcp]" zemble --content all
```

### Instructions (AGENTS.md / CLAUDE.md)

Add the snippet below to your `AGENTS.md` or `CLAUDE.md` so your agent knows when and how to call the zemble CLI:

```markdown
## Code Search

Use `zemble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

​```bash
zemble search "authentication flow" ./my-project --max-snippet-lines 10  # first 10 lines only, concise
zemble search "save_pretrained" ./my-project                          # full chunk content
zemble search "save model to disk" ./my-project --top-k 10           # more results
​```

The index is built on first run (and cached for subsequent runs) and invalidated automatically when files change.

Use `--content docs` to search documentation and prose, `--content config` for config files (yaml, toml, etc.), or `--content all` to search code, docs, and config:

​```bash
zemble search "deployment guide" ./my-project --content docs
zemble search "database host port" ./my-project --content config
zemble search "authentication" ./my-project --content all
​```

Use `zemble find-related` to discover code similar to a known location (pass `file_path` and `line` from a prior search result):

​```bash
zemble find-related src/auth.py 42 ./my-project
​```

`path` defaults to the current directory when omitted; git URLs are accepted.

If `zemble` is not on `$PATH`, use `uvx --from "zemble[mcp]" zemble` in its place.

### Workflow

1. Start with `zemble search` to find relevant chunks. The index is built and cached automatically.
2. Use `--content docs` for documentation, `--content config` for config files, or `--content all` for everything.
3. Navigate directly to the returned file and line — do not re-search or grep for the same content.
4. Optionally use `zemble find-related` with a promising result's `file_path` and `line` to discover related implementations.
5. Use grep only when you need every occurrence of a literal string across the whole repo (e.g., all callers of a renamed function).
```

### Sub-agent

For harnesses that support sub-agents (Claude Code, Cursor, Gemini CLI, Kiro, OpenCode, GitHub Copilot, Codex, Reasonix, Pi, Command Code, Antigravity, ZCode), you can install a dedicated `zemble-search` sub-agent. Copy the appropriate file from [`src/zemble/agents/`](../src/zemble/agents/) to your agent's agents directory:

> **Pi prerequisite:** Pi sub-agents require the Pi agents extension. Run `pi install npm:pi-agents` once before installing.

| Agent | File | Destination |
|---|---|---|
| Claude Code | `claude.md` | `~/.claude/agents/zemble-search.md` |
| Cursor | `cursor.md` | `~/.cursor/agents/zemble-search.md` |
| Gemini CLI | `gemini.md` | `~/.gemini/agents/zemble-search.md` |
| Kiro | `kiro.md` | `~/.kiro/agents/zemble-search.md` |
| OpenCode | `opencode.md` | `~/.config/opencode/agents/zemble-search.md` |
| GitHub Copilot | `copilot.md` | `~/.copilot/agents/zemble-search.agent.md` |
| Codex | `codex.toml` | `~/.codex/agents/zemble-search.toml` |
| Reasonix | `reasonix.md` | `~/.reasonix/skills/zemble-search.md` |
| Pi | `pi.md` | `~/.pi/agents/zemble-search.md` |
| Command Code | `commandcode.md` | `~/.commandcode/agents/zemble-search.md` |
| Antigravity | `antigravity.md` | `~/.gemini/config/skills/zemble-search/SKILL.md` |
| ZCode | `zcode.md` | `~/.zcode/agents/zemble-search.md` |
