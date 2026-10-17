from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable, NoReturn, Sequence, TypeVar

from zemble.installer import nib
from zemble.installer.agents import (
    AGENTS,
    INSTRUCTIONS,
    ZEMBLE_PIN,
    AgentTarget,
    IntegrationType,
    Mode,
    WriteResult,
    is_detected,
)
from zemble.installer.config import (
    has_json_member,
    merge_json_member,
    merge_toml_block,
    remove_json_member,
    remove_marked,
    remove_toml_block,
    replace_or_append_marked,
)

_T = TypeVar("_T")

_GREEN = "\033[32m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_ACTION_DETAIL: dict[str, str] = {
    "skipped": "JSON5 grammar unavailable — add manually (see README)",
    "error": "could not parse or edit config",
    "hosted": "nib hosts zemble for this agent, nothing written",
}


@dataclass(frozen=True)
class _Integration:
    """Descriptor for one installer integration (MCP server, instructions, sub-agent)."""

    id: IntegrationType
    label: str
    desc: str
    apply: Callable[[AgentTarget, Mode], WriteResult | None]
    plan_path: Callable[[AgentTarget], Path | None]


def merge_mcp(agent: AgentTarget) -> WriteResult:
    """Add the zemble MCP entry to the agent's config."""
    assert agent.mcp is not None
    path = agent.mcp.path
    return WriteResult(path, merge_json_member(path, agent.mcp.key, "zemble", agent.mcp.entry))


def remove_mcp(agent: AgentTarget) -> WriteResult:
    """Remove the zemble MCP entry from the agent's config."""
    assert agent.mcp is not None
    path = agent.mcp.path
    return WriteResult(path, remove_json_member(path, agent.mcp.key, "zemble"))


def _apply_mcp(agent: AgentTarget, mode: Mode) -> WriteResult | None:
    """Apply or remove the MCP server integration for one agent."""
    if agent.mcp is None:
        return None
    path = agent.mcp.path
    if agent.mcp.format == "toml":
        return WriteResult(path, merge_toml_block(path) if mode == "install" else remove_toml_block(path))
    return merge_mcp(agent) if mode == "install" else remove_mcp(agent)


def _apply_instructions(agent: AgentTarget, mode: Mode) -> WriteResult | None:
    """Apply or remove the instructions block integration for one agent."""
    path = agent.instructions_path
    if path is None:
        return None
    action = replace_or_append_marked(path, INSTRUCTIONS) if mode == "install" else remove_marked(path)
    return WriteResult(path, action)


def _apply_subagent(agent: AgentTarget, mode: Mode) -> WriteResult | None:
    """Apply or remove the global sub-agent file for one agent."""
    dest = agent.subagent_path
    if dest is None:
        return None
    if mode == "uninstall":
        if not dest.exists():
            return WriteResult(dest, "not-found")
        dest.unlink()
        return WriteResult(dest, "removed")
    existed = dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        src = files("zemble").joinpath(f"agents/{agent.id}{dest.suffix}").read_text(encoding="utf-8")
        dest.write_text(src.replace('"zemble[mcp]"', f'"{ZEMBLE_PIN}"'), encoding="utf-8")
    except Exception:
        return WriteResult(dest, "error")
    return WriteResult(dest, "updated" if existed else "created")


_INTEGRATIONS: list[_Integration] = [
    _Integration(
        IntegrationType.MCP,
        "MCP server",
        "lets the agent call zemble directly as a tool",
        _apply_mcp,
        AgentTarget.resolved_mcp_path,
    ),
    _Integration(
        IntegrationType.INSTRUCTIONS,
        "Instructions",
        "adds CLI usage guidance to AGENTS.md / CLAUDE.md",
        _apply_instructions,
        lambda a: a.instructions_path,
    ),
    _Integration(
        IntegrationType.SUBAGENT,
        "Sub-agent",
        "installs a dedicated zemble-search sub-agent",
        _apply_subagent,
        lambda a: a.subagent_path,
    ),
]


def _tick(ok: bool) -> str:
    """Return a green ✓ or dim – for use in apply output."""
    return f"{_GREEN}✓{_RESET}" if ok else f"{_DIM}–{_RESET}"


def _exit(message: str, code: int = 0) -> NoReturn:
    """Print message and exit with the given code (0 by default, for user-cancelled runs)."""
    print(message)
    sys.exit(code)


def _checkbox(prompt: str, items: Sequence[tuple[str, _T, bool]]) -> list[_T] | None:
    """Show an interactive multi-select checkbox; return selected values or None if cancelled."""
    # prompt_toolkit defaults "selected" to reverse-video (a filled block); override it
    # so checked rows show a clean green ● and the cursor row is just bold.
    import questionary

    style = questionary.Style(
        [
            ("pointer", "bold"),
            ("highlighted", "noreverse bold"),
            ("selected", "noreverse fg:ansigreen"),
        ]
    )
    choices = [questionary.Choice(title=label, value=value, checked=checked) for label, value, checked in items]
    instruction = "(↑↓ move · space select · a all · enter confirm)"
    return questionary.checkbox(prompt, choices=choices, style=style, instruction=instruction).ask()


def _hosted_by_nib(nib_program: str | None, agent: AgentTarget, integ: _Integration) -> bool:
    """True when an install leaves this agent's MCP entry to nib; an uninstall's program is empty, so never then."""
    return bool(nib_program) and integ.id is IntegrationType.MCP and agent.nib_hostable


def _nib_program(mode: Mode, agents: list[AgentTarget], integrations: list[_Integration]) -> str | None:
    """Return the program nib should host as zemble ("" to stop hosting it), or None when nib takes no part."""
    if not any(i.id is IntegrationType.MCP for i in integrations) or not any(a.nib_hostable for a in agents):
        return None
    if not nib.nib_usable():
        return None
    if mode == "uninstall":
        return ""
    program = nib.provider_program()
    if program is None:
        note = "nib can host providers, but no zemble executable was found to hand it; writing MCP entries."
        print(f"  {_DIM}{note}{_RESET}")
    return program


def _print_plan(agents: list[AgentTarget], integrations: list[_Integration], nib_program: str | None = None) -> None:
    """Print what will be written or removed for each selected agent and integration."""
    print(f"\n  {_BOLD}Plan:{_RESET}\n")
    if nib_program is not None:
        print(f"  {_BOLD}nib{_RESET}\n    {shlex.join(nib.setup_command(nib_program))}\n")
    for agent in agents:
        print(f"  {_BOLD}{agent.display_name}{_RESET}")
        for integ in integrations:
            path = integ.plan_path(agent)
            ok = path is not None
            where = "(hosted by nib)" if _hosted_by_nib(nib_program, agent, integ) else path
            print(f"    {integ.label:<13} {_tick(ok)}  {where if ok else '(not supported)'}")
    print()


def _run_nib(mode: Mode, nib_program: str, hosted: list[AgentTarget]) -> None:
    """Run `nib setup` for zemble, say what it means for the agents nib registers, and pass its lines through."""
    command = shlex.join(nib.setup_command(nib_program))
    print(f"  {_BOLD}nib{_RESET}")
    if mode == "install":
        skipped = ", ".join(f"{a.display_name} {a.mcp.key}.zemble in {a.mcp.path}" for a in hosted if a.mcp)
        print(f"    Registered zemble through nib ({command}); skipped writing {skipped}.")
    else:
        print(f"    Stopped hosting zemble in nib ({command}); nib does not put a standalone zemble server back.")
    for line in nib.run_setup(nib_program):
        failed = nib.is_failure(line)
        print(f"    {_RED + '!' + _RESET if failed else _tick(True)} {line}")
    if mode == "install":
        for agent in hosted:
            if agent.mcp and has_json_member(agent.mcp.path, agent.mcp.key, "zemble"):
                print(
                    f"    {_RED}!{_RESET} {agent.mcp.path} still holds {agent.mcp.key}.zemble, which duplicates "
                    "the tools nib hosts; remove that entry."
                )
    print()


def _apply(
    mode: Mode, agents: list[AgentTarget], integrations: list[_Integration], nib_program: str | None = None
) -> None:
    """Execute install or uninstall for all chosen agents and integrations, printing results."""
    print()
    if nib_program is not None:
        _run_nib(mode, nib_program, [a for a in agents if a.nib_hostable])
    for agent in agents:
        print(f"  {_BOLD}{agent.display_name}{_RESET}")
        for integ in integrations:
            if _hosted_by_nib(nib_program, agent, integ):
                result: WriteResult | None = WriteResult(agent.mcp.path, "hosted") if agent.mcp else None
            else:
                result = integ.apply(agent, mode)
            if result is None:
                print(f"    {_DIM}– {integ.id}: not supported{_RESET}")
                continue
            ok = result.action in ("created", "updated", "removed", "unchanged", "hosted")
            detail = _ACTION_DETAIL.get(result.action, "")
            suffix = f" — {detail}" if detail else ""
            print(f"    {_tick(ok)} {integ.id} ({result.action}){suffix} → {result.path}")
        print()


def run(
    mode: Mode,
    agent_ids: list[str] | None = None,
    integration_ids: list[IntegrationType] | None = None,
    yes: bool = False,
) -> None:
    """Install or uninstall zemble across coding agents.

    Prompts interactively unless `agent_ids` is given, in which case it runs unattended
    against those agents (and `integration_ids`, or all integrations if omitted).
    """
    install = mode == "install"
    print(f"\n  {_BOLD}{'Zemble Installer' if install else 'Zemble Uninstaller'}{_RESET}\n")

    if agent_ids is not None:
        chosen_agents = [a for a in AGENTS if a.id in agent_ids] or _exit("No matching agents. Exiting.", code=1)
        chosen_integrations = (
            [i for i in _INTEGRATIONS if i.id in integration_ids]
            if integration_ids is not None
            else list(_INTEGRATIONS)
        ) or _exit("No matching integrations. Exiting.", code=1)
    else:
        agent_items = [
            (f"{a.display_name}{'  (detected)' if (detected := is_detected(a)) else ''}", a, detected and install)
            for a in sorted(AGENTS, key=lambda a: not is_detected(a))
        ]
        chosen_agents = _checkbox(
            f"Select agents to {'configure' if install else 'remove configuration from'}:", agent_items
        ) or _exit("Nothing selected. Exiting.")

        max_label = max(len(i.label) for i in _INTEGRATIONS)
        integ_items = [(f"{i.label:<{max_label}}  —  {i.desc}", i, True) for i in _INTEGRATIONS]
        chosen_integrations = _checkbox(
            f"Select integrations to {'enable' if install else 'remove'}:", integ_items
        ) or _exit("Nothing selected. Exiting.")

    nib_program = _nib_program(mode, chosen_agents, chosen_integrations)
    _print_plan(chosen_agents, chosen_integrations, nib_program)

    if not yes:
        question = "Proceed?" if install else "Remove zemble configuration?"
        import questionary

        if not questionary.confirm(question, default=install).ask():
            _exit("Cancelled.")

    _apply(mode, chosen_agents, chosen_integrations, nib_program)
    footer = " Restart your agents to pick up the changes." if install else ""
    print(f"  {_GREEN}Done!{_RESET}{footer}\n")
