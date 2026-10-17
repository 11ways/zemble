"""Registering zemble as a provider nib hosts, through `nib setup` (docs/nib-provider.md)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

#: The provider name; nib then offers zemble's tools as `mcp__nib__zemble_<tool>`.
PROVIDER_NAME = "zemble"

#: The flag form `nib setup --help` documents once it can host providers; an older nib lacks it.
_PROVIDER_FLAG = "--provider NAME=PROGRAM"

_COULD_NOT_RUN = "nib setup could not run: "
_BAD_EXIT = "nib setup failed with exit code "

#: Fragments of the report lines `nib setup` prints when a step failed (it exits 0 regardless), plus this module's own.
_FAILURE_FRAGMENTS = (
    _COULD_NOT_RUN,
    _BAD_EXIT,
    ": could not be written: ",
    "claude is not on PATH; ",
    "claude mcp remove failed",
    "claude mcp add-json failed",
    "claude mcp add failed",
)

_TIMEOUT_SECONDS = 60


def nib_usable() -> bool:
    """Return True when `nib` is on PATH and its `setup --help` documents `--provider NAME=PROGRAM`."""
    nib = shutil.which("nib")
    if nib is None:
        return False
    try:
        done = subprocess.run(
            [nib, "setup", "--help"], capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return _PROVIDER_FLAG in done.stdout + done.stderr


def provider_program() -> str | None:
    """Return the absolute path of the zemble executable nib should run as `PROGRAM nib-provider`, or None.

    nib takes one program and appends `nib-provider` itself, so a `uvx --from ...` launcher cannot be
    handed over; the `zemble` on PATH wins over the one beside this interpreter, whose environment may be
    a throwaway uvx cache.
    """
    found = shutil.which("zemble")
    if found is not None:
        return str(Path(found).absolute())
    beside = Path(sys.executable).parent / "zemble"
    return str(beside) if beside.is_file() else None


def setup_command(program: str) -> list[str]:
    """Return the `nib setup` argv that hosts `program` as zemble, or stops hosting it for an empty one."""
    return ["nib", "setup", "--user", "--provider", f"{PROVIDER_NAME}={program}"]


def run_setup(program: str) -> list[str]:
    """Run `nib setup` for `program` with this process's environment and return the lines it reported."""
    try:
        done = subprocess.run(
            setup_command(program), capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"{_COULD_NOT_RUN}{exc}"]
    lines = [line for line in (done.stdout + done.stderr).splitlines() if line.strip()]
    if done.returncode != 0:
        lines.append(f"{_BAD_EXIT}{done.returncode}")
    return lines


def is_failure(line: str) -> bool:
    """Return True when a line `run_setup` returned reports a step that did not happen."""
    return any(fragment in line for fragment in _FAILURE_FRAGMENTS)
