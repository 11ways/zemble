"""Cross-module verdicts: what a clone class spanning declared modules should do about it.

Driven by the same `<root>/.zemble/home.toml` the `home` tool reads; a workspace
without one gets no verdicts and no noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from zemble.dedup.model import CloneClass
from zemble.home.config import ConfigError, HomeConfig


class HomeVerdictKind(str, Enum):
    """The three answers for duplication that crosses a module boundary."""

    CANDIDATE_HOME = "candidate-home"
    FORBIDDEN_DEP = "forbidden-dep"
    NO_SHARED_ANCESTOR = "no-shared-ancestor"


@dataclass(frozen=True, slots=True)
class HomeVerdict:
    """One cross-module judgement: the member modules (most core first), the home if any, why."""

    kind: HomeVerdictKind
    modules: tuple[str, ...]
    home: str | None
    detail: str

    def describe(self) -> str:
        """One line for the text report."""
        span = ", ".join(self.modules)
        if self.kind is HomeVerdictKind.CANDIDATE_HOME:
            return f"candidate home {self.home} (spans {span}; {self.detail})"
        if self.kind is HomeVerdictKind.FORBIDDEN_DEP:
            return f"forbidden dependency (spans {span}; {self.detail})"
        if self.kind is HomeVerdictKind.NO_SHARED_ANCESTOR:
            return f"no shared ancestor (spans {span}; {self.detail})"
        raise ValueError(f"Unhandled verdict kind {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        """Render the verdict for the wire."""
        return {
            "verdict": self.kind.value,
            "modules": list(self.modules),
            "home": self.home,
            "detail": self.detail,
        }


def _judge(modules: Sequence[str], config: HomeConfig) -> HomeVerdict:
    """Judge one set of member modules against the declared order and the forbidden rules."""
    ranked = tuple(sorted(modules, key=lambda module: (config.rank(module), module)))
    head = ranked[0]
    if config.rank(head) >= len(config.order):
        return HomeVerdict(
            HomeVerdictKind.NO_SHARED_ANCESTOR, ranked, None, "no member module is in the declared order"
        )
    broken = [rule for module in ranked[1:] if (rule := config.forbids(module, head)) is not None]
    if broken:
        detail = "; ".join(rule.describe() for rule in broken) + f"; a shared home must sit deeper than {head}"
        return HomeVerdict(HomeVerdictKind.FORBIDDEN_DEP, ranked, head, detail)
    return HomeVerdict(
        HomeVerdictKind.CANDIDATE_HOME,
        ranked,
        head,
        f"{head} is the most core member module and every other member may depend on it",
    )


def judge_classes(root: str | Path, classes: Sequence[CloneClass]) -> tuple[dict[str, HomeVerdict], list[str]]:
    """Judge every class spanning more than one declared module.

    :param root: The scanned root, where `home.toml` is looked for.
    :param classes: The report's final classes.
    :return: Verdicts keyed by class key, and any note worth printing.
    """
    try:
        config = HomeConfig.load(root)
    except ConfigError as error:
        return {}, [f"home.toml error, cross-module verdicts skipped: {error}"]
    if config.generic:
        return {}, []
    verdicts: dict[str, HomeVerdict] = {}
    for clone in classes:
        modules = {config.module_of(member.file_path) for member in clone.members}
        if len(modules) < 2:
            continue
        verdicts[clone.key] = _judge(tuple(modules), config)
    return verdicts, []


__all__ = ["HomeVerdict", "HomeVerdictKind", "judge_classes"]
