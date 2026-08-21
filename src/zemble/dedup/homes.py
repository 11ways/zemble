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

from zemble.dedup.model import CloneClass, Unit
from zemble.home.config import ConfigError, HomeConfig
from zemble.home.tables import DeclaredRow, load_rows, row_names_symbol


class HomeVerdictKind(str, Enum):
    """The four answers for duplication that crosses a module boundary."""

    EXISTING_HOME = "existing-home"
    CANDIDATE_HOME = "candidate-home"
    FORBIDDEN_DEP = "forbidden-dep"
    NO_SHARED_ANCESTOR = "no-shared-ancestor"


#: Longest declared-row title the text report prints before it truncates.
_TITLE_LIMIT = 100


def _shorten(text: str, limit: int = _TITLE_LIMIT) -> str:
    """Cut a table cell down to something a report line can carry."""
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


@dataclass(frozen=True, slots=True)
class DeclaredEvidence:
    """One declared capability-table row, as the reason a clone class has an existing home."""

    capability: str
    #: `DeclaredRow.title`: the capability up to its first parenthesis, for the text report.
    title: str
    file: str
    line: int
    kind: str = "declared-row"

    @property
    def file_name(self) -> str:
        """The table file's base name, which is what the text report has room for."""
        return self.file.rsplit("/", 1)[-1]

    def to_dict(self) -> dict[str, Any]:
        """Render the evidence for the wire; the title is derived, so only the capability ships."""
        return {"kind": self.kind, "capability": self.capability, "file": self.file, "line": self.line}


@dataclass(frozen=True, slots=True)
class HomeVerdict:
    """One cross-module judgement: the member modules (most core first), the home if any, why.

    `symbol`, `location` and `evidence` are only filled for `existing-home`, the one
    verdict that names an actual declared mechanism.
    """

    kind: HomeVerdictKind
    modules: tuple[str, ...]
    home: str | None
    detail: str
    #: The clone member the declared row names, as `Type.member`.
    symbol: str | None = None
    #: `file_path:start_line` of that member.
    location: str | None = None
    #: What made this an existing home; declared table rows only.
    evidence: tuple[DeclaredEvidence, ...] = ()

    def describe_lines(self) -> list[str]:
        """The text report's rendering, unindented; only `existing-home` needs more than one line."""
        span = ", ".join(self.modules)
        if self.kind is HomeVerdictKind.EXISTING_HOME:
            lines = [f"existing home {self.home}: {self.symbol}"]
            lines.extend(f"declared by {item.file_name}: {_shorten(item.title)}" for item in self.evidence)
            lines.append("downstream copies should call or extend it")
            return lines
        if self.kind is HomeVerdictKind.CANDIDATE_HOME:
            return [f"candidate home {self.home} (spans {span}; {self.detail})"]
        if self.kind is HomeVerdictKind.FORBIDDEN_DEP:
            return [f"forbidden dependency (spans {span}; {self.detail})"]
        if self.kind is HomeVerdictKind.NO_SHARED_ANCESTOR:
            return [f"no shared ancestor (spans {span}; {self.detail})"]
        raise ValueError(f"Unhandled verdict kind {self.kind!r}")

    def describe(self) -> str:
        """The text report's rendering as one string, for callers that do their own indenting."""
        return "\n".join(self.describe_lines())

    def to_dict(self) -> dict[str, Any]:
        """Render the verdict for the wire, the structured fields only when they are filled."""
        payload: dict[str, Any] = {
            "verdict": self.kind.value,
            "modules": list(self.modules),
            "home": self.home,
            "detail": self.detail,
        }
        if self.symbol is not None:
            payload["symbol"] = self.symbol
        if self.location is not None:
            payload["location"] = self.location
        if self.evidence:
            payload["evidence"] = [item.to_dict() for item in self.evidence]
        return payload


def _declared_match(
    clone: CloneClass, home: str, config: HomeConfig, rows: Sequence[DeclaredRow]
) -> tuple[Unit, tuple[DeclaredRow, ...]] | None:
    """Find the first clone member in the home module that a declared row names.

    Only whole bodies count: a statement window inside a method is not the mechanism
    the table declared, however much of it the window covers. Synthetic members whose
    last name segment is bracketed (`Type.<initializer>`) are skipped too: nothing can
    call an initializer, so a row naming its class never makes it the shared mechanism.
    """
    if not rows:
        return None
    for member in clone.members:
        if not member.is_body or member.name.rsplit(".", 1)[-1].startswith("<"):
            continue
        if config.module_of(member.file_path) != home:
            continue
        naming = tuple(
            row
            for row in rows
            if home in row.home_modules and any(row_names_symbol(symbol, member.name) for symbol in row.symbols)
        )
        if naming:
            return member, naming
    return None


def _judge(clone: CloneClass, modules: Sequence[str], config: HomeConfig, rows: Sequence[DeclaredRow]) -> HomeVerdict:
    """Judge one clone class against the declared order, the forbidden rules and the tables."""
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
    declared = _declared_match(clone, head, config, rows)
    if declared is not None:
        member, naming = declared
        evidence = tuple(
            DeclaredEvidence(capability=row.capability, title=row.title, file=row.file, line=row.line) for row in naming
        )
        titles = ", ".join(f"'{row.title}'" for row in naming)
        return HomeVerdict(
            HomeVerdictKind.EXISTING_HOME,
            ranked,
            head,
            f"{head} already declares {member.name} as the home of {titles}",
            symbol=member.name,
            location=f"{member.file_path}:{member.start_line}",
            evidence=evidence,
        )
    return HomeVerdict(
        HomeVerdictKind.CANDIDATE_HOME,
        ranked,
        head,
        f"{head} is the most core member module and every other member may depend on it",
    )


def _declared_rows(config: HomeConfig) -> tuple[list[DeclaredRow], list[str]]:
    """Load every declared-home row once, degrading to no evidence plus a note.

    A table is documentation: an unreadable or unparseable one loses the
    `existing-home` lane and nothing else, because this is a report, never a gate.
    """
    notes = [
        f"declared-home table {spec.file} could not be read, existing-home evidence skipped"
        for spec in config.tables
        if not (config.root / spec.file).is_file()
    ]
    try:
        rows = load_rows(config)
    except Exception as error:  # noqa: BLE001 - a broken table must not break the report
        return [], [*notes, f"declared-home tables could not be read, existing-home evidence skipped: {error}"]
    return rows, notes


class _RowSource:
    """Loads a workspace's declared rows at most once, and only if a verdict could use them."""

    def __init__(self, config: HomeConfig) -> None:
        """Hold the config; nothing is read until :meth:`rows` is called."""
        self.config = config
        self.notes: list[str] = []
        self._rows: list[DeclaredRow] | None = None

    def rows(self) -> list[DeclaredRow]:
        """The declared rows, loading and noting the unreadable tables on first use."""
        if self._rows is None:
            self._rows, self.notes = _declared_rows(self.config)
        return self._rows


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
    source = _RowSource(config)
    verdicts: dict[str, HomeVerdict] = {}
    for clone in classes:
        modules = {config.module_of(member.file_path) for member in clone.members}
        if len(modules) < 2:
            continue
        verdicts[clone.key] = _judge(clone, tuple(modules), config, source.rows())
    return verdicts, source.notes


__all__ = ["DeclaredEvidence", "HomeVerdict", "HomeVerdictKind", "judge_classes"]
