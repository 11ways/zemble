"""Cross-module verdicts: what a clone class spanning declared modules should do about it.

Driven by the same `<root>/.zemble/home.toml` the `home` tool reads; a workspace
without one gets no verdicts and no noise. Every verdict is reached by one ordered
decision function, and every step of it fails closed: what cannot be proven from the
declarations, the parse and the dependency graph is never called a reusable API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from zemble.dedup.model import CloneClass, CloneKind, Unit
from zemble.home.config import ConfigError, HomeConfig
from zemble.home.tables import DeclaredRow, RowMatchKind, load_rows, row_match_kind


class HomeVerdictKind(str, Enum):
    """Every answer duplication crossing a module boundary can get.

    The two `existing-*` members are deliberately separate: a mechanism can be in the
    right place and still be uncallable from where the copies live, and telling a reader
    to call a private method is worse than telling them nothing.
    """

    EXISTING_REUSABLE_API = "existing-reusable-api"
    EXISTING_IMPLEMENTATION_NOT_API = "existing-implementation-not-api"
    CANDIDATE_HOME = "candidate-home"
    SIBLINGS_NEED_COMMON_HOME = "siblings-need-common-home"
    FORBIDDEN_DEP = "forbidden-dep"
    NO_SHARED_ANCESTOR = "no-shared-ancestor"
    REVIEW_REQUIRED = "review-required"


class EvidenceKind(str, Enum):
    """What one piece of a verdict's reasoning was read from."""

    DECLARED_MEMBER = "declared-member"
    DECLARED_TYPE = "declared-type"
    VISIBILITY = "visibility"
    SOURCE_SET = "source-set"
    DEPENDENCY = "dependency"
    CLONE_KIND = "clone-kind"


#: Longest declared-row title the text report prints before it truncates.
_TITLE_LIMIT = 100


def _shorten(text: str, limit: int = _TITLE_LIMIT) -> str:
    """Cut a table cell down to something a report line can carry."""
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


@dataclass(frozen=True, slots=True)
class Evidence:
    """One kind-tagged reason a verdict is what it is, with the line the report prints."""

    kind: EvidenceKind
    text: str
    #: The whole capability cell, for declared-row evidence only.
    capability: str | None = None
    file: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render the evidence for the wire; the declared-row fields ship only when filled."""
        payload: dict[str, Any] = {"kind": self.kind.value, "text": self.text}
        if self.capability is not None:
            payload |= {"capability": self.capability, "file": self.file, "line": self.line}
        return payload


def visibility_evidence(member: Unit) -> Evidence:
    """The visibility proof for one clone member: what blocks reuse, or that nothing does.

    Both levels have to be PUBLIC, and the member is reported before its declaring type
    because that is the one a reader can act on; anything a profile could not place is
    UNKNOWN and therefore restricted, which is what keeps a new language failing closed.
    """
    owner = member.name.rsplit(".", 1)[0] if "." in member.name else ""
    container = f"declaring type {owner}" if owner else "declaring type"
    for subject, level in (("member", member.visibility), (container, member.container_visibility)):
        if not level.is_public:
            return Evidence(EvidenceKind.VISIBILITY, level.phrase(subject))
    return Evidence(EvidenceKind.VISIBILITY, "public member on a public type")


@dataclass(frozen=True, slots=True)
class _Rendering:
    """How one verdict kind turns into report lines."""

    head: Callable[[HomeVerdict], str]
    #: What to do about it, or None for a verdict that prescribes nothing.
    action: Callable[[HomeVerdict], str] | None = None


@dataclass(frozen=True, slots=True)
class HomeVerdict:
    """One cross-module judgement: the member modules (most core first), the home if any, why."""

    kind: HomeVerdictKind
    modules: tuple[str, ...]
    home: str | None
    detail: str
    #: The clone member this verdict is about, as `Type.member`.
    symbol: str | None = None
    #: `file_path:start_line` of that member.
    location: str | None = None
    #: Every kind-tagged reason behind the verdict.
    evidence: tuple[Evidence, ...] = ()
    #: Where a shared mechanism should go when no member module is it.
    suggested_home: str | None = None

    @property
    def span(self) -> str:
        """The member modules the way every head line names them."""
        return ", ".join(self.modules)

    def describe_lines(self) -> list[str]:
        """The text report's rendering, unindented: head line, evidence, what to do."""
        rendering = _RENDERINGS.get(self.kind)
        if rendering is None:  # pragma: no cover - a new kind without a rendering is a build error
            raise ValueError(f"Unhandled verdict kind {self.kind!r}")
        lines = [rendering.head(self)]
        lines.extend(item.text for item in self.evidence)
        if rendering.action is not None:
            lines.append(rendering.action(self))
        return lines

    def describe(self) -> str:
        """The text report's rendering as one string, for callers that do their own indenting."""
        return "\n".join(self.describe_lines())

    def to_dict(self) -> dict[str, Any]:
        """Render the verdict for the wire.

        `verdict` and `kind` are the same value: `verdict` is the name the first readers
        were written against, `kind` the one every other zemble payload uses.
        """
        return {
            "verdict": self.kind.value,
            "kind": self.kind.value,
            "modules": list(self.modules),
            "home": self.home,
            "detail": self.detail,
            "symbol": self.symbol,
            "location": self.location,
            "suggested_home": self.suggested_home,
            "evidence": [item.to_dict() for item in self.evidence],
            "lines": self.describe_lines(),
        }


def _siblings_action(verdict: HomeVerdict) -> str:
    """What to do with two modules that cannot reach each other."""
    if verdict.suggested_home is None:
        return "common home unknown: no dependency information"
    return f"shared mechanism belongs in {verdict.suggested_home}"


def _candidate_head(verdict: HomeVerdict) -> str:
    """A candidate home names a member only when a row was lexically related to one."""
    if verdict.symbol is None:
        return f"candidate home {verdict.home} (spans {verdict.span}; {verdict.detail})"
    return f"candidate home {verdict.home}: {verdict.symbol}"


#: One rendering per verdict kind; a member without an entry raises rather than printing nothing.
_RENDERINGS: dict[HomeVerdictKind, _Rendering] = {
    HomeVerdictKind.EXISTING_REUSABLE_API: _Rendering(
        head=lambda verdict: f"existing reusable API {verdict.home}: {verdict.symbol}",
        action=lambda verdict: "downstream copies should call or extend it",
    ),
    HomeVerdictKind.EXISTING_IMPLEMENTATION_NOT_API: _Rendering(
        head=lambda verdict: f"existing implementation {verdict.home}: {verdict.symbol} (not a reusable API)",
        action=lambda verdict: "expose it or extract the generic mechanism; do not call it as is",
    ),
    HomeVerdictKind.CANDIDATE_HOME: _Rendering(
        head=_candidate_head,
        action=lambda verdict: "no declared member; review before consolidating",
    ),
    HomeVerdictKind.SIBLINGS_NEED_COMMON_HOME: _Rendering(
        head=lambda verdict: f"siblings {verdict.span}: {verdict.symbol}",
        action=_siblings_action,
    ),
    HomeVerdictKind.FORBIDDEN_DEP: _Rendering(
        head=lambda verdict: f"forbidden dependency (spans {verdict.span}; {verdict.detail})",
    ),
    HomeVerdictKind.NO_SHARED_ANCESTOR: _Rendering(
        head=lambda verdict: f"no shared ancestor (spans {verdict.span}; {verdict.detail})",
    ),
    HomeVerdictKind.REVIEW_REQUIRED: _Rendering(
        head=lambda verdict: f"possible existing mechanism {verdict.home}: {verdict.symbol} (logic clone)",
        action=lambda verdict: "semantic review required; structural similarity is not equivalence",
    ),
}


@dataclass(frozen=True, slots=True)
class _RowHit:
    """One declared row, and the name in it that matched the clone member."""

    row: DeclaredRow
    symbol: str


def _row_evidence(hit: _RowHit, kind: EvidenceKind) -> Evidence:
    """Turn a matched row into evidence, keeping the whole capability cell beside the line."""
    file_name = hit.row.file.rsplit("/", 1)[-1]
    if kind is EvidenceKind.DECLARED_MEMBER:
        text = f"declared by {file_name}: {_shorten(hit.row.title)} (row names {hit.symbol})"
    else:
        text = f"lexically related row: {_shorten(hit.row.title)} (names the type, not this member)"
    return Evidence(kind, text, capability=hit.row.capability, file=hit.row.file, line=hit.row.line)


def _row_hits(member: Unit, home: str, rows: Sequence[DeclaredRow]) -> tuple[list[_RowHit], list[_RowHit]]:
    """Split the rows homed in one module into the ones declaring a member and the ones naming its type."""
    exact: list[_RowHit] = []
    bare: list[_RowHit] = []
    for row in rows:
        if home not in row.home_modules:
            continue
        for symbol in row.symbols:
            kind = row_match_kind(symbol, member.name)
            if kind is RowMatchKind.EXACT_MEMBER:
                exact.append(_RowHit(row, symbol))
                break
            if kind is RowMatchKind.BARE_TYPE:
                bare.append(_RowHit(row, symbol))
                break
    return exact, bare


def _declared_match(
    clone: CloneClass, home: str, config: HomeConfig, rows: Sequence[DeclaredRow]
) -> tuple[Unit, tuple[_RowHit, ...], RowMatchKind] | None:
    """Find the clone member in the home module that a declared row names, best match first.

    Only whole bodies count: a statement window inside a method is not the mechanism
    the table declared, however much of it the window covers. Synthetic members whose
    last name segment is bracketed (`Type.<initializer>`) are skipped too: nothing can
    call an initializer, so a row naming its class never makes it the shared mechanism.
    A bare `Type` row is returned as BARE_TYPE and never as a declaration of the member.
    """
    if not rows:
        return None
    fallback: tuple[Unit, tuple[_RowHit, ...], RowMatchKind] | None = None
    for member in clone.members:
        if not member.is_body or member.name.rsplit(".", 1)[-1].startswith("<"):
            continue
        if config.module_of(member.file_path) != home:
            continue
        exact, bare = _row_hits(member, home, rows)
        if exact:
            return member, tuple(exact), RowMatchKind.EXACT_MEMBER
        if bare and fallback is None:
            fallback = (member, tuple(bare), RowMatchKind.BARE_TYPE)
    return fallback


def _most_core_member(clone: CloneClass, config: HomeConfig) -> Unit:
    """The clone member living closest to the core, bodies before statement windows."""
    return min(
        clone.members,
        key=lambda member: (config.rank(config.module_of(member.file_path)), not member.is_body, member.name),
    )


def _shared_symbol(clone: CloneClass, config: HomeConfig) -> str:
    """The name the copies share, or the most core copy's qualified name when they differ."""
    simple = {member.name.rsplit(".", 1)[-1] for member in clone.members}
    return simple.pop() if len(simple) == 1 else _most_core_member(clone, config).name


def _dependency_home(ranked: tuple[str, ...], config: HomeConfig) -> tuple[str | None, Evidence, bool]:
    """The member module every other member module may reach, with the evidence for it.

    AIDEV-NOTE: `order` ranks modules, it never granted anyone permission to depend on
    anyone. A workspace with no dependency graph therefore gets its most core member
    module as a PLACE for a mechanism, and the caller caps such a verdict below
    `existing-reusable-api`: "core-most" is not proof that the other copies may call it.
    """
    if not config.dependencies.known:
        unknown = Evidence(
            EvidenceKind.DEPENDENCY,
            "dependency reachability unknown: the workspace declares and builds no dependency graph",
        )
        return ranked[0], unknown, False
    shared = [module for module in ranked if all(config.reachable(other, module).usable for other in ranked)]
    if not shared:
        return None, Evidence(EvidenceKind.DEPENDENCY, "no dependency path either way"), True
    home = shared[0]
    reasons = ", ".join(f"{other}: {config.reachable(other, home).value}" for other in ranked if other != home)
    return home, Evidence(EvidenceKind.DEPENDENCY, f"every copy's module reaches {home} ({reasons})"), True


def _forbidden_step(
    clone: CloneClass, ranked: tuple[str, ...], config: HomeConfig, rows: Sequence[DeclaredRow]
) -> HomeVerdict | None:
    """A `[[forbidden]]` rule between any two member modules outranks every other answer."""
    broken = [
        rule
        for consumer in ranked
        for provider in ranked
        if consumer != provider and (rule := config.forbids(consumer, provider)) is not None
    ]
    if not broken:
        return None
    detail = "; ".join(rule.describe() for rule in broken) + f"; a shared home must sit deeper than {ranked[0]}"
    return HomeVerdict(HomeVerdictKind.FORBIDDEN_DEP, ranked, ranked[0], detail)


def _undeclared_step(
    clone: CloneClass, ranked: tuple[str, ...], config: HomeConfig, rows: Sequence[DeclaredRow]
) -> HomeVerdict | None:
    """A member outside the declared architecture cannot be judged by it."""
    declared = set(config.modules)
    stray = [module for module in ranked if module not in declared]
    if not stray:
        return None
    detail = f"{', '.join(stray)} {'is' if len(stray) == 1 else 'are'} not declared in home.toml"
    return HomeVerdict(HomeVerdictKind.NO_SHARED_ANCESTOR, ranked, None, detail)


def _logic_step(
    clone: CloneClass, ranked: tuple[str, ...], config: HomeConfig, rows: Sequence[DeclaredRow]
) -> HomeVerdict | None:
    """A logic clone is a lead, never a duplicate: structural similarity is not equivalence."""
    if clone.kind is not CloneKind.LOGIC:
        return None
    declared: tuple[Evidence, ...] = ()
    member: Unit | None = None
    home: str | None = None
    for module in ranked:
        match = _declared_match(clone, module, config, rows)
        if match is not None and match[2] is RowMatchKind.EXACT_MEMBER:
            member, home = match[0], module
            declared = tuple(_row_evidence(hit, EvidenceKind.DECLARED_MEMBER) for hit in match[1])
            break
    if member is None:
        member = _most_core_member(clone, config)
        home = config.module_of(member.file_path)
    clone_kind = Evidence(EvidenceKind.CLONE_KIND, "logic clone: similar control flow and call set, not the same code")
    return HomeVerdict(
        HomeVerdictKind.REVIEW_REQUIRED,
        ranked,
        home,
        f"{member.name} in {home} is the best-evidenced copy of a logic clone",
        symbol=member.name,
        location=f"{member.file_path}:{member.start_line}",
        evidence=(*declared, clone_kind),
    )


def _siblings_step(
    clone: CloneClass, ranked: tuple[str, ...], config: HomeConfig, rows: Sequence[DeclaredRow]
) -> HomeVerdict | None:
    """Modules that cannot reach each other need a new home, not one of themselves."""
    home, dependency, _known = _dependency_home(ranked, config)
    if home is not None:
        return None
    suggested = config.nearest_common_dependency(ranked)
    where = f"a shared mechanism belongs in {suggested}" if suggested else "they share no reachable module"
    return HomeVerdict(
        HomeVerdictKind.SIBLINGS_NEED_COMMON_HOME,
        ranked,
        None,
        f"no member module may depend on another; {where}",
        symbol=_shared_symbol(clone, config),
        evidence=(dependency,),
        suggested_home=suggested,
    )


def _reusability(
    clone: CloneClass,
    ranked: tuple[str, ...],
    config: HomeConfig,
    home: str,
    member: Unit,
    declared: tuple[Evidence, ...],
    dependency: Evidence,
    known: bool,
) -> HomeVerdict:
    """Decide whether a declared member is callable from every copy, and say what blocks it."""
    reusable = member.visibility.is_public and member.container_visibility.is_public
    visibility = visibility_evidence(member)
    folds = tuple(
        Evidence(
            EvidenceKind.SOURCE_SET,
            f"{config.module_of(other.file_path)} {config.source_set_of(other.file_path).value} cannot use "
            f"{home} {config.source_set_of(member.file_path).value}",
        )
        for other in clone.members
        if other is not member and not config.source_set_compatible(other.file_path, member.file_path)
    )
    # The home module was picked because every other one reaches it, so this re-check is
    # empty by construction; it stays because a home that stops being reachable must
    # cost the reusable verdict rather than pass silently.
    unreachable = (
        tuple(
            Evidence(EvidenceKind.DEPENDENCY, f"{module} cannot reach {home} ({config.reachable(module, home).value})")
            for module in ranked
            if module != home and not config.reachable(module, home).usable
        )
        if known
        else ()
    )
    failures: list[Evidence] = []
    if not reusable:
        failures.append(visibility)
    failures.extend(folds)
    failures.extend(unreachable)
    if not known:
        failures.append(dependency)
    location = f"{member.file_path}:{member.start_line}"
    if not failures:
        return HomeVerdict(
            HomeVerdictKind.EXISTING_REUSABLE_API,
            ranked,
            home,
            f"{home} declares {member.name} and every copy's module may call it",
            symbol=member.name,
            location=location,
            evidence=(*declared, visibility, dependency),
        )
    return HomeVerdict(
        HomeVerdictKind.EXISTING_IMPLEMENTATION_NOT_API,
        ranked,
        home,
        f"{home} declares {member.name}, but it is not reusable from every copy as it stands",
        symbol=member.name,
        location=location,
        evidence=(*declared, *failures),
    )


def _home_step(
    clone: CloneClass, ranked: tuple[str, ...], config: HomeConfig, rows: Sequence[DeclaredRow]
) -> HomeVerdict:
    """The last step: a home module exists, so the answer is what the tables and the parse say."""
    home, dependency, known = _dependency_home(ranked, config)
    if home is None:  # pragma: no cover - the siblings step already answered this case
        raise ValueError("no home module after the siblings step")
    match = _declared_match(clone, home, config, rows)
    if match is None:
        return HomeVerdict(
            HomeVerdictKind.CANDIDATE_HOME,
            ranked,
            home,
            f"{home} is the member module every other member may depend on",
            evidence=(dependency,),
        )
    member, hits, kind = match
    if kind is RowMatchKind.BARE_TYPE:
        return HomeVerdict(
            HomeVerdictKind.CANDIDATE_HOME,
            ranked,
            home,
            f"a row names {member.name.rsplit('.', 1)[0]}, but declares no member of it",
            symbol=member.name,
            location=f"{member.file_path}:{member.start_line}",
            evidence=(*(_row_evidence(hit, EvidenceKind.DECLARED_TYPE) for hit in hits), dependency),
        )
    declared = tuple(_row_evidence(hit, EvidenceKind.DECLARED_MEMBER) for hit in hits)
    return _reusability(clone, ranked, config, home, member, declared, dependency, known)


#: The decision order; the first step that answers wins, and the last one always answers.
_STEPS: tuple[Callable[[CloneClass, tuple[str, ...], HomeConfig, Sequence[DeclaredRow]], HomeVerdict | None], ...] = (
    _forbidden_step,
    _undeclared_step,
    _logic_step,
    _siblings_step,
    _home_step,
)


def _judge(clone: CloneClass, modules: Sequence[str], config: HomeConfig, rows: Sequence[DeclaredRow]) -> HomeVerdict:
    """Run the ordered decision function over one clone class."""
    ranked = tuple(sorted(modules, key=lambda module: (config.rank(module), module)))
    for step in _STEPS:
        verdict = step(clone, ranked, config, rows)
        if verdict is not None:
            return verdict
    raise ValueError("the decision order answered nothing")  # pragma: no cover - _home_step always answers


def _declared_rows(config: HomeConfig) -> tuple[list[DeclaredRow], list[str]]:
    """Load every declared-home row once, degrading to no evidence plus a note.

    A table is documentation: an unreadable or unparseable one loses the declared lane
    and nothing else, because this is a report, never a gate.
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


__all__ = [
    "Evidence",
    "EvidenceKind",
    "HomeVerdict",
    "HomeVerdictKind",
    "judge_classes",
    "visibility_evidence",
]
