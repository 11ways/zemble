"""The shapes duplication detection speaks in: units, clone classes and reports."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from zemble.dedup.languages import Signature, Visibility, body_unit_kinds
from zemble.graph.model import is_test_path


class CloneKind(str, Enum):
    """The duplication channels: three clone kinds from strictest to loosest, then the sub-body and vocabulary ones.

    Every per-kind decision reads :attr:`facts`, whose table names every member; a kind missing
    from it raises instead of falling back to another kind's behaviour.
    """

    EXACT = "exact"
    RENAMED = "renamed"
    LOGIC = "logic"
    HOLED = "holed"
    IDIOM = "idiom"
    REIMPLEMENTS = "reimplements"
    VOCABULARY = "vocabulary"

    @property
    def facts(self) -> KindFacts:
        """What this kind is keyed, ranked and judged by."""
        return _KIND_FACTS[self]


class Ranking(str, Enum):
    """How a kind's classes are weighed against each other."""

    #: tokens x copies x files, the weighting `zenit-dev duplication` uses.
    MASS = "mass"
    #: copies x files, copies counted up to `SPREAD_PER_FILE` per file: a shape at 900 sites in 200 files
    #: outranks a long one at four, and 250 sites packed into 7 migration files rank as the 21 they spread like.
    SPREAD = "spread"


#: Copies per file a spread ranking counts; more sites in one file are local repetition, not spread.
SPREAD_PER_FILE = 3


@dataclass(frozen=True, slots=True)
class KindFacts:
    """The per-kind facts every consumer reads instead of switching over the kind."""

    #: The unit attribute a class of this kind is keyed by.
    key_attribute: str
    ranking: Ranking
    #: Why a class of this kind is only a lead, or None when its members are provably the same code or value.
    lead: str | None
    #: Members a rendered class lists before summing up the rest; None lists every member.
    site_cap: int | None
    #: Whether a focused run answers this kind from the per-root unit index (the others are filtered from a whole run).
    indexed_focus: bool
    #: Whether this kind reads body vectors from the embedder.
    embeds: bool


#: Members a sub-body or vocabulary class prints: a 900-site idiom must not print 900 lines.
SITE_CAP = 12

_KIND_FACTS: dict[CloneKind, KindFacts] = {
    CloneKind.EXACT: KindFacts("exact_hash", Ranking.MASS, None, None, True, False),
    CloneKind.RENAMED: KindFacts("renamed_hash", Ranking.MASS, None, None, True, False),
    # Logic classes have no stream of their own, so they are keyed by the alpha-renamed body hash of each member.
    CloneKind.LOGIC: KindFacts(
        "renamed_hash",
        Ranking.MASS,
        "logic clone: similar control flow and call set, not the same code",
        None,
        True,
        True,
    ),
    CloneKind.HOLED: KindFacts(
        "shape_hash", Ranking.MASS, "holed clone: the same code with different literal values", SITE_CAP, False, False
    ),
    CloneKind.IDIOM: KindFacts(
        "shape_hash", Ranking.SPREAD, "idiom: one call shape repeated at many sites", SITE_CAP, False, False
    ),
    CloneKind.REIMPLEMENTS: KindFacts(
        "renamed_hash",
        Ranking.MASS,
        "re-implementation candidate: similar embedding and call set, not the same code",
        None,
        False,
        True,
    ),
    CloneKind.VOCABULARY: KindFacts("shape_hash", Ranking.SPREAD, None, SITE_CAP, False, False),
}


class Lane(str, Enum):
    """Which source sets a clone class lives in, in the order the report prints them."""

    PRODUCTION = "production"
    MIXED = "mixed"
    TEST = "test"


class Demotion(str, Enum):
    """Why a class is a lesser finding of its kind; each ranks after every actionable class, in this order."""

    #: A copy bound to other constants than its original: a parameter to extract rather than a call to make.
    REBOUND = "rebound"
    #: Copies with no original their module may call: a mechanism to move where both sides can reach it.
    UNREACHABLE = "unreachable"

    @property
    def standing(self) -> int:
        """The class standing it takes, after code-evidenced (0) and inferred (1) classes."""
        match self:
            case Demotion.REBOUND:
                return 2
            case Demotion.UNREACHABLE:
                return 3
        raise ValueError(f"Unhandled demotion {self!r}")


#: Unit kinds that own a whole declaration body, as opposed to a statement window. The
#: language profiles are the home of the vocabulary; a new profile widens this by itself.
BODY_KINDS: tuple[str, ...] = tuple(sorted(body_unit_kinds()))
#: The one kind that is never a body: a run of consecutive statements inside one.
WINDOW_KIND = "window"

#: Copies at or above which a logic class's reasons are aggregated instead of listed per pair.
AGGREGATE_FROM = 3
#: How many outlier members the aggregate names before summing the rest up.
_AGGREGATE_OUTLIERS = 4
#: How many names a brace list shows before an ellipsis.
_NAMES_SHOWN = 5


def _braced(names: list[str]) -> str:
    """Render a sorted name list as `{a, b, ...}`."""
    shown = ", ".join(names[:_NAMES_SHOWN]) + ("" if len(names) <= _NAMES_SHOWN else ", ...")
    return f"{{{shown}}}"


@dataclass(frozen=True, slots=True)
class Unit:
    """One comparable piece of code: a whole body, or a window of statements inside one."""

    file_path: str
    start_line: int
    end_line: int
    kind: str
    name: str
    token_count: int
    exact_hash: str
    renamed_hash: str
    skeleton: tuple[str, ...]
    calls: tuple[str, ...]
    literals: tuple[str, ...]
    text: str | None = None
    #: Declaration modifiers (`public`, `static`, `pub`, ...); reported, never hashed, so a
    #: visibility edit can never move a clone key.
    modifiers: tuple[str, ...] = ()
    #: How far this member itself can be called from, as its language profile reads it.
    visibility: Visibility = Visibility.UNKNOWN
    #: The same for the innermost declaring type, already folded through its enclosing types.
    container_visibility: Visibility = Visibility.UNKNOWN
    #: The hash the holed, idiom and vocabulary channels group by; "" on a unit no such channel reads.
    shape_hash: str = ""
    #: That shape as text with typed holes (`<str>`, `<const>`), or the value a vocabulary site declares.
    shape: str = ""
    #: `Type.member` when the whole body only hands its parameters, in order, to that one callable.
    forwards_to: str | None = None
    #: Whether the member implements a declared contract (`@Override`): a role implementation, not a utility.
    implements_contract: bool = False
    #: Declared types, staticness and documentation, on a shaped body whose profile reads them.
    signature: Signature | None = None
    #: The constant references a shaped body's holes stand for, or an enum's members pass, as written.
    constants: tuple[str, ...] = ()
    #: For a value-set site: the qualified declaration behind each value.
    declares: tuple[str, ...] = ()
    #: Whether a shaped body only hands over to another constructor of its type (`this(...)`, `super(...)`).
    delegates: bool = False

    @property
    def location(self) -> str:
        """File path and line range, the way the report prints it."""
        if self.start_line == self.end_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def is_body(self) -> bool:
        """Whether this unit is a whole declaration body rather than a statement window."""
        return self.kind in BODY_KINDS

    @property
    def is_test(self) -> bool:
        """Whether this unit lives in a test source set, decided by the symbol graph's own rule."""
        return is_test_path(self.file_path)


@dataclass(frozen=True, slots=True)
class PairReason:
    """Why one pair of units inside a logic class was accepted."""

    left: str
    right: str
    reason: str

    def __str__(self) -> str:
        """Render the reason the way the text report prints it."""
        return f"{self.left} ~ {self.right}: {self.reason}"


@dataclass(frozen=True, slots=True)
class CloneClass:
    """A set of units judged to be copies of one another under one kind."""

    kind: CloneKind
    members: tuple[Unit, ...]
    tokens: int
    reasons: tuple[PairReason, ...] = ()
    #: Class-level findings (a representative shape, a suggested home); printed before any pair reason.
    notes: tuple[str, ...] = ()
    #: Whether the class rests on stated intent alone (names, signatures, documentation) rather than on code:
    #: it ranks after every class of its kind that code evidences.
    inferred: bool = False
    #: What the class is rooted at when no member names it well (an idiom's shape); None roots it at its first member.
    root: str | None = None
    #: Why the class is a lesser finding of its kind, if it is: it ranks after the rest (`Demotion.standing`).
    demoted: Demotion | None = None

    @property
    def root_label(self) -> str:
        """What a one-line summary names the class by."""
        if self.root is not None:
            return self.root
        head = self.members[0]
        return f"{head.kind} {head.name}"

    @property
    def standing(self) -> tuple[int, int]:
        """The sort key within one kind: code-evidenced classes first, then inferred, then demoted; then by score."""
        if self.demoted is not None:
            return self.demoted.standing, -self.score
        return (1 if self.inferred else 0), -self.score

    @property
    def files(self) -> int:
        """How many distinct files the class spans."""
        return len({member.file_path for member in self.members})

    @property
    def score(self) -> int:
        """Rank of the class, weighed the way its kind declares (:class:`Ranking`)."""
        files = self.files
        ranking = self.kind.facts.ranking
        if ranking is Ranking.MASS:
            return self.tokens * len(self.members) * files
        if ranking is Ranking.SPREAD:
            return min(len(self.members), SPREAD_PER_FILE * files) * files
        raise ValueError(f"Unhandled ranking {ranking!r}")

    @property
    def shown_members(self) -> tuple[Unit, ...]:
        """The members a rendering lists: all of them, or the first `site_cap` for a many-site kind."""
        cap = self.kind.facts.site_cap
        return self.members if cap is None else self.members[:cap]

    @property
    def lane(self) -> Lane:
        """Whether every member is production code, every member is test code, or both."""
        flags = {member.is_test for member in self.members}
        if flags == {False}:
            return Lane.PRODUCTION
        if flags == {True}:
            return Lane.TEST
        return Lane.MIXED

    @property
    def key(self) -> str:
        """A stable identity for this class: its kind plus a digest of its members' streams.

        File paths and line numbers are deliberately absent, so editing around a clone, moving
        a file, or scanning from a different ancestor root all keep the key; adding or removing
        a copy changes it, which is what makes a stale ignore entry visible.
        """
        attribute = self.kind.facts.key_attribute
        parts = sorted(getattr(member, attribute) for member in self.members)
        digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()
        return f"{self.kind.value}:{digest[:12]}"

    @property
    def aggregate_reasons(self) -> list[str]:
        """One consensus line plus the members that deviate from it, for classes of 3+ copies."""
        shared = set(self.members[0].calls)
        for member in self.members[1:]:
            shared &= set(member.calls)
        flows = {member.skeleton for member in self.members}
        literal_sets = {tuple(sorted(set(member.literals))) for member in self.members}
        calls_note = f"all call {_braced(sorted(shared))}" if shared else "no call shared by every copy"
        flow_note = (
            "control flow identical across all copies" if len(flows) == 1 else f"{len(flows)} control-flow shapes"
        )
        literal_note = "literals identical" if len(literal_sets) == 1 else "literals differ per copy"
        lines = [f"{len(self.members)} copies; {flow_note}; {calls_note}; {literal_note}"]
        outliers = [(member, extra) for member in self.members if (extra := sorted(set(member.calls) - shared))]
        for member, extra in outliers[:_AGGREGATE_OUTLIERS]:
            lines.append(f"outlier {member.name} also calls {_braced(extra)}")
        if len(outliers) > _AGGREGATE_OUTLIERS:
            lines.append(f"and {len(outliers) - _AGGREGATE_OUTLIERS} more member(s) with extra calls")
        return lines

    @property
    def wire_reasons(self) -> list[str]:
        """The reasons for the wire: the class notes, then an aggregate for 3+ copies or deduped pairs below that."""
        notes = list(self.notes)
        if not self.reasons:
            return notes
        if len(self.members) >= AGGREGATE_FROM:
            return notes + self.aggregate_reasons
        verdicts = {reason.reason for reason in self.reasons}
        if len(verdicts) == 1:
            return [*notes, self.reasons[0].reason]
        return notes + [str(reason) for reason in self.reasons]

    def to_dict(self) -> dict[str, Any]:
        """Render the class for the wire; a many-site kind lists only its first `site_cap` members."""
        return {
            "key": self.key,
            "kind": self.kind.value,
            "lane": self.lane.value,
            "score": self.score,
            **({"inferred": True} if self.inferred else {}),
            **({"root": self.root} if self.root is not None else {}),
            **({"demoted": self.demoted.value} if self.demoted is not None else {}),
            "tokens": self.tokens,
            "copies": len(self.members),
            "files": self.files,
            "reasons": self.wire_reasons,
            "members": [
                {
                    "file_path": member.file_path,
                    "start_line": member.start_line,
                    "end_line": member.end_line,
                    "kind": member.kind,
                    "name": member.name,
                    "tokens": member.token_count,
                    "is_test": member.is_test,
                    "modifiers": list(member.modifiers),
                    "visibility": member.visibility.value,
                    "container_visibility": member.container_visibility.value,
                }
                for member in self.shown_members
            ],
        }


@dataclass
class DupeReport:
    """Everything one duplication run found, plus what it cost."""

    root: str
    analyzed_files: int = 0
    #: Files that were walked but whose extraction raised; a missing grammar lands here.
    failed_files: int = 0
    #: Up to a handful of the paths behind `failed_files`, for the note.
    failed_examples: list[str] = field(default_factory=list)
    #: The extensions this run walked, so an empty report can say what it was even looking for.
    supported_extensions: tuple[str, ...] = ()
    units: int = 0
    body_units: int = 0
    elapsed_seconds: float = 0.0
    min_tokens: int = 0
    min_statements: int = 0
    classes: list[CloneClass] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: The kinds this run looked for, so a baseline diff never calls an unscanned kind resolved.
    kinds: tuple[CloneKind, ...] = tuple(CloneKind)
    #: The lane this run was restricted to, or None when every lane was reported.
    lane: Lane | None = None
    #: The paths a focused run reported classes for (`--focus`); empty for a run that reported every class.
    focus: tuple[str, ...] = ()
    #: Classes a justified `.zemble/dupes.ignore` entry took out of the report.
    suppressed: list[CloneClass] = field(default_factory=list)
    #: Ignore-file violations: entries without a justification, and entries matching nothing.
    ignore_problems: list[str] = field(default_factory=list)
    #: Cross-module verdicts keyed by class key (`zemble.dedup.homes.HomeVerdict`); typed loosely
    #: because importing homes here would be a cycle.
    homes: dict[str, Any] = field(default_factory=dict)

    def of_kind(self, kind: CloneKind) -> list[CloneClass]:
        """Return this report's classes of one kind, already ranked."""
        return [clone for clone in self.classes if clone.kind is kind]

    def section(self, lane: Lane, kind: CloneKind) -> list[CloneClass]:
        """Return this report's classes of one lane and kind, already ranked."""
        return [clone for clone in self.classes if clone.kind is kind and clone.lane is lane]

    def clone_dict(self, clone: CloneClass) -> dict[str, Any]:
        """Render one class for the wire, with its cross-module verdict when one exists."""
        payload = clone.to_dict()
        verdict = self.homes.get(clone.key)
        if verdict is not None:
            payload["home"] = verdict.to_dict()
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Render the whole report for the wire."""
        return {
            "root": self.root,
            "analyzed_files": self.analyzed_files,
            "failed_files": self.failed_files,
            "supported_extensions": list(self.supported_extensions),
            "units": self.units,
            "body_units": self.body_units,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "min_tokens": self.min_tokens,
            "min_statements": self.min_statements,
            "kinds": [kind.value for kind in self.kinds],
            "lane": self.lane.value if self.lane is not None else None,
            "focus": list(self.focus),
            "notes": list(self.notes),
            "suppressed": len(self.suppressed),
            "ignore_problems": list(self.ignore_problems),
            "classes": [self.clone_dict(clone) for clone in self.classes],
        }
