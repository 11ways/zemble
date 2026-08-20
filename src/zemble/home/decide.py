"""Turning search hits, graph facts and declared rows into a home verdict.

The scoring is deliberately small and readable: relevance mass per module, a bonus
for a module the workspace already declared as the home of this capability family,
a bonus for sitting closer to the core than the modules that would consume it, and
a penalty for a placement the workspace forbids. Everything it decides is reported
as a sentence, because a verdict nobody can audit is worse than no verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from zemble.home.config import HomeConfig
from zemble.home.tables import RowMatch
from zemble.types import SearchResult

#: A mechanism this close to the best one is still a candidate for "this already exists".
STRONG_SCORE_RATIO = 0.85
# The bonuses are calibrated against the relevance share, which is normalised across
# the modules that were hit and so tops out near 1.0: a declared home has to be able
# to outweigh a module that holds most of the matching code, and a forbidden
# placement has to lose outright rather than merely slip.
#: A module a matched declared row names as the home gets up to this much added,
#: scaled by how well the row matched: a row the description barely resembles must
#: not outweigh every hit in the workspace.
DECLARED_BONUS = 0.8
#: A module already holding this family and sitting closer to the core gets this.
CORE_BONUS = 0.35
#: Each forbidden dependency a placement would create costs this.
FORBIDDEN_PENALTY = 1.0
#: Two candidates within this share of the leader's score make the answer uncertain.
UNCERTAIN_MARGIN = 0.15
#: A lead of at least this share of the leader's score is called high confidence.
CONFIDENT_MARGIN = 0.4
#: How many candidate homes are reported.
MAX_CANDIDATES = 3


class Verdict(str, Enum):
    """What the answer concluded."""

    EXTEND_EXISTING = "EXTEND_EXISTING"
    NEW_MECHANISM = "NEW_MECHANISM"
    UNCERTAIN = "UNCERTAIN"


class Confidence(str, Enum):
    """How much the verdict should be leaned on."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Mechanism:
    """An existing symbol the description might already be describing."""

    label: str
    kind: str
    signature: str
    module: str
    file_path: str
    start_line: int
    end_line: int
    score: float
    #: Distinct modules that call or implement it, its own module excluded.
    consumer_modules: tuple[str, ...] = ()
    caller_count: int = 0
    implementation_count: int = 0
    strong: bool = False
    reasons: tuple[str, ...] = ()

    @property
    def location(self) -> str:
        """File path and line range as a string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        """Render the mechanism as JSON-ready data."""
        return {
            "label": self.label,
            "kind": self.kind,
            "signature": self.signature,
            "module": self.module,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "location": self.location,
            "score": round(self.score, 4),
            "consumer_modules": list(self.consumer_modules),
            "caller_count": self.caller_count,
            "implementation_count": self.implementation_count,
            "strong": self.strong,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ModuleHits:
    """What one module contributed to the search."""

    module: str
    mass: float
    hits: int
    best_score: float
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Render the module's share as JSON-ready data."""
        return {
            "module": self.module,
            "mass": round(self.mass, 4),
            "hits": self.hits,
            "best_score": round(self.best_score, 4),
            "files": list(self.files),
        }


@dataclass(frozen=True)
class Candidate:
    """A module the mechanism could live in, and why it scored what it scored."""

    module: str
    score: float
    reasons: tuple[str, ...]
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Render the candidate as JSON-ready data."""
        return {
            "module": self.module,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class Checklist:
    """What to read and respect before writing the code."""

    rules: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    skills: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the checklist as JSON-ready data."""
        return {
            "rules": list(self.rules),
            "forbidden": list(self.forbidden),
            "skills": {module: list(names) for module, names in self.skills.items()},
        }


@dataclass(frozen=True)
class Similar:
    """A location the best hit resembles, offered as "look here too"."""

    file_path: str
    start_line: int
    end_line: int
    module: str
    score: float

    @property
    def location(self) -> str:
        """File path and line range as a string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        """Render the location as JSON-ready data."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "location": self.location,
            "module": self.module,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class DocHit:
    """A documentation chunk the same description matched."""

    file_path: str
    start_line: int
    end_line: int
    module: str
    score: float
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        """Render the documentation hit as JSON-ready data."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "module": self.module,
            "score": round(self.score, 4),
            "excerpt": self.excerpt,
        }


@dataclass
class HomeAnswer:
    """The whole answer to "does this exist, and where should it live"."""

    description: str
    verdict: Verdict
    confidence: Confidence
    reasons: list[str] = field(default_factory=list)
    home: str | None = None
    extend: Mechanism | None = None
    mechanisms: list[Mechanism] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    module_hits: list[ModuleHits] = field(default_factory=list)
    declared: list[RowMatch] = field(default_factory=list)
    similar: list[Similar] = field(default_factory=list)
    docs: list[DocHit] = field(default_factory=list)
    checklist: Checklist = field(default_factory=Checklist)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render the answer as JSON-ready data."""
        return {
            "description": self.description,
            "verdict": self.verdict.value,
            "confidence": self.confidence.value,
            "home": self.home,
            "extend": self.extend.to_dict() if self.extend else None,
            "reasons": list(self.reasons),
            "mechanisms": [mechanism.to_dict() for mechanism in self.mechanisms],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "module_hits": [entry.to_dict() for entry in self.module_hits],
            "declared": [match.to_dict() for match in self.declared],
            "similar": [entry.to_dict() for entry in self.similar],
            "docs": [entry.to_dict() for entry in self.docs],
            "checklist": self.checklist.to_dict(),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Render the answer as markdown."""
        lines = [f"# Home for: {self.description}", ""]
        for note in self.notes:
            lines += [f"> {note}", ""]
        lines += _render_mechanisms(self)
        lines += _render_candidates(self)
        lines += _render_verdict(self)
        lines += _render_checklist(self.checklist)
        return "\n".join(lines).rstrip() + "\n"


def module_hits_of(config: HomeConfig, hits: Sequence[SearchResult]) -> list[ModuleHits]:
    """Group search hits by the module their file belongs to, heaviest first."""
    mass: dict[str, float] = {}
    counts: dict[str, int] = {}
    best: dict[str, float] = {}
    files: dict[str, list[str]] = {}
    for hit in hits:
        module = config.module_of(hit.chunk.file_path)
        mass[module] = mass.get(module, 0.0) + max(hit.score, 0.0)
        counts[module] = counts.get(module, 0) + 1
        best[module] = max(best.get(module, 0.0), hit.score)
        paths = files.setdefault(module, [])
        if hit.chunk.file_path not in paths:
            paths.append(hit.chunk.file_path)
    grouped = [
        ModuleHits(module=module, mass=mass[module], hits=counts[module], best_score=best[module], files=tuple(paths))
        for module, paths in files.items()
    ]
    grouped.sort(key=lambda entry: (-entry.mass, config.rank(entry.module), entry.module))
    return grouped


def mark_strong(
    config: HomeConfig, mechanisms: Sequence[Mechanism], row_matches: Sequence[RowMatch] = ()
) -> list[Mechanism]:
    """Decide which mechanisms are strong enough to be "this already exists".

    Strong means near the top of the ranking AND carrying the shape of a mechanism
    rather than one caller's private helper: consumers in two or more modules, a
    position closer to the core than everything that uses it, or - the case consumer
    spread cannot see - being the very symbol a matched declared row names.
    """
    if not mechanisms:
        return []
    named_by = _named_rows(row_matches)
    top = max(mechanism.score for mechanism in mechanisms)
    marked = []
    for mechanism in mechanisms:
        reasons: list[str] = []
        near_top = top > 0 and mechanism.score >= STRONG_SCORE_RATIO * top
        spread = len(mechanism.consumer_modules) >= 2
        own_rank = config.rank(mechanism.module)
        below = [module for module in mechanism.consumer_modules if config.rank(module) > own_rank]
        declared = _declaring_row(mechanism.label, named_by)
        if spread:
            reasons.append(
                f"used from {len(mechanism.consumer_modules)} modules: {', '.join(mechanism.consumer_modules)}"
            )
        if below and len(below) == len(mechanism.consumer_modules):
            reasons.append(f"lives closer to the core than all {len(below)} of its consumers")
        if declared is not None:
            reasons.append(f"named by the declared row '{declared.row.title}'")
        strong = bool(near_top and (spread or below or declared is not None))
        marked.append(replace(mechanism, strong=strong, reasons=tuple(reasons)))
    marked.sort(key=lambda mechanism: (-mechanism.score, mechanism.label))
    return marked


def _named_rows(row_matches: Sequence[RowMatch]) -> dict[str, RowMatch]:
    """Index the symbol names the matched rows write, best-matching row first.

    Only rows near the best match may name anything, the same near-the-top rule the
    mechanisms themselves are held to: a row the description faintly resembles names
    its own capability's classes, and letting those speak turns every neighbouring
    row's mechanism into "this already exists".

    The owning class of a `Class.member` name is indexed too: a row that names one
    method of a class is naming that class as the mechanism, and the hit the search
    anchors on is the type at least as often as the member.
    """
    if not row_matches:
        return {}
    best = max(match.score for match in row_matches)
    table: dict[str, RowMatch] = {}
    for match in row_matches:
        if match.score < STRONG_SCORE_RATIO * best:
            continue
        for name in match.row.symbols:
            for key in (name, name.split(".")[0]):
                table.setdefault(key, match)
    return table


def _declaring_row(label: str, named_by: dict[str, RowMatch]) -> RowMatch | None:
    """Return the matched row that names this symbol, if one does.

    AIDEV-NOTE: this is the "by declaration" strong match. A mechanism a human wrote
    into the capability table IS the mechanism, however few modules consume it: the
    wrappers around `PreferenceCookie` all live inside its own module by design, and
    the consumer-spread rule alone reads that as a private helper.
    """
    for key in (label, label.split(".")[0]):
        if key in named_by:
            return named_by[key]
    return None


def candidates_of(
    config: HomeConfig,
    grouped: Sequence[ModuleHits],
    row_matches: Sequence[RowMatch],
) -> list[Candidate]:
    """Score every module the hits touched as a possible home."""
    if not grouped:
        return []
    total = sum(entry.mass for entry in grouped) or 1.0
    declared: dict[str, RowMatch] = {}
    for match in row_matches:
        for module in match.row.home_modules:
            if match.score > declared.get(module, match).score or module not in declared:
                declared[module] = match
    strongest = grouped[0]
    # Only DECLARED modules can be "below" a candidate: an undeclared one has no known
    # position, and ranking it last would make every candidate closer to the core than it.
    declared_modules = set(config.modules)
    candidates = []
    for entry in grouped:
        share = entry.mass / total
        score = share
        reasons = [
            f"{entry.hits} of {sum(g.hits for g in grouped)} hits live here"
            f" ({share:.0%} of the relevance){'' if entry is not strongest else ', the largest share'}"
        ]
        # AIDEV-NOTE: the declared bonus is per MODULE and applied once, so a row whose
        # named symbol also turns up as a hit in that module does not pay twice: the
        # symbol makes the mechanism strong, the row makes the module a candidate.
        match = declared.get(entry.module)
        if match is not None:
            score += DECLARED_BONUS * min(match.score, 1.0)
            reasons.append(f"declared home for '{match.row.title}' (row match {match.score:.0%})")
        below = [
            other.module
            for other in grouped
            if other.module in declared_modules and config.rank(other.module) > config.rank(entry.module)
        ]
        if below and not config.generic:
            score += CORE_BONUS
            reasons.append(
                f"already holds this family and is closer to the core than its {len(below)} consumer module(s):"
                f" {', '.join(below)}"
            )
        # AIDEV-NOTE: a forbidden placement is checked against EVERY other module the
        # description touched, not only the ones further from the core: the module that
        # may not depend on this one is usually the one closer in (zenit-widget before
        # zenit-cms), and scoping the check by order would make the rule unreachable.
        violations = []
        for consumer in [other.module for other in grouped if other.module != entry.module]:
            rule = config.forbids(consumer, entry.module)
            if rule is not None:
                score -= FORBIDDEN_PENALTY
                violations.append(f"would make {consumer} depend on {entry.module}: {rule.why or 'forbidden'}")
        candidates.append(
            Candidate(module=entry.module, score=score, reasons=tuple(reasons), violations=tuple(violations))
        )
    candidates.sort(key=lambda candidate: (-candidate.score, config.rank(candidate.module), candidate.module))
    return candidates[:MAX_CANDIDATES]


def checklist_of(config: HomeConfig, modules: Sequence[str]) -> Checklist:
    """Collect the rules, refusals and skills that concern a set of candidate modules."""
    scope = tuple(modules)
    rules = tuple(rule.text for rule in config.rules if rule.applies_to(scope))
    forbidden = tuple(rule.describe() for rule in config.forbidden if rule.source in scope or rule.target in scope)
    skills = {module: config.skills_for(module) for module in scope if config.skills_for(module)}
    return Checklist(rules=rules, forbidden=forbidden, skills=skills)


def decide(
    config: HomeConfig,
    description: str,
    hits: Sequence[SearchResult],
    mechanisms: Sequence[Mechanism],
    row_matches: Sequence[RowMatch] = (),
    similar: Sequence[Similar] = (),
    docs: Sequence[DocHit] = (),
) -> HomeAnswer:
    """Weigh everything gathered about a feature description into one answer.

    :param config: What the workspace declared about itself.
    :param description: The feature someone is about to build.
    :param hits: Code search results, best first.
    :param mechanisms: The symbols behind the best hits, before strength is judged.
    :param row_matches: Declared-home rows the description looks like.
    :param similar: Locations resembling the best hit.
    :param docs: Documentation chunks the description matched.
    :return: The verdict, its candidates, and everything it was based on.
    """
    grouped = module_hits_of(config, hits)
    judged = mark_strong(config, mechanisms, row_matches)
    candidates = candidates_of(config, grouped, row_matches)
    strong = [mechanism for mechanism in judged if mechanism.strong]
    notes = []
    if config.generic:
        notes.append(
            "No .zemble/home.toml in this workspace: modules are guessed from the first path segment, and no"
            " declared homes, forbidden dependencies, rules or skills were available."
        )
    verdict, confidence, home, extend, reasons = _verdict(config, candidates, strong, row_matches)
    answer = HomeAnswer(
        description=description,
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        home=home,
        extend=extend,
        mechanisms=judged,
        candidates=candidates,
        module_hits=grouped,
        declared=list(row_matches),
        similar=list(similar),
        docs=list(docs),
        checklist=checklist_of(config, [candidate.module for candidate in candidates]),
        notes=notes,
    )
    return answer


def _verdict(
    config: HomeConfig,
    candidates: Sequence[Candidate],
    strong: Sequence[Mechanism],
    row_matches: Sequence[RowMatch],
) -> tuple[Verdict, Confidence, str | None, Mechanism | None, list[str]]:
    """Pick the verdict, its confidence, the home it names and the sentences behind it."""
    reasons: list[str] = []
    if strong:
        best = strong[0]
        reasons.append(f"{best.label} in {best.module} already covers this ({best.location})")
        reasons.extend(best.reasons)
        reasons.append("wire or extend it; do not duplicate it")
        for match in row_matches:
            if best.module in match.row.home_modules:
                reasons.append(f"and {match.row.file} declares {best.module} the home of '{match.row.title}'")
                break
        confidence = Confidence.HIGH if len(strong) == 1 or best.consumer_modules else Confidence.MEDIUM
        return Verdict.EXTEND_EXISTING, confidence, best.module, best, reasons
    if not candidates:
        reasons.append("nothing in this workspace matched the description, so it names no home")
        return Verdict.UNCERTAIN, Confidence.LOW, None, None, reasons
    top = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    lead = top.score - runner_up.score if runner_up else top.score
    relative = lead / top.score if top.score > 0 else 0.0
    if runner_up is not None and relative < UNCERTAIN_MARGIN:
        reasons.append(
            f"{top.module} ({top.score:.2f}) and {runner_up.module} ({runner_up.score:.2f}) score within"
            f" {relative:.0%} of each other; this is a call to make, not to read off"
        )
        reasons.extend(top.reasons)
        reasons.extend(runner_up.reasons)
        return Verdict.UNCERTAIN, Confidence.LOW, None, None, reasons
    reasons.append(f"no existing mechanism matched strongly enough to extend; {top.module} is the best home")
    reasons.extend(top.reasons)
    if top.violations:
        reasons.extend(top.violations)
    confidence = Confidence.HIGH if relative >= CONFIDENT_MARGIN else Confidence.MEDIUM
    if config.generic:
        confidence = Confidence.LOW if confidence is Confidence.MEDIUM else Confidence.MEDIUM
    return Verdict.NEW_MECHANISM, confidence, top.module, None, reasons


def _render_mechanisms(answer: HomeAnswer) -> list[str]:
    """Render the "does it already exist" section."""
    lines = ["## Existing mechanisms", ""]
    if not answer.mechanisms:
        lines += ["Nothing in the index looks like this.", ""]
    for mechanism in answer.mechanisms:
        lines += _render_mechanism(mechanism)
    lines.append("")
    lines += _render_declared(answer.declared)
    if answer.similar:
        lines += ["### Also similar", ""]
        lines += [f"- {entry.location} ({entry.module})" for entry in answer.similar]
        lines.append("")
    if answer.docs:
        lines += ["### Documentation", ""]
        lines += [f"- {entry.file_path}:{entry.start_line} ({entry.module}) - {entry.excerpt}" for entry in answer.docs]
        lines.append("")
    return lines


def _render_mechanism(mechanism: Mechanism) -> list[str]:
    """Render one existing mechanism and what the graph knows about its use."""
    mark = " **(strong match)**" if mechanism.strong else ""
    lines = [f"- `{mechanism.label}` in **{mechanism.module}**{mark}  -  {mechanism.location}"]
    if mechanism.signature:
        lines.append(f"  - `{mechanism.signature}`")
    if mechanism.consumer_modules:
        lines.append(
            f"  - consumers: {', '.join(mechanism.consumer_modules)}"
            f" ({mechanism.caller_count} caller(s), {mechanism.implementation_count} implementation(s))"
        )
    else:
        lines.append("  - no callers or implementations outside its own module")
    lines += [f"  - {reason}" for reason in mechanism.reasons]
    return lines


def _render_declared(matches: Sequence[RowMatch]) -> list[str]:
    """Render the declared-home rows the description looked like."""
    if not matches:
        return []
    lines = ["### Declared homes", ""]
    for match in matches:
        homes = ", ".join(match.row.home_modules) or "(no module named in backticks)"
        lines.append(f"- **{homes}** - '{match.row.title}' ({match.row.file}:{match.row.line})")
        lines.append(f"  - home cell: {match.row.raw_home}")
        if match.row.consumer_modules:
            lines.append(f"  - consumers: {', '.join(match.row.consumer_modules)}")
        lines.append(f"  - matched on: {', '.join(match.shared)}")
    lines.append("")
    return lines


def _render_candidates(answer: HomeAnswer) -> list[str]:
    """Render the ranked candidate homes."""
    lines = ["## Candidate homes", ""]
    if not answer.candidates:
        lines += ["No module scored: the search found nothing to place this beside.", ""]
        return lines
    for position, candidate in enumerate(answer.candidates, 1):
        lines.append(f"{position}. **{candidate.module}** (score {candidate.score:.2f})")
        for reason in candidate.reasons:
            lines.append(f"   - {reason}")
        for violation in candidate.violations:
            lines.append(f"   - FORBIDDEN: {violation}")
    lines.append("")
    return lines


def _render_verdict(answer: HomeAnswer) -> list[str]:
    """Render the verdict and the sentences behind it."""
    head = f"**{answer.verdict.value}** (confidence: {answer.confidence.value})"
    if answer.verdict is Verdict.EXTEND_EXISTING and answer.extend is not None:
        head += f" - extend `{answer.extend.label}` in {answer.extend.module}"
    elif answer.verdict is Verdict.NEW_MECHANISM and answer.home:
        head += f" - new mechanism, home: {answer.home}"
    lines = ["## Verdict", "", head, ""]
    lines += [f"- {reason}" for reason in answer.reasons]
    lines.append("")
    return lines


def _render_checklist(checklist: Checklist) -> list[str]:
    """Render the rules, refusals and skills that apply to the candidates."""
    if not (checklist.rules or checklist.forbidden or checklist.skills):
        return []
    lines = ["## Checklist", ""]
    for module, names in checklist.skills.items():
        lines.append(f"- read the {', '.join(names)} skill(s) before designing in {module}")
    for entry in checklist.forbidden:
        lines.append(f"- {entry}")
    for rule in checklist.rules:
        lines.append(f"- {rule}")
    lines.append("")
    return lines


__all__ = [
    "CORE_BONUS",
    "DECLARED_BONUS",
    "FORBIDDEN_PENALTY",
    "MAX_CANDIDATES",
    "STRONG_SCORE_RATIO",
    "UNCERTAIN_MARGIN",
    "Candidate",
    "Checklist",
    "Confidence",
    "DocHit",
    "HomeAnswer",
    "Mechanism",
    "ModuleHits",
    "Similar",
    "Verdict",
    "candidates_of",
    "checklist_of",
    "decide",
    "mark_strong",
    "module_hits_of",
]
