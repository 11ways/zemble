"""The scoring and the verdict, over hand-built hits so every rule is visible."""

from __future__ import annotations

import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from zemble.home.config import HomeConfig
from zemble.home.decide import Confidence, Mechanism, Verdict, decide
from zemble.home.tables import DeclaredRow, RowMatch, RowMatchKind
from zemble.types import Chunk, SearchResult

_CONFIG = """
    order = ["protoblast", "zenit", "plumage", "zenit-widget", "zenit-cms"]

    [modules]
    protoblast = "protoblast/**"
    zenit = "zenit/**"
    plumage = "plumage/**"
    zenit-widget = "zenit-widget/**"
    zenit-cms = "zenit-cms/**"

    [[forbidden]]
    from = "zenit-widget"
    to = "zenit-cms"
    why = "widget never depends on cms"

    [skills]
    plumage = ["plumage-components"]

    [[rules]]
    text = "Nothing lands without at least one wired consumer and a test"

    [[rules]]
    text = "Anything rendered belongs in plumage"
    modules = ["plumage"]
"""


@pytest.fixture
def config(tmp_path: Path) -> HomeConfig:
    """A five-module workspace with one forbidden dependency, one skill and two rules."""
    (tmp_path / ".zemble").mkdir()
    (tmp_path / ".zemble" / "home.toml").write_text(textwrap.dedent(_CONFIG).strip() + "\n", encoding="utf-8")
    return HomeConfig.load(tmp_path)


def hit(module: str, name: str, score: float) -> SearchResult:
    """A search result in a module's source tree."""
    path = f"{module}/src/main/java/{name}.java"
    chunk = Chunk(content="x", file_path=path, start_line=1, end_line=9, language="java")
    return SearchResult(chunk=chunk, score=score)


def mechanism(module: str, label: str, score: float, consumers: tuple[str, ...] = ()) -> Mechanism:
    """A symbol the graph reported, with the modules that consume it."""
    return Mechanism(
        label=label,
        kind="class",
        signature=f"public class {label}",
        module=module,
        file_path=f"{module}/src/main/java/{label}.java",
        start_line=1,
        end_line=40,
        score=score,
        consumer_modules=consumers,
        caller_count=len(consumers) * 2,
        implementation_count=0,
    )


def test_an_existing_mechanism_with_spread_is_extended(config: HomeConfig) -> None:
    """A top hit used from two modules is the answer: extend it, do not build a second one."""
    hits = [hit("zenit", "PageWindow", 0.9), hit("zenit-cms", "Paging", 0.5)]
    answer = decide(
        config,
        "paginate a list of records",
        hits,
        [mechanism("zenit", "PageWindow", 0.9, ("zenit-widget", "zenit-cms")), mechanism("zenit-cms", "Paging", 0.5)],
    )
    assert answer.verdict is Verdict.EXTEND_EXISTING, "a strong match means extend"
    assert answer.extend is not None and answer.extend.label == "PageWindow", "it names what to extend"
    assert answer.home == "zenit", "and the module that already holds it"
    assert answer.confidence is Confidence.HIGH, "consumers in two modules is a confident call"
    assert any("do not duplicate" in reason for reason in answer.reasons), "the verdict says so out loud"
    assert answer.mechanisms[0].strong and not answer.mechanisms[1].strong, "only the top one is strong"


def test_a_private_helper_is_not_a_mechanism(config: HomeConfig) -> None:
    """A hit with one consumer in its own module is not "this already exists"."""
    hits = [hit("zenit-cms", "ListPage", 0.8), hit("zenit-cms", "DetailPage", 0.6)]
    answer = decide(config, "render a picker for related records", hits, [mechanism("zenit-cms", "ListPage", 0.8)])
    assert answer.verdict is Verdict.NEW_MECHANISM, "no spread and no core position means nothing to extend"
    assert answer.home == "zenit-cms", "the only module with hits is the home"
    assert not answer.mechanisms[0].strong, "the helper is reported, not promoted"


def test_a_symbol_a_declared_row_names_is_strong_without_consumers(config: HomeConfig) -> None:
    """A mechanism the table itself names is the mechanism, however few modules use it."""
    row = DeclaredRow(
        capability="UI preference cookies (`PreferenceCookie.named(...)` = ONE cookie shape)",
        symbols=("PreferenceCookie.named",),
        home_modules=("zenit",),
        home_names=("common/ui",),
        consumer_modules=(),
        file="CLAUDE.md",
        line=17,
        raw_home="`zenit` core (`common/ui`)",
    )
    match = RowMatch(row=row, score=0.6, shared=("cookie", "preference"))
    hits = [hit("zenit", "PreferenceCookie", 0.9), hit("zenit", "Themes", 0.88)]
    found = [mechanism("zenit", "PreferenceCookie", 0.9), mechanism("zenit", "Themes", 0.88)]
    # 1. Its wrappers all live in its own module, so consumer spread says "private helper".
    without = decide(config, "remember a ui preference in a cookie", hits, found)
    assert without.verdict is Verdict.NEW_MECHANISM, "step 1: spread alone cannot see it"

    # 2. The declared row names it, which is the evidence the graph does not have.
    answer = decide(config, "remember a ui preference in a cookie", hits, found, [match])
    assert answer.verdict is Verdict.EXTEND_EXISTING, "step 2: a declared name is a strong match"
    assert answer.extend is not None and answer.extend.label == "PreferenceCookie", "step 2: it names what to extend"
    assert any(
        "named by the declared row 'UI preference cookies'" in reason for reason in answer.mechanisms[0].reasons
    ), "step 2: and says which row named it"

    # 3. The row named `PreferenceCookie.named`, so its owning class matched; a neighbour
    #    that is just as relevant but named nowhere in the table did not.
    assert answer.mechanisms[0].label == "PreferenceCookie", "step 3: the named class is the strong one"
    assert not answer.mechanisms[1].strong, "step 3: an equally relevant unnamed neighbour is only a hit"

    # 4. A row that names no symbol leaves the old rule untouched.
    bare = RowMatch(row=replace(row, symbols=()), score=0.6, shared=("cookie", "preference"))
    plain = decide(config, "remember a ui preference in a cookie", hits, found, [bare])
    assert plain.verdict is Verdict.NEW_MECHANISM, "step 4: no named symbol, no by-declaration match"

    # 5. A row the description only faintly resembles names nothing: every row in the
    #    table names classes, and a distant one would make any neighbour "this exists".
    leader = RowMatch(row=replace(row, capability="Session cookies", symbols=("SessionCookies",)), score=0.9, shared=())
    faint = decide(config, "remember a ui preference in a cookie", hits, found, [leader, match])
    assert faint.verdict is Verdict.NEW_MECHANISM, "step 5: only rows near the best match may name a mechanism"


def test_closer_to_core_wins_but_only_where_the_family_lives(config: HomeConfig) -> None:
    """The core-proximity bonus needs the module to already hold this family."""
    hits = [hit("zenit", "Icon", 0.6), hit("zenit-cms", "IconCell", 0.55), hit("zenit-cms", "IconColumn", 0.5)]
    answer = decide(config, "an icon vocabulary shared by admin tables", hits, [])
    modules = [candidate.module for candidate in answer.candidates]
    assert modules[:2] == ["zenit", "zenit-cms"], "zenit leads on the bonus despite less relevance mass"
    assert any("closer to the core" in reason for reason in answer.candidates[0].reasons), "the bonus is explained"
    assert "protoblast" not in modules, "a module with no hits is never a candidate, however core it is"
    assert not any("closer to the core" in reason for reason in answer.candidates[1].reasons), (
        "the last module in the order has no consumers below it, so it gets no bonus"
    )


def test_a_forbidden_placement_is_penalised_and_named(config: HomeConfig) -> None:
    """Choosing a home that a consumer may not depend on costs it the lead."""
    hits = [
        hit("zenit-cms", "DashboardPanel", 0.9),
        hit("zenit-cms", "PanelPeer", 0.8),
        hit("zenit-widget", "Tree", 0.4),
    ]
    answer = decide(config, "a dashboard panel surface action", hits, [])
    by_module = {candidate.module: candidate for candidate in answer.candidates}
    assert by_module["zenit-cms"].violations, "the forbidden dependency is recorded"
    assert "widget never depends on cms" in by_module["zenit-cms"].violations[0], "with the workspace's own reason"
    assert answer.candidates[0].module == "zenit-widget", "so the placement that is allowed wins"


def test_a_declared_row_names_the_home(config: HomeConfig) -> None:
    """A matching declared-home row lifts the module it names and says which row did it."""
    row = DeclaredRow(
        capability="Pagination arithmetic over a record source",
        symbols=(),
        home_modules=("zenit",),
        home_names=("common/data",),
        consumer_modules=("zenit-cms",),
        file="CLAUDE.md",
        line=42,
        raw_home="`zenit` core (`common/data`)",
    )
    hits = [hit("zenit-cms", "Paging", 0.9), hit("zenit-cms", "PagingBar", 0.8), hit("zenit", "PageWindow", 0.3)]
    answer = decide(config, "pagination arithmetic", hits, [], [RowMatch(row=row, score=0.9, shared=("pagination",))])
    assert answer.candidates[0].module == "zenit", "the declared home outweighs the relevance mass"
    assert any("declared home for 'Pagination arithmetic" in reason for reason in answer.candidates[0].reasons), (
        "and the answer says which row said so"
    )


def test_two_close_candidates_stay_uncertain(config: HomeConfig) -> None:
    """Near-equal candidates are reported as a call to make, never as a verdict."""
    hits = [hit("plumage", "Overlay", 0.5), hit("zenit-widget", "Popover", 1.05)]
    answer = decide(config, "an overlay dismissal substrate", hits, [])
    assert answer.verdict is Verdict.UNCERTAIN, "a photo finish is uncertain"
    assert answer.confidence is Confidence.LOW, "and says so"
    assert answer.home is None, "no home is claimed"
    assert "plumage" in answer.reasons[0] and "zenit-widget" in answer.reasons[0], "both are named"


def test_no_hits_is_uncertain_not_a_guess(config: HomeConfig) -> None:
    """An empty search says nothing matched rather than inventing a module."""
    answer = decide(config, "quantum teleportation", [], [])
    assert answer.verdict is Verdict.UNCERTAIN, "nothing found is uncertain"
    assert answer.candidates == [] and answer.home is None, "and names no home"


def test_generic_mode_says_what_it_could_not_use(tmp_path: Path) -> None:
    """Without a config, modules are path segments and the answer admits it."""
    answer = decide(HomeConfig.load(tmp_path), "session cookies", [hit("zenit", "SessionCookies", 0.7)], [])
    assert answer.notes and "home.toml" in answer.notes[0], "the answer states the workspace declared nothing"
    assert answer.candidates[0].module == "zenit", "the path segment still gives a candidate"
    assert not any("closer to the core" in reason for reason in answer.candidates[0].reasons), (
        "with no declared order there is no core-proximity claim"
    )


def test_the_checklist_carries_the_rules_that_apply(config: HomeConfig) -> None:
    """Only the rules, refusals and skills touching the candidates are echoed."""
    hits = [hit("plumage", "Overlay", 0.9), hit("zenit-cms", "Sheet", 0.2)]
    answer = decide(config, "an overlay", hits, [])
    assert "Nothing lands without at least one wired consumer and a test" in answer.checklist.rules, "unscoped applies"
    assert "Anything rendered belongs in plumage" in answer.checklist.rules, "plumage is a candidate"
    assert answer.checklist.skills["plumage"] == ("plumage-components",), "its skill is named"
    assert any("zenit-widget must not depend on zenit-cms" in entry for entry in answer.checklist.forbidden), (
        "a refusal touching a candidate is echoed"
    )


def test_markdown_renders_the_sections_in_order(config: HomeConfig) -> None:
    """The rendering is the answer: sections, reasons, verdict and checklist."""
    hits = [hit("zenit", "PageWindow", 0.9), hit("zenit-cms", "Paging", 0.4)]
    answer = decide(config, "paginate records", hits, [mechanism("zenit", "PageWindow", 0.9, ("zenit-cms",))])
    text = answer.render()
    headings = ("# Home for:", "## Existing mechanisms", "## Candidate homes", "## Verdict", "## Checklist")
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions), "the sections keep their declared order"
    assert "EXTEND_EXISTING" in text, "the verdict is in the markdown"
    assert "`PageWindow` in **zenit**" in text, "and so is the mechanism it names"
    assert answer.to_dict()["verdict"] == "EXTEND_EXISTING", "the JSON shape carries the same verdict"


_SIBLINGS = """
    order = ["protoblast", "zenit", "zenit-flow", "zenit-widget"]

    [modules]
    protoblast = "protoblast/**"
    zenit = "zenit/**"

    [modules.zenit-flow]
    globs = ["zenit-flow/**"]
    depends_on = ["zenit"]

    [modules.zenit-widget]
    globs = ["zenit-widget/**"]
    depends_on = ["zenit"]
"""


@pytest.fixture
def siblings(tmp_path: Path) -> HomeConfig:
    """A workspace where two modules both depend on zenit and neither depends on the other."""
    (tmp_path / ".zemble").mkdir()
    (tmp_path / ".zemble" / "home.toml").write_text(textwrap.dedent(_SIBLINGS).strip() + "\n", encoding="utf-8")
    return HomeConfig.load(tmp_path)


def _row(capability: str, symbols: tuple[str, ...], home: str) -> DeclaredRow:
    """One declared-home row, as the table parser would have produced it."""
    return DeclaredRow(
        capability=capability,
        symbols=symbols,
        home_modules=(home,),
        home_names=(),
        consumer_modules=(),
        file="CLAUDE.md",
        line=7,
        raw_home=f"`{home}`",
    )


def test_a_sibling_mechanism_is_not_offered_for_extension(siblings: HomeConfig) -> None:
    """Two modules with no dependency path between them get their shared substrate, not each other."""
    hits = [hit("zenit-widget", "WidgetMigrations", 0.9), hit("zenit-flow", "FlowMigrations", 0.7)]
    found = [
        mechanism("zenit-widget", "WidgetMigrations", 0.9, ("zenit-cms", "quirkyquarters")),
        mechanism("zenit-flow", "FlowMigrations", 0.7),
    ]
    answer = decide(siblings, "shared Flow/Widget migration state", hits, found)

    assert answer.verdict is not Verdict.EXTEND_EXISTING, "extending a sibling is not an option"
    assert answer.verdict is Verdict.NEW_MECHANISM, "the shared substrate is a new mechanism there"
    assert answer.suggested_home == "zenit", "which is the highest-ranked module both can depend on"
    assert answer.home == "zenit", "and that is the home the verdict names"
    assert any(
        "zenit-widget and zenit-flow are siblings (no dependency path); the shared mechanism belongs in zenit" in reason
        for reason in answer.reasons
    ), "the reason says it in one line"
    assert answer.to_dict()["suggested_home"] == "zenit", "and the payload carries it"
    assert "belongs in zenit" in answer.render(), "the markdown says where the shared mechanism goes"


def test_a_mechanism_its_consumer_can_reach_is_still_extended(siblings: HomeConfig) -> None:
    """The sibling rule only fires between modules that cannot reach each other."""
    hits = [hit("zenit", "Migrations", 0.9), hit("zenit-flow", "FlowMigrations", 0.6)]
    found = [mechanism("zenit", "Migrations", 0.9, ("zenit-flow", "zenit-widget"))]
    answer = decide(siblings, "record a migration as applied", hits, found)
    assert answer.verdict is Verdict.EXTEND_EXISTING, "zenit-flow depends on zenit, so it may use it"
    assert answer.home == "zenit" and answer.suggested_home is None, "no suggestion is needed"


def test_a_row_matching_on_words_alone_declares_nothing(siblings: HomeConfig) -> None:
    """A capability row sharing vocabulary must not turn a neighbour's class into "this exists"."""
    row = _row("Record comments on a model, including nullable authors", ("RecordComment",), "zenit-widget")
    hits = [hit("zenit", "ActivityLog", 0.9)]
    found = [mechanism("zenit", "ActivityLog", 0.9)]
    answer = decide(
        siblings,
        "look up a record by its nullable model primary key",
        hits,
        found,
        [RowMatch(row=row, score=0.6, shared=("model", "record", "nullable"))],
    )
    assert answer.verdict is not Verdict.EXTEND_EXISTING, "a word overlap is not a declaration"
    assert not any(mech.strong for mech in answer.mechanisms), "and no mechanism is strong because of it"
    assert all(mech.declared is None for mech in answer.mechanisms), "nothing was declared about these symbols"
    assert any("lexically related row" in reason for reason in answer.reasons), "the answer says what it had"


def test_a_row_naming_the_symbol_still_declares_it(siblings: HomeConfig) -> None:
    """The row that writes the symbol down is still the strongest evidence there is."""
    row = _row("Nullable primary-key lookup (`Models.byIdOrNull`)", ("Models.byIdOrNull",), "zenit")
    hits = [hit("zenit", "Models", 0.9)]
    found = [mechanism("zenit", "Models.byIdOrNull", 0.9)]
    answer = decide(
        siblings,
        "look up a record by its nullable model primary key",
        hits,
        found,
        [RowMatch(row=row, score=0.6, shared=("lookup", "nullable"))],
    )
    assert answer.verdict is Verdict.EXTEND_EXISTING, "the row names this exact symbol"
    extend = answer.extend
    assert extend is not None and extend.declared is not None, "the evidence is attached to the mechanism"
    assert extend.declared.kind is RowMatchKind.EXACT_MEMBER, "and it is an exact-member match"
    assert any("declared (CLAUDE.md row names `Models.byIdOrNull`)" in reason for reason in answer.reasons), (
        "the verdict carries the evidence kind"
    )


def test_graph_evidence_is_named_as_such(config: HomeConfig) -> None:
    """A strong match with no declared row says so instead of implying a declaration."""
    hits = [hit("zenit", "PageWindow", 0.9), hit("zenit-cms", "Paging", 0.5)]
    answer = decide(config, "paginate a list of records", hits, [mechanism("zenit", "PageWindow", 0.9, ("zenit-cms",))])
    assert answer.verdict is Verdict.EXTEND_EXISTING, "consumer position still makes it strong"
    assert any("evidence: graph evidence" in reason for reason in answer.reasons), "and the evidence is named"
    assert any("no dependency information for this workspace" in reason for reason in answer.reasons), (
        "a workspace with no dependency facts is told so rather than pretending"
    )


def test_a_consumer_copy_is_not_the_mechanism(siblings: HomeConfig) -> None:
    """A strong match inside a module the demand cannot reach is a copy, not the home."""
    hits = [
        hit("zenit", "Records", 0.9),
        hit("zenit", "RecordSources", 0.8),
        hit("zenit-flow", "FlowRecordSources", 0.75),
    ]
    found = [mechanism("zenit-flow", "FlowRecordSources", 0.75, ("zenit-widget", "zenit-cms"))]
    answer = decide(siblings, "register which module owns a record source", hits, found)
    assert answer.verdict is Verdict.NEW_MECHANISM, "the copy in the consumer is not extended"
    assert answer.home == "zenit" and answer.suggested_home == "zenit", "the demand's own module is the home"
    assert any("cannot depend on zenit-flow" in reason for reason in answer.reasons), "and the answer says why"
