"""The scoring and the verdict, over hand-built hits so every rule is visible."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zemble.home.config import HomeConfig
from zemble.home.decide import Confidence, Mechanism, Verdict, decide
from zemble.home.tables import DeclaredRow, RowMatch
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
