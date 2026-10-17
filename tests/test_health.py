"""Behaviour journeys of `zemble health` over a fixture with every shape it reports and every one it must not."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import git
from zemble.cli import _cli_main
from zemble.graph import cli as graph_cli
from zemble.graph.model import Resolution
from zemble.graph.provider import open_provider
from zemble.graph.store import graph_db_path, graph_present
from zemble.health.answers import COPYING_OPT_IN, HealthRequest, health_payload
from zemble.health.cli import in_process
from zemble.health.collect import collect
from zemble.health.git import changed_lines, history_checkouts
from zemble.health.history import measure_history
from zemble.health.model import Axis, Comparison, Signal, Tier
from zemble.health.score import AXIS_WEIGHTS, SCORE_DISCLAIMER
from zemble.health.signals import PRODUCERS
from zemble.health.style import NameShape, StyleDimension, StyleMeasurement, chi_square_quantile, measure_style
from zemble.home.config import HomeConfig

_CORE = "src/main/java/com/example/core"
_PKG = "com.example"

#: Every finding the fixture must produce, by key, and nothing else.
_EXPECTED = {
    f"long_callable:java:{_PKG}.core.Util.tangled(int,int,int,int,int,int)",
    f"complex_callable:java:{_PKG}.core.Util.tangled(int,int,int,int,int,int)",
    f"deep_nesting:java:{_PKG}.core.Util.tangled(int,int,int,int,int,int)",
    f"many_parameters:java:{_PKG}.core.Util.tangled(int,int,int,int,int,int)",
    "many_types_per_file:python:src/main/python/zoo.py",
    f"forbidden_crossing:java:{_CORE}/Leaker.java->app",
    f"single_implementation_interface:java:{_PKG}.core.Greeter",
    f"single_subclass_abstract:java:{_PKG}.core.AbstractStore",
    f"deep_inheritance:java:{_PKG}.core.Tower.Level7",
    f"unreferenced_symbol:java:{_PKG}.core.Util.deadHelper(int)",
    f"lone_public_callable:java:{_PKG}.core.Ledger.normalize(int)",
    f"package_cycle:java:core:{_CORE}/north",
    f"package_cycle:java:core:{_CORE}/south",
    f"feature_envy:java:{_PKG}.core.Invoice.label()",
}

#: The duplication tier's findings on the fixture, by key: two copied bodies in two files, one
#: vocabulary declared twice, and one public forwarder.
_COPYING = {
    f"clone_mass:java:{_CORE}/Billing.java",
    f"clone_mass:java:{_CORE}/Pricing.java",
    f"cloned_body:java:{_PKG}.core.Billing.total(int[])",
    f"cloned_body:java:{_PKG}.core.Pricing.total(int[])",
    "vocabulary_drift:java:vocabulary:eb364e0ef1f8",
    f"pass_through:java:{_PKG}.core.Facade.compute(int,int)",
}

#: Shapes the fixture holds on purpose that no signal may report.
_NEVER = (
    "Shape:",  # two implementors
    "Plugin",  # one implementor, in another module: an extension point
    "EnglishGreeter.greet",  # overrides, called through the interface only
    "Greeter.greet",  # a contract member is the contract, not a helper
    "FileStore.save",  # overrides, called through the abstract type only
    "Launcher.main",  # the entry point; its many calls are static ones on a type name
    "Util.frameworkHook",  # annotated: a framework reaches it
    "Util.calledTwice",  # two callers
    "Util.calledOnce",  # one caller, in another type
    "Util.neverCalled",  # nothing calls it: an API or dead code, which the graph cannot tell apart
    "Util.hiddenHelper",  # referenced
    "Operator.applyAsInt",  # forwards, but implements a contract (@Override)
    "Calculator.add",  # the forwarders' target, called twice
    "Customer.",  # called from Invoice, outside its own type
    "Invoice.replay",  # calls a ledger's logic, not its getters: no envy
    "Invoice.moves",  # one read, then calls on what each call returned: a chain, not envy
    "Tower.Level6",  # five ancestors: at the bar, not past it
    "Clock",  # one production implementor and a test double: a seam
    "pass_through:java:com.example.core.Leaker.leak()",  # forwards no parameter: it names a value
)


@pytest.fixture
def health_root(tmp_path: Path, graph_cache: Path) -> Path:
    """A private copy of the health fixture, so a test may edit its graph."""
    graph_cli._refreshed.clear()
    return Path(shutil.copytree(Path(__file__).parent / "fixtures" / "health", tmp_path / "health"))


def _report(root: Path, **arguments: Any) -> dict[str, Any]:
    """Refresh the graph (a test may have edited the tree) and answer in process; the daemon's argument names."""
    graph_cli._refreshed.clear()
    return in_process(str(root), HealthRequest.from_args({"limit": 500, **arguments}))


def _keys(payload: dict[str, Any]) -> set[str]:
    """Every finding key of a report."""
    return {finding["key"] for finding in payload["health"]["findings"]}


def test_the_fixture_reports_exactly_its_suspects(health_root: Path) -> None:
    """Every planted shape is reported by key, and every exclusion holds."""
    payload = _report(health_root)
    keys = _keys(payload)

    # 1. The exact set: nothing missing, nothing extra.
    assert keys == _EXPECTED, f"step 1: missing {sorted(_EXPECTED - keys)}, extra {sorted(keys - _EXPECTED)}"

    # 2. The planted non-findings are named nowhere.
    for marker in _NEVER:
        assert not [key for key in keys if marker in key], f"step 2: {marker} must not be reported"

    # 3. A graph finding names its weakest supporting edge; a syntactic one reads exact.
    by_key = {finding["key"]: finding for finding in payload["health"]["findings"]}
    crossing = by_key[f"forbidden_crossing:java:{_CORE}/Leaker.java->app"]
    assert crossing["resolution"] == Resolution.EXACT.value and crossing["value"] >= 1, "step 3: exact edges"
    assert "core must not depend on app" in crossing["reason"], "step 3: the rule rides along"
    lone = by_key[f"lone_public_callable:java:{_PKG}.core.Ledger.normalize(int)"]
    assert lone["reason"].endswith("(2 caller(s), all in Ledger)"), f"step 3: a nested type is its own, {lone}"
    envy = by_key[f"feature_envy:java:{_PKG}.core.Invoice.label()"]
    assert envy["value"] == 3 and "(3 getter calls into Customer, 0 calls into its own type)" in envy["reason"], envy
    cycle = by_key[f"package_cycle:java:core:{_CORE}/north"]
    assert cycle["name"] == f"{_CORE}/north" and cycle["value"] == 2, f"step 3: a directory finding, {cycle}"
    lightest = f"its lightest link into it: 3 edge(s) from North.java into {_CORE}/south)"
    assert cycle["reason"].endswith(f"(in a cycle with {_CORE}/south; {lightest}"), cycle["reason"]
    deep = by_key[f"deep_inheritance:java:{_PKG}.core.Tower.Level7"]
    assert deep["value"] == 6 and deep["reason"].endswith("(up to Level1)"), deep

    # 4. Every surface carries the disclaimer next to the score.
    assert f"({SCORE_DISCLAIMER})" in payload["markdown"], "step 4: the score is never a target"
    assert payload["health"]["score_note"] == SCORE_DISCLAIMER, "step 4: in the JSON too"


def test_percentiles_and_score_are_deterministic_and_follow_the_formula(health_root: Path) -> None:
    """Two runs agree byte for byte, and the composite is the declared weighted mean of the axes."""
    first = _report(health_root)
    second = _report(health_root)
    assert first == second, "step 1: deterministic"

    health = first["health"]
    axes = {axis["axis"]: axis for axis in health["axes"]}
    expected_axes: dict[str, float] = {}
    for name, axis in axes.items():
        fractions = [item["past_mass"] / item["mass"] for item in axis["signals"] if item["mass"]]
        if fractions:
            expected_axes[name] = 100 * (1 - sum(fractions) / len(fractions))
            assert axis["score"] == round(expected_axes[name], 1), f"step 2: {name} follows the formula"
    weights = {axis.value: weight for axis, weight in AXIS_WEIGHTS.items()}
    composite = sum(weights[name] * score for name, score in expected_axes.items()) / sum(
        weights[name] for name in expected_axes
    )
    assert health["score"] == round(composite, 1), "step 3: the composite is the weighted mean"

    # 4. Hand-computed on the fixture. Machinery: single-impl 1/2 (Greeter of Greeter and Shape; Plugin's
    #    one implementor is in `app`, and Clock's test implements it too, so neither is measured),
    #    single-subclass 1/1, deep inheritance 1/35 classes, lone public 1/14 (the public methods outside a
    #    contract with a production caller:
    #    Util.calledOnce, calledTwice, tangled, Leaker.leak, Launcher.version, Calculator.add,
    #    AbstractStore.save, Customer.name, street, city, moved, South.down, Ledger.record and
    #    Ledger.normalize), unreferenced 1/3. Layering is one forbidden pair of three crossing pairs.
    #    Coupling: two cycle directories of five, one envious method of the 26 bodies no contract or
    #    framework places. Size: large_file measures zoo.py alone, every other file holding exactly one
    #    top-level type.
    machinery = 100 * (1 - (1 / 2 + 1 / 1 + 1 / 35 + 1 / 14 + 1 / 3) / 5)
    assert axes["machinery"]["score"] == round(machinery, 1), f"step 4: machinery {axes['machinery']}"
    assert axes["layering"]["score"] == round(100 * (1 - 1 / 3), 1), f"step 4: layering {axes['layering']}"
    coupling = 100 * (1 - (2 / 5 + 1 / 26) / 2)
    assert axes["coupling"]["score"] == round(coupling, 1), f"step 4: coupling {axes['coupling']}"
    sizes = {item["signal"]: item["measured"] for item in axes["size"]["signals"]}
    assert sizes["large_file"] == 1, f"step 4: a one-type file is large_type's, {sizes}"

    # 5. A percentile is the share of the language's population no more suspect, and ranks first.
    findings = health["findings"]
    assert all(0 < finding["percentile"] <= 100 for finding in findings), "step 5: within 0-100"
    ranks = [(-finding["percentile"]) for finding in findings]
    assert ranks == sorted(ranks), "step 5: ranked by percentile"


def test_every_vocabulary_member_has_its_facts_and_its_producer() -> None:
    """A signal without a producer, threshold or axis, or an axis without a weight, cannot ship."""
    assert set(PRODUCERS) == set(Signal), "every signal has a producer"
    assert set(AXIS_WEIGHTS) == set(Axis), "every axis has a weight"
    for signal in Signal:
        facts = signal.facts
        assert isinstance(facts.threshold, int) and facts.source and facts.advice, signal
        assert isinstance(facts.comparison, Comparison) and isinstance(facts.axis, Axis), signal
        if facts.tier is Tier.GRAPH:
            assert facts.floor is not None and facts.floor.at_least(Resolution.UNIQUE_NAME), signal
        else:
            assert facts.floor is None, f"{signal} rests on no edge"
    for dimension in StyleDimension:
        assert callable(dimension.facts.compute) and dimension.facts.description, dimension
    shapes = [dimension.facts.shape for dimension in StyleDimension if dimension.facts.shape is not None]
    assert sorted(shapes) == sorted(NameShape), "every name shape has exactly one dimension"


def test_name_shapes_and_the_chi_square_cut() -> None:
    """A name's shape ignores edge underscores and calls one lower-case word nothing; the cut matches chi-square."""
    shapes = {name: NameShape.of(name) for name in ("_load_file", "loadFile", "LoadFile", "load", "LOAD_ALL", "__x__")}
    assert shapes == {
        "_load_file": NameShape.SNAKE,
        "loadFile": NameShape.CAMEL,
        "LoadFile": NameShape.PASCAL,
        "load": None,
        "LOAD_ALL": None,
        "__x__": None,
    }, shapes
    # Published chi-square 0.99 quantiles: 3 degrees 11.345, 10 degrees 23.209, 14 degrees 29.141.
    for degrees, published in ((3, 11.345), (10, 23.209), (14, 29.141)):
        assert abs(chi_square_quantile(degrees, 0.99) - published) / published < 0.01, degrees


@pytest.fixture
def style_root(tmp_path: Path, graph_cache: Path) -> Path:
    """A private copy of the consistency fixtures, outside any other graph."""
    graph_cli._refreshed.clear()
    return Path(shutil.copytree(Path(__file__).parent / "fixtures" / "health_style", tmp_path / "health_style"))


_STYLE = ["style_outlier_repo", "style_outlier_directory"]


def _style(root: Path) -> StyleMeasurement:
    """Measure a fixture root's style straight from its graph."""
    _report(root, signals=_STYLE)
    provider = open_provider(str(root))
    try:
        return measure_style(collect(provider, HomeConfig.load(root)))
    finally:
        provider.close()


def test_the_one_foreign_file_is_the_one_style_outlier(style_root: Path) -> None:
    """Seven files in one style and one foreign file: only the foreign one is past the cut, and the reason says why."""
    payload = _report(style_root / "repo", signals=_STYLE)
    findings = payload["health"]["findings"]

    # 1. Exactly one finding, the foreign file against its language.
    assert _keys(payload) == {"style_outlier_repo:python:pkg/foreign.py"}, f"step 1: got {sorted(_keys(payload))}"

    # 2. The reason names what reads different: its long callables and its missing comments.
    reason = findings[0]["reason"]
    assert "against 8 python files" in reason, f"step 2: the norm is named, got {reason!r}"
    assert "longer callables" in reason and "less commented (0% vs" in reason, f"step 2: got {reason!r}"

    # 3. The file with two callables is not measured at all: eight of nine files are.
    axis = next(axis for axis in payload["health"]["axes"] if axis["axis"] == Axis.CONSISTENCY.value)
    measured = {item["signal"]: item["measured"] for item in axis["signals"]}
    assert measured["style_outlier_repo"] == 8, f"step 3: {measured}"
    style = _style(style_root / "repo")
    assert "pkg/small.py" not in style.styles and "pkg/foreign.py" in style.styles, "step 3: small.py unmeasured"

    # 4. One directory holds the whole language, so the directory norm adds nothing and measures no file.
    assert measured["style_outlier_directory"] == 0 and not style.directory, f"step 4: {measured}"


def test_style_vectors_are_deterministic_and_constant_dimensions_drop(style_root: Path) -> None:
    """Two measurements agree exactly, and a dimension no file varies on is left out of the norm."""
    first, second = _style(style_root / "repo"), _style(style_root / "repo")
    assert first == second, "step 1: same vectors, norms and distances"

    # 2. Every file declares no type, every name is public and every callable takes two parameters.
    norm = first.norms["python"]
    for constant in (StyleDimension.TOP_LEVEL_TYPES, StyleDimension.PUBLIC_SHARE, StyleDimension.PARAMETERS_MEAN):
        assert constant not in norm.kept, f"step 2: {constant.value} is constant and dropped"
    assert len(first.standardised["pkg/foreign.py"]) == len(norm.kept), "step 2: vectors span the kept dimensions"

    # 3. The foreign file differs on naming too, past the cap on both shapes.
    scores = dict(zip(norm.kept, first.standardised["pkg/foreign.py"], strict=True))
    assert scores[StyleDimension.CAMEL_CASE_NAMES] > 3 and scores[StyleDimension.SNAKE_CASE_NAMES] < -3, scores


def test_a_file_foreign_to_its_directory_is_flagged_against_its_directory_only(style_root: Path) -> None:
    """A house-style file among legacy files reads ordinary for the repo and different for its directory."""
    payload = _report(style_root / "directory", signals=_STYLE)
    keys = _keys(payload)
    newcomer = "legacy/newcomer.py"

    # 1. The newcomer is flagged against its directory and not against its language.
    assert f"style_outlier_directory:python:{newcomer}" in keys, f"step 1: got {sorted(keys)}"
    assert f"style_outlier_repo:python:{newcomer}" not in keys, "step 1: ordinary for the repo"
    finding = next(item for item in payload["health"]["findings"] if item["file_path"] == newcomer)
    assert "against legacy/ (5 python files)" in finding["reason"], finding["reason"]

    # 2. Its legacy neighbours agree with each other: none is flagged against the directory.
    assert [key for key in keys if key.startswith("style_outlier_directory:")] == [
        f"style_outlier_directory:python:{newcomer}"
    ], f"step 2: got {sorted(keys)}"

    # 3. The house files are flagged by neither signal.
    assert not [key for key in keys if ":house/" in key], f"step 3: got {sorted(keys)}"


def test_since_and_paths_judge_against_the_whole_view(style_root: Path) -> None:
    """`--since` and `--paths` narrow what is listed, never the files a style is judged against."""
    root = style_root / "repo"
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    foreign = "style_outlier_repo:python:pkg/foreign.py"

    # 1. A change in a house file lists nothing; the norm and the foreign file's finding still stand.
    orders = root / "pkg" / "orders.py"
    orders.write_text(orders.read_text(encoding="utf-8") + "\n# One more remark.\n", encoding="utf-8")
    delta = _report(root, signals=_STYLE, since="HEAD")["health"]
    assert delta["findings"] == [] and delta["since"]["scope_total"] == 1, f"step 1: {delta['since']}"

    # 2. Touching the foreign file lists it.
    path = root / "pkg" / "foreign.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert _keys(_report(root, signals=_STYLE, since="HEAD")) == {foreign}, "step 2"

    # 3. A run over the foreign file alone still compares it with the other seven.
    alone = _report(root, signals=_STYLE, paths=["pkg/foreign.py"])
    assert _keys(alone) == {foreign} and "against 8 python files" in alone["health"]["findings"][0]["reason"], "step 3"


def test_an_ambiguous_edge_never_supports_a_graph_finding(health_root: Path) -> None:
    """Demoting the supporting edges to AMBIGUOUS withdraws the findings they supported."""
    before = _keys(_report(health_root))
    single = f"single_implementation_interface:java:{_PKG}.core.Greeter"
    crossing = f"forbidden_crossing:java:{_CORE}/Leaker.java->app"
    assert {single, crossing} <= before, "step 1: both findings stand on resolved edges"

    # 2. Demote, in the store itself, the only IMPLEMENTS edge and every edge Leaker.java writes.
    database = graph_db_path(str(health_root))
    assert database is not None
    with sqlite3.connect(database) as connection:
        updated = connection.execute(
            "UPDATE edge_rows SET resolution = ? WHERE kind = 'implements' OR file = "
            "(SELECT key FROM refs WHERE text = ?)",
            (Resolution.AMBIGUOUS.value, f"{_CORE}/Leaker.java"),
        ).rowcount
    assert updated >= 3, f"step 2: the edges were demoted, got {updated}"

    # 3. Read without refreshing (a refresh would resolve them again): both findings are gone.
    provider = open_provider(str(health_root))
    try:
        after = _keys(health_payload(provider, HomeConfig.load(health_root), HealthRequest(limit=500)))
    finally:
        provider.close()
    assert single not in after, "step 3: an ambiguous implementor cannot make an interface single"
    assert crossing not in after, "step 3: an ambiguous edge cannot cross a forbidden boundary"


def test_a_sub_directory_is_reported_in_its_own_paths(health_root: Path) -> None:
    """A sub-directory answers from the ancestor graph, with paths relative to it and nothing outside it."""
    _report(health_root)
    core = health_root / _CORE
    payload = _report(core)
    findings = payload["health"]["findings"]

    # 1. Only in-folder units, spelled relative to the folder.
    paths = {finding["file_path"] for finding in findings}
    assert paths and all(not path.startswith("src/") for path in paths), f"step 1: rebased paths, got {sorted(paths)}"
    assert {"north/North.java", "Util.java"} <= paths, f"step 1: a sub-package keeps its own segment, {sorted(paths)}"
    assert not graph_present(str(core)), "step 1: no graph of its own"

    # 2. Same identities as the root run, minus everything outside the folder; a sub-directory is a
    #    bounded scope, so the duplication tier is measured too, against the whole tree.
    expected = {
        key.replace(f"{_CORE}/", "")
        for key in _EXPECTED | _COPYING
        if "app.Launcher" not in key and "zoo.py" not in key
    }
    assert _keys(payload) == expected, f"step 2: got {sorted(_keys(payload) ^ expected)}"

    # 3. The ancestor's home.toml still judges the crossing.
    assert payload["health"]["config"].endswith(".zemble/home.toml"), "step 3: the ancestor config"


def test_paths_and_signals_narrow_the_report(health_root: Path) -> None:
    """`paths` keeps findings under them; `signals` measures only those signals; a bad name is refused."""
    payload = _report(health_root, paths=["src/main/python"], signals=["many_types_per_file", "long_callable"])
    assert _keys(payload) == {"many_types_per_file:python:src/main/python/zoo.py"}, "step 1: scoped"
    assert payload["health"]["signals"] == ["long_callable", "many_types_per_file"], "step 1: declaration order"
    axes = {axis["axis"]: axis["score"] for axis in payload["health"]["axes"]}
    assert axes["machinery"] is None and axes["layering"] is None, "step 2: unmeasured axes are not 100"
    with pytest.raises(ValueError, match="unknown signal"):
        _report(health_root, signals=["no_such_signal"])


def test_the_duplication_tier_reports_copies_vocabulary_and_pass_throughs(health_root: Path) -> None:
    """With copying on, the planted copies, vocabulary and forwarders are reported by key, and nothing else is."""
    payload = _report(health_root, copying=True)
    keys = _keys(payload)

    # 1. Exactly the graph findings plus the duplication ones.
    expected = _EXPECTED | _COPYING
    assert keys == expected, f"step 1: missing {sorted(expected - keys)}, extra {sorted(keys - expected)}"
    for marker in _NEVER:
        assert not [key for key in keys if marker in key], f"step 1: {marker} must not be reported"

    # 2. Each finding says where else it lives: the vocabulary carries its other declaration.
    by_key = {finding["key"]: finding for finding in payload["health"]["findings"]}
    drift = by_key["vocabulary_drift:java:vocabulary:eb364e0ef1f8"]
    assert drift["value"] == 2 and drift["elsewhere"] == [f"{_CORE}/Status.java:3-8"], f"step 2: {drift}"
    cloned = by_key[f"cloned_body:java:{_PKG}.core.Pricing.total(int[])"]
    assert cloned["value"] == 1 and f"{_CORE}/Billing.java" in cloned["reason"], f"step 2: {cloned}"
    mass = by_key[f"clone_mass:java:{_CORE}/Billing.java"]
    assert mass["value"] > Signal.CLONE_MASS.facts.threshold, f"step 2: {mass}"
    assert "forwards to Calculator.add" in by_key[f"pass_through:java:{_PKG}.core.Facade.compute(int,int)"]["reason"]

    # 3. The two new axes are scored, and a whole-tree run without the flag leaves them out with a note.
    axes = {axis["axis"]: axis["score"] for axis in payload["health"]["axes"]}
    assert axes["copying"] is not None and axes["vocabulary"] == 0.0, f"step 3: {axes}"
    plain = _report(health_root)
    assert not _keys(plain) & _COPYING and COPYING_OPT_IN in plain["health"]["notes"], "step 3: opt-in"
    assert {axis["axis"]: axis["score"] for axis in plain["health"]["axes"]}["copying"] is None, "step 3"

    # 4. A bounded scope turns the tier on by itself, and `copying=False` turns it off.
    scoped = _report(health_root, paths=[_CORE])
    assert _COPYING <= _keys(scoped), "step 4: --paths measures copying"
    assert not _keys(_report(health_root, paths=[_CORE], copying=False)) & _COPYING, "step 4: off"


def test_a_baseline_splits_findings_into_resolved_remaining_and_new(health_root: Path, tmp_path: Path) -> None:
    """A removed finding is resolved, an added one new, the rest remaining, even after moving lines."""
    saved = tmp_path / "health.baseline.json"
    first = _report(health_root, copying=True, save_baseline=str(saved))
    document = json.loads(saved.read_text(encoding="utf-8"))
    assert {entry["key"] for entry in document["findings"]} == _EXPECTED | _COPYING, "step 1: every key saved"
    assert any("finding key(s)" in note for note in first["health"]["notes"]), "step 1: the write is noted"

    # 2. Drop deadHelper, add another dead helper above everything else: every other method moves down.
    util = health_root / _CORE / "Util.java"
    text = util.read_text(encoding="utf-8")
    text = text.replace("    private static int deadHelper(int value) {\n        return value - 2;\n    }\n\n", "")
    text = text.replace(
        "public class Util {\n",
        "public class Util {\n\n\n\n    private static int forgotten(int value) {\n        return value - 3;\n    }\n",
    )
    util.write_text(text, encoding="utf-8")
    payload = _report(health_root, copying=True, baseline=str(saved))
    health = payload["health"]
    dead = f"unreferenced_symbol:java:{_PKG}.core.Util.deadHelper(int)"
    forgotten = f"unreferenced_symbol:java:{_PKG}.core.Util.forgotten(int)"
    assert [entry["key"] for entry in health["baseline"]["resolved"]] == [dead], f"step 2: {health['baseline']}"
    assert health["baseline"]["new"] == [forgotten], "step 2: the added finding is new"
    counts = health["baseline"]["counts"]
    assert counts["remaining"] == len(_EXPECTED | _COPYING) - 1, f"step 2: {counts}"

    # 3. The moved method keeps its key: remaining, at its new line.
    tangled = next(finding for finding in health["findings"] if finding["key"].startswith("long_callable:"))
    assert tangled["baseline"] == "remaining" and tangled["start_line"] > 30, f"step 3: {tangled}"
    markdown = payload["markdown"]
    for title in (
        "== RESOLVED (in the baseline, gone now) ==",
        "== REMAINING (in the baseline, still here) ==",
        "== NEW ==",
    ):
        assert title in markdown, f"step 3: section {title}"

    # 4. A narrowed run never calls what it did not measure resolved.
    narrowed = _report(health_root, copying=True, baseline=str(saved), signals=["long_callable"])
    assert narrowed["health"]["baseline"]["resolved"] == [], "step 4: unmeasured signals are not resolved"
    with pytest.raises(ValueError, match="not a version 1 document"):
        saved.write_text("{}", encoding="utf-8")
        _report(health_root, baseline=str(saved))


def test_since_lists_the_changed_spans_and_keeps_the_whole_score(health_root: Path, tmp_path: Path) -> None:
    """Only findings touching a changed or untracked line are listed; the score is still the whole scope's."""
    git(health_root, "init", "-q")
    git(health_root, "add", "-A")
    git(health_root, "commit", "-q", "-m", "base")
    saved = tmp_path / "base.json"
    _report(health_root, save_baseline=str(saved))

    # 1. Change one method and add one untracked file.
    util = health_root / _CORE / "Util.java"
    util.write_text(
        util.read_text(encoding="utf-8").replace("steps += 30;\n", "steps += 30;\n        steps += 31;\n"),
        encoding="utf-8",
    )
    (health_root / _CORE / "Fresh.java").write_text(
        "package com.example.core;\n\npublic class Fresh {\n\n    public static int fresh(int value) {\n"
        "        return value + 7;\n    }\n\n    public static int twice(int value) {\n"
        "        return fresh(fresh(value));\n    }\n}\n",
        encoding="utf-8",
    )
    changes = changed_lines(str(health_root), "HEAD")
    assert changes.files == (f"{_CORE}/Fresh.java", f"{_CORE}/Util.java"), f"step 1: {changes.ranges}"

    # 2. Only the changed method's findings and the new file's are listed.
    payload = _report(health_root, since="HEAD")
    health = payload["health"]
    tangled = f"{_PKG}.core.Util.tangled(int,int,int,int,int,int)"
    expected = {
        f"{signal}:java:{tangled}"
        for signal in ("long_callable", "complex_callable", "deep_nesting", "many_parameters")
    } | {f"lone_public_callable:java:{_PKG}.core.Fresh.fresh(int)"}
    assert _keys(payload) == expected, f"step 2: got {sorted(_keys(payload) ^ expected)}"
    assert health["since"] == {"ref": "HEAD", "files": 2, "scope_total": len(_EXPECTED) + 1}, health["since"]

    # 3. The score is the whole scope's: the same as a whole run without the duplication tier, which a
    #    --since run measures over the changed files only and so leaves out of the score.
    whole = _report(health_root, copying=False)["health"]
    assert health["score"] == whole["score"], "step 3: the composite is not recomputed over the delta"
    assert set(health["unscored"]) == {s.value for s in Signal if s.facts.tier is Tier.DUPLICATION}, "step 3"
    assert "Since HEAD: 5 finding(s)" in payload["markdown"], "step 3: the delta count is printed"

    # 4. Against the baseline saved at the commit, the delta reads remaining and new.
    marked = _report(health_root, since="HEAD", baseline=str(saved))["health"]
    statuses = {finding["key"]: finding["baseline"] for finding in marked["findings"]}
    assert statuses.pop(f"lone_public_callable:java:{_PKG}.core.Fresh.fresh(int)") == "new", "step 4"
    assert set(statuses.values()) == {"remaining"}, f"step 4: {statuses}"

    # 5. A ref that names no commit, and --since with --save-baseline, are refused.
    with pytest.raises(ValueError, match="names no commit"):
        _report(health_root, since="no-such-ref")
    with pytest.raises(ValueError, match="takes no --save-baseline"):
        _report(health_root, since="HEAD", save_baseline=str(saved))


def test_changed_lines_mark_deletions_and_merge_ranges(tmp_path: Path) -> None:
    """A pure deletion marks the line it follows; adjacent changed lines join into one range."""
    git(tmp_path, "init", "-q")
    source = tmp_path / "a.txt"
    source.write_text("".join(f"line {number}\n" for number in range(1, 11)), encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    lines = source.read_text(encoding="utf-8").split("\n")
    lines[1], lines[2] = "two", "three"  # lines 2-3 changed
    del lines[6]  # line 7 removed: follows line 6
    source.write_text("\n".join(lines), encoding="utf-8")
    nested = tmp_path / "inner"  # a nested checkout git would add as a gitlink, never as files
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "copy.txt").write_text("a nested checkout\n", encoding="utf-8")
    changes = changed_lines(str(tmp_path), "HEAD")
    assert changes.ranges == {"a.txt": ((2, 3), (6, 6))}, changes.ranges
    assert changes.touches("a.txt", 3, 5) and changes.touches("a.txt", 6, 9) and not changes.touches("a.txt", 4, 5)
    with pytest.raises(ValueError, match="takes a git ref"):
        changed_lines(str(tmp_path), "--output=x")


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    """Run the real entry point and return its exit code and stdout."""
    monkeypatch.setattr(sys, "argv", ["zemble", *argv])
    with pytest.raises(SystemExit) as exit_info:
        _cli_main()
    return int(exit_info.value.code or 0), capsys.readouterr().out


def test_cli_answers_in_both_shapes_and_builds_a_missing_graph(
    health_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`zemble health` exits 0 with findings, prints the score line, and `--json` parses."""
    code, out = _run_cli(["health", str(health_root), "--no-daemon"], capsys, monkeypatch)
    assert code == 0, "step 1: findings are a report, never a failure"
    assert "Score: " in out and SCORE_DISCLAIMER in out, f"step 1: the score line, got {out[:200]!r}"

    code, out = _run_cli(["health", str(health_root), "--json", "--limit", "3", "--no-daemon"], capsys, monkeypatch)
    payload = json.loads(out)
    assert code == 0 and len(payload["findings"]) == 3 and payload["truncated"], "step 2: capped JSON"
    assert payload["total"] == len(_EXPECTED), "step 2: the total counts every finding"

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "tool.py").write_text("def run(value):\n    return value + 1\n", encoding="utf-8")
    assert not graph_present(str(plain)), "step 3: no graph yet"
    code, out = _run_cli(["health", str(plain), "--no-daemon"], capsys, monkeypatch)
    assert code == 0 and graph_present(str(plain)), "step 3: the graph was built and the report answered"

    code, _ = _run_cli(["health", str(health_root), "--signals", "bogus", "--no-daemon"], capsys, monkeypatch)
    assert code == 1, "step 4: an unknown signal is refused"
    code, _ = _run_cli(["health", str(health_root), "--since", "main", "--no-daemon"], capsys, monkeypatch)
    assert code == 1, "step 4: --since outside a git checkout is refused"

    # 5. The new flags reach the payload: copying, then a baseline saved and read back.
    saved = tmp_path / "cli.baseline.json"
    argv = ["health", str(health_root), "--copying", "--json", "--no-daemon", "--save-baseline", str(saved)]
    code, out = _run_cli(argv, capsys, monkeypatch)
    assert code == 0 and json.loads(out)["counts"]["cloned_body"] == 2 and saved.is_file(), "step 5: copying, saved"
    argv = ["health", str(health_root), "--copying", "--no-daemon", "--baseline", str(saved)]
    code, out = _run_cli(argv, capsys, monkeypatch)
    total = len(_EXPECTED | _COPYING)
    assert code == 0 and f"Baseline: {total} finding(s); 0 resolved, {total} remaining, 0 new" in out, out


#: A Python function of cognitive complexity 16: for +1, if +2, for +3, if +4, if +5, `and` +1.
_TANGLED = """def tangled(a, b, c, d):
    total = 0
    for i in range(a):
        if i > b:
            for j in range(c):
                if j > d:
                    if i and j:
                        total += 1
    return total
"""
_HISTORY = ["hotspot", "boundary_co_change"]


def _edit(root: Path, *relatives: str, message: str, date: str | None = None) -> None:
    """Append one line to each file and commit them together."""
    for relative in relatives:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"# {message}\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message, date=date)


@pytest.fixture
def history_root(tmp_path: Path, graph_cache: Path) -> Path:
    """A scripted repository of every history shape: churn with and without complexity, co-change in and across modules.

    A complex and a simple file are committed 12 times, a test file too; a cross-module and a same-module
    pair change together in 6 of 8 commits; 20 files are committed once; a complex file has 12 commits
    500 days old.
    """
    graph_cli._refreshed.clear()
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    for days in range(512, 500, -1):
        stamp = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        _edit(root, "app/old.py", message=f"old {days}", date=stamp)
    (root / "app/old.py").write_text(_TANGLED.replace("tangled", "aged"), encoding="utf-8")
    (root / "app/hot.py").write_text(_TANGLED, encoding="utf-8")
    (root / "app/simple.py").write_text("def plain(value):\n    return value + 1\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests/test_hot.py").write_text(_TANGLED.replace("tangled", "test_tangled"), encoding="utf-8")
    for index in range(20):
        (root / f"app/cold_{index}.py").write_text(f"def cold_{index}():\n    return {index}\n", encoding="utf-8")
    for module, first, second in (("app", "app/a.py", "lib/b.py"), ("lib", "lib/c.py", "lib/d.py")):
        for path in (first, second):
            (root / path).parent.mkdir(exist_ok=True)
            (root / path).write_text(f"def {Path(path).stem}():\n    return '{module}'\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    for index in range(11):
        _edit(root, "app/hot.py", "app/simple.py", "tests/test_hot.py", message=f"hot {index}")
    for index in range(7):
        # Together in the base commit and five of these seven: 6 shared of 8 either, each pair on its own.
        _edit(root, "app/a.py", *(("lib/b.py",) if index < 5 else ()), message=f"cross {index}")
        _edit(root, "lib/c.py", *(("lib/d.py",) if index < 5 else ()), message=f"inside {index}")
    return root


def test_history_reports_a_churning_complex_file_and_a_cross_module_pair(history_root: Path) -> None:
    """A hotspot needs both churn and complexity; a co-change pair is reported only across modules."""
    payload = _report(history_root, signals=_HISTORY)
    health = payload["health"]
    by_key = {finding["key"]: finding for finding in health["findings"]}

    # 1. The complex file committed 12 times is a hotspot; the simple one beside it is not, nor is the test file,
    #    and the old complex file with one commit in the window is too short a history to rank.
    assert set(by_key) == {"hotspot:python:app/hot.py", "boundary_co_change:python:app/a.py<->lib/b.py"}, by_key
    hot = by_key["hotspot:python:app/hot.py"]
    assert hot["value"] > hot["threshold"], "step 1: past the top decile"
    assert "12 commits in 365 days; complexity 16 in hot.tangled" in hot["reason"], hot["reason"]

    # 2. The pair in two modules changed together in 6 of 8 commits; the pair inside `lib` is not measured.
    pair = by_key["boundary_co_change:python:app/a.py<->lib/b.py"]
    assert pair["value"] == 750 and "changed together in 6 of 8 commits with lib/b.py" in pair["reason"], pair
    assert pair["elsewhere"] == ["lib/b.py:1-7"], pair["elsewhere"]
    history = next(axis for axis in health["axes"] if axis["axis"] == "history")
    measured = {item["signal"]: (item["past"], item["measured"]) for item in history["signals"]}
    # Every one of the 27 changed production files is measured; only the hot one ranks.
    assert measured == {"hotspot": (1, 27), "boundary_co_change": (1, 1)}, f"step 2: {measured}"

    # 3. A path restriction keeps the pair when either file is inside it, anchored there.
    narrowed = _report(history_root, signals=_HISTORY, paths=["lib"])
    assert _keys(narrowed) == {"boundary_co_change:python:app/a.py<->lib/b.py"}, _keys(narrowed)
    assert narrowed["health"]["findings"][0]["file_path"] == "lib/b.py", "step 3: anchored inside the scope"


def test_the_history_window_excludes_older_commits(history_root: Path) -> None:
    """The old complex file has one commit in a year and thirteen in a thousand days."""
    year = {finding["key"] for finding in _report(history_root, signals=["hotspot"])["health"]["findings"]}
    assert "hotspot:python:app/old.py" not in year, "step 1: its 12 old commits are outside the default window"
    longer = _report(history_root, signals=["hotspot"], history_window=1000)["health"]
    old = next(finding for finding in longer["findings"] if finding["key"] == "hotspot:python:app/old.py")
    assert "13 commits in 1000 days" in old["reason"], old["reason"]
    with pytest.raises(ValueError, match="positive number of days"):
        _report(history_root, history_window=0)


def test_history_off_or_outside_git_is_not_measured(
    history_root: Path, health_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-history` and a root in no checkout both leave the axis out, with a note, and still exit 0."""
    off = _report(history_root, history=False)["health"]
    assert not set(off["signals"]) & set(_HISTORY) and "history not measured: turned off" in off["notes"]

    outside = _report(health_root)["health"]
    history = next(axis for axis in outside["axes"] if axis["axis"] == "history")
    assert history["score"] is None and not set(outside["signals"]) & set(_HISTORY), "step 2: not measured"
    assert any(note.startswith("history not measured: not in a git checkout") for note in outside["notes"])
    code, out = _run_cli(["health", str(health_root), "--no-daemon"], capsys, monkeypatch)
    assert code == 0 and "- history: not measured (weight 1)" in out, "step 2: the CLI says so and exits 0"
    argv = ["health", str(history_root), "--no-daemon", "--history-window", "1000", "--signals", "hotspot"]
    code, out = _run_cli(argv, capsys, monkeypatch)
    assert code == 0 and "13 commits in 1000 days" in out, "step 3: the window reaches the request"


def test_a_workspace_of_checkouts_reads_each_one(history_root: Path, tmp_path: Path) -> None:
    """A root in no checkout reads the checkouts directly under it, paths prefixed by their directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.move(str(history_root), str(workspace / "repo"))
    assert history_checkouts(str(workspace)) == ((str(workspace / "repo"), "repo/"),)
    keys = _keys(_report(workspace, signals=["hotspot"]))
    assert keys == {"hotspot:python:repo/app/hot.py"}, keys


def test_a_checkout_holding_checkouts_reads_itself_and_each_child(history_root: Path, tmp_path: Path) -> None:
    """A root checkout of changelogs whose children are checkouts reads all of them, each commit once."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.move(str(history_root), str(workspace / "repo"))
    # A checkout two levels down and a plain directory beside the child checkout.
    deep = workspace / "group" / "deep"
    deep.mkdir(parents=True)
    git(deep, "init", "-q")
    _edit(deep, "lib/deep.py", message="deep")
    git(workspace, "init", "-q")
    for index in range(3):
        # `add -A` records `repo` as a gitlink: its files never enter the root's own log.
        _edit(workspace, "changelog/notes.py", message=f"notes {index}")

    # 1. The root itself, then each direct child checkout; the deeper one is not looked for.
    assert history_checkouts(str(workspace)) == ((str(workspace), ""), (str(workspace / "repo"), "repo/")), "step 1"

    # 2. Each file's commits come from its own checkout, counted once.
    history = measure_history(str(workspace), ["repo/app/hot.py", "changelog/notes.py", "group/deep/lib/deep.py"], 365)
    assert history.churn("repo/app/hot.py") == 12 and history.churn("changelog/notes.py") == 3, history.commits
    assert history.churn("group/deep/lib/deep.py") == 0, "step 2: a checkout under a plain directory is not read"
    assert history.checkout_of("repo/app/hot.py") == "repo/" and history.checkout_of("changelog/notes.py") == ""

    # 3. The report measures the history axis there.
    keys = _keys(_report(workspace, signals=["hotspot"]))
    assert keys == {"hotspot:python:repo/app/hot.py"}, keys


def test_components_and_ancestor_chains_survive_size_and_cycles() -> None:
    """Both walks are iterative: a 5000-node chain passes the recursion limit, and a cycle in bad data ends."""
    from zemble.health.structure import ancestor_depths, strongly_connected

    # 1. A ring of 5000 is one component; a tail hanging off it is one component per node.
    ring = {index: [(index + 1) % 5000] for index in range(5000)}
    ring[5000] = [0]
    components = strongly_connected(range(5001), ring)
    assert sorted(len(component) for component in components) == [1, 5000], "step 1"
    assert strongly_connected(["a", "b"], {"a": ["b", "elsewhere"], "b": []}) == [["b"], ["a"]], "step 1: unknowns"

    # 2. A 5000-deep chain counts every ancestor; a two-node cycle stops instead of looping.
    chain = {index: [index - 1] for index in range(1, 5000)}
    depths = ancestor_depths(chain)
    assert depths[4999][0] == 4999 and depths[4999][1][-1] == 0, "step 2: the longest chain, topmost last"
    cyclic = ancestor_depths({"a": ["b"], "b": ["a"]})
    assert set(cyclic) == {"a", "b"} and all(depth <= 1 for depth, _ in cyclic.values()), cyclic


def test_read_estimate_sizes_a_graph_on_the_high_side(health_root: Path) -> None:
    """The daemon sizes a health read from row counts: above every measured read, growing with the graph."""
    from zemble.graph.model import RowCounts
    from zemble.health.collect import read_estimate_mb

    _report(health_root)
    provider = open_provider(str(health_root))
    try:
        counts = provider.row_counts()
    finally:
        provider.close()
    assert counts.symbols > 0 and counts.edges > 0 and counts.metrics > 0 and counts.files > 0, counts
    small = read_estimate_mb(counts)
    javaweb = read_estimate_mb(RowCounts(symbols=122269, edges=1258533, metrics=67762, files=7909))
    zenit = read_estimate_mb(RowCounts(symbols=246682, edges=2694116, metrics=0, files=14593))
    # Measured VmData growth of a fresh process: javaweb 209 MiB, the zenit workspace 642 MiB.
    assert small < javaweb < zenit and javaweb > 209 and zenit > 642, (small, javaweb, zenit)


def test_an_outgrown_daemon_answer_is_read_in_process(
    health_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daemon that cannot hold the read answers `outgrown`; the CLI and the MCP tool then read the stored graph."""
    from zemble.daemon import client
    from zemble.daemon.protocol import CommandOutgrown
    from zemble.index_cache import IndexCache
    from zemble.mcp import create_server

    _report(health_root)  # the stored graph the fallback reads
    outgrown = CommandOutgrown("health needs ~300 MiB of private memory and this daemon has 10 MiB free.")
    asked: list[str] = []

    def daemon(cmd: str, args: Any = None, **_: Any) -> Any:
        asked.append(cmd)
        raise outgrown

    monkeypatch.setattr(client, "call", daemon)
    monkeypatch.setattr(client, "_disabled_reason", None)
    monkeypatch.delenv("ZEMBLE_DAEMON", raising=False)
    monkeypatch.setattr("zemble.health.cli.ensure_graph", lambda *a, **k: None)

    # 1. The CLI answers from the stored graph, exits 0, and says on stderr why it did not come from the daemon.
    monkeypatch.setattr(sys, "argv", ["zemble", "health", str(health_root), "--limit", "3"])
    with pytest.raises(SystemExit) as exit_info:
        _cli_main()
    captured = capsys.readouterr()
    assert exit_info.value.code in (0, None) and asked == ["health"], (exit_info.value.code, asked)
    assert SCORE_DISCLAIMER in captured.out, captured.out[:300]
    assert str(outgrown) in captured.err and "Answering in this process from the stored graph" in captured.err

    # 2. The MCP tool does the same and carries the reason as a note above the report.
    async def outgrown_call(cmd: str, args: dict[str, Any]) -> Any:
        asked.append(f"mcp:{cmd}")
        raise outgrown

    monkeypatch.setattr("zemble.mcp._daemon_call", outgrown_call)
    instance = create_server(IndexCache())
    result = asyncio.run(instance.call_tool("health", {"repo": str(health_root), "limit": 3}))
    text = result[0].text
    assert text.startswith(f"Note: {outgrown} Answered in the MCP server's process"), text[:300]
    assert SCORE_DISCLAIMER in text and asked[-1] == "mcp:health", (text[:300], asked)
