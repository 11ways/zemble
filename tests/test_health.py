"""Behaviour journeys of `zemble health` over a fixture with every shape it reports and every one it must not."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
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
from zemble.health.git import changed_lines
from zemble.health.model import Axis, Comparison, Signal, Tier
from zemble.health.score import AXIS_WEIGHTS, SCORE_DISCLAIMER
from zemble.health.signals import PRODUCERS
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
    f"unreferenced_symbol:java:{_PKG}.core.Util.deadHelper(int)",
    f"lone_public_callable:java:{_PKG}.core.Util.neverCalled(int)",
    f"lone_public_callable:java:{_PKG}.core.Util.calledOnce(int)",
    f"lone_public_callable:java:{_PKG}.core.Util.tangled(int,int,int,int,int,int)",
    f"lone_public_callable:java:{_PKG}.core.Leaker.leak()",
    f"lone_public_callable:java:{_PKG}.app.Launcher.version()",
    f"lone_public_callable:java:{_PKG}.core.Greeter.greet(String)",
    f"lone_public_callable:java:{_PKG}.core.Shape.area()",
    f"lone_public_callable:java:{_PKG}.core.AbstractStore.save()",
    f"lone_public_callable:java:{_PKG}.core.Facade.compute(int,int)",
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
    "EnglishGreeter.greet",  # overrides, called through the interface only
    "FileStore.save",  # overrides, called through the abstract type only
    "Launcher.main",  # the entry point
    "Util.frameworkHook",  # annotated: a framework reaches it
    "Util.calledTwice",  # two callers
    "Util.hiddenHelper",  # referenced
    "Operator.applyAsInt",  # forwards, but implements a contract (@Override)
    "Calculator.add",  # the forwarders' target, called twice
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
    lone = by_key[f"lone_public_callable:java:{_PKG}.core.Util.neverCalled(int)"]
    assert lone["reason"].endswith("(never called)"), f"step 3: the absence says so, got {lone['reason']!r}"

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

    # 4. Hand-computed on the fixture: machinery is single-impl 1/2, single-subclass 1/1, lone 9/11,
    #    unreferenced 1/3; layering is one forbidden pair of two crossing pairs.
    machinery = 100 * (1 - (1 / 2 + 1 / 1 + 9 / 11 + 1 / 3) / 4)
    assert axes["machinery"]["score"] == round(machinery, 1), f"step 4: machinery {axes['machinery']}"
    assert axes["layering"]["score"] == 50.0, f"step 4: layering {axes['layering']}"

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
    assert paths and all("/" not in path for path in paths), f"step 1: rebased paths, got {sorted(paths)}"
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
        "        return value + 7;\n    }\n}\n",
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
        for signal in ("long_callable", "complex_callable", "deep_nesting", "many_parameters", "lone_public_callable")
    } | {f"lone_public_callable:java:{_PKG}.core.Fresh.fresh(int)"}
    assert _keys(payload) == expected, f"step 2: got {sorted(_keys(payload) ^ expected)}"
    assert health["since"] == {"ref": "HEAD", "files": 2, "scope_total": len(_EXPECTED) + 1}, health["since"]

    # 3. The score is the whole scope's: the same as a whole run without the duplication tier, which a
    #    --since run measures over the changed files only and so leaves out of the score.
    whole = _report(health_root, copying=False)["health"]
    assert health["score"] == whole["score"], "step 3: the composite is not recomputed over the delta"
    assert set(health["unscored"]) == {s.value for s in Signal if s.facts.tier is Tier.DUPLICATION}, "step 3"
    assert "Since HEAD: 6 finding(s)" in payload["markdown"], "step 3: the delta count is printed"

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
