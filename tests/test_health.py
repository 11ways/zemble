"""Behaviour journeys of `zemble health` over a fixture with every shape it reports and every one it must not."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from zemble.cli import _cli_main
from zemble.graph import cli as graph_cli
from zemble.graph.model import Resolution
from zemble.graph.provider import open_provider
from zemble.graph.store import graph_db_path, graph_present
from zemble.health.answers import health_payload
from zemble.health.cli import in_process
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
)


@pytest.fixture
def health_root(tmp_path: Path, graph_cache: Path) -> Path:
    """A private copy of the health fixture, so a test may edit its graph."""
    graph_cli._refreshed.clear()
    return Path(shutil.copytree(Path(__file__).parent / "fixtures" / "health", tmp_path / "health"))


def _report(root: Path, **kwargs: Any) -> dict[str, Any]:
    """Build (or refresh) the graph and answer in process."""
    return in_process(str(root), kwargs.pop("signals", None), kwargs.pop("paths", []), kwargs.pop("limit", 500))


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

    # 4. Hand-computed on the fixture: machinery is single-impl 1/2, single-subclass 1/1, lone 8/9,
    #    unreferenced 1/3; layering is one forbidden pair of two crossing pairs.
    machinery = 100 * (1 - (1 / 2 + 1 / 1 + 8 / 9 + 1 / 3) / 4)
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
        after = _keys(health_payload(provider, HomeConfig.load(health_root), limit=500))
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

    # 2. Same identities as the root run, minus everything outside the folder.
    expected = {key.replace(f"{_CORE}/", "") for key in _EXPECTED if "app.Launcher" not in key and "zoo.py" not in key}
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
    assert code == 1, "step 4: an unknown signal is the one refusal"
