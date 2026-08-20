"""Behaviour journeys over the graph facts overlay."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from zemble.cli import _cli_main
from zemble.graph.facts import FactsFormatError, discover_facts_files, load_facts_file, matches_facts_glob
from zemble.graph.store import build_graph, connect

CONSUMER = "src/main/java/com/example/app/Consumer.java"
CIRCLE = "src/main/java/com/example/core/Circle.java"
MEASURE = "com.example.app.Consumer#measure(com.example.core.Circle)"
GENERATED = "build/generated-sources/com/example/gen/Tpl_Page.java"


def _workspace(source: Path, destination: Path) -> Path:
    """Copy the fixture workspace so a test can write facts into it."""
    shutil.copytree(source, destination)
    return destination


def _sha(workspace: Path, relative: str) -> str:
    """Hex sha256 of a workspace file's current content."""
    return hashlib.sha256((workspace / relative).read_bytes()).hexdigest()


def _write_facts(
    workspace: Path, lines: list[dict], name: str = "javac-facts.jsonl", where: str = ".zemble/facts"
) -> Path:
    """Write a hand-made facts file into the workspace."""
    path = workspace / where / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def _header(tool: str = "javac-facts", version: int = 1) -> dict:
    """A facts header pointing at the workspace two folders above the facts file."""
    return {
        "zemble_facts": version,
        "tool": tool,
        "tool_version": "0.1.0",
        "generated_at": "2026-08-20T09:00:00Z",
        "language": "java",
        "root": "../..",
    }


def _edges(workspace: Path, where: str) -> list[dict]:
    """Every edge stored for one source file."""
    connection = connect(str(workspace))
    try:
        rows = connection.execute("SELECT * FROM edges WHERE file_path = ?", (where,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def _bucket(payload: dict, name: str) -> dict:
    """One skipped-fact bucket out of a `--json` status payload."""
    return next(bucket for bucket in payload["skipped"] if bucket["bucket"] == name)


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Run the zemble CLI with the given arguments and return its exit code."""
    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    monkeypatch.setattr("sys.argv", ["zemble", *argv])
    try:
        _cli_main()
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    return 0


def test_overlay_journey(graph_fixture_root: Path, graph_cache: Path, tmp_path: Path) -> None:
    """Fresh facts replace the extracted edges of their file, and stale ones do not."""
    workspace = _workspace(graph_fixture_root, tmp_path / "ws")

    # 1. Without facts, `circle.area()` in Consumer.measure cannot be graded: two Circles.
    build_graph(str(workspace))
    calls = [edge for edge in _edges(workspace, CONSUMER) if edge["kind"] == "calls"]
    assert [edge["resolution"] for edge in calls] == ["ambiguous"], "step 1: by-name resolution is ambiguous"
    assert calls[0]["source"] == "tree-sitter", "step 1: and the extractor is named as its source"

    # 2. A facts file for that one source file replaces its call edges.
    _write_facts(
        workspace,
        [
            _header(),
            {"t": "file", "path": CONSUMER, "sha256": _sha(workspace, CONSUMER)},
            {"t": "call", "from": MEASURE, "to": "com.example.core.Circle#area()", "path": CONSUMER, "line": 7},
            {"t": "call", "from": MEASURE, "to": "java.util.List#add(java.lang.Object)", "path": CONSUMER, "line": 7},
            {"t": "call", "from": MEASURE, "to": "com.example.core.Circle#nope()", "path": CONSUMER, "line": 7},
            {"t": "wobble", "nonsense": True},
        ],
    )
    stats = build_graph(str(workspace))
    calls = sorted(
        (edge for edge in _edges(workspace, CONSUMER) if edge["kind"] == "calls"), key=lambda e: e["dst_name"]
    )
    assert len(calls) == 2, "step 2: the two mappable facts replaced the one ambiguous edge"

    # 3. The resolved one is exact and says which tool said so.
    resolved = next(edge for edge in calls if edge["dst_name"] == "area")
    assert resolved["resolution"] == "exact", "step 3: a fact edge is exact"
    assert resolved["source"] == "javac-facts", "step 3: and carries the tool as its source"
    assert "core/Circle.java" in resolved["dst_id"], "step 3: on the Circle the tool meant, not the other one"

    # 4. A target outside the workspace is kept as an external callee under its full ref.
    external = next(edge for edge in calls if edge["dst_id"] is None)
    assert external["dst_name"] == "java.util.List#add(java.lang.Object)", "step 4: the ref is kept verbatim"
    assert external["resolution"] == "unresolved", "step 4: graded like any JDK target"

    # 5. A ref the workspace cannot answer is reported, not guessed at.
    assert stats.facts["unmapped"] == 1, "step 5: the bogus ref is counted"
    assert stats.facts["files_fresh"] == 1, "step 5: one file is covered"

    # 6. The unknown fact kind was skipped rather than refused.
    connection = connect(str(workspace))
    status = connection.execute("SELECT * FROM facts_status").fetchall()
    connection.close()
    assert len(status) == 1 and status[0]["tool"] == "javac-facts", "step 6: the facts file is recorded"
    assert status[0]["files_declared"] == 1 and status[0]["unmapped"] == 1, "step 6: with its counts"

    # 7. Editing the source makes the facts stale: the extracted edges come back.
    consumer = workspace / CONSUMER
    consumer.write_text(consumer.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    stats = build_graph(str(workspace))
    assert stats.facts["files_stale"] == 1 and stats.facts["files_fresh"] == 0, "step 7: staleness is counted"
    calls = [edge for edge in _edges(workspace, CONSUMER) if edge["kind"] == "calls"]
    assert [edge["source"] for edge in calls] == ["tree-sitter"], "step 7: the extractor is back in charge"
    assert calls[0]["resolution"] == "ambiguous", "step 7: with the honest grade it always had"

    # 8. Rewriting the facts against the new content covers the file again, with no edit to it.
    _write_facts(
        workspace,
        [
            _header(),
            {"t": "file", "path": CONSUMER, "sha256": _sha(workspace, CONSUMER)},
            {"t": "call", "from": MEASURE, "to": "com.example.core.Circle#area()", "path": CONSUMER, "line": 7},
        ],
    )
    build_graph(str(workspace))
    calls = [edge for edge in _edges(workspace, CONSUMER) if edge["kind"] == "calls"]
    assert [edge["source"] for edge in calls] == ["javac-facts"], "step 8: a changed facts file re-resolves its files"

    # 9. Deleting the facts file hands the file back to the extractor.
    (workspace / ".zemble/facts/javac-facts.jsonl").unlink()
    build_graph(str(workspace))
    calls = [edge for edge in _edges(workspace, CONSUMER) if edge["kind"] == "calls"]
    assert [edge["source"] for edge in calls] == ["tree-sitter"], "step 9: no facts, no overlay"


def test_two_facts_files_union_their_edges(graph_fixture_root: Path, graph_cache: Path, tmp_path: Path) -> None:
    """One source file compiled by two tasks contributes the union of both fact sets."""
    workspace = _workspace(graph_fixture_root, tmp_path / "ws")
    shared = {"t": "file", "path": CONSUMER, "sha256": _sha(workspace, CONSUMER)}
    call = {"t": "call", "from": MEASURE, "to": "com.example.core.Circle#area()", "path": CONSUMER, "line": 7}
    _write_facts(workspace, [_header(), shared, call], name="common.jsonl", where="build/zemble")
    _write_facts(
        workspace,
        [
            _header(),
            shared,
            call,
            {"t": "call", "from": MEASURE, "to": "com.example.util.Circle#area()", "path": CONSUMER, "line": 8},
        ],
        name="server.jsonl",
        where="build/zemble",
    )
    stats = build_graph(str(workspace))
    calls = _edges(workspace, CONSUMER)
    edges = sorted(edge["dst_id"] for edge in calls if edge["kind"] == "calls")
    assert len(edges) == 2, "the identical edge is not doubled and the extra one is kept"
    assert stats.facts["facts_files"] == 2, "both files under build/zemble were discovered"


def test_a_wrong_header_is_refused(graph_fixture_root: Path, graph_cache: Path, tmp_path: Path) -> None:
    """A facts file of another format version is refused whole, not read half way."""
    workspace = _workspace(graph_fixture_root, tmp_path / "ws")
    path = _write_facts(
        workspace,
        [
            _header(version=99),
            {"t": "file", "path": CONSUMER, "sha256": _sha(workspace, CONSUMER)},
            {"t": "call", "from": MEASURE, "to": "com.example.core.Circle#area()", "path": CONSUMER, "line": 7},
        ],
    )
    with pytest.raises(FactsFormatError, match="version"):
        load_facts_file(path, workspace)
    stats = build_graph(str(workspace))
    assert stats.facts["errors"], "the build reports the refused file"
    calls = [edge for edge in _edges(workspace, CONSUMER) if edge["kind"] == "calls"]
    assert [edge["source"] for edge in calls] == ["tree-sitter"], "and nothing it said was applied"


def test_discovery_finds_both_conventions_and_honours_the_config(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path
) -> None:
    """Both default globs are walked, build output included, and graph.toml overrides them."""
    workspace = _workspace(graph_fixture_root, tmp_path / "ws")
    _write_facts(workspace, [_header()], name="a.jsonl")
    _write_facts(workspace, [_header()], name="b.jsonl", where="module/build/zemble")
    _write_facts(workspace, [_header()], name="c.jsonl", where="out/facts")
    found = {path.name for path in discover_facts_files(workspace)}
    assert found == {"a.jsonl", "b.jsonl"}, "the two default conventions are found and nothing else"
    assert matches_facts_glob(workspace, workspace / "module/build/zemble/b.jsonl"), "a facts path is recognised"
    assert not matches_facts_glob(workspace, workspace / "out/facts/c.jsonl"), "an unconfigured one is not"

    (workspace / ".zemble/graph.toml").write_text('[facts]\nsources = ["out/facts/*.jsonl"]\n', encoding="utf-8")
    found = {path.name for path in discover_facts_files(workspace)}
    assert found == {"c.jsonl"}, "a configured glob replaces the defaults"


def test_facts_status_reports_what_the_graph_used(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """`zemble graph facts status` names the files, the coverage and the unmapped refs."""
    workspace = _workspace(graph_fixture_root, tmp_path / "ws")

    # 1. With no facts at all it says so, and names where it looked.
    assert _run(monkeypatch, "graph", "facts", "status", str(workspace)) == 0, "step 1: status succeeds"
    output = capsys.readouterr().out
    assert "0 file(s) found" in output and ".zemble/facts/*.jsonl" in output, "step 1: it says where it looked"

    # 2. With facts it names the tool, the coverage and the refs it could not map.
    _write_facts(
        workspace,
        [
            _header(),
            {"t": "file", "path": CONSUMER, "sha256": _sha(workspace, CONSUMER)},
            {"t": "call", "from": MEASURE, "to": "com.example.core.Circle#area()", "path": CONSUMER, "line": 7},
            {"t": "call", "from": MEASURE, "to": "com.example.core.Circle#nope()", "path": CONSUMER, "line": 7},
            {"t": "file", "path": CIRCLE, "sha256": "0" * 64},
            {"t": "call", "from": "com.example.core.Circle#area()", "to": "x.Y#z()", "path": CIRCLE, "line": 20},
        ],
    )
    _run(monkeypatch, "graph", "facts", "status", str(workspace))
    output = capsys.readouterr().out
    assert "javac-facts 0.1.0" in output, "step 2: the tool and its version are named"
    assert "2 file(s) declared, 1 fresh, 1 stale" in output, "step 2: freshness is counted per file"
    assert f"stale: {CIRCLE} (content changed)" in output, "step 2: and the stale file says why"
    assert "com.example.core.Circle#nope()" in output, "step 2: the unmapped ref is listed"
    assert "calls in covered files: exact 1" in output, "step 2: covered calls are graded apart"

    assert "unmapped (top 1 of 1)" in output, "step 2: and lives in the unmapped bucket"

    # 3. The JSON form carries the same numbers, bucket by bucket.
    _run(monkeypatch, "graph", "facts", "status", str(workspace), "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["files"][0]["tool"] == "javac-facts", "step 3: the file is described"
    assert payload["coverage"]["files_fresh"] == 1, "step 3: coverage counts survive"
    assert _bucket(payload, "unmapped")["top"][0]["subject"] == "com.example.core.Circle#nope()", (
        "step 3: with the unmapped detail"
    )
    assert _bucket(payload, "stale")["count"] == 1, "step 3: the stale file's one fact is counted apart"
    assert _bucket(payload, "source_ignored")["count"] == 0, "step 3: and every bucket is present, even at zero"

    # 4. A query over a covered file names the tool in its reason.
    _run(monkeypatch, "graph", "callers", str(workspace), "com.example.core.Circle.area", "--json")
    payload = json.loads(capsys.readouterr().out)
    reasons = [hit["reason"] for hit in payload["results"] if hit["source"] == "javac-facts"]
    assert reasons and "javac-facts" in reasons[0], "step 4: the hit says which tool resolved it"

    # 5. Facts about generated code the index never walked are their own bucket, not unmapped.
    generated = workspace / GENERATED
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text("package com.example.gen;\nclass Tpl_Page { void render() {} }\n", encoding="utf-8")
    _write_facts(
        workspace,
        [
            _header(),
            {"t": "file", "path": CONSUMER, "sha256": _sha(workspace, CONSUMER)},
            {"t": "call", "from": MEASURE, "to": "com.example.core.Circle#area()", "path": CONSUMER, "line": 7},
            {"t": "file", "path": GENERATED, "sha256": _sha(workspace, GENERATED)},
            {
                "t": "call",
                "from": "com.example.gen.Tpl_Page#render()",
                "to": "com.example.core.Circle#area()",
                "path": GENERATED,
                "line": 2,
            },
        ],
    )
    _run(monkeypatch, "graph", "facts", "status", str(workspace), "--json")
    payload = json.loads(capsys.readouterr().out)
    ignored = _bucket(payload, "source_ignored")
    assert ignored["count"] == 1, "step 5: the generated file's fact is bucketed by where it came from"
    assert ignored["top"][0]["subject"] == GENERATED, "step 5: listed under the file, which is the lever"
    assert "generated/ignored" in ignored["top"][0]["reason"], "step 5: and says so in words"
    assert _bucket(payload, "unmapped")["count"] == 0, "step 5: it is not counted as an unmapped ref"

    # 6. The human report names every bucket on its own line.
    _run(monkeypatch, "graph", "facts", "status", str(workspace))
    output = capsys.readouterr().out
    assert "skipped facts: source outside the index 0, source ignored by the index 1" in output, (
        "step 6: the headline splits the buckets"
    )
    assert "source ignored by the index (top 1 of 1):" in output, "step 6: with a top-N list of its own"


def test_hierarchy_and_override_facts_replace_the_derived_ones(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path
) -> None:
    """A covered file's supertype and override edges come from the facts alone."""
    workspace = _workspace(graph_fixture_root, tmp_path / "ws")
    _write_facts(
        workspace,
        [
            _header(tool="acme"),
            {"t": "file", "path": CIRCLE, "sha256": _sha(workspace, CIRCLE)},
            {"t": "implements", "from": "com.example.core.Circle", "to": "com.example.core.Shape"},
            {"t": "override", "from": "com.example.core.Circle#area()", "to": "com.example.core.Shape#area()"},
        ],
    )
    build_graph(str(workspace))
    edges = _edges(workspace, CIRCLE)
    supertypes = [edge for edge in edges if edge["kind"] in ("extends", "implements")]
    assert [edge["source"] for edge in supertypes] == ["acme"], "the supertype edge comes from the tool"
    assert supertypes[0]["line"] == 5, "an edge with no line falls back to the declaration's own line"
    overrides = [edge for edge in edges if edge["kind"] == "overrides"]
    assert len(overrides) == 1 and overrides[0]["source"] == "acme", "the override is not derived a second time"


REGISTRY = "src/main/java/com/example/core/Registry.java"
POINT = "src/main/java/com/example/core/Point.java"


def test_javac_spellings_land_on_zemble_symbols(graph_fixture_root: Path, graph_cache: Path, tmp_path: Path) -> None:
    """The shapes javac spells differently - flat names, initializers, record accessors - map."""
    workspace = _workspace(graph_fixture_root, tmp_path / "ws")
    _write_facts(
        workspace,
        [
            _header(),
            {"t": "file", "path": REGISTRY, "sha256": _sha(workspace, REGISTRY)},
            # An anonymous class body: javac numbers it per outermost class.
            {
                "t": "call",
                "from": "com.example.core.Registry$1#area()",
                "to": "com.example.util.Helpers#twice(double)",
                "path": REGISTRY,
                "line": 26,
            },
            # A local class, named as well as numbered.
            {
                "t": "call",
                "from": "com.example.core.Registry$1Local#go()",
                "to": "com.example.core.Registry.Entry#name()",
                "path": REGISTRY,
                "line": 36,
            },
            # A static initializer, which has no symbol of its own.
            {
                "t": "call",
                "from": "com.example.core.Registry#<clinit>()",
                "to": "com.example.util.Helpers#twice(double)",
                "path": REGISTRY,
                "line": 6,
            },
            # `new Anon(){}`: the anonymous class declares no constructor.
            {
                "t": "call",
                "from": "com.example.core.Registry#anonymousShape()",
                "to": "com.example.core.Registry$1#<init>()",
                "path": REGISTRY,
                "line": 23,
            },
            {"t": "file", "path": POINT, "sha256": _sha(workspace, POINT)},
            # A record accessor: implicit in javac, a component field in zemble.
            {
                "t": "call",
                "from": "com.example.core.Point#sum()",
                "to": "com.example.core.Point#x()",
                "path": POINT,
                "line": 6,
            },
        ],
    )
    stats = build_graph(str(workspace))
    assert stats.facts["unmapped"] == 0, "every javac spelling found its zemble symbol"

    sources = {edge["src_id"] for edge in _edges(workspace, REGISTRY) if edge["kind"] == "calls"}
    assert any("$anon@23.area" in source for source in sources), "the flat anonymous name became the anon method"
    assert any("localHelper.Local.go" in source for source in sources), "the flat local name became the local method"
    assert any(source.endswith("#com.example.core.Registry") for source in sources), (
        "a static initializer is attributed to its type"
    )

    targets = {edge["dst_id"] for edge in _edges(workspace, REGISTRY) if edge["kind"] == "calls"}
    assert any(target.endswith("$anon@23(1)") for target in targets), "the implicit constructor is the type itself"

    accessor = next(edge for edge in _edges(workspace, POINT) if edge["kind"] == "calls")
    assert accessor["dst_id"].endswith("Point.x"), "the record accessor landed on the component"
