"""Behaviour journeys over duplication detection."""

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zemble.dedup.baseline import diff_baseline, load_baseline, save_baseline
from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.homes import Evidence, EvidenceKind, HomeVerdict, HomeVerdictKind, visibility_evidence
from zemble.dedup.languages import PROFILES, Visibility
from zemble.dedup.model import CloneKind, Lane
from zemble.dedup.report import format_baseline_diff, format_report, report_json
from zemble.dedup.structure import check_pair, edit_distance, jaccard
from zemble.dedup.units import extract_units

FIXTURES = Path(__file__).parent / "fixtures" / "dedup"
LANES = Path(__file__).parent / "fixtures" / "dedup_lanes"
_CLI_ENTRY = "from zemble.cli import main; main()"


class BagOfWordsEmbedder:
    """A deterministic stand-in for the real embedder: hashed bag of identifiers, L2 normalized."""

    model_id = "stub:bag-of-words"
    dimensions = 64

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in re.findall(r"[A-Za-z_]+", text):
            vector[sum(word.encode()) % self.dimensions] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / (norm or 1.0)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents."""
        return np.vstack([self._vector(text) for text in texts]).astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed queries the same way."""
        return self.embed_documents(texts)


def _units(name: str, **kwargs: object) -> list:
    """Extract the units of one fixture file."""
    path = FIXTURES / "src" / name
    return extract_units(path.read_bytes(), name, **kwargs)  # type: ignore[arg-type]


def _locations(report, kind: CloneKind) -> list[set[str]]:
    """The member locations of every class of one kind."""
    return [{member.location for member in clone.members} for clone in report.of_kind(kind)]


def test_unit_extraction_journey() -> None:
    """A file becomes body units and statement windows, with spans, calls and control flow."""
    # 1. A method body is one unit, spanning from its declaration to its closing brace.
    bodies = [unit for unit in _units("RenamedA.java") if unit.is_body]
    assert len(bodies) == 1, "step 1: the file holds exactly one body long enough to compare"
    body = bodies[0]
    assert (body.kind, body.name) == ("method", "RenamedA.gather"), "step 1: the unit names its declaration"
    assert (body.start_line, body.end_line) == (8, 17), "step 1: the span covers the whole declaration"

    # 2. The control-flow skeleton and the call set are recorded, not the local names.
    assert body.skeleton == ("for", "if", "return"), "step 2: the skeleton is the control keyword sequence"
    assert body.calls == (), "step 2: the body calls nothing"
    assert set(body.literals) == {"0", "3"}, "step 2: literals are kept verbatim"

    # 3. Bodies below the token floor are dropped, so getters never form clone classes.
    assert _units("RenamedA.java", min_tokens=1000) == [], "step 3: the token floor drops everything short"

    # 4. Windows appear inside a long body, and only there.
    windows = [unit for unit in _units("WindowA.java") if not unit.is_body]
    assert windows, "step 4: a nine-statement body yields statement windows"
    assert all(unit.kind == "window" for unit in windows), "step 4: windows are marked as such"
    assert all(unit.token_count >= 30 for unit in windows), "step 4: windows respect the token floor too"

    # 5. Windows can be switched off entirely.
    assert all(unit.is_body for unit in _units("WindowA.java", windows=False)), "step 5: --no-windows means bodies only"


def test_exact_and_renamed_journey() -> None:
    """Exact and alpha-renamed classes separate copies from near misses, and report each once."""
    report = find_duplication(FIXTURES, DupeOptions(kinds=(CloneKind.EXACT, CloneKind.RENAMED)))

    # 1. The verbatim copy across two files is one exact class.
    exact = _locations(report, CloneKind.EXACT)
    assert {"src/ExactA.java:8-17", "src/ExactB.java:9-18"} in exact, "step 1: the planted exact pair is found"

    # 2. Comments and whitespace are not part of the token stream that hashed it.
    exact_units = [unit for unit in _units("ExactB.java") if unit.is_body]
    assert exact_units[0].token_count == 49, "step 2: the comment and the extra spaces are gone"

    # 3. The alpha-renamed twin joins them, because only declared locals differ.
    renamed = _locations(report, CloneKind.RENAMED)
    assert len(renamed) == 1, "step 3: exactly one renamed class exists"
    assert "src/RenamedA.java:8-17" in renamed[0], "step 3: the renamed twin is a member"

    # 4. A differing literal is a different decision and never matches.
    assert not any("src/LiteralDiff.java:8-17" in members for members in renamed), (
        "step 4: the one-literal near miss is not a renamed clone"
    )

    # 5. A differing FIELD name is not a local, so it does not match either.
    assert not any("src/FieldDiff.java:8-17" in members for members in renamed), (
        "step 5: the one-field near miss is not a renamed clone"
    )

    # 6. A constructor's `this.field = field` run does not collapse: a member name is not a local.
    assert not any({"src/CtorA.java:13-20", "src/CtorB.java:13-20"} <= members for members in renamed), (
        "step 6: two constructors with different FIELD names are not renamed clones"
    )

    # 7. Pure exact duplication is reported once: no renamed class repeats it alone.
    window_pair = {"src/WindowA.java:7-13", "src/WindowB.java:8-14"}
    assert window_pair in exact, "step 7: the copied statement window is an exact class"
    assert window_pair not in renamed, "step 7: and it is not repeated under renamed"


def test_ranking_journey() -> None:
    """Classes are ranked by weight, subsumed window classes drop out, and filters narrow the report."""
    report = find_duplication(FIXTURES, DupeOptions(kinds=(CloneKind.EXACT,)))

    # 1. The many overlapping windows of one copied run collapse into a single class.
    window_classes = [clone for clone in report.of_kind(CloneKind.EXACT) if clone.members[0].kind == "window"]
    assert len(window_classes) == 1, "step 1: only the widest window survives subsumption"
    assert window_classes[0].tokens == 49, "step 1: and it is the widest one, not a nested slice"

    # 2. Weight is tokens x copies x files, so a wider class outranks a narrower one.
    scores = [clone.score for clone in report.of_kind(CloneKind.EXACT)]
    assert scores == sorted(scores, reverse=True), "step 2: classes come back best first"
    assert report.of_kind(CloneKind.EXACT)[0].score == 49 * 2 * 2, "step 2: the score is tokens x copies x files"

    # 3. A cross-file floor above what the fixture spans empties the report.
    narrowed = find_duplication(FIXTURES, DupeOptions(kinds=(CloneKind.EXACT,), min_files=3))
    assert narrowed.classes == [], "step 3: no fixture class spans three files"

    # 4. Restricting the paths restricts what is even read.
    scoped = find_duplication(
        FIXTURES, DupeOptions(kinds=(CloneKind.EXACT,), paths=(str(FIXTURES / "src" / "ExactA.java"),))
    )
    assert scoped.analyzed_files == 1, "step 4: only the named file is scanned"
    assert scoped.classes == [], "step 4: one file alone duplicates nothing"


def test_logic_journey() -> None:
    """Logic mode reports only embedding candidates that also pass the structural check."""
    options = DupeOptions(kinds=(CloneKind.LOGIC,), windows=False, logic_threshold=0.5)
    report = find_duplication(FIXTURES, options, embedder=BagOfWordsEmbedder())
    classes = report.of_kind(CloneKind.LOGIC)

    # 1. The planted logic pair is one class, despite different locals and one extra statement.
    pair = next(
        (clone for clone in classes if {"src/LogicA.java:6-15", "src/LogicB.java:6-16"} <= _members(clone)), None
    )
    assert pair is not None, "step 1: the logic pair is reported"

    # 2. The pair states its reason: control flow, shared calls, literals.
    assert any("control flow identical" in reason.reason for reason in pair.reasons), "step 2: control flow named"
    assert any("shared" in reason.reason for reason in pair.reasons), "step 2: the reason names the call overlap"
    assert all(reason.left and reason.right for reason in pair.reasons), "step 2: each reason names its two units"

    # 2b. The wire form states one shared reason per class rather than repeating it per pair.
    verdicts = {reason.reason for reason in pair.reasons}
    assert len(pair.wire_reasons) == (1 if len(verdicts) == 1 else len(pair.reasons)), "step 2b: reasons are deduped"

    # 3. The non-pair shares every call but no control flow, so it is never reported.
    assert not any("src/LogicC.java:6-17" in _members(clone) for clone in classes), (
        "step 3: same calls with a different control flow is not a logic clone"
    )

    # 4. Exact and renamed members are not re-reported as logic clones.
    assert not any({"src/ExactA.java:8-17", "src/ExactB.java:9-18"} <= _members(clone) for clone in classes), (
        "step 4: an exact pair is reported under exact alone"
    )

    # 5. The run says what it embedded and how many pairs survived.
    assert any("stub:bag-of-words" in note for note in report.notes), "step 5: the note names the embedder"

    # 6. A threshold no candidate can reach yields nothing at all.
    strict = find_duplication(
        FIXTURES,
        DupeOptions(kinds=(CloneKind.LOGIC,), windows=False, logic_threshold=1.01),
        embedder=BagOfWordsEmbedder(),
    )
    assert strict.of_kind(CloneKind.LOGIC) == [], "step 6: nothing passes an impossible threshold"


def _members(clone) -> set[str]:
    """The member locations of one class."""
    return {member.location for member in clone.members}


def test_structural_check_journey() -> None:
    """The structural check accepts and refuses for stated, separate reasons."""
    logic_a = [unit for unit in _units("LogicA.java", include_text=True) if unit.is_body][0]
    logic_b = [unit for unit in _units("LogicB.java", include_text=True) if unit.is_body][0]
    logic_c = [unit for unit in _units("LogicC.java", include_text=True) if unit.is_body][0]
    renamed = [unit for unit in _units("RenamedA.java") if unit.is_body][0]

    # 1. Identical control flow plus identical calls is accepted.
    verdict = check_pair(logic_a, logic_b)
    assert verdict.accepted, "step 1: the planted pair passes"

    # 2. A different control flow is refused before anything else is considered.
    refused = check_pair(logic_a, logic_c)
    assert not refused.accepted and refused.reason == "control flow differs", "step 2: control flow gates first"

    # 3. A shared control flow with no call overlap is refused on the call set.
    disjoint = check_pair(logic_a, renamed)
    assert not disjoint.accepted, "step 3: disjoint call sets are refused"
    assert "call overlap" in disjoint.reason, "step 3: and the reason says so"

    # 4. The primitives behave: distance gives up past its limit, Jaccard is a ratio.
    assert edit_distance(("if", "return"), ("if", "for", "while", "return"), 2) == 2, "step 4: two insertions"
    assert edit_distance(("if",), ("for", "while", "switch", "try"), 2) == 3, "step 4: past the limit it gives up"
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3), "step 4: Jaccard is a ratio"


def test_report_and_cli_journey(tmp_path: Path) -> None:
    """The report renders like the reference tool, and the CLI is a report, never a gate."""
    report = find_duplication(FIXTURES, DupeOptions(kinds=(CloneKind.EXACT, CloneKind.RENAMED)))

    # 1. The text form leads with the counts and prints one section per kind.
    text = format_report(report)
    assert text.startswith("Analyzed 12 file(s)"), "step 1: the header counts the files"
    assert "-- EXACT --" in text and "copies x" in text, "step 1: the sections mirror zenit-dev duplication"
    assert "== PRODUCTION" in text, "step 1: lane sections wrap the kind sections"

    # 2. The limit caps a section and says what it hid.
    capped = format_report(report, limit=1)
    assert "showing the top 1 of 2" in capped, "step 2: the limit is announced"

    # 3. The JSON form carries the same classes, machine-readable.
    payload = report_json(report)
    assert payload["class_counts"]["exact"] == 2, "step 3: the counts survive the wire"
    assert payload["classes"][0]["members"][0]["file_path"].endswith(".java"), "step 3: members carry their file"

    # 4. The CLI exits 0 however much duplication there is.
    result = subprocess.run(
        [sys.executable, "-c", _CLI_ENTRY, "dupes", str(FIXTURES), "--json", "--kind", "all"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"step 4: a report never fails the build ({result.stderr[-400:]})"
    parsed = json.loads(result.stdout)
    assert parsed["analyzed_files"] == 12, "step 4: the JSON names what was scanned"

    # 5. An unknown kind is a usage error, not a silent empty report.
    bad = subprocess.run(
        [sys.executable, "-c", _CLI_ENTRY, "dupes", str(FIXTURES), "--kind", "nonsense"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0 and "Unknown --kind" in bad.stderr, "step 5: a bad kind is refused loudly"

    # 6. A missing root is refused too.
    missing = find_duplication
    with pytest.raises(FileNotFoundError):
        missing(tmp_path / "nope")


def test_mcp_server_registers_the_dupes_tool() -> None:
    """The duplication tool joins the existing MCP server without replacing anything."""
    import asyncio

    from zemble.index_cache import IndexCache
    from zemble.mcp import create_server

    server = create_server(IndexCache())
    tools = {tool.name for tool in asyncio.run(server.list_tools())}

    # 1. The tool is registered.
    assert "dupes" in tools, "step 1: the dupes tool is served"

    # 2. The tools that were there before still are.
    assert {"search", "find_related", "graph_definition"} <= tools, "step 2: nothing was replaced"

    # 3. By default it answers with the text report the CLI prints, not with JSON in a string.
    content, structured = asyncio.run(server.call_tool("dupes", {"repo": str(FIXTURES), "kind": "exact"}))
    text = content[0].text
    assert text.startswith("Analyzed "), "step 3: the default format is the CLI's own report"
    assert structured["result"] == text, "step 3: the text is handed over once, unencoded"

    # 4. `brief` trims it to one line per class, still as text.
    brief = asyncio.run(server.call_tool("dupes", {"repo": str(FIXTURES), "kind": "exact", "brief": True}))[0][0].text
    assert "-- EXACT --" not in brief and "#1  exact " in brief, "step 4: brief is the header and the class lines"

    # 5. `format="json"` returns the structured object itself, never a JSON string.
    content, structured = asyncio.run(
        server.call_tool("dupes", {"repo": str(FIXTURES), "kind": "exact", "format": "json"})
    )
    payload = structured["result"]
    assert isinstance(payload, dict), "step 5: the JSON form is an object on the wire"
    assert payload["class_counts"]["exact"] == 2, "step 5: the tool returns the ranked classes"
    assert json.loads(content[0].text) == payload, "step 5: the text content is that object, encoded once"

    # 6. The lane and exclude arguments reach the scan.
    lane_only = asyncio.run(
        server.call_tool("dupes", {"repo": str(LANES), "kind": "exact", "lane": "test", "format": "json"})
    )[1]["result"]
    assert {clone["lane"] for clone in lane_only["classes"]} == {"test"}, "step 6: --lane is available over MCP"


def test_nothing_scanned_is_never_a_clean_report(tmp_path: Path) -> None:
    """A run that walked no supported file says what it looked for instead of "No duplication"."""
    from zemble.dedup.mcp import _options, _run

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "notes.txt").write_text("nothing to parse here\n")

    # 1. The text report refuses to call an empty scan a clean one.
    report = find_duplication(tmp_path, DupeOptions(kinds=(CloneKind.EXACT,)))
    text = format_report(report)
    assert report.analyzed_files == 0, "step 1: nothing was walked"
    assert "No duplication found." not in text, "step 1: the misleading line is gone"
    assert f"Scanned 0 supported file(s) under {report.root}" in text, "step 1: it names the root"
    assert "(supported: .java, .zig)" in text, "step 1: it names the extensions it walks"
    assert "check --paths/--exclude/ignore files" in text, "step 1: it names the likely cause"

    # 2. Brief mode says it too; a piped report must not read as a pass either.
    assert "Scanned 0 supported file(s)" in format_report(report, brief=True), "step 2: brief says it as well"

    # 3. The JSON form carries it machine-readably, counts and all.
    payload = report_json(report)
    assert payload["analyzed_files"] == 0 and payload["failed_files"] == 0, "step 3: the counts are on the wire"
    assert payload["supported_extensions"] == [".java", ".zig"], "step 3: so are the extensions"
    assert any("Scanned 0 supported file(s)" in note for note in payload["notes"]), "step 3: and the note"

    # 4. The CLI still exits 0: this is a report, not a gate.
    result = subprocess.run(
        [sys.executable, "-c", _CLI_ENTRY, "dupes", str(tmp_path), "--kind", "exact"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"step 4: an empty scan is not a failure ({result.stderr[-400:]})"
    assert "Scanned 0 supported file(s)" in result.stdout, "step 4: the refusal is on stdout"
    assert "No duplication found." not in result.stdout, "step 4: and the misleading line is not"

    # 5. MCP says the same thing, in both of its shapes.
    options = _options("exact", "all", None, None, 1)
    mcp_text = _run(options, str(tmp_path), 25, "text", False, False, False)
    assert "Scanned 0 supported file(s)" in mcp_text, "step 5: the MCP text form carries it"
    mcp_json = _run(options, str(tmp_path), 25, "json", False, False, False)
    assert any("Scanned 0 supported file(s)" in note for note in mcp_json["notes"]), "step 5: so does the JSON"


def test_extraction_failures_are_counted_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file that will not parse is named in the report; a missing grammar is never a clean run."""
    import zemble.dedup.detect as detect

    (tmp_path / "src").mkdir()
    for name in ("Alpha.java", "Beta.java"):
        (tmp_path / "src" / name).write_text("class X { void m() { int a = 1; } }\n")

    def _boom(source: bytes, file_path: str, **kwargs: object) -> list:
        raise RuntimeError("No tree-sitter java grammar available")

    monkeypatch.setattr(detect, "extract_units", _boom)
    report = find_duplication(tmp_path, DupeOptions(kinds=(CloneKind.EXACT,), jobs=1))

    # 1. The files were walked, and every one of them failed.
    assert report.analyzed_files == 2 and report.failed_files == 2, "step 1: the failures are counted"
    assert report.failed_examples == ["src/Alpha.java", "src/Beta.java"], "step 1: and named"

    # 2. Both surfaces say so rather than reporting a clean workspace.
    text = format_report(report)
    assert "extraction failed for 2 file(s): src/Alpha.java, src/Beta.java" in text, "step 2: the text says it"
    payload = report_json(report)
    assert payload["failed_files"] == 2, "step 2: the JSON counts it"
    assert any("extraction failed for 2 file(s)" in note for note in payload["notes"]), "step 2: and notes it"


def test_relative_paths_resolve_against_the_scan_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--paths` and `--exclude` are root-relative, whatever directory the process happens to be in."""
    from zemble.dedup.mcp import _options, _run

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    # 1. A root-relative --paths finds the files even though the CWD holds no such directory.
    report = find_duplication(FIXTURES, DupeOptions(kinds=(CloneKind.EXACT,), paths=("src",)))
    assert report.analyzed_files == 12, "step 1: `src` is resolved under the root, not under the CWD"

    # 2. An absolute path keeps working exactly as before.
    absolute = find_duplication(FIXTURES, DupeOptions(kinds=(CloneKind.EXACT,), paths=(str(FIXTURES / "src"),)))
    assert absolute.analyzed_files == 12, "step 2: an absolute restriction is taken as given"

    # 3. A path that only exists relative to the CWD selects nothing, which is the point.
    (elsewhere / "src").mkdir()
    (elsewhere / "src" / "Stray.java").write_text("class Stray { void m() { int a = 1; } }\n")
    assert find_duplication(FIXTURES, DupeOptions(paths=("src",))).analyzed_files == 12, "step 3: the CWD is ignored"

    # 4. --exclude is root-relative too.
    excluded = find_duplication(FIXTURES, DupeOptions(kinds=(CloneKind.EXACT,), exclude=("src/Exact*.java",)))
    assert excluded.analyzed_files == 10, "step 4: the exclude matched two root-relative paths"

    # 5. The CLI agrees, run from that same unrelated directory.
    result = subprocess.run(
        [sys.executable, "-c", _CLI_ENTRY, "dupes", str(FIXTURES), "--kind", "exact", "--paths", "src", "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=elsewhere,
    )
    assert result.returncode == 0, f"step 5: the run succeeded ({result.stderr[-400:]})"
    assert json.loads(result.stdout)["analyzed_files"] == 12, "step 5: the CLI resolves against the root"

    # 6. And so does MCP, which only ever knows the workspace.
    payload = _run(_options("exact", "all", ["src"], None, 1), str(FIXTURES), 25, "json", False, False, False)
    assert payload["analyzed_files"] == 12, "step 6: the MCP tool resolves against the repo"


def _by_lane(report, lane: Lane) -> list:
    """The classes of one lane, ranked."""
    return [clone for clone in report.classes if clone.lane is lane]


def test_lane_and_exclude_journey(tmp_path: Path) -> None:
    """Test scaffolding is reported apart from production code, and a glob can drop files entirely."""
    report = find_duplication(LANES, DupeOptions(kinds=(CloneKind.EXACT,)))

    # 1. Every class is classified by where its members live, using the graph's own test rule.
    lanes = {clone.lane for clone in report.classes}
    assert lanes == {Lane.PRODUCTION, Lane.MIXED, Lane.TEST}, "step 1: all three lanes are recognised"
    assert _by_lane(report, Lane.TEST)[0].members[0].name.endswith("setUp"), "step 1: the fixture copy is test-only"
    assert all(member.is_test for member in _by_lane(report, Lane.TEST)[0].members), "step 1: every member is a test"

    # 2. The text report prints production first and test-only last, however the scores fall.
    text = format_report(report)
    order = [text.index(f"== {name}") for name in ("PRODUCTION", "MIXED", "TEST")]
    assert order == sorted(order), "step 2: production duplication can never sit below scaffolding"

    # 3. --lane restricts the report without touching any score.
    only_test = find_duplication(LANES, DupeOptions(kinds=(CloneKind.EXACT,), lane=Lane.TEST))
    assert [clone.key for clone in only_test.classes] == [clone.key for clone in _by_lane(report, Lane.TEST)], (
        "step 3: a lane filter selects, it does not rerank"
    )

    # 4. An exclude glob drops the files before anything is parsed.
    excluded = find_duplication(LANES, DupeOptions(kinds=(CloneKind.EXACT,), exclude=("generated/**",)))
    assert excluded.analyzed_files == report.analyzed_files - 1, "step 4: the generated file is never read"
    production = _by_lane(excluded, Lane.PRODUCTION)[0]
    assert len(production.members) == 2, "step 4: the generated copy is gone from the class"
    assert all("generated" not in member.file_path for member in production.members), "step 4: no generated member"

    # 5. --paths still restricts the scan, and the two flags are independent.
    restricted = find_duplication(LANES, DupeOptions(kinds=(CloneKind.EXACT,), paths=(str(LANES / "src" / "test"),)))
    assert {clone.lane for clone in restricted.classes} == {Lane.TEST}, "step 5: only the test tree was scanned"

    # 6. An exclude pattern matching nothing changes nothing.
    untouched = find_duplication(LANES, DupeOptions(kinds=(CloneKind.EXACT,), exclude=("nothing/here/**",)))
    assert [clone.key for clone in untouched.classes] == [clone.key for clone in report.classes], (
        "step 6: a pattern that matches nothing is not a filter"
    )
    assert tmp_path.exists(), "step 6: no temporary state was needed"


def test_class_key_stability_journey(tmp_path: Path) -> None:
    """A class key survives line drift and moves when the class itself changes."""
    workspace = tmp_path / "keys"
    (workspace / "src").mkdir(parents=True)
    source = (LANES / "src" / "main" / "java" / "fixtures" / "Alpha.java").read_text()
    (workspace / "src" / "Alpha.java").write_text(source)
    (workspace / "src" / "Beta.java").write_text(source.replace("class Alpha", "class Beta"))
    options = DupeOptions(kinds=(CloneKind.EXACT,))

    # 1. The class has one stable key, printed on its head line.
    first = find_duplication(workspace, options)
    key = first.classes[0].key
    assert key.startswith("exact:") and len(key) == len("exact:") + 12, "step 1: the key names its kind"
    assert f"key: {key}" in format_report(first), "step 1: the report prints the key"

    # 2. Shifting every member down by comment lines does not move the key.
    for name in ("Alpha.java", "Beta.java"):
        path = workspace / "src" / name
        path.write_text("// a new banner comment\n// and another\n" + path.read_text())
    shifted = find_duplication(workspace, options)
    assert shifted.classes[0].key == key, "step 2: line numbers are not part of the identity"
    assert shifted.classes[0].members[0].start_line != first.classes[0].members[0].start_line, "step 2: lines moved"

    # 3. Adding a copy is a different class, and says so.
    (workspace / "src" / "Delta.java").write_text(source.replace("class Alpha", "class Delta"))
    grown = find_duplication(workspace, options)
    grown_key = grown.classes[0].key
    assert grown_key != key, "step 3: a third copy is a new class"
    assert len(grown.classes[0].members) == 3, "step 3: the copy did join the class"

    # 4. Moving a file does not move the key: the identity is the content, not the paths.
    (workspace / "src" / "moved").mkdir()
    (workspace / "src" / "Delta.java").rename(workspace / "src" / "moved" / "Delta.java")
    moved = find_duplication(workspace, options)
    assert moved.classes[0].key == grown_key, "step 4: a file move keeps the key"

    # 5. Scanning from an ancestor root gives the same key, so per-repo ignore entries hold.
    parent = find_duplication(tmp_path, options)
    assert parent.classes[0].key == grown_key, "step 5: the key is scan-root independent"


def _ignore_file(root: Path, *lines: str) -> Path:
    """Write a `.zemble/dupes.ignore` under a workspace root."""
    path = root / ".zemble" / "dupes.ignore"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_suppression_journey(tmp_path: Path) -> None:
    """A justified ignore entry hides a class; an unjustified or stale one is itself reported."""
    workspace = tmp_path / "suppress"
    workspace.mkdir()
    for name in ("Alpha.java", "Beta.java", "Gamma.java", "AlphaTest.java", "BetaTest.java", "GammaTest.java"):
        target = workspace / "src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = next(LANES.rglob(name))
        target.write_text(source.read_text())
    options = DupeOptions(kinds=(CloneKind.EXACT,))
    plain = find_duplication(workspace, options)
    assert len(plain.classes) >= 2, "setup: there is something to suppress"
    victim = plain.classes[0].key

    # 1. A justified entry takes the class out of the report and counts it in the trailer.
    _ignore_file(workspace, "# deliberate", f"{victim}  the driver forces these two bodies apart")
    suppressed = find_duplication(workspace, options)
    assert victim not in {clone.key for clone in suppressed.classes}, "step 1: the class is gone from the report"
    assert [clone.key for clone in suppressed.suppressed] == [victim], "step 1: it is counted as suppressed"
    assert "suppressed: 1" in format_report(suppressed), "step 1: the trailer says how many were hidden"
    assert not suppressed.ignore_problems, "step 1: a justified entry is not a violation"

    # 2. --show-suppressed prints them, so a reader can audit the file.
    shown = format_report(suppressed, show_suppressed=True)
    assert victim in shown, "step 2: the suppressed class is printable on demand"

    # 3. An entry without a justification suppresses nothing and is a violation of its own.
    _ignore_file(workspace, victim)
    bare = find_duplication(workspace, options)
    assert victim in {clone.key for clone in bare.classes}, "step 3: a bare entry hides nothing"
    assert any("no justification" in problem for problem in bare.ignore_problems), "step 3: the bare entry is named"
    assert "ignore-file violation" in format_report(bare), "step 3: the report says so"

    # 4. An entry matching nothing is stale, and the line number points at it.
    _ignore_file(workspace, "exact:000000000000  a class that no longer exists")
    stale = find_duplication(workspace, options)
    assert any("is stale" in problem and ":1:" in problem for problem in stale.ignore_problems), "step 4: stale"

    # 5. A kind that was not scanned is never called stale.
    _ignore_file(workspace, "renamed:000000000000  suppressed in the renamed report")
    other_kind = find_duplication(workspace, options)
    assert not other_kind.ignore_problems, "step 5: --kind exact does not judge renamed entries"

    # 6. A nested repo's own ignore file is honoured by a workspace scan of the ancestor root.
    body = (
        "package inner;\n\npublic class Inner {\n"
        "    public String weave(String input) {\n"
        "        StringBuilder builder = new StringBuilder();\n"
        '        builder.append("inner");\n'
        "        builder.append(input.length());\n"
        "        builder.append(input.trim());\n"
        "        builder.append(input);\n"
        "        return builder.toString();\n"
        "    }\n}\n"
    )
    for repo, name in (("repo-a", "Inner"), ("repo-b", "InnerCopy")):
        target = workspace / repo / "src" / f"{name}.java"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body.replace("class Inner", f"class {name}"))
    spanning = find_duplication(workspace, options)
    inner_key = next(
        clone.key for clone in spanning.classes if any("repo-a" in member.file_path for member in clone.members)
    )
    nested = workspace / "repo-a" / ".zemble" / "dupes.ignore"
    nested.parent.mkdir(parents=True)
    nested.write_text(f"{inner_key}  the two repos ship separately, reviewed\n")
    honoured = find_duplication(workspace, options)
    assert inner_key in {clone.key for clone in honoured.suppressed}, "step 6: the nested entry suppresses"
    assert inner_key not in {clone.key for clone in honoured.classes}, "step 6: and the class left the report"

    # 7. A violation in a nested file is named by that file's own path.
    nested.write_text(f"{inner_key}\n")
    bare_nested = find_duplication(workspace, options)
    assert any(problem.startswith("repo-a/.zemble/dupes.ignore:1:") for problem in bare_nested.ignore_problems), (
        "step 7: the violation names the nested file"
    )


def test_baseline_journey(tmp_path: Path) -> None:
    """A baseline turns a second run into resolved / remaining / new instead of two lists to eyeball."""
    workspace = tmp_path / "baseline"
    workspace.mkdir()
    source = (LANES / "src" / "main" / "java" / "fixtures" / "Alpha.java").read_text()
    setup = (LANES / "src" / "test" / "java" / "fixtures" / "AlphaTest.java").read_text()
    (workspace / "src").mkdir()
    (workspace / "src" / "Alpha.java").write_text(source)
    (workspace / "src" / "Beta.java").write_text(source.replace("class Alpha", "class Beta"))
    (workspace / "src" / "AlphaTest.java").write_text(setup)
    (workspace / "src" / "BetaTest.java").write_text(setup.replace("class AlphaTest", "class BetaTest"))
    describe = (LANES / "src" / "main" / "java" / "fixtures" / "Gamma.java").read_text()
    (workspace / "src" / "Gamma.java").write_text(describe)
    (workspace / "src" / "GammaCopy.java").write_text(describe.replace("class Gamma", "class GammaCopy"))
    options = DupeOptions(kinds=(CloneKind.EXACT,))
    before = find_duplication(workspace, options)
    assert len(before.classes) == 3, "setup: three clone classes to compare against"
    saved = save_baseline(tmp_path / "dupes-baseline.json", before)

    # 1. The baseline holds one entry per class, keyed the way the report prints it.
    baseline = load_baseline(saved)
    assert baseline.keys == {clone.key for clone in before.classes}, "step 1: every class is remembered"

    # 2. Fixing one class, editing another and adding a third gives resolved, changed and new.
    (workspace / "src" / "Beta.java").write_text("package fixtures;\n\npublic class Beta {\n}\n")
    (workspace / "src" / "Delta.java").write_text(setup.replace("class AlphaTest", "class Delta"))
    fresh = (
        "package fixtures;\n\npublic class Epsilon {\n"
        "    public String weave(String input) {\n"
        "        StringBuilder builder = new StringBuilder();\n"
        '        builder.append("epsilon");\n'
        "        builder.append(input.length());\n"
        "        builder.append(input.trim());\n"
        "        builder.append(input);\n"
        "        return builder.toString();\n"
        "    }\n}\n"
    )
    (workspace / "src" / "Epsilon.java").write_text(fresh)
    (workspace / "src" / "EpsilonCopy.java").write_text(fresh.replace("class Epsilon", "class EpsilonCopy"))
    after = find_duplication(workspace, options)
    difference = diff_baseline(after, baseline)
    resolved = {entry.members[0] for entry in difference.resolved}
    assert "src/Alpha.java:6-15" in resolved, "step 2: the class whose copy was deleted is resolved"
    assert difference.resolved[0].kind == "exact", "step 2: a resolved entry still describes itself"
    assert len(difference.remaining) == 1, "step 2: the untouched class is remaining"
    assert len(difference.new) == 1, "step 2: only the genuinely new pair counts as new"
    assert "src/Epsilon.java" in {member.file_path for member in difference.new[0].members}, (
        "step 2: the new class is the planted pair"
    )

    # 2b. The grown class re-keyed but spans the old files, so it is CHANGED, not resolved-plus-new.
    assert len(difference.changed) == 1, "step 2b: the grown class is paired with its old entry"
    change = difference.changed[0]
    assert change.was.key != change.now.key, "step 2b: the key really did move"
    assert len(change.now.members) == 3 and change.score_delta > 0, "step 2b: the delta says it grew"

    # 3. The rendered diff names all four sections and stays a report.
    text = format_baseline_diff(after, baseline)
    assert "== RESOLVED" in text and "== CHANGED" in text, "step 3: resolved and changed sections"
    assert "== REMAINING" in text and "== NEW ==" in text, "step 3: remaining and new sections"
    assert "changed" in text and f"{change.was.key} -> {change.now.key}" in text, "step 3: the pairing is printed"
    unchanged = format_baseline_diff(before, baseline)
    assert unchanged.count("  none") == 3, "step 3: an empty section says so rather than vanishing"

    # 4. A suppressed class is neither new nor changed, and its re-keyed old entry is not resolved.
    _ignore_file(workspace, f"{change.now.key}  three copies of one fixture, on purpose for now")
    quiet = find_duplication(workspace, options)
    quiet_difference = diff_baseline(quiet, baseline)
    assert not quiet_difference.changed, "step 4: a suppressed class is not changed"
    assert not any(entry.key == change.now.key for entry in quiet_difference.resolved), "step 4: nor resolved as-is"
    assert not any(entry.key == change.was.key for entry in quiet_difference.resolved), (
        "step 4: the old entry it re-keyed from is not called resolved either: it is still there, on purpose"
    )

    # 5. A narrowed run never calls what it did not look for resolved.
    everything = find_duplication(LANES, DupeOptions(kinds=(CloneKind.EXACT,)))
    lanes_baseline = load_baseline(save_baseline(tmp_path / "lanes.json", everything))
    production_only = find_duplication(LANES, DupeOptions(kinds=(CloneKind.EXACT,), lane=Lane.PRODUCTION))
    assert diff_baseline(production_only, lanes_baseline).resolved == (), "step 5: a lane filter resolves nothing"
    other_kind = find_duplication(LANES, DupeOptions(kinds=(CloneKind.RENAMED,)))
    assert diff_baseline(other_kind, lanes_baseline).resolved == (), "step 5: a kind filter resolves nothing either"

    # 6. A baseline written by another version is refused loudly.
    broken = tmp_path / "broken.json"
    broken.write_text('{"version": 99, "classes": []}')
    with pytest.raises(ValueError):
        load_baseline(broken)


def test_brief_journey() -> None:
    """--brief is the pipe-friendly form: header plus one line per class, nothing else."""
    report = find_duplication(LANES, DupeOptions(kinds=(CloneKind.EXACT,)))
    brief = format_report(report, brief=True)
    lines = brief.rstrip("\n").split("\n")

    # 1. One header line, then exactly one line per class.
    assert len(lines) == 1 + len(report.classes), "step 1: nothing but the header and the class lines"
    assert lines[0].startswith("Analyzed "), "step 1: the header still leads"

    # 2. Every class line carries rank, kind, lane, size, score, root symbol, files and key.
    for index, line in enumerate(lines[1:], start=1):
        assert line.startswith(f"#{index}  "), "step 2: the classes are ranked"
        for part in ("copies x", "score ", "root: ", "files: ", "key: "):
            assert part in line, f"step 2: {part!r} is on the line"
    assert any(f" {lane.value} " in line for lane in Lane for line in lines[1:]), "step 2: the lane is named"

    # 3. No member locations and no reasons, which is what makes it grep-able.
    assert ".java:" not in brief, "step 3: member paths are left out"
    assert "reason:" not in brief, "step 3: reasons are left out"

    # 4. The full report still has them.
    assert ".java:" in format_report(report), "step 4: --brief is the only thing that drops them"


def _synthetic_unit(name: str, calls: tuple[str, ...], literals: tuple[str, ...] = ("1",)):
    """Build one in-memory unit for reason-aggregation tests."""
    from zemble.dedup.model import Unit

    return Unit(
        file_path=f"src/{name}.java",
        start_line=1,
        end_line=9,
        kind="method",
        name=f"{name}.apply",
        token_count=40,
        exact_hash=f"exact-{name}",
        renamed_hash=f"renamed-{name}",
        skeleton=("if", "return"),
        calls=calls,
        literals=literals,
    )


def test_reason_aggregation_journey() -> None:
    """Three or more copies get one consensus line plus outliers; a pair keeps its pair reason."""
    from zemble.dedup.model import CloneClass, PairReason

    alpha = _synthetic_unit("Alpha", ("applyAttribute", "setStringAttribute"))
    beta = _synthetic_unit("Beta", ("applyAttribute", "setStringAttribute", "toBooleanValue"))
    gamma = _synthetic_unit("Gamma", ("applyAttribute", "setStringAttribute"), literals=("1", "2"))
    reasons = (
        PairReason(left=alpha.location, right=beta.location, reason="pair reason one"),
        PairReason(left=alpha.location, right=gamma.location, reason="pair reason two"),
    )
    clone = CloneClass(kind=CloneKind.LOGIC, members=(alpha, beta, gamma), tokens=40, reasons=reasons)

    # 1. The wire form leads with one consensus line over the whole class.
    wire = clone.wire_reasons
    assert wire[0].startswith("3 copies;"), "step 1: the aggregate counts the copies"
    assert "control flow identical across all copies" in wire[0], "step 1: the flow consensus is stated"
    assert "all call {applyAttribute, setStringAttribute}" in wire[0], "step 1: the shared call set is stated"
    assert "literals differ per copy" in wire[0], "step 1: the literal spread is stated"

    # 2. Only the member that deviates from the consensus is named.
    assert wire[1] == "outlier Beta.apply also calls {toBooleanValue}", "step 2: the outlier and its extra calls"
    assert len(wire) == 2, "step 2: agreeing members are not repeated"

    # 3. A two-copy class keeps the pair format: the aggregate would say nothing more.
    pair = CloneClass(kind=CloneKind.LOGIC, members=(alpha, beta), tokens=40, reasons=(reasons[0],))
    assert pair.wire_reasons == ["pair reason one"], "step 3: a pair stays a pair"

    # 4. The text report prints the aggregate lines too.
    from zemble.dedup.report import _class_lines

    lines = _class_lines(1, clone)
    assert any("reason: 3 copies;" in line for line in lines), "step 4: the report prints the aggregate"
    assert any("outlier Beta.apply" in line for line in lines), "step 4: and the outlier"


_HOME_TOML = """
order = ["core", "app", "widget"]

[modules]
core = ["core/**"]
app = ["app/**"]
widget = ["widget/**"]

[[forbidden]]
from = "widget"
to = "app"
why = "widgets ship standalone"
"""

_JAVA_BODY = """package %s;

public class %s {
    public String weave(String input) {
        StringBuilder builder = new StringBuilder();
        builder.append("%s");
        builder.append(input.length());
        builder.append(input.trim());
        builder.append(input);
        return builder.toString();
    }
}
"""


def _plant_pair(
    workspace: Path,
    left: str,
    right: str,
    salt: str,
    *,
    body: str = _JAVA_BODY,
    folds: tuple[str, str] = ("src", "src"),
) -> None:
    """Write one identical body into two module directories, optionally into named source sets."""
    for module, name, fold in ((left, "One" + salt.title(), folds[0]), (right, "Two" + salt.title(), folds[1])):
        target = workspace / module / fold / f"{name}.java"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body % (module.replace("-", ""), name, salt))


def _verdict_for(report, module: str):
    """The home verdict of the class with a member under one module directory."""
    clone = next(c for c in report.classes if any(m.file_path.startswith(f"{module}/") for m in c.members))
    return clone, report.homes.get(clone.key)


def test_home_verdicts_journey(tmp_path: Path) -> None:
    """A clone class spanning declared modules carries a verdict driven by home.toml."""
    workspace = tmp_path / "workspace"
    _plant_pair(workspace, "core", "app", "shared")
    _plant_pair(workspace, "app", "widget", "leaky")
    _plant_pair(workspace, "stray-one", "stray-two", "orphan")
    options = DupeOptions(kinds=(CloneKind.EXACT,))

    # 1. Without a home.toml there are no verdicts and no noise.
    silent = find_duplication(workspace, options)
    assert len(silent.classes) == 3, "setup: three cross-module classes"
    assert silent.homes == {} and not silent.notes, "step 1: no config, no verdicts, no note"

    # 2. The most core member module that everyone may depend on is the candidate home.
    (workspace / ".zemble").mkdir()
    (workspace / ".zemble" / "home.toml").write_text(_HOME_TOML)
    report = find_duplication(workspace, options)
    _, candidate = _verdict_for(report, "core")
    assert candidate is not None and candidate.kind.value == "candidate-home", "step 2: core+app has a home"
    assert candidate.home == "core" and candidate.modules == ("core", "app"), "step 2: core is the home, rank order"

    # 3. A forbidden dependency is named, rule and all, instead of a naive extract-upward.
    _, forbidden = _verdict_for(report, "widget")
    assert forbidden is not None and forbidden.kind.value == "forbidden-dep", "step 3: widget must not depend on app"
    assert "widgets ship standalone" in forbidden.detail, "step 3: the rule's why is carried"
    assert "deeper than app" in forbidden.detail, "step 3: the direction of the fix is stated"

    # 4. Sibling directories outside the declared order have no shared ancestor.
    _, orphan = _verdict_for(report, "stray-one")
    assert orphan is not None and orphan.kind.value == "no-shared-ancestor", "step 4: undeclared siblings"

    # 5. The text and JSON forms carry the verdicts.
    text = format_report(report)
    assert "home: candidate home core" in text, "step 5: the report prints the verdict"
    payload = report_json(report)
    judged = [clone for clone in payload["classes"] if "home" in clone]
    assert {clone["home"]["verdict"] for clone in judged} == {
        "candidate-home",
        "forbidden-dep",
        "no-shared-ancestor",
    }, "step 5: every verdict survives the wire"

    # 6. A malformed home.toml is a note, never a crash: this is still a report.
    (workspace / ".zemble" / "home.toml").write_text("order = [")
    broken = find_duplication(workspace, options)
    assert broken.homes == {}, "step 6: no verdicts from a config that cannot be trusted"
    assert any("home.toml error" in note for note in broken.notes), "step 6: and the report says why"


def test_mcp_baseline_journey(tmp_path: Path) -> None:
    """Over MCP the baseline lives at a fixed path, so save and diff are two booleans."""
    import asyncio

    from zemble.index_cache import IndexCache
    from zemble.mcp import create_server

    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    source = (LANES / "src" / "main" / "java" / "fixtures" / "Alpha.java").read_text()
    (workspace / "src" / "Alpha.java").write_text(source)
    (workspace / "src" / "Beta.java").write_text(source.replace("class Alpha", "class Beta"))
    server = create_server(IndexCache())

    def call(**arguments):
        return asyncio.run(server.call_tool("dupes", {"repo": str(workspace), "kind": "exact", **arguments}))

    # 1. Asking for a diff before any baseline exists is answered, not raised.
    content, _ = call(baseline=True)
    assert "baseline: none at .zemble/dupes.baseline.json" in content[0].text, "step 1: the miss is named"
    assert content[0].text.startswith("Analyzed "), "step 1: the plain report still comes back"

    # 2. save_baseline writes the fixed per-workspace file.
    content, _ = call(save_baseline=True)
    assert (workspace / ".zemble" / "dupes.baseline.json").is_file(), "step 2: the baseline is on disk"
    assert "baseline: wrote" in content[0].text, "step 2: and the answer says so"

    # 3. The next run diffs against it.
    content, _ = call(baseline=True)
    assert "== RESOLVED" in content[0].text and "remaining" in content[0].text, "step 3: the diff form"

    # 4. The JSON form is the diff object, once-encoded.
    _, structured = call(baseline=True, format="json")
    payload = structured["result"]
    assert payload["counts"] == {"resolved": 0, "changed": 0, "remaining": 1, "new": 0}, "step 4: the buckets"


_TABLE_SECTION = """
[[tables]]
file = "ARCH.md"
capability = "Capability"
home = "Mechanism home"
consumers = "Consumers"
"""

_HOME_TOML_TABLE = _HOME_TOML + _TABLE_SECTION

#: The same three modules, with the dependency graph declared: app and widget both use
#: core, and neither of them may reach the other.
_DEPENDS_TOML = """
order = ["core", "app", "widget"]

[dependencies]
source = "declared"

[modules.core]
globs = ["core/**"]
depends_on = []

[modules.app]
globs = ["app/**"]
depends_on = ["core"]

[modules.widget]
globs = ["widget/**"]
depends_on = ["core"]
"""

_ARCH_HEAD = """# Architecture

| Capability | Mechanism home | Consumers |
| --- | --- | --- |
"""

_JAVA_PRIVATE_BODY = _JAVA_BODY.replace("public String weave", "private String weave")
#: The same body as an interface default method: implicitly public, with no `public` keyword.
_JAVA_INTERFACE_BODY = _JAVA_BODY.replace("public class %s {", "public interface %s {").replace(
    "public String weave", "default String weave"
)
_JAVA_PRIVATE_INTERFACE_BODY = _JAVA_INTERFACE_BODY.replace("default String weave", "private String weave")
#: A public method whose declaring class nothing outside the package can name.
_JAVA_PACKAGE_TYPE_BODY = _JAVA_BODY.replace("public class %s", "class %s")
#: An implicitly public static nested class inside an interface, holding a public method.
_JAVA_NESTED_BODY = """package %s;

public interface %s {
    class Weaving {
        public String weave(String input) {
            StringBuilder builder = new StringBuilder();
            builder.append("%s");
            builder.append(input.length());
            builder.append(input.trim());
            builder.append(input);
            return builder.toString();
        }
    }
}
"""
#: The Zig body every Zig home test clones, at file level and inside a struct. The top-level
#: copy is TitleCase because a declared-home row only reads a backticked name as a symbol when
#: it starts with a capital, and a file-level function has no type to hang the row off.
_ZIG_MEMBER = """pub fn %s(items: []const u32) u32 {
    var total: u32 = 0;
    var index: usize = 0;
    while (index < items.len) : (index += 1) {
        if (items[index] > 10) {
            total += items[index];
        }
    }
    return total;
}
"""
_ZIG_TOP_LEVEL = _ZIG_MEMBER % "Weave"
_ZIG_HIDDEN_CONTAINER = "const Hidden = struct {\n" + _ZIG_MEMBER % "weave" + "};\n"


def _declare(workspace: Path, capability: str, *, table: bool = True, toml: str = _HOME_TOML) -> None:
    """Write a home.toml, and the one-row declared-home table it points at."""
    (workspace / ".zemble").mkdir(parents=True, exist_ok=True)
    (workspace / ".zemble" / "home.toml").write_text(toml + _TABLE_SECTION if table else toml)
    if table:
        (workspace / "ARCH.md").write_text(f"{_ARCH_HEAD}| {capability} | `core` | `app` |\n")


def _existing_home_workspace(
    tmp_path: Path,
    capability: str,
    *,
    table: bool = True,
    toml: str = _HOME_TOML,
    body: str = _JAVA_BODY,
    folds: tuple[str, str] = ("src", "src"),
) -> Path:
    """A core+app clone whose core copy the table may or may not declare."""
    workspace = tmp_path / "workspace"
    _plant_pair(workspace, "core", "app", "shared", body=body, folds=folds)
    _declare(workspace, capability, table=table, toml=toml)
    return workspace


def test_existing_reusable_api_from_declared_row(tmp_path: Path) -> None:
    """A declared PUBLIC member every copy's module may reach is the one verdict that says call it."""
    capability = "Weaving a string (`OneShared.weave`)"
    workspace = _existing_home_workspace(tmp_path, capability, toml=_DEPENDS_TOML)
    options = DupeOptions(kinds=(CloneKind.EXACT,))

    # 1. The verdict names the mechanism, not just the module.
    report = find_duplication(workspace, options)
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "existing-reusable-api", "step 1: declared, public, reachable"
    assert verdict.home == "core" and verdict.symbol == "OneShared.weave", "step 1: the symbol is the core body"
    assert verdict.location == "core/src/OneShared.java:4", "step 1: the location is file:start_line"

    # 2. The evidence is the row, the visibility and the dependency proof, each kind-tagged.
    assert [item.kind.value for item in verdict.evidence] == ["declared-member", "visibility", "dependency"], (
        "step 2: three proofs, in the order the report prints them"
    )
    assert verdict.evidence[0].capability == capability, "step 2: the row's capability is kept whole"
    assert verdict.evidence[0].file == "ARCH.md" and verdict.evidence[0].line == 5, "step 2: and where it stands"
    assert verdict.evidence[1].text == "public member on a public type", "step 2: both levels are proven"
    assert verdict.evidence[2].text == "every copy's module reaches core (app: direct)", "step 2: named reachability"
    assert hash(verdict), "step 2: the verdict stays hashable, so evidence is not a bare dict"

    # 3. Text and JSON both carry it.
    text = format_report(report)
    assert "    home: existing reusable API core: OneShared.weave" in text, "step 3: the head line names it"
    assert "          declared by ARCH.md: Weaving a string (row names OneShared.weave)" in text, (
        "step 3: the evidence line carries the row's TITLE and the name that matched"
    )
    assert capability not in text, "step 3: the full capability cell stays out of the text report"
    assert "          downstream copies should call or extend it" in text, "step 3: and what to do about it"
    payload = report_json(report)
    judged = next(clone["home"] for clone in payload["classes"] if "home" in clone)
    assert judged["kind"] == judged["verdict"] == "existing-reusable-api", "step 3: the wire agrees, under both names"
    assert judged["symbol"] == "OneShared.weave" and judged["location"] == verdict.location, "step 3: symbol on wire"
    assert judged["evidence"] == [item.to_dict() for item in verdict.evidence], "step 3: evidence on the wire"
    assert set(judged["evidence"][0]) == {"kind", "text", "capability", "file", "line"}, "step 3: declared-row shape"
    assert set(judged["evidence"][1]) == {"kind", "text"}, "step 3: every other evidence item is kind plus line"
    assert judged["lines"] == verdict.describe_lines(), "step 3: the rendered lines ship too"


def test_private_declared_member_is_not_a_reusable_api(tmp_path: Path) -> None:
    """A declared mechanism nothing outside its class can call is not something to call."""
    workspace = _existing_home_workspace(
        tmp_path, "Weaving a string (`OneShared.weave`)", toml=_DEPENDS_TOML, body=_JAVA_PRIVATE_BODY
    )
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "existing-implementation-not-api", "private is not an API"
    assert verdict.symbol == "OneShared.weave", "the mechanism is still named"
    kinds = [item.kind.value for item in verdict.evidence]
    assert kinds == ["declared-member", "visibility"], "the row and the visibility that blocks it"
    assert verdict.evidence[1].text == "member is private", "the level, and which of the two carries it"
    text = format_report(report)
    assert "home: existing implementation core: OneShared.weave (not a reusable API)" in text, "the head line says so"
    assert "expose it or extract the generic mechanism; do not call it as is" in text, "and what to do instead"
    assert "call or extend" not in text, "a private member is never something to call"


def test_an_interface_method_is_public_without_saying_so(tmp_path: Path) -> None:
    """A declared default method carries no `public` keyword and is a reusable API all the same."""
    workspace = _existing_home_workspace(
        tmp_path, "Weaving a string (`OneShared.weave`)", toml=_DEPENDS_TOML, body=_JAVA_INTERFACE_BODY
    )
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "existing-reusable-api", "an interface member is public"
    assert verdict.evidence[1].text == "public member on a public type", "and the evidence says both levels are"
    assert "downstream copies should call or extend it" in format_report(report), "so it is something to call"


def test_a_private_interface_method_is_not_an_api(tmp_path: Path) -> None:
    """Java 9's private interface method is the one interface member the implicit rule must not claim."""
    workspace = _existing_home_workspace(
        tmp_path, "Weaving a string (`OneShared.weave`)", toml=_DEPENDS_TOML, body=_JAVA_PRIVATE_INTERFACE_BODY
    )
    _, verdict = _verdict_for(find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,))), "core")
    assert verdict is not None and verdict.kind.value == "existing-implementation-not-api", "explicit beats implicit"
    assert verdict.evidence[1].text == "member is private", "and the member's own level is what blocks it"


def test_a_package_private_declaring_type_blocks_a_public_member(tmp_path: Path) -> None:
    """A public method is only reachable if its class is: the declaring type is checked too."""
    workspace = _existing_home_workspace(
        tmp_path, "Weaving a string (`OneShared.weave`)", toml=_DEPENDS_TOML, body=_JAVA_PACKAGE_TYPE_BODY
    )
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "existing-implementation-not-api", "the type blocks it"
    assert verdict.evidence[1].text == "declaring type OneShared is package-private", "and it is named, with its level"
    assert "call or extend" not in format_report(report), "an unreachable type is never something to call"


def test_a_nested_type_in_an_interface_stays_public(tmp_path: Path) -> None:
    """A nested class of an interface is implicitly public static, so its public method is reusable."""
    workspace = _existing_home_workspace(
        tmp_path, "Weaving a string (`Weaving.weave`)", toml=_DEPENDS_TOML, body=_JAVA_NESTED_BODY
    )
    _, verdict = _verdict_for(find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,))), "core")
    assert verdict is not None and verdict.kind.value == "existing-reusable-api", "the nesting keeps it public"
    assert verdict.symbol == "OneShared.Weaving.weave", "and the whole nested path is the mechanism"


def _zig_workspace(tmp_path: Path, source: str, capability: str) -> Path:
    """A core+app Zig clone under the declared dependency graph, with a row naming its member."""
    workspace = tmp_path / "workspace"
    for module in ("core", "app"):
        target = workspace / module / "src" / "weave.zig"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    _declare(workspace, capability, toml=_DEPENDS_TOML)
    return workspace


def test_a_zig_pub_member_of_a_private_container_is_not_an_api(tmp_path: Path) -> None:
    """`pub fn` inside a file-private struct is only public within the file that declares it."""
    workspace = _zig_workspace(tmp_path, _ZIG_HIDDEN_CONTAINER, "Weaving (`Hidden.weave`)")
    _, verdict = _verdict_for(find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,))), "core")
    assert verdict is not None and verdict.kind.value == "existing-implementation-not-api", "the container blocks it"
    assert verdict.evidence[1].text == "declaring type Hidden is private", "and the container is named"


def test_a_zig_top_level_pub_member_is_a_reusable_api(tmp_path: Path) -> None:
    """A `pub fn` of the file struct itself has no container to hide it."""
    workspace = _zig_workspace(tmp_path, _ZIG_TOP_LEVEL, "Weaving (`Weave`)")
    _, verdict = _verdict_for(find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,))), "core")
    assert verdict is not None and verdict.kind.value == "existing-reusable-api", "nothing restricts it"
    assert verdict.evidence[1].text == "public member on a public type", "the file itself is the public container"


def test_unknown_dependencies_cap_a_declared_member(tmp_path: Path) -> None:
    """Order is not permission: with no dependency graph, may-call-it is never proven."""
    workspace = _existing_home_workspace(tmp_path, "Weaving a string (`OneShared.weave`)")
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "existing-implementation-not-api", "capped, not promoted"
    assert [item.kind.value for item in verdict.evidence] == ["declared-member", "dependency"], "the row and the gap"
    assert "dependency reachability unknown" in verdict.evidence[1].text, "and the gap is named as unknown"
    assert "call or extend" not in format_report(report), "an unproven dependency never says call it"


def test_incompatible_source_sets_block_a_declared_member(tmp_path: Path) -> None:
    """A server-fold mechanism is unreachable from a common-fold copy, however close the modules are."""
    workspace = _existing_home_workspace(
        tmp_path,
        "Weaving a string (`OneShared.weave`)",
        toml=_DEPENDS_TOML,
        folds=("src/server/java", "src/common/java"),
    )
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "existing-implementation-not-api", "the fold blocks reuse"
    folds = [item for item in verdict.evidence if item.kind.value == "source-set"]
    assert folds and folds[0].text == "app common cannot use core server", "the incompatible pair is named"


def test_bare_class_row_is_never_a_declaration(tmp_path: Path) -> None:
    """A row naming the type says the row is ABOUT that class, not that this member is its API."""
    workspace = _existing_home_workspace(tmp_path, "Weaving a string (`OneShared`)", toml=_DEPENDS_TOML)
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "candidate-home", "a bare type declares no member"
    assert verdict.symbol == "OneShared.weave", "the lexically related member is still named"
    assert [item.kind.value for item in verdict.evidence] == ["declared-type", "dependency"], "tagged as a type row"
    text = format_report(report)
    assert "home: candidate home core: OneShared.weave" in text, "the head line stays a candidate"
    assert "(names the type, not this member)" in text, "and says exactly what the row proved"
    assert "call or extend" not in text, "a bare row never authorises calling anything"
    assert "no declared member; review before consolidating" in text, "the action is review, not reuse"


def test_siblings_need_a_common_home(tmp_path: Path) -> None:
    """Two modules that cannot reach each other need a third one, not one of themselves."""
    workspace = tmp_path / "workspace"
    _plant_pair(workspace, "app", "widget", "leaky")
    _declare(workspace, "unused", table=False, toml=_DEPENDS_TOML)
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "widget")
    assert verdict is not None and verdict.kind.value == "siblings-need-common-home", "neither may depend on the other"
    assert verdict.suggested_home == "core", "the module both reach is where it belongs"
    assert verdict.symbol == "weave", "the shared member name is what they both have"
    text = format_report(report)
    assert "home: siblings app, widget: weave" in text, "the head line names both siblings"
    assert "no dependency path either way" in text, "the evidence is the missing path"
    assert "shared mechanism belongs in core" in text, "and the answer is a third module"


def test_logic_clones_are_a_review_lead_not_a_duplicate(tmp_path: Path) -> None:
    """Structural similarity is not equivalence, so a logic clone never says call it."""
    workspace = tmp_path / "workspace"
    for module, name in (("core", "LogicA"), ("app", "LogicB")):
        target = workspace / module / "src" / f"{name}.java"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((FIXTURES / "src" / f"{name}.java").read_text())
    _declare(workspace, "Rendering an order (`LogicA.render`)", toml=_DEPENDS_TOML)
    report = find_duplication(
        workspace,
        DupeOptions(kinds=(CloneKind.LOGIC,), windows=False, logic_threshold=0.5),
        embedder=BagOfWordsEmbedder(),
    )
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "review-required", "a logic clone is never stronger"
    assert verdict.home == "core" and verdict.symbol == "LogicA.render", "the declared copy is the best evidence"
    assert [item.kind.value for item in verdict.evidence] == ["declared-member", "clone-kind"], "the clone kind shows"
    text = format_report(report)
    assert "home: possible existing mechanism core: LogicA.render (logic clone)" in text, "the head line hedges"
    assert "semantic review required; structural similarity is not equivalence" in text, "and says why"
    assert "call or extend" not in text, "a logic clone authorises nothing"


@pytest.mark.parametrize("kind", list(HomeVerdictKind))
def test_every_verdict_kind_renders_and_ships(kind: HomeVerdictKind) -> None:
    """The verdict vocabulary has one home: every member renders and reaches the wire."""
    verdict = HomeVerdict(
        kind,
        ("core", "app"),
        "core",
        "why this happened",
        symbol="OneShared.weave",
        location="core/src/OneShared.java:4",
        evidence=(Evidence(EvidenceKind.DEPENDENCY, "every copy's module reaches core (app: direct)"),),
        suggested_home="core",
    )
    lines = verdict.describe_lines()
    assert lines and all(line.strip() for line in lines), f"{kind.value} renders real lines"
    payload = verdict.to_dict()
    assert set(payload) == {
        "verdict",
        "kind",
        "modules",
        "home",
        "detail",
        "symbol",
        "location",
        "suggested_home",
        "evidence",
        "lines",
    }, f"{kind.value} carries the whole wire shape"
    assert payload["kind"] == payload["verdict"] == kind.value, f"{kind.value} ships under both names"
    assert ("downstream copies should call or extend it" in lines) is (kind is HomeVerdictKind.EXISTING_REUSABLE_API), (
        "only a proven reusable API is something to call"
    )


#: One whole-body member per language, public and inside a public container, for the drift test.
_VISIBILITY_FIXTURES: dict[str, tuple[str, str]] = {
    "java": (
        "Vis.java",
        "public class Vis {\n"
        "    public String weave(String input) {\n"
        "        StringBuilder builder = new StringBuilder();\n"
        "        builder.append(input.length());\n"
        "        return builder.toString();\n"
        "    }\n"
        "}\n",
    ),
    "zig": (
        "vis.zig",
        "pub fn sumAll(values: []const u32) u32 {\n"
        "    var total: u32 = 0;\n"
        "    for (values) |value| { total += value; }\n"
        "    return total;\n"
        "}\n",
    ),
}


@pytest.mark.parametrize("profile", sorted(set(PROFILES.values()), key=lambda p: p.name), ids=lambda p: p.name)
def test_every_language_places_a_whole_body_member(profile) -> None:
    """A language whose profile cannot place a member fails closed, and adding one must not be forgotten."""
    from zemble.dedup.units import extract_units

    assert profile.name in _VISIBILITY_FIXTURES, f"{profile.name} has no visibility fixture"
    name, source = _VISIBILITY_FIXTURES[profile.name]
    units = extract_units(source.encode(), f"src/{name}", windows=False, min_tokens=5)
    bodies = [unit for unit in units if unit.is_body]
    assert bodies, f"{profile.name}: the fixture must produce a whole-body unit"
    for unit in bodies:
        assert unit.visibility is Visibility.PUBLIC, f"{profile.name}: a public member is placed as public"
        assert unit.container_visibility is Visibility.PUBLIC, f"{profile.name}: so is its public container"


@pytest.mark.parametrize("level", list(Visibility), ids=lambda level: level.value)
def test_the_reusability_step_handles_every_visibility(level: Visibility) -> None:
    """Every level is judged, only PUBLIC is reusable, and the evidence names the level and its owner."""
    member = replace(_synthetic_unit("Alpha", ()), visibility=level, container_visibility=Visibility.PUBLIC)
    owner = replace(member, visibility=Visibility.PUBLIC, container_visibility=level)
    reusable = level is Visibility.PUBLIC
    assert (member.visibility.is_public and member.container_visibility.is_public) is reusable, "only public reuses"
    assert visibility_evidence(member).kind is EvidenceKind.VISIBILITY, "the evidence is tagged as visibility"
    if reusable:
        assert visibility_evidence(member).text == "public member on a public type", "nothing blocks it"
        return
    assert visibility_evidence(member).text == level.phrase("member"), "the member is named before its type"
    assert visibility_evidence(owner).text == level.phrase("declaring type Alpha"), "the type is named too"
    assert ("visibility unknown" in visibility_evidence(member).text) is (level is Visibility.UNKNOWN), (
        "an unplaceable level keeps the wording that says so"
    )


def test_candidate_home_without_a_declared_row(tmp_path: Path) -> None:
    """Being the most core member module alone never claims a mechanism exists."""
    workspace = _existing_home_workspace(tmp_path, "unused", table=False, toml=_DEPENDS_TOML)
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "candidate-home", "no table, no existing home"
    assert verdict.symbol is None, "and no mechanism is named"
    assert [item.kind.value for item in verdict.evidence] == ["dependency"], "only the topology is proven"


def test_declared_row_near_miss_stays_candidate(tmp_path: Path) -> None:
    """A row naming a different type's member of the same name does not claim the copy."""
    workspace = _existing_home_workspace(tmp_path, "Weaving a string (`Other.weave`)")
    _, verdict = _verdict_for(find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,))), "core")
    assert verdict is not None and verdict.kind.value == "candidate-home", "symbol matching fails closed"


def test_forbidden_dep_outranks_every_declaration(tmp_path: Path) -> None:
    """A forbidden dependency is still the answer, however well declared the core copy is."""
    workspace = tmp_path / "workspace"
    _plant_pair(workspace, "app", "widget", "leaky")
    (workspace / ".zemble").mkdir(parents=True)
    (workspace / ".zemble" / "home.toml").write_text(_HOME_TOML_TABLE)
    (workspace / "ARCH.md").write_text(f"{_ARCH_HEAD}| Weaving a string (`OneLeaky.weave`) | `app` | `widget` |\n")
    _, verdict = _verdict_for(find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,))), "widget")
    assert verdict is not None and verdict.kind.value == "forbidden-dep", "the rule outranks the declaration"
    assert "widgets ship standalone" in verdict.detail, "and the rule text is still the detail"
    assert verdict.symbol is None and not verdict.evidence, "a refused home names no mechanism"


def test_no_shared_ancestor_survives_the_table(tmp_path: Path) -> None:
    """Undeclared siblings stay undeclared siblings when a table is present."""
    workspace = tmp_path / "workspace"
    _plant_pair(workspace, "stray-one", "stray-two", "orphan")
    _declare(workspace, "Weaving a string (`OneOrphan.weave`)")
    _, verdict = _verdict_for(find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,))), "stray-one")
    assert verdict is not None and verdict.kind.value == "no-shared-ancestor", "no member is in the declared order"
    assert "not declared in home.toml" in verdict.detail, "and the verdict says which modules it could not place"


def test_one_undeclared_member_is_no_shared_ancestor(tmp_path: Path) -> None:
    """A copy outside the declared architecture cannot be judged by it, whatever its neighbour is."""
    workspace = tmp_path / "workspace"
    _plant_pair(workspace, "core", "stray-one", "mixed")
    _declare(workspace, "unused", table=False, toml=_DEPENDS_TOML)
    _, verdict = _verdict_for(find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,))), "core")
    assert verdict is not None and verdict.kind.value == "no-shared-ancestor", "one stray module is enough"
    assert "stray-one is not declared" in verdict.detail, "and it is named"


def test_missing_table_file_degrades_to_a_note(tmp_path: Path) -> None:
    """A declared table that is not there loses the declared lane and nothing else."""
    workspace = _existing_home_workspace(tmp_path, "Weaving a string (`OneShared.weave`)")
    (workspace / "ARCH.md").unlink()
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "candidate-home", "no evidence, no promotion"
    assert any("ARCH.md could not be read" in note for note in report.notes), "the report says the table is missing"


def test_malformed_table_file_degrades_to_candidate(tmp_path: Path) -> None:
    """A table file with no recognisable header is simply no evidence."""
    workspace = _existing_home_workspace(tmp_path, "Weaving a string (`OneShared.weave`)")
    (workspace / "ARCH.md").write_text("| nonsense |\n| ---- |\n| `OneShared.weave` |\n")
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "candidate-home", "an unparseable table declares nothing"
    assert not report.notes, "a readable file is not a missing one, so there is nothing to report"


def test_declared_rows_do_not_move_any_key(tmp_path: Path) -> None:
    """Clone, baseline and ignore keys are content-only: a table cannot move them."""
    options = DupeOptions(kinds=(CloneKind.EXACT,))
    plain = _existing_home_workspace(tmp_path / "a", "unused", table=False)
    declared = _existing_home_workspace(tmp_path / "b", "Weave (`OneShared.weave`)")
    without = find_duplication(plain, options)
    with_table = find_duplication(declared, options)
    assert [clone.key for clone in without.classes] == [clone.key for clone in with_table.classes], "same clone keys"
    save_baseline(tmp_path / "a.json", without)
    save_baseline(tmp_path / "b.json", with_table)
    assert load_baseline(tmp_path / "a.json").keys == load_baseline(tmp_path / "b.json").keys, "same baseline keys"

    # An ignore entry written against the tableless run still suppresses the declared one.
    key = without.classes[0].key
    (declared / ".zemble" / "dupes.ignore").write_text(f"{key}  # declared on purpose\n")
    suppressed = find_duplication(declared, options)
    assert not suppressed.classes and len(suppressed.suppressed) == 1, "the ignore key did not move either"
    assert not suppressed.ignore_problems, "and it is not stale"


def test_every_verdict_ships_the_same_wire_shape(tmp_path: Path) -> None:
    """One renderer, one shape: every verdict on the wire carries the same keys."""
    workspace = tmp_path / "workspace"
    _plant_pair(workspace, "core", "app", "shared")
    _plant_pair(workspace, "app", "widget", "leaky")
    _plant_pair(workspace, "stray-one", "stray-two", "orphan")
    _declare(workspace, "Weaving a string (`Nothing.matches`)")
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    payload = report_json(report)
    seen = {clone["home"]["kind"]: set(clone["home"]) for clone in payload["classes"] if "home" in clone}
    assert set(seen) == {"candidate-home", "forbidden-dep", "no-shared-ancestor"}, "no row matches, so no promotion"
    for keys in seen.values():
        assert keys == {
            "verdict",
            "kind",
            "modules",
            "home",
            "detail",
            "symbol",
            "location",
            "suggested_home",
            "evidence",
            "lines",
        }, "every verdict carries the whole shape, filled or null"


def test_existing_home_never_indexes_or_embeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Classification is declarations, the parse and the dependency graph: never search or embeddings."""
    import zemble.embedding.registry
    import zemble.index.create
    import zemble.search

    def explode(*args, **kwargs):
        raise AssertionError("dupes must not index, embed or search to judge a home")

    monkeypatch.setattr(zemble.embedding.registry, "load_embedder", explode)
    monkeypatch.setattr(zemble.embedding.registry, "build_embedder", explode)
    monkeypatch.setattr(zemble.index.create, "create_index_from_path", explode)
    monkeypatch.setattr(zemble.search, "search", explode)

    workspace = _existing_home_workspace(tmp_path, "Weaving a string (`OneShared.weave`)", toml=_DEPENDS_TOML)
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.kind.value == "existing-reusable-api", "reached without retrieval"


_JAVA_INITIALIZER = """package %s;

public class %s {
    static final StringBuilder BUILDER = new StringBuilder();

    static {
        BUILDER.append("one");
        BUILDER.append("two");
        BUILDER.append("three");
        BUILDER.append("four");
        BUILDER.append("five");
    }
}
"""


def test_initializers_are_never_an_existing_home(tmp_path: Path) -> None:
    """A row naming the class does not promote its `<initializer>`: nothing can call one."""
    workspace = tmp_path / "workspace"
    for module, name in (("core", "OneInit"), ("app", "TwoInit")):
        target = workspace / module / "src" / f"{name}.java"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_JAVA_INITIALIZER % (module, name))
    _declare(workspace, "Shared builder priming (`OneInit`)", toml=_DEPENDS_TOML)
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    clone, verdict = _verdict_for(report, "core")
    assert any(member.name.endswith(".<initializer>") for member in clone.members), (
        "setup: the copies are static blocks"
    )
    assert verdict is not None and verdict.kind.value == "candidate-home", "an initializer is not a mechanism"
    assert verdict.symbol is None, "and nothing was named"


def test_no_cross_module_class_reads_no_table(tmp_path: Path) -> None:
    """Rows load on the first spanning class, so a same-module scan cannot report a missing table."""
    workspace = tmp_path / "workspace"
    _plant_pair(workspace, "core", "core", "inner")
    _declare(workspace, "Weaving a string (`OneInner.weave`)")
    (workspace / "ARCH.md").unlink()
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    assert report.classes, "setup: there is still a clone class, it just does not span modules"
    assert report.homes == {}, "one module is no cross-module verdict"
    assert not report.notes, "and no table was ever looked for, so there is nothing to note"


def test_long_capability_titles_are_truncated_in_the_text(tmp_path: Path) -> None:
    """A table cell is prose and can run for hundreds of characters; the report line cannot."""
    long_title = "Priming a shared string builder from every module that needs one " * 3
    workspace = _existing_home_workspace(tmp_path, f"{long_title.strip()} (`OneShared.weave`)", toml=_DEPENDS_TOML)
    report = find_duplication(workspace, DupeOptions(kinds=(CloneKind.EXACT,)))
    line = next(line for line in format_report(report).splitlines() if "declared by ARCH.md:" in line)
    title = line.strip().removeprefix("declared by ARCH.md: ").split(" (row names ")[0]
    assert title.endswith("...") and len(title) <= 100, "the title is cut off at a readable length"
    _, verdict = _verdict_for(report, "core")
    assert verdict is not None and verdict.evidence[0].capability.startswith(long_title.strip()), (
        "the whole capability survives in data"
    )
