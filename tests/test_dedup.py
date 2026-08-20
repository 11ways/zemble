"""Behaviour journeys over duplication detection."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from zemble.dedup.baseline import diff_baseline, load_baseline, save_baseline
from zemble.dedup.detect import DupeOptions, find_duplication
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
    assert grown.classes[0].key != key, "step 3: a third copy is a new class"
    assert len(grown.classes[0].members) == 3, "step 3: the copy did join the class"


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

    # 2. Fixing one duplication and adding another gives resolved, remaining and new.
    (workspace / "src" / "Beta.java").write_text("package fixtures;\n\npublic class Beta {\n}\n")
    (workspace / "src" / "Delta.java").write_text(setup.replace("class AlphaTest", "class Delta"))
    after = find_duplication(workspace, options)
    difference = diff_baseline(after, baseline)
    resolved = {entry.members[0] for entry in difference.resolved}
    assert "src/Alpha.java:6-15" in resolved, "step 2: the class whose copy was deleted is resolved"
    assert difference.resolved[0].kind == "exact", "step 2: a resolved entry still describes itself"
    assert len(difference.remaining) == 1, "step 2: the untouched class is remaining"
    assert len(difference.new) == 1 and len(difference.new[0].members) == 3, "step 2: the grown class counts as new"

    # 3. The rendered diff names all three sections and stays a report.
    text = format_baseline_diff(after, baseline)
    assert "== RESOLVED" in text and "== REMAINING" in text and "== NEW ==" in text, "step 3: three sections"
    assert "resolved" in text and "remaining" in text, "step 3: the counts lead the diff"
    unchanged = format_baseline_diff(before, baseline)
    assert unchanged.count("  none") == 2, "step 3: an empty section says so rather than vanishing"

    # 4. A suppressed class is neither new nor remaining, and is not called resolved either.
    _ignore_file(workspace, f"{difference.new[0].key}  three copies of one fixture, on purpose for now")
    quiet = find_duplication(workspace, options)
    quiet_difference = diff_baseline(quiet, baseline)
    assert not quiet_difference.new, "step 4: a suppressed class is not new"
    assert not any(entry.key == difference.new[0].key for entry in quiet_difference.resolved), "step 4: nor resolved"

    # 5. A baseline written by another version is refused loudly.
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
