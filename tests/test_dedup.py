"""Behaviour journeys over duplication detection."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.model import CloneKind
from zemble.dedup.report import format_report, report_json
from zemble.dedup.structure import check_pair, edit_distance, jaccard
from zemble.dedup.units import extract_units

FIXTURES = Path(__file__).parent / "fixtures" / "dedup"
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
    assert any("control flow identical" in reason for reason in pair.reasons), "step 2: the reason names control flow"
    assert any("shared" in reason for reason in pair.reasons), "step 2: the reason names the call overlap"

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
    assert "== EXACT ==" in text and "copies x" in text, "step 1: the sections mirror zenit-dev duplication"

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

    # 3. It answers with the same JSON the CLI prints.
    payload = json.loads(asyncio.run(server.call_tool("dupes", {"repo": str(FIXTURES), "kind": "exact"}))[1]["result"])
    assert payload["class_counts"]["exact"] == 2, "step 3: the tool returns the ranked classes"
