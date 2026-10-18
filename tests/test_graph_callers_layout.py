"""A callers answer groups its call sites by file and caller, shows the lines of the first few, and keeps the total."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zemble.graph.mcp import SHOWN_SITES, answer, callers_parts, callers_text, graph_parts


def _hit(path: str, line: int, caller: str, resolution: str = "exact", grade: str = "exact match") -> dict[str, Any]:
    """One callers hit as the wire carries it."""
    return {
        "qualified_name": f"pkg.{caller}",
        "display": caller,
        "kind": "method",
        "file_path": path,
        "line": line,
        "edge_kind": "calls",
        "resolution": resolution,
        "grade": grade,
        "depth": 1,
        "source": "tree-sitter",
        "reason": f"called from {caller} (line {line}, {grade})",
    }


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Three files long enough for every line the payloads point at."""
    for name in ("A.java", "B.java", "C.java", "T.java"):
        (tmp_path / name).write_text("".join(f"line {number}\n" for number in range(1, 60)), encoding="utf-8")
    return tmp_path


def _payload(results: list[dict[str, Any]], total: int | None = None) -> dict[str, Any]:
    """A callers payload around these hits, capped the way `answer` caps one."""
    payload: dict[str, Any] = {
        "symbol": {"kind": "method", "qualified_name": "pkg.T.target", "file_path": "T.java", "line": 3},
        "display": "T.target",
        "results": results,
        "total": total if total is not None else len(results),
    }
    if total is not None and total > len(results):
        payload["truncated"] = f"showing {len(results)} of {total}; raise `limit` to see the rest"
    return payload


def test_sites_are_grouped_by_file_and_caller_with_one_block_of_lines_per_file(tree: Path) -> None:
    """Each file is one prose line naming its callers, then its lines with no prose between them."""
    results = [
        _hit("A.java", 10, "A.one"),
        _hit("A.java", 20, "A.one"),
        _hit("B.java", 5, "B.two", "ambiguous", "ambiguous, 2 candidates"),
        _hit("A.java", 30, "A.three"),
    ]
    parts = callers_parts(str(tree), _payload(results))
    assert parts is not None
    texts = [part["text"] for part in parts if "text" in part]
    assert texts[0] == "T.target: method pkg.T.target"
    assert texts[1] == "4 call site(s) from 3 caller(s) in 2 file(s) (exact 3, ambiguous 1)"
    assert texts[2] == "A.java: A.one L10, L20; A.three L30"
    assert texts[3] == "B.java: B.two L5 (ambiguous, 2 candidates)"
    assert texts[-1] == "total: 4"
    a_lines = [part["start"] for part in parts if part.get("path") == str(tree / "A.java")]
    assert a_lines == [10, 20, 30], "one file's lines follow each other, in line order"
    first_a = next(index for index, part in enumerate(parts) if part.get("path") == str(tree / "A.java"))
    assert all("path" in part for part in parts[first_a : first_a + 3]), "with no prose between them"


def test_only_the_first_sites_show_lines_and_the_rest_stay_named(tree: Path) -> None:
    """Past SHOWN_SITES a site is named by file, caller and line; the total and the cap stay exact."""
    results = [_hit("A.java", line, "A.many") for line in range(1, SHOWN_SITES + 1)]
    results += [_hit("C.java", 40, "C.late"), _hit("C.java", 41, "C.late")]
    parts = callers_parts(str(tree), _payload(results, total=137))
    assert parts is not None
    located = [part for part in parts if "path" in part and part["path"] != str(tree / "T.java")]
    assert len(located) == SHOWN_SITES, "only the first sites print their lines"
    texts = [part["text"] for part in parts if "text" in part]
    assert f"the lines of the first {SHOWN_SITES}" in texts[1]
    assert "2 more call site(s), lines not shown:\nC.java: C.late L40, L41" in texts
    assert texts[-1] == "total: 137\nshowing 12 of 137; raise `limit` to see the rest", "the cap is never silent"


def test_plain_text_names_every_listed_site(tree: Path) -> None:
    """A client without nib reads the same grouping as text, every listed site included."""
    results = [_hit("A.java", line, "A.many") for line in range(1, SHOWN_SITES + 3)]
    text = callers_text(_payload(results))
    assert text.splitlines()[0] == "T.target: method pkg.T.target  T.java:3"
    assert "A.java: A.many L1, L2, L3" in text and f"L{SHOWN_SITES + 2}" in text
    assert text.endswith(f"total: {SHOWN_SITES + 2}")
    refusal = {"error": "No symbol named 'x'."}
    assert callers_text(refusal) is refusal, "a refusal stays the payload it was"


def test_an_older_daemon_payload_lays_out_from_resolution_and_qualified_name(tree: Path) -> None:
    """Without `grade` and `display`, a non-exact site is marked by its resolution and named in full."""
    old = _hit("B.java", 7, "B.old", "unique_name")
    del old["grade"], old["display"]
    parts = callers_parts(str(tree), _payload([old]))
    assert parts is not None
    assert "B.java: pkg.B.old L7 (unique_name)" in [part.get("text") for part in parts]


def test_a_tool_written_exact_edge_is_counted_once_not_noted_per_site(tree: Path) -> None:
    """An exact edge a tool wrote carries no per-site note; the heading counts the tool; a `via` grade stays."""
    javac = _hit("A.java", 3, "A.one", grade="zemble-javac-facts")
    javac["source"] = "zemble-javac-facts"
    mapped = _hit("A.java", 4, "A.two", grade="exact match via render(Ctx)")
    text = callers_text(_payload([javac, mapped]))
    assert "2 call site(s) from 2 caller(s) in 1 file(s) (1 from zemble-javac-facts)" in text
    assert "A.java: A.one L3; A.two L4 (exact match via render(Ctx))" in text


def test_an_ambiguous_name_lists_its_candidates_as_lines() -> None:
    """A refusal with candidates is prose, one line per candidate, instead of a JSON dump."""
    refusal = {
        "error": "'Len' is ambiguous; pass a qualified name.",
        "candidates": [
            {"kind": "function", "qualified_name": "js.Len", "file_path": "a/js.go", "line": 29},
            {"kind": "function", "qualified_name": "js.Len", "file_path": "b/js.go", "line": 29},
        ],
    }
    for build in (graph_parts, callers_parts):
        assert build("/root", refusal) == [
            {
                "text": "'Len' is ambiguous; pass a qualified name.\n2 candidate(s):\n"
                "  function js.Len  a/js.go:29\n  function js.Len  b/js.go:29"
            }
        ]


def test_real_callers_carry_grade_and_display(built_graph: Any, graph_fixture_root: Path) -> None:
    """Over the fixture graph each hit names its caller shortly and words its resolution."""
    payload = answer(str(graph_fixture_root), "Helpers.twice", "callers")
    assert payload["results"], "the fixture calls Helpers.twice"
    for hit in payload["results"]:
        assert hit["display"] and "." in hit["display"]
        assert hit["grade"] and hit["grade"] in hit["reason"], "the grade is the reason's own wording"
