"""The default outline lists the API a file offers, counts what it leaves out, and locates each member's naming line."""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from zemble.evidence.answers import outline_payload
from zemble.evidence.mcp import declaration_line, outline_parts
from zemble.evidence.outline import outline
from zemble.graph import SqliteGraphProvider, build_graph

_JAVA = textwrap.dedent(
    """\
    package p;

    public class Api {
        public static final int LIMIT = 3;
        private final String name;

        public Api(String name) {
            this.name = name;
        }

        @Override
        public String toString() {
            return name;
        }

        protected void hook() {
        }

        void packageLevel() {
        }

        private void secret() {
        }

        private static class Hidden {
            void inside() {
            }
        }

        public interface Port {
            void open();
        }
    }
    """
)

_PYTHON = textwrap.dedent(
    """\
    class Box:
        def __init__(self, size):
            self.size = size

        def _helper(self):
            return 1

        def public(self):
            return self._helper()
    """
)


@pytest.fixture
def workspace(tmp_path: Path, graph_cache: Path) -> Iterator[tuple[Path, SqliteGraphProvider]]:
    """A Java and a Python file with every visibility an outline weighs, and a graph over them."""
    root = tmp_path / "ws"
    (root / "src/p").mkdir(parents=True)
    (root / "src/p/Api.java").write_text(_JAVA, encoding="utf-8")
    (root / "box.py").write_text(_PYTHON, encoding="utf-8")
    build_graph(str(root))
    provider = SqliteGraphProvider(str(root))
    yield root, provider
    provider.close()


def _names(found: object) -> list[str]:
    """The member names an outline lists, in order."""
    return [entry.symbol.name for entry in found.entries]  # type: ignore[attr-defined]


def test_default_outline_lists_the_api_and_counts_the_rest(workspace: tuple[Path, SqliteGraphProvider]) -> None:
    """Public, protected and package-private members are listed; private ones and fields are counted."""
    _, graph = workspace
    found = outline(graph, "src/p/Api.java")
    names = _names(found)
    for listed in ("Api", "toString", "hook", "packageLevel", "Port", "open"):
        assert listed in names, f"{listed} is API code outside the file can use"
    for hidden in ("secret", "Hidden", "inside", "LIMIT", "name"):
        assert hidden not in names, f"{hidden} is private or a field"
    assert (found.omitted.private, found.omitted.fields) == (3, 2), "a private type counts with what it holds"
    assert found.render().endswith(
        "(not listed: 3 private member(s) and 2 field(s); pass `all` to list them, or `members` to find one by name)"
    )
    assert found.to_dict()["omitted"] == {"private": 3, "fields": 2}, "the counts are data too"


def test_all_and_members_reach_what_the_default_leaves_out(workspace: tuple[Path, SqliteGraphProvider]) -> None:
    """`all` lists everything with no note; a name filter searches private members and fields as well."""
    _, graph = workspace
    everything = outline(graph, "src/p/Api.java", everything=True)
    assert {"secret", "Hidden", "inside", "LIMIT", "name"} <= set(_names(everything))
    assert everything.omitted.private == everything.omitted.fields == 0
    assert "not listed" not in everything.render()
    assert "secret" in _names(outline(graph, "src/p/Api.java", "secret")), "a filter finds a private member"
    assert "LIMIT" in _names(outline(graph, "src/p/Api.java", "LIMIT")), "and a field"


def test_python_hides_underscore_members_but_keeps_runtime_hooks(workspace: tuple[Path, SqliteGraphProvider]) -> None:
    """`_helper` is private by convention; `__init__` is the constructor callers use."""
    _, graph = workspace
    names = _names(outline(graph, "box.py"))
    assert "__init__" in names and "public" in names
    assert "_helper" not in names


def test_outline_parts_locate_the_line_naming_each_member(workspace: tuple[Path, SqliteGraphProvider]) -> None:
    """An annotated member is located at its signature, not at its `@Override` line, and the note follows."""
    root, graph = workspace
    payload = outline_payload(graph, "src/p/Api.java", None)["outline"]
    parts = outline_parts(str(root), payload)
    assert parts is not None
    lines = (root / "src/p/Api.java").read_text(encoding="utf-8").split("\n")
    shown = [lines[part["start"] - 1].strip() for part in parts if "path" in part]
    assert "public String toString() {" in shown and "@Override" not in shown
    assert all(part["start"] == part["end"] for part in parts if "path" in part), "one row per member"
    assert parts[-1]["text"].startswith("(not listed: 3 private member(s)"), "the parts say what was left out"


def test_outline_parts_take_an_older_payload_without_counts(workspace: tuple[Path, SqliteGraphProvider]) -> None:
    """A daemon older than `omitted` lists everything; its outline lays out with no note."""
    root, graph = workspace
    payload = outline_payload(graph, "src/p/Api.java", None, everything=True)["outline"]
    del payload["omitted"]
    parts = outline_parts(str(root), payload)
    assert parts is not None and all("not listed" not in part.get("text", "") for part in parts)


def test_declaration_line_skips_annotation_lines() -> None:
    """The naming line is the first one past annotations, even one whose argument spells the name."""
    lines = ['@Named("run")', "@Override", "public void run() {", "}"]
    assert declaration_line(lines, 1, 4, "run") == 3
    assert declaration_line(lines, 1, 4, "absent") == 1, "no naming line falls back to the start"
    assert declaration_line(None, 7, 9, "run") == 7, "an unreadable file falls back to the start"
    assert declaration_line(["int runner = 1;", "void run() {"], 1, 2, "run") == 2, "a longer word is not the name"
    typed = ["@Override", "@NonNull Box box() {", "    return box;", "}"]
    assert declaration_line(typed, 1, 4, "box") == 2, "an annotation before the type does not hide the declaration"
