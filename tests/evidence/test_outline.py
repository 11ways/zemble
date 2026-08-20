"""Outlines and signatures over the fixture Java workspace."""

from __future__ import annotations

import pytest

from zemble.evidence.outline import OutlineError, outline, signatures
from zemble.evidence.tokens import estimate_tokens
from zemble.graph.cli import select_symbol
from zemble.graph.provider import SqliteGraphProvider


def test_outline_journey(built_graph: SqliteGraphProvider) -> None:
    """Outline a type, a file and a filtered member set."""
    # 1. A type outline names its package, its declaration and every member once.
    rendered = outline(built_graph, "Registry")
    text = rendered.render()
    assert "package com.example.core" in text, "step 1: the package is stated"
    assert "class Registry<T extends Shape>" in text, "step 1: the type keeps its generic signature"
    assert "class class" not in text, "step 1: the kind is not repeated when the signature already says it"
    assert "[@Marker]" in text, "step 1: annotations are carried"
    assert text.count("method String localHelper()") == 1, "step 1: a member appears once"

    # 2. Nesting is indentation, and anonymous classes are not members.
    lines = [line for line in text.splitlines() if "Entry" in line]
    assert lines[0].startswith("  class Entry"), "step 2: a nested type is indented under its owner"
    assert any(line.startswith("    ") for line in lines), "step 2: its members are indented again"
    assert "$anon@" not in text, "step 2: anonymous classes are noise, not members"

    # 3. An outline is cheap: a whole class for a few hundred tokens.
    assert estimate_tokens(text) < 300, "step 3: an outline of a typical class stays small"

    # 4. A file path outlines the file, not a type.
    by_file = outline(built_graph, "src/main/java/com/example/core/Circle.java").render()
    assert "class Circle implements Shape" in by_file, "step 4: a path resolves to its declarations"
    assert "method double scale(double factor, int times)" in by_file, "step 4: overloads are listed separately"

    # 5. --members narrows to matching members and drops the types left empty.
    filtered = outline(built_graph, "src/main/java/com/example/core/Circle.java", "scale").render()
    assert "scale(double factor)" in filtered, "step 5: matching members survive"
    assert "label()" not in filtered, "step 5: others do not"
    empty = outline(built_graph, "Registry", "nothing-matches-this").render()
    assert "class Registry" not in empty, "step 5: a type with no surviving member is pruned"

    # 6. An unknown or ambiguous target is a loud refusal, not an empty answer.
    with pytest.raises(OutlineError):
        outline(built_graph, "NoSuchTypeAnywhere")
    with pytest.raises(OutlineError) as ambiguous:
        outline(built_graph, "Circle")
    assert len(ambiguous.value.candidates) == 2, "step 6: both Circle declarations are named"


def test_signatures_lists_exact_callers(built_graph: SqliteGraphProvider) -> None:
    """A signature answer shows the declaration and the call sites the graph is sure about."""
    chosen, _ = select_symbol(built_graph.definition("Helpers.twice"), "Helpers.twice")
    assert chosen is not None, "the fixture declares Helpers.twice"

    answer = signatures(built_graph, chosen)
    rendered = answer.render()
    assert "method double twice(double value)" in rendered, "the signature is the first line"
    assert "src/main/java/com/example/core/Circle.java:19  Circle.area" in rendered, "each caller is one line"
    assert all(entry["caller"] for entry in answer.to_dict()["callers"]), "the JSON form names every caller"
    assert answer.ambiguous == 0, "the fixture's calls all resolve exactly"
