"""Behaviour journeys over a workspace mixing Python, TypeScript, Rust and Go."""

from pathlib import Path

import pytest

from zemble.evidence.outline import outline
from zemble.graph.model import Resolution, SymbolKind
from zemble.graph.provider import SqliteGraphProvider
from zemble.graph.store import build_graph

POLYGLOT = Path(__file__).parent / "fixtures" / "polyglot"


@pytest.fixture
def polyglot_graph(graph_cache: Path) -> SqliteGraphProvider:
    """A graph over the polyglot fixture, built from scratch."""
    stats = build_graph(str(POLYGLOT), force=True)
    assert stats.skipped_by_language == {}, "every file has a grammar"
    provider = SqliteGraphProvider(str(POLYGLOT))
    yield provider
    provider.close()


def _one(provider: SqliteGraphProvider, name: str, suffix: str):
    """The single definition of a name in files of one suffix."""
    found = [symbol for symbol in provider.definition(name) if symbol.file_path.endswith(suffix)]
    assert len(found) == 1, f"{name} in *{suffix}: {[symbol.qualified_name for symbol in found]}"
    return found[0]


def test_definitions_and_calls_stay_within_a_language(polyglot_graph: SqliteGraphProvider) -> None:
    """A name declared in four languages is four definitions, and each language's calls reach its own."""
    graph = polyglot_graph
    # 1. `Circle` exists once per language, with the kind each language gives it.
    kinds = {symbol.file_path: symbol.kind for symbol in graph.definition("Circle")}
    assert kinds == {
        "app/shapes.py": SymbolKind.CLASS,
        "app/shapes.ts": SymbolKind.CLASS,
        "app/shapes.rs": SymbolKind.STRUCT,
        "app/shapes.go": SymbolKind.STRUCT,
    }, "step 1: one Circle per language"

    # 2. `scale` is called from each language's `Circle.area`, and only from that language.
    for suffix, expected in (
        (".py", {"app.shapes.Circle.area", "app.support.unused"}),
        (".ts", {"app.shapes.Circle.area"}),
        (".rs", {"app.shapes.Circle.area"}),
        (".go", {"app.Circle.Area"}),
    ):
        scale = _one(graph, "scale", suffix)
        callers = {hit.symbol.qualified_name for hit in graph.callers(scale.id)}
        assert callers == expected, f"step 2: {suffix} callers of scale"
        assert all(hit.symbol.file_path.endswith(suffix) for hit in graph.callers(scale.id)), (
            f"step 2: {suffix} never resolves into another language"
        )

    # 3. The Python call resolved through its import, exactly; the Rust one by name.
    python_hits = graph.callers(_one(graph, "scale", ".py").id)
    assert {hit.resolution for hit in python_hits} == {Resolution.EXACT}, "step 3: `from app.support import scale`"
    rust_hits = graph.callers(_one(graph, "scale", ".rs").id)
    assert {hit.resolution for hit in rust_hits} == {Resolution.EXACT}, "step 3: a file's own function is exact"


def test_hierarchy_journey(polyglot_graph: SqliteGraphProvider) -> None:
    """Subtypes, overrides and interface implementations work off the generic edges."""
    graph = polyglot_graph
    shape = _one(graph, "Shape", ".py")
    assert [hit.symbol.qualified_name for hit in graph.implementations(shape.id)] == ["app.shapes.Circle"], (
        "step 1: `class Circle(Shape)` is a subtype"
    )
    name = _one(graph, "Circle.name", ".py")
    assert [hit.symbol.qualified_name for hit in graph.overrides_of(name.id)] == ["app.shapes.Shape.name"], (
        "step 2: a redeclared method overrides the parent's"
    )
    area = _one(graph, "Shape.area", ".py")
    assert [hit.symbol.qualified_name for hit in graph.overridden_by(area.id)] == ["app.shapes.Circle.area"]
    ts_shape = _one(graph, "Shape", ".ts")
    assert [hit.symbol.qualified_name for hit in graph.implementations(ts_shape.id)] == ["app.shapes.Circle"], (
        "step 3: `implements Shape` in TypeScript"
    )
    rust_trait = _one(graph, "Shape", ".rs")
    assert [hit.symbol.qualified_name for hit in graph.implementations(rust_trait.id)] == ["app.shapes.Circle"], (
        "step 4: `impl Shape for Circle` in Rust"
    )
    go_area = _one(graph, "Circle.Area", ".go")
    assert graph.overrides_of(go_area.id) == [], "step 5: Go has no declared hierarchy to override through"


def test_tests_journey(polyglot_graph: SqliteGraphProvider) -> None:
    """Test modules are found by name (`test_shapes`, `shapes_test`) and by what they exercise."""
    graph = polyglot_graph
    largest = _one(graph, "largest", ".py")
    hits = graph.tests_of(largest.id)
    assert [hit.symbol.name for hit in hits] == ["test_shapes"], "step 1: the Python test module covers it"
    assert hits[0].symbol.is_test, "step 1: a `tests/` path is a test"
    go_area = _one(graph, "Circle.Area", ".go")
    go_hits = graph.tests_of(go_area.id)
    assert [hit.symbol.name for hit in go_hits] == ["shapes_test"], "step 2: `shapes_test.go` exercises Area"
    assert go_hits[0].symbol.is_test, "step 2: a `*_test.go` file name is a test"
    test_function = _one(graph, "test_largest", ".py")
    assert test_function.is_test and test_function.kind is SymbolKind.FUNCTION


def test_outline_and_coverage(polyglot_graph: SqliteGraphProvider) -> None:
    """An outline works on a file or a module of any language, and the graph says what it covers."""
    graph = polyglot_graph
    rendered = outline(graph, "app/shapes.py").render()
    assert "module app.shapes" in rendered and "class Circle(Shape)" in rendered, "step 1: a Python file outline"
    assert "constructor def __init__(self, radius)" in rendered, "step 1: with its members"
    assert "function def largest(shapes)" in rendered
    by_name = outline(graph, "app.support").render()
    assert "function def scale(value)" in by_name, "step 2: a module outline by qualified name"
    rust = outline(graph, "app/shapes.rs").render()
    assert "interface pub trait Shape" in rust and "method fn area(&self) -> f64" in rust, "step 3: Rust"
    assert "languages" in graph.coverage_note(), "step 4: the coverage note counts languages"
