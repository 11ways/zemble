"""Behaviour journeys over the per-file Java extractor."""

from pathlib import Path

import pytest

from zemble.graph.java import extract_java_file
from zemble.graph.model import EdgeKind, SymbolKind, is_test_path


def _extract(root: Path, relative: str):
    """Extract one fixture file by its workspace-relative path."""
    return extract_java_file((root / relative).read_bytes(), relative)


MAIN = "src/main/java/com/example"


def test_type_declaration_journey(graph_fixture_root: Path) -> None:
    """Walk one file through every kind of type declaration it can hold."""
    # 1. An interface with an abstract, a default and a static method.
    shape = _extract(graph_fixture_root, f"{MAIN}/core/Shape.java")
    kinds = {symbol.qualified_name: symbol.kind for symbol in shape.symbols}
    assert kinds["com.example.core.Shape"] is SymbolKind.INTERFACE, "step 1: Shape is an interface"
    assert kinds["com.example.core.Shape.describe"] is SymbolKind.METHOD, "step 1: default methods are methods"
    assert kinds["com.example.core.Shape.unit"] is SymbolKind.METHOD, "step 1: static methods are methods"

    # 2. A record turns its components into fields and keeps its declared body method.
    point = _extract(graph_fixture_root, f"{MAIN}/core/Point.java")
    record_kinds = {symbol.name: symbol.kind for symbol in point.symbols}
    assert record_kinds["Point"] is SymbolKind.RECORD, "step 2: Point is a record"
    assert record_kinds["x"] is SymbolKind.FIELD, "step 2: record components become fields"
    assert record_kinds["sum"] is SymbolKind.METHOD, "step 2: record body methods survive"

    # 3. An enum yields constants, and a constant with a body yields its own members.
    palette = _extract(graph_fixture_root, f"{MAIN}/core/Palette.java")
    palette_kinds = {symbol.qualified_name: symbol.kind for symbol in palette.symbols}
    assert palette_kinds["com.example.core.Palette.RED"] is SymbolKind.ENUM_CONSTANT, "step 3: RED is a constant"
    assert palette_kinds["com.example.core.Palette.GREEN.tag"] is SymbolKind.METHOD, (
        "step 3: a constant body's method attaches to the constant"
    )

    # 4. An annotation type and its element.
    marker = _extract(graph_fixture_root, f"{MAIN}/core/Marker.java")
    marker_kinds = {symbol.name: symbol.kind for symbol in marker.symbols}
    assert marker_kinds["Marker"] is SymbolKind.ANNOTATION, "step 4: @interface is an annotation"
    assert marker_kinds["value"] is SymbolKind.METHOD, "step 4: annotation elements are methods"


def test_nested_anonymous_and_local_type_journey(graph_fixture_root: Path) -> None:
    """Registry holds a nested type, an anonymous type and a local type at once."""
    registry = _extract(graph_fixture_root, f"{MAIN}/core/Registry.java")
    by_qualified = {symbol.qualified_name: symbol for symbol in registry.symbols}

    # 1. A nested static class keeps the enclosing type in its qualified name.
    assert "com.example.core.Registry.Entry" in by_qualified, "step 1: nested types are qualified by their outer type"

    # 2. An anonymous class is named after its enclosing type and its line.
    anonymous = [name for name, symbol in by_qualified.items() if "$anon@" in name and symbol.kind is SymbolKind.CLASS]
    assert len(anonymous) == 1, "step 2: exactly one anonymous class in Registry"
    assert anonymous[0].startswith("com.example.core.Registry$anon@"), "step 2: named <enclosing>$anon@line"

    # 3. The anonymous type's own method is its member, not the enclosing method's.
    area = by_qualified[f"{anonymous[0]}.area"]
    assert area.container_id == by_qualified[anonymous[0]].id, "step 3: anonymous members belong to the anonymous type"

    # 4. That anonymous type hangs off the method that declares it.
    assert by_qualified[anonymous[0]].container_id == by_qualified["com.example.core.Registry.anonymousShape"].id, (
        "step 4: the anonymous type attaches to its enclosing method"
    )

    # 5. A local class is qualified by the method that declares it.
    assert "com.example.core.Registry.localHelper.Local" in by_qualified, "step 5: local types keep their method scope"

    # 6. A lambda-free method body still attributes its calls to the enclosing callable.
    callers = {edge.src_id for edge in registry.edges if edge.kind is EdgeKind.CALLS and edge.dst_name == "twice"}
    assert callers == {area.id}, "step 6: calls inside the anonymous body attribute to its method"


def test_overload_and_generic_signature_journey(graph_fixture_root: Path) -> None:
    """Overloads stay distinct and generics survive in signatures but not in ids."""
    circle = _extract(graph_fixture_root, f"{MAIN}/core/Circle.java")
    scales = [symbol for symbol in circle.symbols if symbol.name == "scale"]

    # 1. Two overloads, two symbols.
    assert len(scales) == 2, "step 1: both scale overloads are extracted"

    # 2. Their ids differ by the erased parameter types.
    assert {symbol.id.rsplit("(", 1)[-1] for symbol in scales} == {"double)", "double,int)"}, (
        "step 2: overloads are disambiguated by erased parameter types"
    )

    # 3. Constructors are their own kind, and both arities are present.
    constructors = [symbol for symbol in circle.symbols if symbol.kind is SymbolKind.CONSTRUCTOR]
    assert sorted(symbol.arity for symbol in constructors) == [0, 1], "step 3: both constructors are extracted"

    # 4. A generic declaration keeps its type parameters in the signature.
    registry = _extract(graph_fixture_root, f"{MAIN}/core/Registry.java")
    outer = next(symbol for symbol in registry.symbols if symbol.qualified_name == "com.example.core.Registry")
    assert outer.signature == "class Registry<T extends Shape>", "step 4: type parameters stay in the signature"

    # 5. A type variable is not a reference to a workspace type.
    referenced = {edge.dst_name for edge in registry.edges if edge.kind is EdgeKind.REFERENCES_TYPE}
    assert "T" not in referenced, "step 5: type variables are not type references"


def test_import_and_annotation_journey(graph_fixture_root: Path) -> None:
    """Every import form and the @Override marker are captured."""
    registry = _extract(graph_fixture_root, f"{MAIN}/core/Registry.java")

    # 1. A wildcard import records the package.
    assert registry.imports.wildcards == ["com.example.util"], "step 1: wildcard imports are recorded"

    # 2. A single static member import maps the member to its owner.
    assert registry.imports.static_members == {"twice": "com.example.util.Helpers"}, (
        "step 2: static member imports map member to owner"
    )

    # 3. A type-level annotation becomes an ANNOTATED_WITH edge.
    annotated = {
        (edge.src_id.split("#")[1], edge.dst_name) for edge in registry.edges if edge.kind is EdgeKind.ANNOTATED_WITH
    }
    assert ("com.example.core.Registry", "Marker") in annotated, "step 3: type annotations become edges"

    # 4. @Override is visible on the method that carries it.
    circle = _extract(graph_fixture_root, f"{MAIN}/core/Circle.java")
    area = next(symbol for symbol in circle.symbols if symbol.name == "area")
    assert area.annotations == ["Override"], "step 4: @Override is recorded on the method"

    # 5. Explicit single-type imports map simple name to fully qualified name.
    assert circle.imports.explicit == {"Helpers": "com.example.util.Helpers"}, "step 5: explicit imports are recorded"


def test_reference_capture_journey(graph_fixture_root: Path) -> None:
    """Calls carry their receiver, their arity and whether they construct."""
    circle = _extract(graph_fixture_root, f"{MAIN}/core/Circle.java")
    calls = {edge.dst_name: edge for edge in circle.edges if edge.kind is EdgeKind.CALLS}

    # 1. A static call on a type name records that type as the receiver type.
    assert calls["twice"].receiver_type == "Helpers", "step 1: static receivers are captured as a type"

    # 2. Argument count is recorded so overloads can be told apart later.
    assert calls["twice"].arity == 1, "step 2: call arity is captured"

    # 3. A chained constructor call is marked as constructing.
    assert calls["this"].is_new is True, "step 3: this(...) is a constructor call"

    # 4. An implicit-this call has no receiver at all.
    assert calls["describe"].receiver is None, "step 4: implicit this calls carry no receiver"

    # 5. Comments and string literals never produce references.
    shape = _extract(graph_fixture_root, f"{MAIN}/core/Shape.java")
    assert not [edge for edge in shape.edges if edge.dst_name == "shape "], "step 5: string content is not a reference"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/test/java/com/example/core/CircleTest.java", True),
        ("src/browserTest/java/com/example/core/PaletteBrowserCheck.java", True),
        ("src/integrationTest/java/com/example/Foo.java", True),
        ("src/main/java/com/example/core/Circle.java", False),
        ("src/main/java/com/example/testing/Contest.java", False),
    ],
)
def test_test_source_set_detection(path: str, expected: bool) -> None:
    """A file is a test file when a directory segment names a test source set."""
    assert is_test_path(path) is expected, f"{path} should have is_test={expected}"
