"""Extraction, resolution and query journeys for Hawkeye templates in the symbol graph."""

from pathlib import Path
from typing import Any

import pytest

from zemble.graph import SqliteGraphProvider, build_graph
from zemble.graph.hwk import TAG_MODIFIER, extract_hwk_file
from zemble.graph.java import extract_java_file
from zemble.graph.model import EdgeKind, Resolution, SymbolKind

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "hwk"
TEMPLATES = FIXTURE_ROOT / "demo" / "src" / "common" / "templates"
DASHBOARD = "demo/src/common/templates/pages/dashboard.hwk"
CARD = "demo/src/common/templates/components/card.hwk"


@pytest.fixture
def hwk_graph(graph_cache: Path) -> Any:  # noqa: ARG001
    """A freshly built graph over the Hawkeye fixture workspace."""
    build_graph(str(FIXTURE_ROOT))
    provider = SqliteGraphProvider(str(FIXTURE_ROOT))
    yield provider
    provider.close()


def test_extract_page_symbols_and_edges() -> None:
    """A page becomes one template symbol plus a symbol per block, with edges for what it uses."""
    extraction = extract_hwk_file((TEMPLATES / "pages" / "dashboard.hwk").read_bytes(), DASHBOARD)

    template = extraction.symbols[0]
    assert template.kind is SymbolKind.TEMPLATE, "step 1: the file itself is a template symbol"
    assert template.qualified_name == DASHBOARD, "step 2: a page with no tag is named by its path"
    assert TAG_MODIFIER not in template.modifiers, "step 3: a page declares no custom element"

    blocks = [symbol for symbol in extraction.symbols if symbol.kind is SymbolKind.BLOCK]
    assert [symbol.name for symbol in blocks] == ["main", "footer"], "step 4: one symbol per block"
    assert all(symbol.container_id == template.id for symbol in blocks), "step 5: blocks live in the file"

    written = {(edge.kind, edge.dst_name) for edge in extraction.edges}
    assert (EdgeKind.EXTENDS, "demo:base") in written, "step 6: the parent template"
    assert (EdgeKind.IMPORTS, "demo:pages/row") in written, "step 7: the rendered partial"
    assert (EdgeKind.REFERENCES_TYPE, "demo-card") in written, "step 8: the element it uses"
    assert (EdgeKind.CALLS, "label") in written, "step 9: the function it calls"


def test_extract_component_is_named_by_its_tag() -> None:
    """A file declaring one custom element IS that element as far as the graph is concerned."""
    extraction = extract_hwk_file((TEMPLATES / "components" / "card.hwk").read_bytes(), CARD)

    template = extraction.symbols[0]
    assert template.qualified_name == "demo-card", "step 1: looked up by the tag"
    assert TAG_MODIFIER in template.modifiers, "step 2: marked as a declaration"
    assert template.signature == "template components/card", "step 3: the template id is kept"


def test_java_extractor_captures_annotation_literals() -> None:
    """A registration annotation's string arguments survive extraction; a constant does not."""
    java = FIXTURE_ROOT / "demo" / "src" / "common" / "java" / "com" / "example" / "ui"
    functions = extract_java_file((java / "DemoFunctions.java").read_bytes(), "demo/DemoFunctions.java")
    element = extract_java_file((java / "DemoWidgetElement.java").read_bytes(), "demo/DemoWidgetElement.java")

    label = next(symbol for symbol in functions.symbols if symbol.name == "label")
    assert label.annotation_args["HawkeyeFunction"]["namespace"] == "Demo", "step 1: the namespace"
    assert label.annotation_args["HawkeyeFunction"]["name"] == "label", "step 2: the function name"

    overload = [symbol for symbol in functions.symbols if symbol.name == "label"][-1]
    assert overload.annotation_args == {}, "step 3: an unannotated overload carries nothing"

    widget = next(symbol for symbol in element.symbols if symbol.name == "DemoWidgetElement")
    assert widget.annotation_args["HawkeyeCustomElement"]["tag"] == "demo-widget", "step 4: the tag"


def test_resolution_ladder(hwk_graph: Any) -> None:
    """Each template edge lands on the rung its evidence supports, and no better."""
    dashboard = next(symbol for symbol in hwk_graph.symbols_in_file(DASHBOARD) if symbol.kind is SymbolKind.TEMPLATE)
    hits = {(hit.edge_kind, hit.symbol.name): hit for hit in hwk_graph.neighbors(dashboard.id)}

    parent = hits[(EdgeKind.EXTENDS, "base")]
    assert parent.resolution is Resolution.EXACT, "step 1: namespace and path both agree"

    element = hits[(EdgeKind.REFERENCES_TYPE, "demo-card")]
    assert element.symbol.file_path == CARD, "step 2: a tag resolves to the template declaring it"
    assert element.resolution is Resolution.EXACT, "step 3: a tag is a unique registration key"

    hand_written = hits[(EdgeKind.REFERENCES_TYPE, "DemoWidgetElement")]
    assert hand_written.resolution is Resolution.EXACT, "step 4: an annotated class claims its tag"

    call = hits[(EdgeKind.CALLS, "label")]
    assert call.resolution is Resolution.EXACT, "step 5: namespace plus name pins the function"

    assert (EdgeKind.REFERENCES_TYPE, "not-a-real-element") not in hits, "step 6: an unknown tag stays unresolved"


def test_a_bare_call_resolves_by_name(hwk_graph: Any) -> None:
    """A call with no namespace matches the global template function of that name."""
    dashboard = next(symbol for symbol in hwk_graph.symbols_in_file(DASHBOARD) if symbol.kind is SymbolKind.TEMPLATE)

    translate = next(hit for hit in hwk_graph.neighbors(dashboard.id) if hit.symbol.name == "translate")

    assert translate.edge_kind is EdgeKind.CALLS, "step 1: it is a call"
    assert translate.resolution is Resolution.EXACT, "step 2: the global namespace is a match, not a guess"
    assert translate.symbol.annotation_args["HawkeyeFunction"]["name"] == "t", "step 3: matched on the declared name"


def test_callers_of_a_template_function_include_templates(hwk_graph: Any) -> None:
    """The question the extractor exists for: who calls this `@HawkeyeFunction`."""
    label = next(symbol for symbol in hwk_graph.definition("DemoFunctions.label") if symbol.param_types == ["String"])

    callers = hwk_graph.callers(label.id)

    assert {hit.symbol.file_path for hit in callers} == {
        CARD,
        DASHBOARD,
        "demo/src/common/templates/pages/row.hwk",
    }, "every calling template is listed, and nothing else"


def test_references_of_a_custom_element(hwk_graph: Any) -> None:
    """A component's users are the templates that write its tag."""
    card = next(symbol for symbol in hwk_graph.definition("demo-card"))

    references = hwk_graph.references(card.id)

    assert [(hit.symbol.file_path, hit.edge_kind) for hit in references] == [(DASHBOARD, EdgeKind.REFERENCES_TYPE)], (
        "the page that uses <demo-card> is the answer"
    )


def test_graph_build_counts_templates(hwk_graph: Any, graph_cache: Path) -> None:  # noqa: ARG001
    """Templates are extracted, not skipped, and their language is recorded as covered."""
    stats = build_graph(str(FIXTURE_ROOT), force=True)

    assert "hwk" not in stats.skipped_by_language, "step 1: templates are no longer skipped"
    assert stats.extracted_files == 6, "step 2: four templates and two Java files"


def test_cli_answers_a_template_question(graph_cache: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # noqa: ARG001, ANN001
    """`zemble graph references <tag>` names the templates that write a custom element."""
    from zemble.cli import _cli_main
    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    monkeypatch.setattr("sys.argv", ["zemble", "graph", "references", str(FIXTURE_ROOT), "demo-card"])
    with pytest.raises(SystemExit) as exited:
        _cli_main()
    output = capsys.readouterr().out

    assert exited.value.code in (0, None), "step 1: the tag is a symbol the CLI can answer for"
    assert "demo-card  [template]" in output, "step 2: the element is reported as a template"
    assert "pages/dashboard.hwk" in output, "step 3: its user is listed"
