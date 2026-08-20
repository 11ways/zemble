"""Chunking and capsule journeys for Hawkeye `.hwk` templates."""

from pathlib import Path

from zemble.chunking.capsule import CapsuleLevel, FileContext, capsule
from zemble.chunking.chunking import chunk_source
from zemble.hwk import scan, template_id_path, to_kebab_case
from zemble.index.files import ContentType, detect_language, get_extensions
from zemble.types import Chunk

FIXTURES = Path(__file__).parent / "fixtures" / "hwk" / "demo" / "src" / "common" / "templates"


def test_hwk_is_a_code_language() -> None:
    """A `.hwk` file is detected as its own language and lands in the code lane."""
    assert detect_language(Path("a/button.hwk")) == "hwk", "step 1: the extension is registered"
    assert ".hwk" in get_extensions([ContentType.CODE]), "step 2: the code lane walks it"
    assert ".hwk" not in get_extensions([ContentType.DOCS]), "step 3: it is not a docs file"


def test_tag_name_derivation() -> None:
    """A tag class name kebabs exactly the way the Hawkeye compiler does."""
    assert to_kebab_case("PlButton") == "pl-button", "step 1: two words"
    assert to_kebab_case("ZfQueryBuilder") == "zf-query-builder", "step 2: three words"
    assert to_kebab_case("PersistentCounter") == "persistent-counter", "step 3: no vendor prefix"


def test_template_id_path() -> None:
    """A template id is the path below the last `templates/` segment, minus the extension."""
    assert template_id_path("zenit-cms/src/common/templates/pages/list.hwk") == "pages/list"
    assert template_id_path("a/src/browserTest/templates/test/x.hwk") == "test/x"
    assert template_id_path("loose/file.hwk") == "loose/file", "a file outside a root keeps its path"


def test_scan_reads_a_component() -> None:
    """A component file yields its tag, and its style block never reads as a function call."""
    facts = scan((FIXTURES / "components" / "card.hwk").read_text())

    assert [(tag.class_name, tag.tag) for tag in facts.tags] == [("DemoCard", "demo-card")], "step 1: one tag"
    assert facts.tag == "demo-card", "step 2: a single-tag file is that tag"
    assert ("Demo", "label") in {(call.namespace, call.name) for call in facts.calls}, "step 3: the call is seen"
    assert "var" not in {call.name for call in facts.calls}, "step 4: SCSS var() is not a template call"


def test_scan_reads_a_page() -> None:
    """A page yields its parent, its blocks, its partials, its elements and its calls."""
    facts = scan((FIXTURES / "pages" / "dashboard.hwk").read_text())

    assert facts.extends is not None and facts.extends.target == "demo:base", "step 1: the parent"
    assert [block.name for block in facts.blocks] == ["main", "footer"], "step 2: both blocks, in order"
    assert [render.target for render in facts.renders] == ["demo:pages/row"], "step 3: the partial"
    used = facts.tags_used_between(1, facts.line_count)
    assert used == ["demo-card", "demo-widget", "not-a-real-element"], "step 4: every hyphenated tag"
    assert ("Demo", "label") in {(call.namespace, call.name) for call in facts.calls}, "step 5: namespaced call"
    assert (None, "t") in {(call.namespace, call.name) for call in facts.calls}, "step 6: bare call"

    main = facts.block_at(facts.blocks[0].start_line + 1)
    assert main is not None and main.name == "main", "step 7: a line inside a block names it"


def test_chunking_uses_the_html_grammar() -> None:
    """A template chunks through the borrowed html grammar while keeping its own language."""
    source = (FIXTURES / "pages" / "dashboard.hwk").read_text()
    chunks = chunk_source(source, "demo/src/common/templates/pages/dashboard.hwk", "hwk")

    assert chunks, "step 1: the file produces chunks"
    assert {chunk.language for chunk in chunks} == {"hwk"}, "step 2: the chunk keeps the hwk language"
    assert chunks[0].start_line == 1, "step 3: chunking starts at the first line"


def test_capsule_describes_a_page() -> None:
    """A page chunk's capsule names the path, the parent template and the enclosing block."""
    path = "demo/src/common/templates/pages/dashboard.hwk"
    source = (FIXTURES / "pages" / "dashboard.hwk").read_text()
    context = FileContext(file_path=path, source=source, language="hwk")
    inside_main = Chunk(content="", file_path=path, start_line=5, end_line=9, language="hwk")

    text = capsule(inside_main, context, CapsuleLevel.FULL)

    assert path in text, "step 1: the path segment is still first"
    assert "extends demo:base" in text, "step 2: the parent template is named"
    assert "block main" in text, "step 3: the enclosing block is named"
    assert "uses demo-card" in text, "step 4: the elements written in the chunk are named"


def test_capsule_describes_a_component() -> None:
    """A component chunk's capsule names the custom element the lines belong to."""
    path = "demo/src/common/templates/components/card.hwk"
    source = (FIXTURES / "components" / "card.hwk").read_text()
    context = FileContext(file_path=path, source=source, language="hwk")
    chunk = Chunk(content="", file_path=path, start_line=12, end_line=14, language="hwk")

    text = capsule(chunk, context, CapsuleLevel.FULL)

    assert "tag <demo-card> DemoCard" in text, "step 1: the element and its class are named"
    assert "extends" not in text, "step 2: a component declares no parent template"


def test_capsule_survives_a_missing_grammar() -> None:
    """Template facts are lexical, so a capsule still describes a file that never parsed."""
    path = "demo/src/common/templates/pages/dashboard.hwk"
    source = (FIXTURES / "pages" / "dashboard.hwk").read_text()
    context = FileContext(file_path=path, source=source, language="hwk", root=None)
    chunk = Chunk(content="", file_path=path, start_line=6, end_line=9, language="hwk")

    assert "block main" in capsule(chunk, context, CapsuleLevel.FULL), "no parse tree is needed"
