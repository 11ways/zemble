"""Behaviour journeys over the grammar specs every non-Java language is read through."""

from pathlib import Path

import pytest
from semble_grammars import available_languages, get_language, get_parser

from zemble.graph.model import EdgeKind, SymbolKind
from zemble.graph.store import GRAPH_LANGUAGES, extractor_for
from zemble.index.files import detect_language, get_extensions
from zemble.languages.catalog import SPECS, family_of, spec_for
from zemble.languages.paths import resolve, resolve_all, text
from zemble.languages.read import split_qualified, type_name
from zemble.types import ContentType

SAMPLES = Path(__file__).parent / "fixtures" / "polyglot_samples"
#: Languages with a hand-written graph extractor instead of a spec.
HAND_WRITTEN = frozenset({"java"})


def _code_languages() -> set[str]:
    """Every language an indexed code extension maps to."""
    return {
        language
        for extension in get_extensions([ContentType.CODE])
        for language in [detect_language(Path(f"x{extension}"))]
        if language is not None
    }


def test_every_spec_names_real_grammar_nodes() -> None:
    """Drift guard: a spec may only name node kinds its grammar actually has, paths included."""
    for spec in SPECS.values():
        grammar = get_language(spec.language)
        missing = [
            kind
            for kind in sorted(spec.node_kinds())
            if grammar.id_for_node_kind(kind, True) is None and grammar.id_for_node_kind(kind, False) is None
        ]
        assert not missing, f"{spec.language} names node kinds its grammar does not have: {missing}"


def test_every_bundled_code_grammar_is_read() -> None:
    """Drift guard: a code language with a bundled grammar has a spec or a hand-written extractor."""
    code = _code_languages()
    unread = sorted(language for language in available_languages() if language in code and language not in SPECS)
    assert set(unread) <= HAND_WRITTEN, f"bundled code grammars without a reader: {unread}"
    assert set(GRAPH_LANGUAGES) >= set(SPECS), "the graph reads every spec"
    assert "java" in GRAPH_LANGUAGES and "hwk" in GRAPH_LANGUAGES, "and keeps its hand-written lanes"


def test_families_group_interoperating_languages() -> None:
    """A Kotlin call may resolve into Java; a Python one never into either."""
    assert family_of("kotlin") == family_of("groovy") == "jvm", "JVM languages share a family"
    assert family_of("typescript") == family_of("javascript") == "js", "so do the JS dialects"
    assert family_of("python") != family_of("ruby"), "unrelated languages do not"
    assert family_of("java") == "java", "a language without a spec is its own family"
    assert spec_for("cobol") is None and family_of(None) is None, "no grammar, no spec, no family"


def test_path_language_journey() -> None:
    """Every path step reaches what its documentation promises, on one real Python tree."""
    source = b"class Foo(Base):\n    def run(self, x):\n        return helper(x)\n\n\ndef top(n):\n    pass\n"
    root = get_parser("python").parse(source).root_node
    klass = root.named_children[0]

    # 1. Fields, typed children, indexes and descendants.
    assert text(resolve(klass, "name"), source) == "Foo", "step 1: a field step"
    assert text(resolve(klass, "superclasses/@identifier"), source) == "Base", "step 1: a typed child step"
    assert text(resolve(klass, "body/#0/name"), source) == "run", "step 1: an index step"
    assert text(resolve(klass, "**call/function"), source) == "helper", "step 1: a descendant step"

    # 2. Lists fan out; `field+` collects every child in a field.
    run = resolve(klass, "body/@function_definition")
    assert [text(node, source) for node in resolve_all(run, "parameters/@identifier*")] == ["self", "x"], (
        "step 2: `@kind*` yields every child of the kind"
    )
    assert len(resolve_all(root, "@function_definition*")) == 1, "step 2: only top-level function definitions"

    # 3. Siblings, parents, self, alternatives, and the chain steps.
    assert text(resolve(klass, ">/name"), source) == "top", "step 3: the next sibling"
    assert resolve(resolve(klass, ">"), "</name") is not None, "step 3: and the previous one"
    assert resolve(run, "../..") == klass, "step 3: parents"
    assert resolve(run, ".") is run, "step 3: self"
    assert text(resolve(klass, "missing|name"), source) == "Foo", "step 3: the first alternative that resolves wins"
    assert resolve(klass, "missing") is None and resolve(None, "name") is None, "step 3: nothing is nothing"
    assert resolve(klass, "body~block") is not None, "step 3: `field~kind` stops at the kind"

    # 4. Names written with a path in them split on their last separator.
    assert split_qualified("a.b.C") == ("a.b", "C") and split_qualified("std::fmt") == ("std", "fmt")
    assert split_qualified("plain") == (None, "plain")
    assert type_name("Box<int>*") == "Box" and type_name("Base()") == "Base", "step 4: generics and calls drop"


@pytest.mark.parametrize("sample", sorted(SAMPLES.iterdir()), ids=lambda path: path.name)
def test_every_sample_extracts_through_its_spec(sample: Path) -> None:
    """Every bundled grammar's sample yields a module symbol and never raises."""
    extract = extractor_for(sample)
    assert extract is not None, f"{sample.name} has a grammar and a spec"
    extraction = extract(sample.read_bytes(), f"src/{sample.name}")
    assert extraction.symbols[0].kind is SymbolKind.MODULE, "the file itself is the first symbol"
    assert all(symbol.start_line <= symbol.end_line for symbol in extraction.symbols), "spans are sane"


def _symbols(name: str) -> dict[str, SymbolKind]:
    """Qualified name -> kind for one sample."""
    sample = SAMPLES / name
    extract = extractor_for(sample)
    assert extract is not None
    return {symbol.qualified_name: symbol.kind for symbol in extract(sample.read_bytes(), name).symbols}


def _edges(name: str) -> list[tuple[str, str, str]]:
    """(source qualified name, kind, target) for one sample."""
    sample = SAMPLES / name
    extract = extractor_for(sample)
    assert extract is not None
    return [
        (edge.src_id.split("#", 1)[1].split("(")[0], edge.kind.value, edge.dst_name)
        for edge in extract(sample.read_bytes(), name).edges
    ]


def test_declaration_shapes_journey() -> None:
    """The shapes that differ most between grammars all land on the shared vocabulary."""
    # 1. Python: a class, its constructor, a decorated static method and a module constant.
    python = _symbols("sample.py")
    assert python["sample.Foo"] is SymbolKind.CLASS and python["sample.Foo.__init__"] is SymbolKind.CONSTRUCTOR
    assert python["sample.Foo.build"] is SymbolKind.METHOD and python["sample.top"] is SymbolKind.FUNCTION
    assert python["sample.CONST"] is SymbolKind.FIELD, "step 1: a module-level assignment is a field"

    # 2. TypeScript: interfaces, enums, type aliases, namespaces and arrow-function constants.
    ts = _symbols("sample.ts")
    assert ts["sample.Shape"] is SymbolKind.INTERFACE and ts["sample.Color.Green"] is SymbolKind.ENUM_CONSTANT
    assert ts["sample.Alias"] is SymbolKind.TYPE and ts["NS.inner"] is SymbolKind.FUNCTION
    assert ts["sample.arrow"] is SymbolKind.FUNCTION, "step 2: `const f = () => ...` is a function"

    # 3. Go: a package header qualifies names, receiver methods attach to their struct.
    go = _symbols("sample.go")
    assert go["store.Point"] is SymbolKind.STRUCT and go["store.Point.Area"] is SymbolKind.METHOD
    assert go["store.Shape.Area"] is SymbolKind.METHOD, "step 3: interface methods are members"

    # 4. Rust: `impl Trait for Type` attaches methods to the struct and implements the trait.
    rust = _symbols("sample.rs")
    assert rust["sample.Point.area"] is SymbolKind.METHOD and rust["sample.Area"] is SymbolKind.INTERFACE
    assert ("sample.Point", EdgeKind.IMPLEMENTS.value, "Area") in _edges("sample.rs"), "step 4: the trait edge"

    # 5. C: `typedef struct {...} name_t` and prototypes are told apart from definitions.
    c = _symbols("sample.c")
    assert c["sample.thing_t"] is SymbolKind.STRUCT and c["sample.helper"] is SymbolKind.FUNCTION
    assert sum(1 for name in c if name == "sample.helper") == 1, "step 5: the prototype is skipped"

    # 6. C#: a namespace with a body names itself; a `Point(...)` inside `Point` is the constructor.
    csharp = _symbols("sample.cs")
    assert csharp["App.Core"] is SymbolKind.MODULE and csharp["App.Core.Point.Point"] is SymbolKind.CONSTRUCTOR
    assert csharp["App.Core.Person"] is SymbolKind.RECORD

    # 7. Ruby: `include` is a supertype, `attr_reader` a field, `initialize` the constructor.
    ruby = _symbols("sample.rb")
    assert ruby["Store.Point.initialize"] is SymbolKind.CONSTRUCTOR and ruby["Store.Point.x"] is SymbolKind.FIELD
    assert ("Store.Point", EdgeKind.EXTENDS.value, "Comparable") in _edges("sample.rb")

    # 8. Elixir: every declaration is a `call`; `defp` marks it private.
    elixir = _symbols("sample.ex")
    assert elixir["Store.Point"] is SymbolKind.MODULE and elixir["Store.Point.helper"] is SymbolKind.FUNCTION

    # 9. Kotlin: a `val` primary-constructor parameter is a field of the class.
    kotlin = _symbols("sample.kt")
    assert kotlin["sample.P.x"] is SymbolKind.FIELD and kotlin["sample.Shape"] is SymbolKind.INTERFACE


def test_call_shapes_journey() -> None:
    """Receivers, constructors and arity survive the grammars' different call shapes."""
    python = {(kind, target): src for src, kind, target in _edges("sample.py")}
    assert python[("calls", "build")] == "sample.Foo.run", "step 1: `Foo.build(1)` is a call from run"
    assert python[("calls", "Foo")] == "sample.Foo.build", "step 1: a capitalised call is a constructor"
    ruby = [edge for edge in _edges("sample.rb") if edge[1] == "calls"]
    assert ("Store.Point.area", "calls", "compute") in ruby, "step 2: a receiver call keeps its name"
    dart = [edge for edge in _edges("sample.dart") if edge[1] == "calls"]
    assert ("sample.Point.run", "calls", "compute") in dart and ("sample.Point.run", "calls", "area") in dart, (
        "step 3: Dart's selector chains are calls on the previous sibling"
    )
    assert len([edge for edge in dart if edge[2] == "helper"]) == 1, "step 3: a sibling body is walked once"
    perl = [edge for edge in _edges("sample.pl") if edge[1] == "calls"]
    assert perl.count(("Store.Point.area", "calls", "helper")) == 1, "step 4: a bracketed bareword call once"
