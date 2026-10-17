"""Cognitive complexity, nesting, statements and comment lines, against hand-counted fixtures."""

from pathlib import Path

import pytest
from semble_grammars import get_parser

from zemble.dedup.languages import LanguageProfile, profile_for
from zemble.graph.generic import extract_generic_file
from zemble.graph.java import extract_java_file
from zemble.index.files import detect_language
from zemble.languages.catalog import JAVA_COMPLEXITY, complexity_for, spec_for
from zemble.languages.metrics import CallableMetrics, Called, CallReader, comment_lines, measure_callable

FIXTURES = Path(__file__).parent / "fixtures" / "metrics"
SAMPLES = Path(__file__).parent / "fixtures" / "polyglot_samples"

#: The shapes every fixture writes, as (complexity, nesting), counted by hand with Sonar's rules.
COMMON = {
    "seq": (0, 0),  # straight-line code
    "nested": (6, 3),  # loop +1, if +2 (nesting 1), if +3 (nesting 2)
    "branches": (3, 1),  # if +1, else if +1, else +1: branches never take a nesting increment
    "booleans": (5, 1),  # if +1 with `a && b && c` +1; if +1 with `a && b || c` +2
    "ternary": (1, 1),  # ternary +1
    "tryCatch": (3, 2),  # try nothing, catch +1, if inside it +2
    "fact": (2, 1),  # if +1, the recursive call +1
    "lambdaInLoop": (4, 3),  # loop +1, the lambda nests, if inside it +3
}

#: Per language: which COMMON shapes it cannot write, its own extra shapes, its spelling of names.
EXPECTED: dict[str, dict[str, tuple[int, int]]] = {
    "javascript": dict(COMMON),
    "typescript": dict(COMMON),
    "cpp": dict(COMMON),
    "php": dict(COMMON),
    "python": dict(COMMON),
    "ruby": dict(COMMON),
    "csharp": {name[0].upper() + name[1:]: value for name, value in COMMON.items()},
    "c": {
        **{name: COMMON[name] for name in ("seq", "nested", "branches", "booleans", "ternary", "fact")},
        "jumper": (2, 1),  # if +1, goto +1
    },
    "go": {
        **{name: COMMON[name] for name in ("seq", "nested", "branches", "booleans", "fact", "lambdaInLoop")},
        "switcher": (1, 1),  # switch +1; its cases add nothing
    },
    "rust": {
        **{name: COMMON[name] for name in ("seq", "nested", "branches", "booleans", "fact", "lambdaInLoop")},
        "matcher": (1, 1),  # match +1
    },
    "kotlin": {
        **{
            name: COMMON[name] for name in ("seq", "nested", "branches", "booleans", "tryCatch", "fact", "lambdaInLoop")
        },
        "switcher": (1, 1),  # when +1
    },
    "zig": {
        **{name: COMMON[name] for name in ("seq", "nested", "branches", "booleans", "tryCatch", "fact")},
        "ternary": (2, 1),  # Zig's ternary is an if expression: if +1, else +1
        "switcher": (3, 2),  # switch +1, a while inside a case +2
    },
}
_SNAKE = {"tryCatch": "try_catch", "lambdaInLoop": "lambda_in_loop"}
for _language in ("python", "ruby", "rust"):
    EXPECTED[_language] = {_SNAKE.get(name, name): value for name, value in EXPECTED[_language].items()}

#: Java checked against Sonar's cognitive complexity white paper, statement counts included.
JAVA = {
    # (complexity, nesting, statements)
    "seq": (0, 0, 3),  # no structure; three statements
    "nested": (6, 3, 4),  # for +1, if +2 (nesting 1), if +3 (nesting 2); one statement per block
    "branches": (3, 1, 4),  # if +1, else if +1 (hybrid, no nesting increment), else +1
    "booleans": (5, 1, 4),  # if +1, `&&` run +1; if +1, `&&` +1, `||` +1 (each operator change)
    "ternary": (1, 1, 1),  # ternary operator +1
    "tryCatch": (3, 2, 4),  # try adds nothing, catch +1, the if inside the catch +2
    "fact": (2, 1, 3),  # if +1, recursion +1
    "lambdaInLoop": (4, 3, 4),  # for +1, the lambda raises nesting to 2, the if inside it +3
    "labelled": (7, 3, 4),  # for +1, for +2, if +3, `continue outer` +1 (a jump to a label)
    # switch +1, the while in a case +2; a plain `break` adds nothing. The two case groups are
    # statement blocks whose labels count as statements: 1 + 3 + 2 + 1.
    "switcher": (3, 2, 7),
    "work": (0, 0, 0),
    "run": (0, 0, 0),
}

#: Comment lines per fixture: Java has a line comment and a two-line block comment.
COMMENT_LINES = {language: 1 for language in EXPECTED} | {"java": 3}


def _bare_calls(profile: LanguageProfile) -> CallReader:
    """A duplication profile's call names as receiverless calls: the fixtures' recursion is a bare call."""
    return lambda node, source: [Called(name, None) for name in profile.call_names(node, source)]


def _measure(path: Path, profile: LanguageProfile) -> tuple[int, dict[str, CallableMetrics]]:
    """Measure every top-level callable of one file through its duplication profile's readers."""
    source = path.read_bytes()
    parser = profile.parser()
    assert parser is not None
    tree = parser.parse(source)
    root = tree.root_node
    spec = complexity_for(profile.name)
    assert spec is not None, f"{profile.name} is measured"
    measured: dict[str, CallableMetrics] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        body = profile.member_body(node, source) if profile.member_kind(node, source) is not None else None
        if body is None:
            stack.extend(node.children)
            continue
        name = profile.member_name(node, source)
        measured[name] = measure_callable(
            body, spec, name, source, block_kinds=profile.block_kinds, calls=_bare_calls(profile)
        )
    return comment_lines(tree, spec), measured


def _fixture(language: str) -> tuple[int, dict[str, CallableMetrics]]:
    """Measure the one fixture file of a language."""
    (path,) = (FIXTURES / language).iterdir()
    profile = profile_for(path)
    assert profile is not None and profile.name == language
    return _measure(path, profile)


@pytest.mark.parametrize("language", sorted(EXPECTED))
def test_fixture_complexity_is_hand_counted(language: str) -> None:
    """Every shape a language can write measures what Sonar's rules say, in every grammar."""
    comments, measured = _fixture(language)
    got = {name: (measured[name].complexity, measured[name].nesting) for name in EXPECTED[language]}
    assert got == EXPECTED[language]
    assert measured["Seq" if language == "csharp" else "seq"].statements == 3, (
        "three statements in the straight-line function"
    )
    assert comments == COMMENT_LINES[language]


def test_java_against_sonar() -> None:
    """The Java fixture, every function, matches the white paper's increments by hand."""
    comments, measured = _fixture("java")
    got = {name: (metric.complexity, metric.nesting, metric.statements) for name, metric in measured.items()}
    assert got == JAVA
    assert measured["seq"].lines == 5, "the body spans its braces"
    assert comments == COMMENT_LINES["java"]


def _java_body(source: bytes):
    """The body of the first method in a Java snippet."""
    root = get_parser("java").parse(source).root_node
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "method_declaration":
            return node.child_by_field_name("body")
        stack.extend(node.children)
    raise AssertionError("no method")


def test_recursion_needs_a_name_and_a_call_reader() -> None:
    """Without the callable's own name or a way to read calls, recursion is not counted."""
    source = b"class A { int f(int n) { return f(n - 1); } }"
    body = _java_body(source)
    java = profile_for("A.java")
    assert java is not None
    counted = measure_callable(
        body, JAVA_COMPLEXITY, "f", source, block_kinds=java.block_kinds, calls=_bare_calls(java)
    )
    assert counted.complexity == 1, "a call to its own name counts"
    assert measure_callable(body, JAVA_COMPLEXITY, None, source, block_kinds=java.block_kinds).complexity == 0
    assert measure_callable(body, JAVA_COMPLEXITY, "f", source, block_kinds=java.block_kinds).complexity == 0


def _complexity_by_name(extraction) -> dict[str, int]:
    """Each measured callable's complexity, keyed by its qualified name."""
    names = {symbol.id: symbol.qualified_name for symbol in extraction.symbols}
    return {names[symbol_id]: metric.complexity for symbol_id, metric in extraction.metrics.items()}


def test_java_recursion_is_a_call_on_its_own_receiver() -> None:
    """`items.size()` inside `size()` is another object's method; `this.size()`, `Box.size()` and `size()` recurse."""
    source = b"""class Box {
    java.util.List<String> items;
    int size() { return items.size(); }
    int self(int n) { return this.self(n - 1); }
    int bare(int n) { return bare(n - 1); }
    static int typed(int n) { return Box.typed(n - 1); }
    int parent() { return super.parent(); }
}
"""
    got = _complexity_by_name(extract_java_file(source, "src/main/java/Box.java"))
    assert got["Box.size"] == 0, "a call on a field is not recursion"
    assert got["Box.self"] == 1, "a call on `this` is"
    assert got["Box.bare"] == 1, "so is a bare call"
    assert got["Box.typed"] == 1, "and a call on the enclosing type's own name"
    assert got["Box.parent"] == 0, "`super.f()` calls the parent's method"


def test_python_recursion_is_a_call_on_self() -> None:
    """`self.walk()` inside `walk` recurses; `other.walk()` walks another object."""
    source = b"""class Tree:
    def walk(self, n):
        return self.walk(n - 1)

    def visit(self, other):
        return other.visit(self)

    def size(self):
        return len(self.items.size())
"""
    got = _complexity_by_name(extract_generic_file(source, "tree.py", "python"))
    assert got["tree.Tree.walk"] == 1, "a call on self is recursion"
    assert got["tree.Tree.visit"] == 0, "a call on another receiver is not"
    assert got["tree.Tree.size"] == 0, "nor is a call on an attribute of self"


def test_go_method_receivers_are_not_counted() -> None:
    """A Go receiver parameter is a name the writer chose, so `s.walk()` is not read as recursion; a bare call is."""
    source = b"""package tree

type Store struct{}

func (s *Store) walk(n int) int { return s.walk(n - 1) }

func count(n int) int { return count(n - 1) }
"""
    got = _complexity_by_name(extract_generic_file(source, "tree.go", "go"))
    assert got["tree.Store.walk"] == 0, "the receiver parameter is not a self spelling"
    assert got["tree.count"] == 1, "a bare call to the function itself is recursion"


def test_a_long_else_if_chain_is_measured_without_recursing() -> None:
    """An else-if chain nests one tree level per branch; the walk must not hit the recursion limit."""
    branches = 3000
    chain = " else ".join(f"if (n == {index}) {{ n++; }}" for index in range(branches))
    source = f"class A {{ void f(int n) {{ {chain} }} }}".encode()
    measured = measure_callable(_java_body(source), JAVA_COMPLEXITY, None, source, block_kinds=frozenset({"block"}))
    assert (measured.complexity, measured.nesting) == (branches, 1), "every else-if is one flat increment"


@pytest.mark.parametrize("sample", sorted(SAMPLES.iterdir()), ids=lambda path: path.name)
def test_every_polyglot_sample_measures(sample: Path) -> None:
    """Measuring any bundled sample never raises and never answers a negative number."""
    language = detect_language(sample)
    if complexity_for(language) is None:
        spec = spec_for(language)
        assert spec is None or not spec.has_callable_bodies(), f"{language} has callables but is not measured"
        return
    profile = profile_for(sample)
    assert profile is not None, f"{sample.name} has a duplication profile to read its callables"
    comments, measured = _measure(sample, profile)
    assert comments >= 0
    assert measured, f"{sample.name} declares callables"
    for name, metric in measured.items():
        assert min(metric.statements, metric.complexity, metric.nesting) >= 0, name
        assert metric.lines >= 1, name
