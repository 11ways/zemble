"""Behaviour journeys over signature-qualified graph queries: `Type.method(Param,...)` separates overloads."""

import hashlib
import json
from pathlib import Path

import pytest

from zemble.cli import _cli_main
from zemble.graph.cli import EXIT_AMBIGUOUS, EXIT_NOT_FOUND

DATA = "src/main/java/com/example/data"
APP = "src/main/java/com/example/app"

SOURCES = {
    f"{DATA}/Ref.java": "package com.example.data;\n\npublic class Ref {\n}\n",
    f"{DATA}/RecordSource.java": (
        "package com.example.data;\n"
        "\n"
        "public class RecordSource {\n"
        "    public static RecordSource of(Class<?> type) {\n"
        "        return new RecordSource();\n"
        "    }\n"
        "\n"
        "    public static RecordSource of(Ref ref) {\n"
        "        return new RecordSource();\n"
        "    }\n"
        "}\n"
    ),
    f"{DATA}/Opener.java": (
        "package com.example.data;\n"
        "\n"
        "public interface Opener {\n"
        "    void open(Class<?> type);\n"
        "\n"
        "    void open(Ref ref);\n"
        "}\n"
    ),
    f"{DATA}/FileOpener.java": (
        "package com.example.data;\n"
        "\n"
        "public class FileOpener implements Opener {\n"
        "    public void open(Class<?> type) {\n"
        "    }\n"
        "\n"
        "    public void open(Ref ref) {\n"
        "    }\n"
        "}\n"
    ),
    f"{APP}/ByClass.java": (
        "package com.example.app;\n"
        "\n"
        "import com.example.data.RecordSource;\n"
        "\n"
        "public class ByClass {\n"
        "    public RecordSource fromClass() {\n"
        "        return RecordSource.of(String.class);\n"
        "    }\n"
        "}\n"
    ),
    f"{APP}/ByRef.java": (
        "package com.example.app;\n"
        "\n"
        "import com.example.data.RecordSource;\n"
        "import com.example.data.Ref;\n"
        "\n"
        "public class ByRef {\n"
        "    public RecordSource fromRef() {\n"
        "        return RecordSource.of(new Ref());\n"
        "    }\n"
        "}\n"
    ),
    # No facts cover this file, so tree-sitter cannot tell which overload it calls.
    f"{APP}/Loose.java": (
        "package com.example.app;\n"
        "\n"
        "import com.example.data.RecordSource;\n"
        "\n"
        "public class Loose {\n"
        "    public RecordSource guess(Object value) {\n"
        "        return RecordSource.of(value);\n"
        "    }\n"
        "}\n"
    ),
}


def _sha(workspace: Path, relative: str) -> str:
    """Hex sha256 of a workspace file's current content."""
    return hashlib.sha256((workspace / relative).read_bytes()).hexdigest()


def _workspace(root: Path) -> Path:
    """Write the overload workspace plus javac facts for every file but `Loose.java`."""
    for relative, text in SOURCES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    by_class, by_ref, file_opener = f"{APP}/ByClass.java", f"{APP}/ByRef.java", f"{DATA}/FileOpener.java"
    lines = [
        {
            "zemble_facts": 1,
            "tool": "javac-facts",
            "tool_version": "0.1.0",
            "generated_at": "2026-08-20T09:00:00Z",
            "language": "java",
            "root": "../..",
        },
        {"t": "file", "path": by_class, "sha256": _sha(root, by_class)},
        {
            "t": "call",
            "from": "com.example.app.ByClass#fromClass()",
            "to": "com.example.data.RecordSource#of(java.lang.Class)",
            "path": by_class,
            "line": 7,
        },
        {"t": "file", "path": by_ref, "sha256": _sha(root, by_ref)},
        {
            "t": "call",
            "from": "com.example.app.ByRef#fromRef()",
            "to": "com.example.data.RecordSource#of(com.example.data.Ref)",
            "path": by_ref,
            "line": 8,
        },
        {"t": "file", "path": file_opener, "sha256": _sha(root, file_opener)},
        {
            "t": "override",
            "from": "com.example.data.FileOpener#open(java.lang.Class)",
            "to": "com.example.data.Opener#open(java.lang.Class)",
        },
        {
            "t": "override",
            "from": "com.example.data.FileOpener#open(com.example.data.Ref)",
            "to": "com.example.data.Opener#open(com.example.data.Ref)",
        },
    ]
    facts = root / ".zemble" / "facts" / "javac-facts.jsonl"
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return root


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Run the zemble CLI with the given arguments and return its exit code."""
    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    monkeypatch.setattr("sys.argv", ["zemble", *argv])
    try:
        _cli_main()
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    return 0


def test_a_signature_query_parses_into_a_name_and_erased_parameters() -> None:
    """The written form splits at the top-level commas and erases what a stored parameter type never holds."""
    from zemble.graph.signature import SymbolQuery

    assert SymbolQuery.parse("RecordSource.of") == SymbolQuery("RecordSource.of", None), "a bare name has no params"
    assert SymbolQuery.parse("Foo.bar()") == SymbolQuery("Foo.bar", ()), "empty parentheses mean no parameters"
    parsed = SymbolQuery.parse(" Foo.bar( java.util.Map<String, List<Integer>> , String... , int[] ) ")
    assert parsed == SymbolQuery("Foo.bar", ("Map", "String[]", "int[]")), "generics erased, varargs as arrays"
    assert SymbolQuery.parse("Foo.bar(Map<String").params is None, "an unbalanced form is a plain name"


def test_overloads_are_separated_by_a_signature_query(
    tmp_path: Path, graph_cache: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Callers, definition, overrides and signatures answer one overload when the query names its parameters."""
    from zemble.evidence.answers import signatures_payload
    from zemble.graph.mcp import answer
    from zemble.graph.provider import open_provider

    root = str(_workspace(tmp_path / "ws"))

    # 1. A bare name with two overloads still refuses, but names each overload's signature to narrow with.
    assert _run(monkeypatch, "graph", "callers", root, "RecordSource.of") == EXIT_AMBIGUOUS, "step 1: exits 2"
    error = capsys.readouterr().err
    assert "2 overloads" in error, "step 1: the refusal says the candidates are overloads"
    assert "RecordSource.of(Class)" in error and "RecordSource.of(Ref)" in error, "step 1: and lists them"

    # 2. The `of(Class)` signature answers only the `of(Class)` call site, exactly, from the compiler facts.
    assert _run(monkeypatch, "graph", "callers", root, "RecordSource.of(Class)", "--json") == 0, "step 2: answers"
    payload = json.loads(capsys.readouterr().out)
    assert payload["symbol"]["id"].endswith("RecordSource.of(Class)"), "step 2: the of(Class) overload is chosen"
    assert [(hit["qualified_name"], hit["resolution"]) for hit in payload["results"]] == [
        ("com.example.app.ByClass.fromClass", "exact")
    ], "step 2: only the of(Class) caller, graded exact"
    assert payload["overloads"] == ["RecordSource.of(Class)", "RecordSource.of(Ref)"], "step 2: siblings are named"
    assert "1 ambiguous call site" in payload["overload_note"], "step 2: the unattributable call is counted"

    # 3. A qualified parameter type selects the other overload, and the text answer states the same.
    _run(monkeypatch, "graph", "callers", root, "RecordSource.of(com.example.data.Ref)")
    output = capsys.readouterr().out
    assert "called from ByRef.fromRef" in output, "step 3: the of(Ref) caller is listed"
    assert "ByClass" not in output, "step 3: and the of(Class) caller is not"
    assert "1 ambiguous call site" in output, "step 3: the text answer names the unattributed call too"

    # 4. Definition narrows to the one declaration; a signature nothing declares lists what is declared.
    provider = open_provider(root)
    try:
        assert [symbol.param_types for symbol in provider.definition("RecordSource.of(Class)")] == [["Class"]], (
            "step 4: definition keeps the matching overload only"
        )
        described = signatures_payload(provider, "RecordSource.of(Ref)")
    finally:
        provider.close()
    assert _run(monkeypatch, "graph", "callers", root, "RecordSource.of(String)") == EXIT_NOT_FOUND, "step 4: exits 1"
    error = capsys.readouterr().err
    assert "No overload" in error and "RecordSource.of(Ref)" in error, "step 4: and names the declared overloads"

    # 5. The MCP payload answers the same question the same way.
    payload = answer(root, "RecordSource.of(Ref)", "callers")
    assert [hit["qualified_name"] for hit in payload["results"]] == ["com.example.app.ByRef.fromRef"], "step 5"
    refusal = answer(root, "RecordSource.of", "callers")
    assert refusal["overloads"] == ["RecordSource.of(Class)", "RecordSource.of(Ref)"], "step 5: refusal lists them"

    # 6. Overrides of one overload are that overload's only.
    payload = answer(root, "Opener.open(Ref)", "overridden_by")
    assert [hit["line"] for hit in payload["results"]] == [7], "step 6: only FileOpener.open(Ref), at line 7"

    # 7. Signatures accept the same form (described in step 4, while the provider was open).
    assert "error" not in described, "step 7: one overload is described"
    assert "Ref ref" in described["text"], "step 7: and it is the of(Ref) one"
