import textwrap
from pathlib import Path

import orjson
import pytest

from tests.conftest import FakeEmbedder
from zemble.chunking import chunk_source
from zemble.chunking.capsule import (
    DEFAULT_LEVEL,
    CapsuleLevel,
    CapsuleOptions,
    FileContext,
    capsule,
    capsule_without_path,
    embedding_text,
)
from zemble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS
from zemble.chunking.core import chunk_with_tree
from zemble.index import ZembleIndex
from zemble.types import Chunk

JAVA_SOURCE = textwrap.dedent(
    """\
    package be.elevenways.zenit.common.http;

    import java.util.List;
    import java.util.Map;
    import org.checkerframework.checker.nullness.qual.NonNull;

    /** Doc comment for the outer type. */
    public class EntityTags extends BaseTags implements Comparable<EntityTags>, Cloneable {

        private static final String WEAK_PREFIX = "W/";

        /** Doc comment for the method. */
        @Override
        @NonNull
        public static boolean matchesIfNoneMatch(String ifNoneMatch, String quotedEtag) {
            List<String> parts = List.of(ifNoneMatch);
            return parts.contains(quotedEtag);
        }

        private String stripWeak(String tag) {
            return tag;
        }

        public static class Inner implements Runnable {
            public void run() {
                Map<String, String> empty = Map.of();
            }
        }

        public record Pair(String left, String right) {
            public String joined() {
                return left + right;
            }
        }

        public enum Weakness {
            WEAK,
            STRONG;

            public boolean isWeak() {
                return this == WEAK;
            }
        }

        public @interface Marker {
            String value();
        }
    }
    """
)


def java_context(source: str = JAVA_SOURCE, path: str = "zenit/src/common/java/EntityTags.java") -> FileContext:
    """Build a FileContext over Java source, parsed exactly the way chunking parses it."""
    parsed = chunk_with_tree(source, "java", _DESIRED_CHUNK_LENGTH_CHARS)
    assert parsed is not None, "the bundled Java grammar must be available"
    return FileContext(file_path=path, source=source, language="java", root=parsed[1])


def line_at(source: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing a needle."""
    for number, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} not found in source")


def capsule_for(needle: str, *, end: str | None = None, level: CapsuleLevel = CapsuleLevel.FULL) -> str:
    """Build the capsule of a synthetic chunk that starts at one line and ends at another."""
    start_line = line_at(JAVA_SOURCE, needle)
    end_line = line_at(JAVA_SOURCE, end) if end else start_line
    lines = JAVA_SOURCE.splitlines()[start_line - 1 : end_line]
    chunk = Chunk(
        content="\n".join(lines),
        file_path="zenit/src/common/java/EntityTags.java",
        start_line=start_line,
        end_line=end_line,
        language="java",
    )
    return capsule(chunk, java_context(), level)


def test_capsule_names_path_package_type_and_signature() -> None:
    """A chunk inside a method carries path words, package, type chain, annotations and signature."""
    built = capsule_for("List<String> parts")
    assert built.startswith("zenit/src/common/java/EntityTags.java zenit src common java EntityTags")
    assert "package be.elevenways.zenit.common.http" in built
    assert "class EntityTags extends BaseTags implements Comparable<EntityTags>, Cloneable" in built
    assert "@Override @NonNull" in built
    assert "public static boolean matchesIfNoneMatch(String ifNoneMatch, String quotedEtag)" in built
    # Only the imports actually used inside the chunk are listed.
    assert "uses List" in built
    assert "Map" not in built.split("uses ")[1]


def test_capsule_chains_nested_types() -> None:
    """A chunk in a nested type names the whole enclosing type chain, outermost first."""
    built = capsule_for("Map<String, String> empty")
    assert "class EntityTags extends BaseTags implements Comparable<EntityTags>, Cloneable > class Inner" in built
    assert "implements Runnable" in built
    assert "public void run()" in built


def test_capsule_of_chunk_spanning_two_methods_names_the_one_it_starts_in() -> None:
    """The START decides: a chunk running from one method into the next is credited to the first."""
    built = capsule_for("List<String> parts", end="return tag;")
    assert "matchesIfNoneMatch" in built
    assert "stripWeak" not in built


def test_capsule_of_chunk_starting_mid_method_still_names_the_method() -> None:
    """A chunk opening on a statement inside a body names that body's member."""
    built = capsule_for("return parts.contains")
    assert "public static boolean matchesIfNoneMatch(String ifNoneMatch, String quotedEtag)" in built


def test_capsule_of_chunk_starting_on_a_doc_comment_names_what_it_documents() -> None:
    """A doc comment is credited to the member below it, not left contextless."""
    built = capsule_for("Doc comment for the method")
    assert "matchesIfNoneMatch" in built


def test_capsule_names_record_enum_and_annotation_types() -> None:
    """Records, enums and annotation types are named with their own keyword."""
    assert "record Pair" in capsule_for("return left + right;")
    assert "enum Weakness" in capsule_for("return this == WEAK;")
    assert "@interface Marker" in capsule_for("String value();")


def test_capsule_names_a_field_declaration() -> None:
    """A chunk on a field carries that field's declaration as its signature."""
    built = capsule_for("WEAK_PREFIX")
    assert "private static final String WEAK_PREFIX" in built


def test_capsule_lite_drops_signatures_and_imports() -> None:
    """The lite level keeps path and type chain only."""
    built = capsule_for("List<String> parts", level=CapsuleLevel.LITE)
    assert "class EntityTags" in built
    assert "matchesIfNoneMatch" not in built
    assert "uses" not in built


def test_capsule_off_is_empty() -> None:
    """The off level builds nothing at all."""
    assert capsule_for("List<String> parts", level=CapsuleLevel.OFF) == ""


def test_capsule_of_line_chunked_file_is_path_only() -> None:
    """A file with no parse tree yields the path segment and nothing else."""
    context = FileContext(file_path="data/values.pkl", source="a=1\nb=2\n", language=None, root=None)
    chunk = Chunk(content="a=1\nb=2\n", file_path="data/values.pkl", start_line=1, end_line=2)
    assert capsule(chunk, context) == "data/values.pkl data values pkl"


def test_capsule_of_non_java_language_uses_the_generic_subset() -> None:
    """A tree-sitter language other than Java gets path plus the enclosing definition chain."""
    source = textwrap.dedent(
        """\
        class Config:
            def load(self, path):
                return path
        """
    )
    parsed = chunk_with_tree(source, "python", _DESIRED_CHUNK_LENGTH_CHARS)
    assert parsed is not None
    context = FileContext(file_path="src/config.py", source=source, language="python", root=parsed[1])
    chunk = Chunk(content="        return path", file_path="src/config.py", start_line=3, end_line=3)
    built = capsule(chunk, context)
    assert built == "src/config.py src config py | class Config > function load"


def test_chunk_source_stamps_capsules_only_when_enabled() -> None:
    """chunk_source carries the capsule on the chunk; the off level leaves it empty."""
    on = chunk_source(JAVA_SOURCE, "src/EntityTags.java", "java", CapsuleOptions(CapsuleLevel.FULL))
    off = chunk_source(JAVA_SOURCE, "src/EntityTags.java", "java", CapsuleOptions(CapsuleLevel.OFF))
    assert all(chunk.context for chunk in on)
    assert all(chunk.context == "" for chunk in off)
    assert [chunk.content for chunk in on] == [chunk.content for chunk in off]


def test_capsule_without_path_drops_the_leading_segment() -> None:
    """BM25 enrichment reuses everything but the path, which it already contributes itself."""
    assert capsule_without_path("a/b.java a b | package p | class C") == "package p | class C"
    assert capsule_without_path("a/b.java a b") == ""
    assert capsule_without_path("") == ""


def test_embedding_text_prefixes_the_capsule() -> None:
    """The embedded text is capsule then content; a chunk without a capsule embeds unchanged."""
    with_context = Chunk(content="body", file_path="a.java", start_line=1, end_line=1, context="a.java a java")
    assert embedding_text(with_context) == "a.java a java\nbody"
    assert embedding_text(Chunk(content="body", file_path="a.java", start_line=1, end_line=1)) == "body"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("off", (CapsuleLevel.OFF, False)),
        ("full+bm25", (CapsuleLevel.FULL, True)),
        ("nonsense", (CapsuleLevel.OFF, False)),
    ],
)
def test_capsule_options_round_trip_through_their_key(key: str, expected: tuple[CapsuleLevel, bool]) -> None:
    """The metadata key parses back into the options that produced it, and refuses nonsense."""
    options = CapsuleOptions.from_key(key)
    assert (options.level, options.in_bm25) == expected


def test_capsule_options_resolve_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment override drives the level and the BM25 flag; explicit options win."""
    monkeypatch.setenv("ZEMBLE_CAPSULE", "lite")
    monkeypatch.setenv("ZEMBLE_CAPSULE_BM25", "1")
    assert CapsuleOptions.resolve() == CapsuleOptions(CapsuleLevel.LITE, True)
    explicit = CapsuleOptions(CapsuleLevel.OFF, False)
    assert CapsuleOptions.resolve(explicit) == explicit
    monkeypatch.setenv("ZEMBLE_CAPSULE", "bogus")
    assert CapsuleOptions.resolve().level == DEFAULT_LEVEL


def test_chunk_serialization_round_trips_the_context() -> None:
    """The capsule survives to_dict/from_dict and the orjson form the index is persisted as."""
    chunk = Chunk(
        content="body", file_path="a/B.java", start_line=2, end_line=4, language="java", context="a/B.java | class B"
    )
    assert Chunk.from_dict(chunk.to_dict()) == chunk
    assert Chunk.from_dict(orjson.loads(orjson.dumps(chunk))) == chunk


def test_chunk_without_context_loads_from_an_older_dict() -> None:
    """A persisted chunk that predates the field loads with an empty capsule."""
    restored = Chunk.from_dict(
        {"content": "body", "file_path": "a.py", "start_line": 1, "end_line": 1, "language": "python"}
    )
    assert restored.context == ""


def test_index_with_capsules_returns_original_content(tmp_path: Path, mock_embedder: FakeEmbedder) -> None:
    """End to end: capsules reach the embedder, but search results carry the untouched source."""
    (tmp_path / "Greeter.java").write_text(
        textwrap.dedent(
            """\
            package com.example.greeting;

            public class Greeter {
                public String greet(String name) {
                    return "hello " + name;
                }
            }
            """
        )
    )
    index = ZembleIndex.from_path(
        tmp_path, embedder=mock_embedder, capsules=CapsuleOptions(CapsuleLevel.FULL, in_bm25=True)
    )

    embedded = [text for call in mock_embedder.document_calls for text in call]
    assert any("package com.example.greeting" in text and "class Greeter" in text for text in embedded)
    assert all(chunk.context for chunk in index.chunks)

    results = index.search("greet", top_k=5)
    assert results
    for result in results:
        assert result.chunk.context not in result.chunk.content
        assert result.chunk.content in (tmp_path / "Greeter.java").read_text()

    # The capsule is a BM25 signal too: the package name appears in no chunk body.
    assert any("Greeter.java" in result.chunk.file_path for result in index.search("greeting package", top_k=5))


def test_find_related_embeds_the_seed_with_its_capsule(tmp_path: Path, mock_embedder: FakeEmbedder) -> None:
    """The seed chunk is embedded the way the index embedded it, so both sides share one convention."""
    (tmp_path / "Greeter.java").write_text(
        textwrap.dedent(
            """\
            package com.example.greeting;

            public class Greeter {
                public String greet(String name) {
                    return "hello " + name;
                }
            }
            """
        )
    )
    index = ZembleIndex.from_path(tmp_path, embedder=mock_embedder, capsules=CapsuleOptions(CapsuleLevel.FULL))
    seed = index.chunks[0]
    mock_embedder.query_calls.clear()
    index.find_related(seed, top_k=3)
    assert mock_embedder.query_calls == [[embedding_text(seed)]]
    assert seed.context and seed.context in mock_embedder.query_calls[0][0]


def _repo_workspace(root: Path) -> Path:
    """Write a workspace holding one git repo and one directory that is in no repo."""
    (root / "zenit" / ".git").mkdir(parents=True)
    (root / "zenit" / "src").mkdir()
    (root / "zenit" / "src" / "Greeter.java").write_text(
        textwrap.dedent(
            """\
            package be.example.greeting;

            public class Greeter {
                public String greet(String name) {
                    return "hello " + name;
                }
            }
            """
        ),
        encoding="utf-8",
    )
    (root / "loose").mkdir()
    (root / "loose" / "notes.py").write_text("def note():\n    return 1\n", encoding="utf-8")
    return root


def test_the_capsule_path_is_repo_relative_not_index_root_relative(tmp_path: Path) -> None:
    """A file in a git repo names itself the same way under every index root.

    This is what makes the embedding cache hit when a sub-repo is indexed on its own: the
    capsule is part of the embedded text, so a path that moves with the index root re-embeds
    the whole sub-tree for nothing.
    """
    from zemble.index.create import plan_files

    workspace = _repo_workspace(tmp_path / "work")

    # 1. Indexed as part of the workspace, the Java file is named by its repo and its inner path.
    from_workspace = {planned.indexed_path: planned.chunks for planned in plan_files(workspace, display_root=workspace)}
    assert "zenit/src/Greeter.java" in from_workspace, "the workspace stores root-relative paths"
    workspace_contexts = [chunk.context for chunk in from_workspace["zenit/src/Greeter.java"]]
    assert workspace_contexts[0].startswith("zenit/src/Greeter.java "), "the capsule names the repo"

    # 2. Indexed as its own root, the same file produces byte-identical capsules.
    repo_root = workspace / "zenit"
    from_repo = {planned.indexed_path: planned.chunks for planned in plan_files(repo_root, display_root=repo_root)}
    assert "src/Greeter.java" in from_repo, "chunk file paths stay relative to the index root"
    assert [chunk.context for chunk in from_repo["src/Greeter.java"]] == workspace_contexts, (
        "the capsule text does not depend on which root the index was built from"
    )

    # 3. Outside any git repo the capsule falls back to the index-root-relative path.
    loose_from_workspace = next(
        planned for planned in plan_files(workspace, display_root=workspace) if planned.indexed_path.startswith("loose")
    )
    assert loose_from_workspace.chunks[0].context.startswith("loose/notes.py "), "no repo, no rewrite"
    loose_root = workspace / "loose"
    loose_from_itself = next(iter(plan_files(loose_root, display_root=loose_root)))
    assert loose_from_itself.chunks[0].context.startswith("notes.py "), "and it stays root-relative there"
