"""Context capsules: a short description of WHERE a chunk lives, built at chunking time.

The capsule is embedding text, never search output: it is stored on the chunk so the
dense vector (and optionally BM25) can see the package, the enclosing type chain and
the member signature that the chunk body itself never repeats.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

from tree_sitter import Node

from zemble.hwk import TemplateFacts, scan
from zemble.types import Chunk

#: Separator between capsule segments; the first segment is always the path.
SEGMENT_SEPARATOR = " | "

_MAX_IMPORTS = 10
_MAX_HWK_TAGS = 8
_MAX_HWK_BLOCKS = 6
_MAX_SIGNATURE_CHARS = 200
_MAX_TYPE_CHAIN = 4
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SPACE_BEFORE_BRACKET = re.compile(r"\s+([(<])")

#: Java type declarations, mapped to the keyword a reader expects to see.
_JAVA_TYPE_NODES = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "annotation_type_declaration": "@interface",
}

#: Java members whose signature is worth naming when a chunk starts inside one.
_JAVA_MEMBER_NODES = frozenset(
    {
        "method_declaration",
        "constructor_declaration",
        "compact_constructor_declaration",
        "field_declaration",
        "annotation_type_element_declaration",
        "enum_constant",
    }
)

#: Node-type suffixes stripped when naming a definition in a non-Java grammar.
_GENERIC_SUFFIXES = ("_declaration", "_definition", "_specifier", "_item", "_statement")

#: Node-type stems that count as an enclosing definition in a non-Java grammar.
_GENERIC_STEMS = (
    "class",
    "function",
    "method",
    "struct",
    "interface",
    "module",
    "impl",
    "enum",
    "trait",
    "namespace",
    "object",
    "type",
    "union",
    "protocol",
    "record",
)


class CapsuleLevel(str, Enum):
    """How much context a capsule carries."""

    OFF = "off"
    LITE = "lite"
    FULL = "full"


#: The shipped default, measured: full capsules in both lanes beat every other variant.
DEFAULT_LEVEL = CapsuleLevel.FULL

#: Whether capsule tokens are appended to the BM25 document by default.
DEFAULT_IN_BM25 = True


@dataclass(frozen=True)
class CapsuleOptions:
    """The capsule knobs an index was built with; part of the cache identity."""

    level: CapsuleLevel = DEFAULT_LEVEL
    in_bm25: bool = DEFAULT_IN_BM25

    @property
    def key(self) -> str:
        """A stable string identifying this configuration, stored in index metadata."""
        return f"{self.level.value}+bm25" if self.in_bm25 else self.level.value

    @classmethod
    def from_key(cls, key: str) -> CapsuleOptions:
        """Parse a stored :attr:`key` back into options; an unreadable key reads as OFF."""
        level_name, _, suffix = key.partition("+")
        try:
            level = CapsuleLevel(level_name)
        except ValueError:
            return cls(level=CapsuleLevel.OFF, in_bm25=False)
        return cls(level=level, in_bm25=suffix == "bm25")

    @classmethod
    def resolve(cls, options: CapsuleOptions | None = None) -> CapsuleOptions:
        """Return the explicit options, else the environment override, else the defaults.

        ``ZEMBLE_CAPSULE`` takes ``off``/``lite``/``full``; ``ZEMBLE_CAPSULE_BM25`` is a
        boolean. Both exist so a benchmark can sweep variants without an API change.
        """
        if options is not None:
            return options
        raw_level = os.environ.get("ZEMBLE_CAPSULE")
        level = DEFAULT_LEVEL
        if raw_level:
            try:
                level = CapsuleLevel(raw_level.strip().lower())
            except ValueError:
                level = DEFAULT_LEVEL
        raw_bm25 = os.environ.get("ZEMBLE_CAPSULE_BM25")
        in_bm25 = DEFAULT_IN_BM25 if raw_bm25 is None else raw_bm25.strip().lower() in ("1", "true", "yes", "on")
        return cls(level=level, in_bm25=in_bm25)


def _text(node: Node | None) -> str:
    """Return a node's source text, collapsed to single spaces."""
    if node is None or node.text is None:
        return ""
    return _WHITESPACE.sub(" ", node.text.decode("utf-8", errors="replace")).strip()


def _child_name(node: Node) -> str:
    """Return the ``name`` field of a node, or an empty string."""
    return _text(node.child_by_field_name("name"))


def _path_words(file_path: str) -> str:
    """Return the path's own segments as words, de-duplicated, order preserved."""
    seen: list[str] = []
    for word in _WORD.findall(file_path.replace("/", " ").replace("\\", " ")):
        if word not in seen:
            seen.append(word)
    return " ".join(seen)


@dataclass
class FileContext:
    """The per-file facts every capsule in that file is assembled from."""

    file_path: str
    source: str = ""
    language: str | None = None
    root: Node | None = None
    _line_offsets: list[int] = field(default_factory=list, repr=False)
    _data: bytes = field(default=b"", repr=False)

    def __post_init__(self) -> None:
        """Encode the source once and record the byte offset each 1-based line starts at.

        The encoded form is kept because every chunk in the file anchors against it; re-encoding
        per chunk made indexing a large workspace quadratic in file size.
        """
        self._data = self.source.encode("utf-8")
        offsets = [0]
        for line in self._data.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        self._line_offsets = offsets

    @property
    def is_java(self) -> bool:
        """Whether the Java-specific capsule lane applies."""
        return self.language == "java"

    @property
    def is_hwk(self) -> bool:
        """Whether the Hawkeye-template capsule lane applies."""
        return self.language == "hwk"

    @cached_property
    def hwk_facts(self) -> TemplateFacts:
        """The template's declared tags, parent, blocks and used elements."""
        return scan(self.source)

    def anchor_byte(self, start_line: int) -> int:
        """Return the byte offset of the first non-whitespace character at or after a 1-based line."""
        last = len(self._line_offsets) - 1
        line_index = min(max(start_line, 1), max(last, 1))
        offset = self._line_offsets[line_index - 1]
        data = self._data
        while offset < len(data) and data[offset : offset + 1].isspace():
            offset += 1
        return min(offset, max(len(data) - 1, 0))

    @cached_property
    def package(self) -> str:
        """The Java package declaration text, or an empty string."""
        if self.root is None or not self.is_java:
            return ""
        for child in self.root.children:
            if child.type == "package_declaration":
                return _text(child).rstrip(";")
        return ""

    @cached_property
    def imports(self) -> list[str]:
        """Simple names of every Java import, wildcard imports excluded."""
        if self.root is None or not self.is_java:
            return []
        names: list[str] = []
        for child in self.root.children:
            if child.type != "import_declaration":
                continue
            text = _text(child).rstrip(";")
            simple = text.rsplit(".", 1)[-1].strip()
            if simple and simple != "*" and simple not in names:
                names.append(simple)
        return names


def _java_type_label(node: Node) -> str:
    """Render one Java type declaration as ``class Foo extends Bar implements Baz``."""
    parts = [_JAVA_TYPE_NODES[node.type], _child_name(node)]
    superclass = node.child_by_field_name("superclass")
    if superclass is not None:
        parts.append(_text(superclass))
    interfaces = node.child_by_field_name("interfaces")
    if interfaces is not None:
        parts.append(_text(interfaces))
    return " ".join(part for part in parts if part)


def _generic_definition_label(node: Node) -> str:
    """Render a definition node from a non-Java grammar as ``kind name``, or nothing."""
    node_type = node.type
    for suffix in _GENERIC_SUFFIXES:
        if node_type.endswith(suffix):
            node_type = node_type[: -len(suffix)]
            break
    if not any(stem in node_type for stem in _GENERIC_STEMS):
        return ""
    name = _child_name(node)
    return f"{node_type.replace('_', ' ')} {name}".strip() if name else ""


def _annotations(node: Node) -> str:
    """Return the annotations declared on a Java member, space separated."""
    modifiers = node.child_by_field_name("modifiers")
    if modifiers is None:
        for child in node.children:
            if child.type == "modifiers":
                modifiers = child
                break
    if modifiers is None:
        return ""
    names = [_text(child) for child in modifiers.children if child.type.endswith("annotation")]
    return " ".join(name for name in names if name)


def _java_member_signature(node: Node) -> str:
    """Return a member's signature: everything up to its body, annotations removed."""
    body = node.child_by_field_name("body") or node.child_by_field_name("value")
    end = body.start_byte if body is not None else node.end_byte
    pieces: list[str] = []
    for child in node.children:
        if child.start_byte >= end:
            break
        if child.type == "modifiers":
            pieces.extend(_text(part) for part in child.children if not part.type.endswith("annotation"))
            continue
        pieces.append(_text(child))
    signature = _WHITESPACE.sub(" ", " ".join(piece for piece in pieces if piece)).strip().rstrip(";")
    return _SPACE_BEFORE_BRACKET.sub(r"\1", signature)[:_MAX_SIGNATURE_CHARS]


def _enclosing_nodes(root: Node, anchor: int) -> list[Node]:
    """Return the ancestor chain of the smallest node containing the anchor byte.

    The chunk's START decides, never its span: a chunk that spans two methods belongs to the
    one it opens in, and a chunk starting mid-method still names that method. A chunk that
    opens on a doc comment is credited to what the comment documents, not to the comment.
    """
    node = root.descendant_for_byte_range(anchor, anchor)
    if node is None:
        return []
    if node.type.endswith("comment") and (following := node.next_named_sibling) is not None:
        node = following
    chain: list[Node] = []
    while node is not None:
        chain.append(node)
        node = node.parent
    return list(reversed(chain))


def _used_imports(context: FileContext, content: str) -> list[str]:
    """Return the imported simple names that literally appear inside the chunk."""
    if not context.imports:
        return []
    words = set(_WORD.findall(content))
    return [name for name in context.imports if name in words][:_MAX_IMPORTS]


def capsule(chunk: Chunk, file_context: FileContext, level: CapsuleLevel = CapsuleLevel.FULL) -> str:
    """Describe where a chunk lives: path, package, type chain, member signature, imports.

    A file with no tree (line chunking, or an unknown grammar) yields the path segment only.
    """
    if level is CapsuleLevel.OFF:
        return ""

    segments = [f"{file_context.file_path} {_path_words(file_context.file_path)}".strip()]
    if file_context.is_hwk:
        # A template's structure is lexical, not syntactic: the borrowed html grammar knows
        # nothing about `{% tag %}`, `extend` or `block`, so the capsule is built from the
        # template's own facts and works even where no grammar loaded at all.
        segments.extend(_hwk_segments(chunk, file_context.hwk_facts, level))
        return SEGMENT_SEPARATOR.join(segment for segment in segments if segment)
    root = file_context.root
    if root is None:
        return segments[0]

    chain = _enclosing_nodes(root, file_context.anchor_byte(chunk.start_line))

    if file_context.is_java:
        if file_context.package:
            segments.append(file_context.package)
        types = [_java_type_label(node) for node in chain if node.type in _JAVA_TYPE_NODES]
        member = next((node for node in reversed(chain) if node.type in _JAVA_MEMBER_NODES), None)
    else:
        types = [label for node in chain if (label := _generic_definition_label(node))]
        member = None

    if types:
        segments.append(" > ".join(types[-_MAX_TYPE_CHAIN:]))

    if level is CapsuleLevel.FULL:
        segments.extend(_member_segments(member, chunk, file_context))

    return SEGMENT_SEPARATOR.join(segment for segment in segments if segment)


def _hwk_segments(chunk: Chunk, facts: TemplateFacts, level: CapsuleLevel) -> list[str]:
    """Describe where a chunk lives inside a Hawkeye template.

    The custom element the chunk is inside beats the file's other declarations, because a
    component file may declare several tags and only one of them owns these lines.
    """
    segments = []
    declaration = facts.tag_at(chunk.start_line)
    if declaration is not None:
        segments.append(f"tag <{declaration.tag}> {declaration.class_name}")
    if facts.extends is not None:
        segments.append(f"extends {facts.extends.target}")
    block = facts.block_at(chunk.start_line)
    if block is not None:
        segments.append(f"block {block.name}")
    elif facts.blocks:
        segments.append("blocks " + " ".join(item.name for item in facts.blocks[:_MAX_HWK_BLOCKS]))
    if level is CapsuleLevel.FULL:
        used = facts.tags_used_between(chunk.start_line, chunk.end_line)[:_MAX_HWK_TAGS]
        if used:
            segments.append(f"uses {' '.join(used)}")
    return segments


def _member_segments(member: Node | None, chunk: Chunk, file_context: FileContext) -> list[str]:
    """Return the FULL-level segments: the member's annotations, its signature, its used imports."""
    segments = []
    if member is not None:
        segments.append(_annotations(member))
        segments.append(_java_member_signature(member))
    used = _used_imports(file_context, chunk.content)
    if used:
        segments.append(f"uses {' '.join(used)}")
    return [segment for segment in segments if segment]


def capsule_without_path(context: str) -> str:
    """Return a capsule minus its leading path segment, whose tokens BM25 already enriches with."""
    if not context:
        return ""
    _, separator, rest = context.partition(SEGMENT_SEPARATOR)
    return rest if separator else ""


def embedding_text(chunk: Chunk) -> str:
    """Return the text a chunk is embedded as: its capsule, then its content."""
    return f"{chunk.context}\n{chunk.content}" if chunk.context else chunk.content
