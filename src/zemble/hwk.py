"""Lexical facts about a Hawkeye ``.hwk`` template.

A Hawkeye template is HTML carrying ``{% ... %}`` statements and ``{{ ... }}`` interpolations,
and a component file wraps its entire markup in one ``{% tag PascalName { ... } %}`` block, so
the statement delimiters NEST. The scanner here is deliberately lexical rather than a parser:
it reads the handful of facts the index and the graph need - the custom elements a file
declares, the template it extends, its blocks, the partials it renders, the custom element tags
it uses and the function calls it makes - and never builds a tree.

Both the capsule builder (`zemble.chunking.capsule`) and the graph extractor
(`zemble.graph.hwk`) read their facts from here, so the two can never drift.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

#: `{# ... #}` and `{#- ... -#}` template comments. Blanked before anything else is scanned.
_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)
#: The head of a `style { ... }` block, whose SCSS body is blanked: its `rgba(`/`var(` calls
#: are not template function calls, and its braces are not statement delimiters.
_STYLE = re.compile(r"(?m)^[ \t]*style[ \t]*\{")

_OPEN = "{%"
_CLOSE = "%}"
#: `{{ expression }}` interpolation, which carries calls exactly like a statement does.
_INTERPOLATION = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)

#: `//` (never `://`) and `/* */` comments inside a statement's code.
_CODE_COMMENT = re.compile(r"(?<!:)//[^\n]*|/\*.*?\*/", re.DOTALL)

_IDENTIFIER = re.compile(r"[A-Za-z_/][A-Za-z0-9_]*")
_QUOTED = re.compile(r"""["']([^"']*)["']""")
#: `{% tag PlButton {` and the brace-block form `tag TimeScope {` written inside one big
#: `{% ... %}`, optionally `abstract`. Hawkeye hoists tags globally, so a file may declare many.
_TAG_DECLARATION = re.compile(r"(?m)^[ \t]*(?:\{%[ \t]*)?(?:abstract[ \t]+)?tag[ \t]+([A-Za-z_][A-Za-z0-9_]*)")
#: A custom element is a tag name containing a hyphen; that is a platform rule, not a convention.
_CUSTOM_TAG = re.compile(r"<([a-z][a-z0-9]*(?:-[a-z0-9]+)+)")
#: `String.presence(...)`: a namespaced template function call. The leading `:` exclusion keeps
#: `use:List.moveUp(index: i)` - a directive, not a call - out of the results.
_NAMESPACED_CALL = re.compile(r"(?<![A-Za-z0-9_.$:])([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
#: `t("add")`: a call in the global namespace, which the resolver matches on name alone.
_BARE_CALL = re.compile(r"(?<![A-Za-z0-9_.$:])([a-z][A-Za-z0-9_]*)\s*\(")
#: Hawkeye's own PascalCase -> kebab-case rule (`TypeUtils.toKebabCase`), copied exactly.
_KEBAB = re.compile(r"([a-z])([A-Z]+)")

#: The path segment every template root ends with, used to derive a template id.
TEMPLATES_SEGMENT = "/templates/"

#: The Java annotation that registers a static method as a template function, and the
#: arguments naming the call it answers to. Declared here because the template side and the
#: Java side must agree on them, and a template writes exactly `namespace.name(...)`.
FUNCTION_ANNOTATION = "HawkeyeFunction"
FUNCTION_NAME_ARGUMENT = "name"
FUNCTION_NAMESPACE_ARGUMENT = "namespace"

#: The Java annotation that registers a hand-written class as a custom element, and the
#: argument carrying the element tag.
ELEMENT_ANNOTATION = "HawkeyeCustomElement"
ELEMENT_TAG_ARGUMENT = "tag"


def to_kebab_case(pascal_case: str) -> str:
    """Convert a PascalCase tag class name to the element tag Hawkeye registers for it."""
    return _KEBAB.sub(r"\1-\2", pascal_case).lower()


@dataclass(frozen=True)
class Reference:
    """A name written in a template, with the line it was written on."""

    target: str
    line: int


@dataclass(frozen=True)
class Call:
    """A template function call: `String.presence(x)`, or a bare `t("add")`."""

    name: str
    line: int
    namespace: str | None = None


@dataclass(frozen=True)
class Block:
    """A `{% block "name" %}` region and the lines it spans."""

    name: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class TagDeclaration:
    """A custom element declared by a template, and the lines its declaration owns."""

    class_name: str
    tag: str
    start_line: int
    end_line: int


@dataclass
class TemplateFacts:
    """Everything the index and the graph read out of one template file."""

    line_count: int = 1
    tags: list[TagDeclaration] = field(default_factory=list)
    extends: Reference | None = None
    blocks: list[Block] = field(default_factory=list)
    renders: list[Reference] = field(default_factory=list)
    used_tags: list[Reference] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)

    @property
    def tag(self) -> str | None:
        """The custom element this file declares, or None when it declares none or several."""
        return self.tags[0].tag if len(self.tags) == 1 else None

    def tag_at(self, line: int) -> TagDeclaration | None:
        """Return the tag declaration whose region contains a line."""
        for declaration in self.tags:
            if declaration.start_line <= line <= declaration.end_line:
                return declaration
        return None

    def block_at(self, line: int) -> Block | None:
        """Return the narrowest block region containing a line."""
        containing = [block for block in self.blocks if block.start_line <= line <= block.end_line]
        return min(containing, key=lambda block: block.end_line - block.start_line) if containing else None

    def tags_used_between(self, start_line: int, end_line: int) -> list[str]:
        """Return the distinct custom element tags written inside a line range, in source order."""
        seen: list[str] = []
        for reference in self.used_tags:
            if start_line <= reference.line <= end_line and reference.target not in seen:
                seen.append(reference.target)
        return seen


def template_id_path(relative_path: str) -> str:
    """Return the namespace-less template id of a `.hwk` file, e.g. `pages/resource-list`.

    Hawkeye addresses a template as `namespace:path/below/templates`, and every template root
    in the workspace ends in a `templates/` segment. A file outside such a root has no id, so
    its own extension-less path is returned instead.
    """
    path = relative_path[: -len(".hwk")] if relative_path.endswith(".hwk") else relative_path
    marker = path.rfind(TEMPLATES_SEGMENT)
    return path[marker + len(TEMPLATES_SEGMENT) :] if marker >= 0 else path


def _blank(text: str) -> str:
    """Return a run of spaces the same length as some text, keeping its line breaks."""
    return "".join(character if character == "\n" else " " for character in text)


def _blank_comments(source: str) -> str:
    """Replace template comments with spaces, keeping every offset and line break intact."""
    return _COMMENT.sub(lambda match: _blank(match.group(0)), source)


def _blank_styles(source: str) -> str:
    """Replace the body of every `style { ... }` block with spaces.

    Brace matching is enough here because a style body is SCSS, whose braces balance; an
    unbalanced one simply blanks to the end of the file, which is the safe direction.
    """
    result = source
    for match in reversed(list(_STYLE.finditer(source))):
        if result[match.end() - 1] != "{":
            # Already blanked: this is a nested `style {` inside an outer style body.
            continue
        depth = 0
        end = match.end() - 1
        for index in range(match.end() - 1, len(result)):
            if result[index] == "{":
                depth += 1
            elif result[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        else:
            end = len(result) - 1
        start = match.end()
        result = result[:start] + _blank(result[start:end]) + result[end:]
    return result


def _blank_code_comments(text: str) -> str:
    """Blank `//` and `/* */` comments inside a statement's code, keeping every offset.

    A tag body is Java-like code whose prose comments are full of English words followed by
    a bracket, which would otherwise read as bare function calls. `://` is left alone so a
    URL literal does not swallow the rest of its line.
    """
    return _CODE_COMMENT.sub(lambda match: _blank(match.group(0)), text)


def _line_starts(source: str) -> list[int]:
    """Return the offset every 1-based line starts at."""
    offsets = [0]
    for index, character in enumerate(source):
        if character == "\n":
            offsets.append(index + 1)
    return offsets


def _statement_regions(source: str) -> list[tuple[int, int]]:
    """Return the interior span of every `{% ... %}` statement.

    A region ends at the first `%}` OR at the first nested `{%`, whichever comes first: a
    component file's outer `{% tag X { ... } %}` closes only at the end of the file, and
    treating its whole body as one statement would hide every statement written inside it.
    """
    regions: list[tuple[int, int]] = []
    index = 0
    while True:
        start = source.find(_OPEN, index)
        if start < 0:
            return regions
        interior = start + len(_OPEN)
        close = source.find(_CLOSE, interior)
        following = source.find(_OPEN, interior)
        if close < 0 and following < 0:
            regions.append((interior, len(source)))
            return regions
        if close < 0 or (following >= 0 and following < close):
            regions.append((interior, following))
            index = following
        else:
            regions.append((interior, close))
            index = close + len(_CLOSE)


def _first_quoted(text: str) -> str | None:
    """Return the first quoted string of a statement's arguments, or None if it starts unquoted."""
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "\"'":
        return None
    match = _QUOTED.match(stripped)
    return match.group(1) if match else None


class _Scanner:
    """Walks one template's text once and collects its lexical facts."""

    def __init__(self, source: str) -> None:
        """Prepare a scanner over a template's text."""
        self.source = _blank_styles(_blank_comments(source))
        self.starts = _line_starts(self.source)
        self.facts = TemplateFacts(line_count=len(self.starts))
        self._open_blocks: list[tuple[str, int]] = []

    def line_of(self, offset: int) -> int:
        """Return the 1-based line an offset falls on."""
        return bisect_right(self.starts, offset)

    def run(self) -> TemplateFacts:
        """Scan the template and return its facts."""
        self._tag_declarations()
        for start, end in _statement_regions(self.source):
            self._statement(self.source[start:end], start)
        for match in _INTERPOLATION.finditer(self.source):
            self._expressions(match.group(1), match.start(1))
        for name, line in self._open_blocks:
            self.facts.blocks.append(Block(name=name, start_line=line, end_line=self.facts.line_count))
        self.facts.blocks.sort(key=lambda block: block.start_line)
        self.facts.calls = list(dict.fromkeys(self.facts.calls))
        self._custom_tags()
        return self.facts

    def _tag_declarations(self) -> None:
        """Record every custom element the file declares.

        A tag's region runs to the next declaration or to the end of the file: its closing
        `} %}` cannot be found lexically, because string literals carry braces too.
        """
        starts = [(match.group(1), self.line_of(match.start())) for match in _TAG_DECLARATION.finditer(self.source)]
        for index, (class_name, line) in enumerate(starts):
            end = starts[index + 1][1] - 1 if index + 1 < len(starts) else self.facts.line_count
            self.facts.tags.append(
                TagDeclaration(
                    class_name=class_name,
                    tag=to_kebab_case(class_name),
                    start_line=line,
                    end_line=max(end, line),
                )
            )

    def _statement(self, text: str, offset: int) -> None:
        """Read one statement's interior."""
        stripped = text.lstrip()
        match = _IDENTIFIER.match(stripped)
        if match is None:
            self._expressions(text, offset)
            return
        handler = getattr(self, f"_keyword_{match.group(0).replace('/', 'end_')}", None)
        if handler is None:
            self._expressions(text, offset)
            return
        rest_offset = offset + (len(text) - len(stripped)) + match.end()
        handler(stripped[match.end() :], rest_offset, self.line_of(offset))

    def _keyword_extend(self, rest: str, rest_offset: int, line: int) -> None:  # noqa: ARG002
        """Record the parent template of `{% extend "namespace:path" %}`."""
        target = _first_quoted(rest)
        if target and self.facts.extends is None:
            self.facts.extends = Reference(target=target, line=line)

    def _keyword_render(self, rest: str, rest_offset: int, line: int) -> None:
        """Record a `{% render "namespace:path" %}` partial include.

        A `render` whose argument is an expression (`{% render field.templateId %}`) or a
        snippet call names no template readable from the source, so it is scanned for calls
        instead of being recorded as an include.
        """
        target = _first_quoted(rest)
        if target:
            self.facts.renders.append(Reference(target=target, line=line))
            return
        self._expressions(rest, rest_offset)

    def _keyword_block(self, rest: str, rest_offset: int, line: int) -> None:  # noqa: ARG002
        """Open a `{% block "name" %}` region."""
        name = _first_quoted(rest)
        if name is None:
            match = _IDENTIFIER.match(rest.strip())
            name = match.group(0) if match else ""
        if name:
            self._open_blocks.append((name, line))

    def _keyword_end_block(self, rest: str, rest_offset: int, line: int) -> None:  # noqa: ARG002
        """Close the innermost open block region."""
        if self._open_blocks:
            name, start = self._open_blocks.pop()
            self.facts.blocks.append(Block(name=name, start_line=start, end_line=line))

    def _expressions(self, text: str, offset: int) -> None:
        """Record every function call written inside an expression region."""
        text = _blank_code_comments(text)
        for match in _NAMESPACED_CALL.finditer(text):
            self.facts.calls.append(
                Call(name=match.group(2), namespace=match.group(1), line=self.line_of(offset + match.start()))
            )
        for match in _BARE_CALL.finditer(text):
            # `function tick() {` declares a function, it does not call one.
            if text[: match.start()].rstrip().endswith("function"):
                continue
            self.facts.calls.append(Call(name=match.group(1), line=self.line_of(offset + match.start())))

    def _custom_tags(self) -> None:
        """Record every hyphenated element tag written in the markup."""
        for match in _CUSTOM_TAG.finditer(self.source):
            self.facts.used_tags.append(Reference(target=match.group(1), line=self.line_of(match.start())))


def scan(source: str) -> TemplateFacts:
    """Read one Hawkeye template's lexical facts."""
    return _Scanner(source).run()
