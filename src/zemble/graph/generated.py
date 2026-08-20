"""Map facts about generated Hawkeye code back onto the `.hwk` template they came from.

A Hawkeye workspace compiles every template into a Java class under
`<module>/build/generated-sources/hawkeye/<source set>/java/...`, and a javac emitter writes
true facts about those classes. The index never walks a `build/` directory, so without this
module every one of those facts is dropped as "source ignored by the index" - which is where
the bulk of a Hawkeye workspace's compiler knowledge goes to die.

The compiler leaves two things behind, and this module is written against both:

- `// @hwk:<template line>` comments in the generated Java, one before each transpiled unit,
  which is the line map. The `Tpl_*.sourcemap.json` sidecar beside the class is BUILT FROM
  those same comments, but it is built BEFORE the compiler injects the source-map
  self-registration into the class, so its `javaLine` numbers are two lines short of the file
  javac actually compiled. The sidecar is therefore read for identity - `templatePath`,
  `generatedClass` - and the markers in the `.java` for the positions.
- The two deterministic class-naming rules: a template class is `Tpl_` plus the camel-cased
  template id, and a tag class is the PascalCase tag name, whose kebab-case form is the
  element tag. Both are applied FORWARD - from a template zemble already extracted to the
  class name Hawkeye would have generated - so nothing is ever guessed backwards out of a
  class name.
"""

from __future__ import annotations

import json
import logging
import re
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from zemble.graph.hwk import TAG_MODIFIER
from zemble.graph.model import Symbol, SymbolKind
from zemble.hwk import template_id_path, to_kebab_case

logger = logging.getLogger(__name__)

#: The path segments a Hawkeye generated source root always has, in order.
GENERATED_ROOT = "/build/generated-sources/hawkeye/"
#: The directory that separates the source-set name from the package path.
_JAVA_SEGMENT = "java"
#: Sidecar written beside each generated template class, replacing the `.java` suffix.
SOURCEMAP_SUFFIX = ".sourcemap.json"
#: The comment `IRTranspiler` emits before each transpiled unit; the number is a template line.
SOURCE_MARKER = re.compile(r"//\s*@hwk:(\d+)")
#: Prefix of a generated template class, both for a template file and for a declared tag.
TEMPLATE_CLASS_PREFIX = "Tpl_"
#: Suffix of the generated implementation class of a declared tag.
TAG_IMPL_SUFFIX = "Impl"
#: The package segment every generated tag class lives under.
TAG_PACKAGE_SEGMENT = ".tags."

#: Reason a whole generated file's facts are dropped because its template moved on.
TEMPLATE_NEWER_REASON = "template newer than generated source"


def camel_case_identifier(identifier: str) -> str:
    """Camel-case a template id the way Hawkeye's `IRTranspiler` names its class.

    Every non-alphanumeric character is dropped and capitalises the next one, so
    `pages/resource-list` becomes `PagesResourceList`.
    """
    result: list[str] = []
    capitalize = True
    for character in identifier:
        if character.isalnum():
            result.append(character.upper() if capitalize else character)
            capitalize = False
        else:
            capitalize = True
    return "".join(result)


def template_is_newer(root: Path, template_path: str, source_path: str) -> bool:
    """Return whether a template was edited after the class was generated from it.

    Hawkeye records no hash of the template inside its output, so modification time is the
    only handle. A missing file answers False: the sha of the generated source already
    decided that javac saw what is on disk, and inventing staleness out of a stat error
    would drop true facts.

    This is the whole freshness rule for mapped templates, and an incremental build re-decides
    it with these two stats rather than by re-resolving every template a facts file touched.
    """
    try:
        template = (root / template_path).stat().st_mtime_ns
        generated = (root / source_path).stat().st_mtime_ns
    except OSError:
        return False
    return template > generated


@dataclass(frozen=True)
class GeneratedOrigin:
    """The pieces of a generated Java path a template is looked up by."""

    #: Workspace-relative directory of the module that generated it, "" for the root itself.
    module: str
    #: The Gradle source set the templates were compiled from: `common`, `browserTest`, ...
    source_set: str
    #: Dotted package of the generated class.
    package: str
    #: Simple name of the generated class.
    class_name: str

    @property
    def is_tag_class(self) -> bool:
        """Return whether the class was generated for a custom element rather than a file."""
        return TAG_PACKAGE_SEGMENT in f".{self.package}."

    @property
    def qualified_class(self) -> str:
        """The fully qualified name of the generated class."""
        return f"{self.package}.{self.class_name}" if self.package else self.class_name

    @property
    def template_root(self) -> str:
        """Where the templates this class was generated from live, by Hawkeye's layout."""
        prefix = f"{self.module}/" if self.module else ""
        return f"{prefix}src/{self.source_set}/templates/"


def parse_generated_path(relative_path: str) -> GeneratedOrigin | None:
    """Split a workspace-relative path into its generated-source pieces, or None.

    :param relative_path: A workspace-relative posix path.
    :return: The origin, or None when the path is not a Hawkeye generated Java file.
    """
    if not relative_path.endswith(".java"):
        return None
    marker = f"/{relative_path}".find(GENERATED_ROOT)
    if marker < 0:
        return None
    module = relative_path[: max(marker - 1, 0)]
    parts = relative_path[marker + len(GENERATED_ROOT) - 1 :].split("/")
    if len(parts) < 3 or parts[1] != _JAVA_SEGMENT:
        return None
    return GeneratedOrigin(
        module=module,
        source_set=parts[0],
        package=".".join(parts[2:-1]),
        class_name=parts[-1][: -len(".java")],
    )


@dataclass(frozen=True)
class GeneratedMapping:
    """What one position in a generated Java file maps back to.

    `symbol_id` is None when nothing was mapped, and `reason` then says why in the words the
    status report prints. `stale` separates the one refusal that is about freshness - the
    template was edited after the class was generated - from the ones that are about shape.
    """

    symbol_id: str | None
    template_path: str | None
    line: int
    reason: str = ""
    stale: bool = False
    #: The generated class the fact was written about, kept so a reader can see the detour.
    generated_class: str = ""

    @property
    def mapped(self) -> bool:
        """True when the fact can be attributed to a template symbol."""
        return self.symbol_id is not None


@dataclass
class _GeneratedFile:
    """What one generated Java file resolved to, cached for every fact about it."""

    origin: GeneratedOrigin
    #: The symbol a fact lands on when no line map narrows it further.
    symbol: Symbol | None = None
    #: The file-level TEMPLATE symbol, which line refinement starts from.
    file_symbol: Symbol | None = None
    template_path: str | None = None
    #: Generated Java line -> template line, sorted by Java line.
    java_lines: list[int] = field(default_factory=list)
    template_lines: list[int] = field(default_factory=list)
    reason: str = ""
    stale: bool = False


class GeneratedSourceMapper:
    """Resolves a generated Hawkeye Java position back to the template symbol behind it.

    The index it builds is entirely FORWARD: every `.hwk` template zemble already extracted is
    filed under the template id, the class name and the element tag Hawkeye's own naming rules
    would give it. A generated class then looks itself up, which is why an unrecognised class
    is reported rather than approximated.
    """

    def __init__(self, root: Path, symbols: Iterable[Symbol]) -> None:
        """Index the workspace's template symbols by every name a generated class can carry."""
        self.root = root
        self._by_id: dict[str, list[Symbol]] = {}
        self._by_class: dict[str, list[Symbol]] = {}
        self._by_tag: dict[str, list[Symbol]] = {}
        self._owners: dict[str, list[Symbol]] = {}
        self._by_symbol_id: dict[str, Symbol] = {}
        self._files: dict[str, _GeneratedFile] = {}
        for symbol in symbols:
            if symbol.kind is SymbolKind.BLOCK:
                self._owners.setdefault(symbol.file_path, []).append(symbol)
                continue
            if symbol.kind is not SymbolKind.TEMPLATE:
                continue
            self._by_symbol_id[symbol.id] = symbol
            if TAG_MODIFIER in symbol.modifiers:
                self._by_tag.setdefault(symbol.qualified_name, []).append(symbol)
            if symbol.container_id:
                self._owners.setdefault(symbol.file_path, []).append(symbol)
                continue
            template_id = template_id_path(symbol.file_path)
            self._by_id.setdefault(template_id, []).append(symbol)
            self._by_class.setdefault(camel_case_identifier(template_id), []).append(symbol)

    # ---- public face -----------------------------------------------------

    def recognises(self, source_path: str) -> bool:
        """Return whether a source file is generated Hawkeye Java this mapper speaks for."""
        return parse_generated_path(source_path) is not None

    def resolve(self, source_path: str, java_line: int) -> GeneratedMapping:
        """Map a position in a generated Java file back onto a template symbol.

        :param source_path: Workspace-relative path of the generated Java file.
        :param java_line: The line in that file, or 0 when the fact carries none.
        :return: The mapping, whose `symbol_id` is None when nothing could be attributed.
        """
        described = self._describe(source_path)
        if described.symbol is None:
            return GeneratedMapping(
                symbol_id=None,
                template_path=described.template_path,
                line=0,
                reason=described.reason,
                stale=described.stale,
                generated_class=described.origin.qualified_class,
            )
        template_line = self._template_line(described, java_line)
        symbol = described.symbol
        if template_line is not None and described.file_symbol is not None:
            symbol = self._owner(described, template_line)
        return GeneratedMapping(
            symbol_id=symbol.id,
            template_path=described.template_path,
            line=template_line if template_line is not None else symbol.start_line,
            generated_class=described.origin.qualified_class,
        )

    @property
    def mapped_templates(self) -> set[str]:
        """Every template path a generated file has been resolved to so far."""
        return {found.template_path for found in self._files.values() if found.template_path and found.symbol}

    # ---- resolution ------------------------------------------------------

    def _describe(self, source_path: str) -> _GeneratedFile:
        """Resolve one generated file to its template, caching the whole answer."""
        cached = self._files.get(source_path)
        if cached is None:
            cached = self._build(source_path)
            self._files[source_path] = cached
        return cached

    def _build(self, source_path: str) -> _GeneratedFile:
        """Do the work `_describe` caches."""
        origin = parse_generated_path(source_path)
        if origin is None:  # pragma: no cover - callers ask `recognises` first
            raise ValueError(f"{source_path} is not a Hawkeye generated source")
        found = _GeneratedFile(origin=origin)
        sidecar = self._read_sidecar(source_path)
        candidates = self._candidates(origin, sidecar)
        if not candidates:
            found.reason = f"no template answers to generated class {origin.qualified_class}"
            return found
        if len(candidates) > 1:
            found.reason = f"{len(candidates)} templates answer to generated class {origin.qualified_class}"
            return found
        found.file_symbol = self._file_symbol(candidates[0])
        found.symbol = candidates[0]
        found.template_path = candidates[0].file_path
        if self._template_is_newer(found.template_path, source_path):
            found.symbol = None
            found.stale = True
            found.reason = TEMPLATE_NEWER_REASON
            return found
        if sidecar is not None:
            self._load_line_map(found, source_path, sidecar)
        return found

    def _file_symbol(self, symbol: Symbol) -> Symbol | None:
        """Return the file-level TEMPLATE symbol a match belongs to, when there is one."""
        return symbol if not symbol.container_id else self._by_symbol_id.get(symbol.container_id)

    def _candidates(self, origin: GeneratedOrigin, sidecar: dict | None) -> list[Symbol]:
        """Find the template symbols a generated class could have come from."""
        found = self._by_name(origin, sidecar)
        if len(found) <= 1:
            return found
        # Two modules may hold a template of the same id; the generated file names its own.
        inside = [symbol for symbol in found if symbol.file_path.startswith(origin.template_root)]
        return inside or found

    def _by_name(self, origin: GeneratedOrigin, sidecar: dict | None) -> list[Symbol]:
        """Look a generated class up under every name Hawkeye's naming rules give a template."""
        if sidecar is not None:
            declared = sidecar.get("templatePath")
            if isinstance(declared, str) and declared:
                return list(self._by_id.get(declared, []))
        name = origin.class_name
        if name.startswith(TEMPLATE_CLASS_PREFIX):
            name = name[len(TEMPLATE_CLASS_PREFIX) :]
        elif origin.is_tag_class and name.endswith(TAG_IMPL_SUFFIX):
            name = name[: -len(TAG_IMPL_SUFFIX)]
        if origin.is_tag_class:
            return list(self._by_tag.get(to_kebab_case(name), []))
        return list(self._by_class.get(name, []))

    def _read_sidecar(self, source_path: str) -> dict | None:
        """Read the `.sourcemap.json` beside a generated class, or None when it has none."""
        path = self.root / f"{source_path[: -len('.java')]}{SOURCEMAP_SUFFIX}"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _load_line_map(self, found: _GeneratedFile, source_path: str, sidecar: dict) -> None:
        """Read the line map out of the generated Java itself, falling back to the sidecar.

        The markers in the `.java` are the positions javac compiled; the sidecar's `javaLine`
        numbers were recorded before the compiler injected its source-map registration into the
        same file, so they sit two lines higher than the truth. The sidecar is only trusted for
        positions when the generated file cannot be read at all.
        """
        pairs = self._markers(source_path)
        if not pairs:
            pairs = [
                (entry["javaLine"], entry["templateLine"])
                for entry in sidecar.get("mappings", [])
                if isinstance(entry, dict)
                and isinstance(entry.get("javaLine"), int)
                and isinstance(entry.get("templateLine"), int)
            ]
        pairs.sort()
        found.java_lines = [java for java, _ in pairs]
        found.template_lines = [template for _, template in pairs]

    def _markers(self, source_path: str) -> list[tuple[int, int]]:
        """Collect the `// @hwk:` markers of a generated file as (Java line, template line)."""
        try:
            text = (self.root / source_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        found: list[tuple[int, int]] = []
        for number, line in enumerate(text.splitlines(), start=1):
            match = SOURCE_MARKER.search(line)
            if match is not None:
                found.append((number, int(match.group(1))))
        return found

    def _template_is_newer(self, template_path: str, source_path: str) -> bool:
        """Return whether the template was edited after the class was generated from it."""
        return template_is_newer(self.root, template_path, source_path)

    def _template_line(self, found: _GeneratedFile, java_line: int) -> int | None:
        """Look a generated Java line up in the line map, floor-style like Hawkeye does."""
        if not found.java_lines or java_line <= 0:
            return None
        index = bisect_right(found.java_lines, java_line) - 1
        return found.template_lines[index] if index >= 0 else None

    def _owner(self, found: _GeneratedFile, template_line: int) -> Symbol:
        """Return the narrowest block or tag region of the template containing a line."""
        assert found.file_symbol is not None  # noqa: S101 - guarded by the caller
        containing = [
            symbol
            for symbol in self._owners.get(found.file_symbol.file_path, [])
            if symbol.start_line <= template_line <= symbol.end_line
        ]
        if not containing:
            return found.file_symbol
        return min(containing, key=lambda symbol: (symbol.end_line - symbol.start_line, symbol.start_line))
