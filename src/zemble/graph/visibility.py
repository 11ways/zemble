"""How far a graph symbol reaches, from its modifiers, its container and its language's conventions."""

from __future__ import annotations

from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import PurePosixPath

from zemble.graph.model import DECLARED_TYPE_KINDS, Symbol, SymbolKind
from zemble.index.files import detect_language
from zemble.languages import Visibility, spec_for
from zemble.languages.read import visibility_of
from zemble.languages.visibility import VISIBILITY_WORDS

#: Container kinds whose members are part of a contract: as visible as the container unless they say otherwise.
CONTRACT_KINDS = frozenset({SymbolKind.INTERFACE, SymbolKind.ANNOTATION})

#: Looks up a symbol's declaring container, or None when it has none or the caller does not hold it.
ContainerOf = Callable[[Symbol], Symbol | None]


def in_contract(symbol: Symbol, container_of: ContainerOf) -> bool:
    """Whether a symbol is a member of an interface, trait, protocol or annotation type."""
    container = container_of(symbol)
    return container is not None and container.kind in CONTRACT_KINDS


def symbol_visibility(symbol: Symbol, container_of: ContainerOf) -> Visibility:
    """How far a symbol reaches; UNKNOWN for a language no rule places.

    A member of an interface (trait, protocol) without a word of its own is as visible as the
    interface: it is part of the contract, in every language that has one.
    """
    container = container_of(symbol)
    written = set(symbol.modifiers)
    for word, level in VISIBILITY_WORDS:
        if word in written:
            return level
    if container is not None and container.kind in CONTRACT_KINDS:
        return symbol_visibility(container, container_of)
    language = detect_language(PurePosixPath(symbol.file_path))
    if language == "java":
        return Visibility.PACKAGE
    spec = spec_for(language) if language is not None else None
    if spec is None:
        return Visibility.UNKNOWN
    in_type = container is not None and container.kind in DECLARED_TYPE_KINDS
    return visibility_of(spec, symbol.name, symbol.modifiers, in_type=in_type)


def is_runtime_hook(symbol: Symbol) -> bool:
    """Whether the language runtime calls this callable by its name alone (`__repr__`)."""
    language = detect_language(PurePosixPath(symbol.file_path))
    spec = spec_for(language) if language is not None else None
    return spec is not None and any(fnmatch(symbol.name, pattern) for pattern in spec.runtime_hook_patterns)


__all__ = ["CONTRACT_KINDS", "ContainerOf", "in_contract", "is_runtime_hook", "symbol_visibility"]
