"""Signature-qualified symbol queries, `Type.method(Param,...)`, and the parameter-type normal form they match on.

A symbol stores its parameter types erased (`List<String>` as `List`, a type variable `M extends Model`
as `Model`, javac's erasure) and a compiler ref spells them fully qualified, so both sides, and whatever
a caller types, meet on the simple erased name: `java.util.List<String>` and `List` are the same
parameter here. A type variable written in a query is not reinterpreted: `of(M)` names a type `M`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from zemble.graph.model import CALLABLE_KINDS, Symbol


def simple_type_name(name: str) -> str:
    """Reduce a possibly qualified type name to its last segment, keeping array brackets."""
    base, _, arrays = name.partition("[")
    suffix = f"[{arrays}" if arrays else ""
    return base.rsplit(".", 1)[-1].strip() + suffix


def _erase(written: str) -> str | None:
    """Drop type arguments and spell varargs as an array; None when the angle brackets do not balance."""
    depth = 0
    kept: list[str] = []
    for char in written:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0:
            kept.append(char)
    if depth != 0:
        return None
    text = "".join(kept).replace(" ", "")
    return text[:-3] + "[]" if text.endswith("...") else text


def parameter_key(written: str) -> str | None:
    """The form one parameter type is compared in: erased and simple; None when it cannot be read."""
    erased = _erase(written)
    return simple_type_name(erased) if erased else None


def same_parameters(stored: Sequence[str], wanted: Sequence[str]) -> bool:
    """Whether two parameter lists name the same types, compared by `parameter_key`."""
    return len(stored) == len(wanted) and all(
        parameter_key(have) == parameter_key(want) for have, want in zip(stored, wanted)
    )


def _split_top_level(inner: str) -> list[str] | None:
    """Split a parameter list at the commas outside type arguments; None when the brackets do not balance."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in inner:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth < 0:
                return None
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if depth != 0:
        return None
    parts.append("".join(current))
    return parts


@dataclass(frozen=True)
class SymbolQuery:
    """A written symbol name, plus the parameter types it was narrowed to when it carried a signature."""

    name: str
    #: The parameter keys (`parameter_key`) the query names; None when no signature was written.
    params: tuple[str, ...] | None = None

    @classmethod
    def parse(cls, written: str) -> SymbolQuery:
        """Read `name` or `name(Param,...)`; a form that does not parse stays one plain name and matches nothing."""
        text = written.strip()
        opening = text.find("(")
        if opening <= 0 or not text.endswith(")"):
            return cls(text)
        inner = text[opening + 1 : -1]
        if not inner.strip():
            return cls(text[:opening].strip(), ())
        pieces = _split_top_level(inner)
        keys = [parameter_key(piece) for piece in pieces] if pieces is not None else None
        if keys is None or any(key is None for key in keys):
            return cls(text)
        return cls(text[:opening].strip(), tuple(key for key in keys if key is not None))

    def accepts(self, symbol: Symbol) -> bool:
        """Whether a symbol found by `name` fits the signature: always without one, only a callable with one."""
        if self.params is None:
            return True
        return symbol.kind in CALLABLE_KINDS and same_parameters(symbol.param_types, self.params)


def signature_form(display: str, symbol: Symbol) -> str:
    """Spell a callable the way a query narrows to it: `Type.member(Param,...)` with its stored parameter types."""
    return f"{display}({','.join(symbol.param_types)})"


__all__ = ["SymbolQuery", "parameter_key", "same_parameters", "signature_form", "simple_type_name"]
