"""The languages duplication detection can compare, keyed by file extension.

Java and Zig have hand-written profiles; every other language with a grammar spec gets a
profile derived from that spec. Adding a language is adding one spec to
:mod:`zemble.languages.catalog`; nothing downstream of the unit extractor knows it exists.
"""

from __future__ import annotations

from pathlib import Path

from zemble.dedup.languages.base import Container, LanguageProfile, Visibility, node_text
from zemble.dedup.languages.generic import profile_from_spec
from zemble.dedup.languages.java import JAVA
from zemble.dedup.languages.zig import ZIG
from zemble.index.files import extensions_for_language
from zemble.languages.catalog import SPECS
from zemble.languages.spec import LanguageSpec, Role
from zemble.types import ContentType

_HAND_WRITTEN: tuple[LanguageProfile, ...] = (JAVA, ZIG)


def _comparable(spec: LanguageSpec) -> bool:
    """Whether a spec declares callables with bodies: without them there is nothing to compare."""
    return any(rule.role is Role.CALLABLE and rule.body is not None for rule in spec.rules)


_ALL: tuple[LanguageProfile, ...] = (
    *_HAND_WRITTEN,
    *(
        profile_from_spec(spec, tuple(extensions_for_language(language, (ContentType.CODE,))))
        for language, spec in SPECS.items()
        if language not in {profile.name for profile in _HAND_WRITTEN}
        and _comparable(spec)
        and extensions_for_language(language, (ContentType.CODE,))
    ),
)

#: Every supported extension mapped to the profile that owns it.
PROFILES: dict[str, LanguageProfile] = {extension: profile for profile in _ALL for extension in profile.extensions}


def profile_for(path: str | Path) -> LanguageProfile | None:
    """Return the profile owning a path's extension, or None when nothing claims it."""
    return PROFILES.get(Path(path).suffix.lower())


def supported_extensions() -> list[str]:
    """The extensions a duplication scan walks, sorted, as the report prints them."""
    return sorted(PROFILES)


def body_unit_kinds() -> frozenset[str]:
    """Every unit kind that owns a whole declaration body, declared by the profiles themselves."""
    return frozenset(kind for profile in _ALL for kind in profile.member_kinds.values())


def supported_languages() -> list[str]:
    """The language names a duplication scan understands, sorted."""
    return sorted(profile.name for profile in _ALL)


__all__ = [
    "PROFILES",
    "Container",
    "LanguageProfile",
    "Visibility",
    "body_unit_kinds",
    "node_text",
    "profile_for",
    "supported_extensions",
    "supported_languages",
]
