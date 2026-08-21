"""The languages duplication detection can compare, keyed by file extension.

Adding a language is adding one module here and one entry to :data:`_ALL`; nothing
downstream of the unit extractor knows a language exists.
"""

from __future__ import annotations

from pathlib import Path

from zemble.dedup.languages.base import Container, LanguageProfile, node_text
from zemble.dedup.languages.java import JAVA
from zemble.dedup.languages.zig import ZIG

_ALL: tuple[LanguageProfile, ...] = (JAVA, ZIG)

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
    "body_unit_kinds",
    "node_text",
    "profile_for",
    "supported_extensions",
    "supported_languages",
]
