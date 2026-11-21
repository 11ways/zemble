"""The one home of a stored vector's key: the same chunk under any path is one key and one purchase."""

from __future__ import annotations

import hashlib
import re

#: A stored vector's key, whichever kind: a sha256 hex digest.
KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: Prefixed to a keyed text's material, so a path-free key never equals the whole-text hash of another text.
_PATHLESS_DOMAIN = "zemble-pathless-key-1\0"


def text_hash(text: str) -> str:
    """Return the sha256 hex digest of a whole text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingText(str):
    """A text to embed that carries its own cache key, which leaves out what must not split a vector (its path).

    AIDEV-NOTE: a string operation on it returns a plain str, which keys by its whole text again: a
    lost key costs one purchase, never a wrong vector. Its reduce keeps the key across pickling.
    """

    key: str

    def __new__(cls, text: str, key: str) -> EmbeddingText:
        """Carry `key` beside `text`."""
        made = super().__new__(cls, text)
        made.key = key
        return made

    def __reduce__(self) -> tuple[type[EmbeddingText], tuple[str, str]]:
        """Pickle the text with its key."""
        return EmbeddingText, (str(self), self.key)


def keyed_text(text: str, material: str) -> EmbeddingText:
    """Return `text` keyed by the hash of `material`: the text without the parts that must not split its key."""
    return EmbeddingText(text, text_hash(_PATHLESS_DOMAIN + material))


def cache_key(text: str) -> str:
    """Return the key a document vector for this text is stored under: its carried key, else its whole text's hash."""
    return text.key if isinstance(text, EmbeddingText) else text_hash(text)


def legacy_key(text: str) -> str | None:
    """Return the whole-text hash older zemble stored this text under, or None when that is its key."""
    if not isinstance(text, EmbeddingText):
        return None
    whole = text_hash(text)
    return whole if whole != text.key else None


__all__ = ["KEY_PATTERN", "EmbeddingText", "cache_key", "keyed_text", "legacy_key", "text_hash"]
