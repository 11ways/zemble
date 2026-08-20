"""The one place a spec string becomes a :class:`~zemble.rerank.base.Reranker`, plus its knobs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from zemble.rerank.base import Reranker

#: Env var holding a full spec string.
SPEC_ENV = "ZEMBLE_RERANKER"
#: Env var holding how many head candidates are rescored.
TOP_K_ENV = "ZEMBLE_RERANK_K"
#: Env var holding the blend weight given to the reranker score.
ALPHA_ENV = "ZEMBLE_RERANK_ALPHA"
#: Env var holding the passage shape: ``context`` or ``content``.
PASSAGE_ENV = "ZEMBLE_RERANK_PASSAGE"

#: The spec that turns the pass off; also the default.
NONE_SPEC = "none"

SCHEMES = ("cross", "voyage")

DEFAULT_TOP_K = 50
DEFAULT_ALPHA = 1.0


class RerankerSpecError(ValueError):
    """A spec string could not be parsed."""


class PassageMode(str, Enum):
    """What text of a chunk the reranker is shown."""

    #: The context capsule, a newline, then the chunk content.
    CONTEXT = "context"
    #: The chunk content alone.
    CONTENT = "content"


@dataclass(frozen=True)
class RerankSettings:
    """How the rerank pass is applied once a reranker exists."""

    #: Head of the ranked list that gets rescored; everything below keeps its order.
    top_k: int = DEFAULT_TOP_K
    #: Weight of the normalized reranker score; ``1 - alpha`` stays with the fused score.
    alpha: float = DEFAULT_ALPHA
    #: The passage shape shown to the reranker.
    passage: PassageMode = PassageMode.CONTEXT

    @classmethod
    def from_env(cls: type[RerankSettings]) -> RerankSettings:
        """Read the knobs from the environment, falling back to the defaults; an unusable value is refused.

        :return: The resolved settings.
        """
        return cls(
            top_k=_positive_int(TOP_K_ENV, DEFAULT_TOP_K),
            alpha=_unit_float(ALPHA_ENV, DEFAULT_ALPHA),
            passage=_passage(PASSAGE_ENV, PassageMode.CONTEXT),
        )


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer from the environment.

    :param name: The environment variable.
    :param default: The value to use when it is unset or empty.
    :return: The parsed value.
    :raises RerankerSpecError: If the value is not a positive integer.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RerankerSpecError(f"{name}={raw!r} is not an integer") from None
    if value <= 0:
        raise RerankerSpecError(f"{name}={value} must be positive")
    return value


def _unit_float(name: str, default: float) -> float:
    """Read a float in ``[0, 1]`` from the environment.

    :param name: The environment variable.
    :param default: The value to use when it is unset or empty.
    :return: The parsed value.
    :raises RerankerSpecError: If the value is not a float between 0 and 1.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise RerankerSpecError(f"{name}={raw!r} is not a number") from None
    if not 0.0 <= value <= 1.0:
        raise RerankerSpecError(f"{name}={value} must be between 0 and 1")
    return value


def _passage(name: str, default: PassageMode) -> PassageMode:
    """Read a passage mode from the environment.

    :param name: The environment variable.
    :param default: The value to use when it is unset or empty.
    :return: The parsed mode.
    :raises RerankerSpecError: If the value names no known mode.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    try:
        return PassageMode(raw)
    except ValueError:
        known = ", ".join(mode.value for mode in PassageMode)
        raise RerankerSpecError(f"{name}={raw!r} is not a passage mode; expected one of {known}") from None


def parse_reranker_spec(spec: str) -> Reranker | None:
    """Turn a spec string into a ready reranker.

    Grammar::

        none
        cross:<hf-model>
        voyage:<model>

    :param spec: The spec string.
    :return: A reranker, or None for ``none``.
    :raises RerankerSpecError: If the spec is empty, has an unknown scheme, or names no model.
    """
    spec = spec.strip()
    if not spec:
        raise RerankerSpecError("Empty reranker spec")
    if spec.lower() == NONE_SPEC:
        return None
    scheme, separator, rest = spec.partition(":")
    if not separator or scheme not in SCHEMES:
        expected = ", ".join([NONE_SPEC, *(f"{s}:..." for s in SCHEMES)])
        raise RerankerSpecError(f"Unknown reranker spec {spec!r}; expected one of {expected}")
    if not rest.strip():
        raise RerankerSpecError(f"Reranker spec {spec!r} names no model")

    if scheme == "cross":
        from zemble.rerank.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(rest.strip())

    from zemble.rerank.voyage import VoyageReranker

    return VoyageReranker(rest.strip())


def resolve_reranker_spec(spec: str | None = None) -> str:
    """Resolve the spec to use: explicit, then ``ZEMBLE_RERANKER``, then ``none``.

    :param spec: An explicit spec, or None to read the environment.
    :return: The spec string to build from.
    """
    if spec:
        return spec
    return os.environ.get(SPEC_ENV, "").strip() or NONE_SPEC


def load_reranker(spec: str | None = None) -> Reranker | None:
    """Resolve a spec (or the environment default) and build the reranker.

    The model itself is loaded lazily on the first :meth:`Reranker.score` call, so building
    one costs nothing when no query is ever run.

    :param spec: An explicit spec, or None to read the environment.
    :return: A reranker, or None when reranking is off.
    """
    return parse_reranker_spec(resolve_reranker_spec(spec))
