"""The one place a spec string becomes an :class:`~zemble.embedding.base.Embedder`."""

from __future__ import annotations

import os
from dataclasses import dataclass

from zemble.embedding.base import Embedder
from zemble.utils import DEFAULT_MODEL_NAME

DEFAULT_EMBEDDER_SPEC = f"model2vec:{DEFAULT_MODEL_NAME}"

#: Env var holding a full spec string.
SPEC_ENV = "ZEMBLE_EMBEDDER"
#: Legacy env var holding a bare Model2Vec model name.
LEGACY_MODEL_ENV = "ZEMBLE_MODEL_NAME"
#: Set to "0" to bypass the sqlite embedding cache for remote providers.
CACHE_ENV = "ZEMBLE_EMBED_CACHE"
#: Names the environment variable an OpenAI-compatible endpoint's API key lives in.
KEY_ENV_ENV = "ZEMBLE_EMBEDDER_KEY_ENV"

SCHEMES = ("model2vec", "voyage", "openai")
#: Families whose vectors cost money or a round trip, so they are worth a sqlite lookup.
_CACHED_SCHEMES = frozenset({"voyage", "openai"})


@dataclass(frozen=True)
class ResolvedEmbedder:
    """An embedder together with the two strings the cache and the price table are keyed by."""

    spec: str
    embedder: Embedder
    scheme: str
    family: str


class EmbedderSpecError(ValueError):
    """A spec string could not be parsed."""


def _split_dimensions(rest: str) -> tuple[str, int | None]:
    """Split a trailing ``@<dims>`` off a spec body.

    :param rest: The spec body after the scheme.
    :return: The body without dimensions and the parsed width, or None.
    :raises EmbedderSpecError: If the dimensions are not a positive integer.
    """
    if "@" not in rest:
        return rest, None
    body, _, raw = rest.rpartition("@")
    try:
        dimensions = int(raw)
    except ValueError:
        raise EmbedderSpecError(f"Invalid dimensions {raw!r} in embedder spec; expected an integer") from None
    if dimensions <= 0:
        raise EmbedderSpecError(f"Invalid dimensions {dimensions} in embedder spec; must be positive")
    return body, dimensions


@dataclass(frozen=True)
class _ParsedSpec:
    """A spec string taken apart, before anything is built from it."""

    scheme: str
    family: str
    model: str
    dimensions: int | None
    base_url: str = ""


def _parse(spec: str) -> _ParsedSpec:
    """Take a spec apart into the keys the cache, the price table and the builder read.

    :param spec: The spec string.
    :return: Its scheme, cache family, model, width and (OpenAI-compatible only) base URL.
    :raises EmbedderSpecError: If the spec is empty, has an unknown scheme, or is malformed.
    """
    spec = spec.strip()
    if not spec:
        raise EmbedderSpecError("Empty embedder spec")
    scheme, separator, rest = spec.partition(":")
    if not separator or scheme not in SCHEMES:
        raise EmbedderSpecError(
            f"Unknown embedder spec {spec!r}; expected one of {', '.join(f'{s}:...' for s in SCHEMES)}"
        )
    if not rest.strip():
        raise EmbedderSpecError(f"Embedder spec {spec!r} names no model")

    if scheme == "model2vec":
        if "@" in rest:
            raise EmbedderSpecError(
                f"Embedder spec {spec!r} sets dimensions, but a Model2Vec model's width is fixed by the model"
            )
        return _ParsedSpec(scheme, f"model2vec:{rest}", rest, None)

    if scheme == "voyage":
        model, dimensions = _split_dimensions(rest)
        if not model:
            raise EmbedderSpecError(f"Embedder spec {spec!r} names no model")
        return _ParsedSpec(scheme, f"voyage:{model}", model, dimensions)

    base_url, separator, tail = rest.partition("#")
    if not separator or not base_url.strip() or not tail.strip():
        raise EmbedderSpecError(f"Embedder spec {spec!r} must be openai:<base_url>#<model>[@<dims>]")
    model, dimensions = _split_dimensions(tail)
    if not model:
        raise EmbedderSpecError(f"Embedder spec {spec!r} names no model")
    base_url = base_url.rstrip("/")
    return _ParsedSpec(scheme, f"openai:{base_url}#{model}", model, dimensions, base_url)


def cached_family(spec: str) -> str | None:
    """Return the embedding-cache family a spec (or an index's stored model id) reads and writes, or None.

    :return: The family, or None for a scheme that is never cached or a spec that does not parse.
    """
    try:
        parsed = _parse(spec)
    except EmbedderSpecError:
        return None
    return parsed.family if parsed.scheme in _CACHED_SCHEMES else None


def _build(spec: str) -> tuple[Embedder, str, str]:
    """Build the raw (uncached) embedder for a spec.

    :param spec: The spec string.
    :return: The embedder, its scheme, and its cache family key.
    :raises EmbedderSpecError: If the spec is empty, has an unknown scheme, or is malformed.
    """
    parsed = _parse(spec)
    if parsed.scheme == "model2vec":
        from zemble.embedding.model2vec import Model2VecEmbedder

        return Model2VecEmbedder(parsed.model), parsed.scheme, parsed.family

    if parsed.scheme == "voyage":
        from zemble.embedding.voyage import VoyageEmbedder

        return VoyageEmbedder(parsed.model, parsed.dimensions), parsed.scheme, parsed.family

    from zemble.embedding.openai_compat import DEFAULT_API_KEY_ENV, OpenAICompatibleEmbedder

    key_env = os.environ.get(KEY_ENV_ENV, "").strip() or DEFAULT_API_KEY_ENV
    return (
        OpenAICompatibleEmbedder(parsed.base_url, parsed.model, parsed.dimensions, api_key_env=key_env),
        parsed.scheme,
        parsed.family,
    )


def caching_enabled() -> bool:
    """Return whether remote embedders should be wrapped in the sqlite cache."""
    return os.environ.get(CACHE_ENV, "1").strip().lower() not in {"0", "false", "no"}


def build_provider(spec: str) -> tuple[Embedder, str, str]:
    """Build the bare provider for a spec, bypassing both the local cache and an embedding server.

    The embedding server buys through this, so its own provider calls can never be routed back
    through a client of itself.

    :param spec: The spec string.
    :return: The embedder, its scheme, and its cache family key.
    """
    return _build(spec)


def build_embedder(spec: str) -> ResolvedEmbedder:
    """Build an embedder from a spec and keep its scheme and cache family alongside it.

    The family (scheme plus model, no dimensions) is what the sqlite cache file and the
    price table are keyed by, and neither can be recovered from an :class:`Embedder`. A paid
    family goes through the embedding server when ``ZEMBLE_EMBED_SERVER`` names one, whatever
    ``ZEMBLE_EMBED_CACHE`` says, because the server always caches; otherwise through the local
    sqlite cache.

    :param spec: The spec string.
    :return: The embedder and its keys.
    """
    embedder, scheme, family = _build(spec)
    if scheme in _CACHED_SCHEMES:
        from zemble.embedding.wire import server_settings

        settings = server_settings()
        if settings is not None:
            from zemble.embedding.served import ServerEmbedder

            served = ServerEmbedder(spec, family, settings, embedder)
            return ResolvedEmbedder(spec=spec, embedder=served, scheme=scheme, family=family)
        if caching_enabled():
            from zemble.embedding.cache import CachingEmbedder

            embedder = CachingEmbedder(embedder, family)
    return ResolvedEmbedder(spec=spec, embedder=embedder, scheme=scheme, family=family)


def parse_embedder_spec(spec: str) -> Embedder:
    """Turn a spec string into a ready embedder, wrapping remote providers in the content-hash cache.

    Grammar::

        model2vec:<hf-model>
        voyage:<model>[@<dims>]
        openai:<base_url>#<model>[@<dims>]

    :param spec: The spec string.
    :return: An embedder.
    """
    return build_embedder(spec).embedder


def resolve_embedder_spec(spec: str | None = None) -> str:
    """Resolve the spec to use: explicit, then ``ZEMBLE_EMBEDDER``, then the legacy ``ZEMBLE_MODEL_NAME``.

    ``ZEMBLE_MODEL_NAME`` predates the seam and holds a bare Model2Vec model name; it is
    kept as an alias and maps to ``model2vec:<name>``.

    :param spec: An explicit spec, or None to read the environment.
    :return: The spec string to build from.
    """
    if spec:
        return spec
    from_env = os.environ.get(SPEC_ENV, "").strip()
    if from_env:
        return from_env
    legacy = os.environ.get(LEGACY_MODEL_ENV, "").strip()
    if legacy:
        return legacy if legacy.partition(":")[0] in SCHEMES else f"model2vec:{legacy}"
    return DEFAULT_EMBEDDER_SPEC


def load_embedder(spec: str | None = None) -> Embedder:
    """Resolve a spec (or the environment default) and build the embedder.

    :param spec: An explicit spec, or None to read the environment.
    :return: An embedder.
    """
    return parse_embedder_spec(resolve_embedder_spec(spec))
