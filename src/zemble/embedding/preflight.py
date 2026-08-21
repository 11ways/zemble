"""What a build would embed, and what it would cost, without embedding anything.

The report chunks the tree through the same walker, capsules and mtime-based reuse rule a
build uses, then asks the sqlite cache which of the would-be-embedded texts are already paid
for. Nothing here ever contacts a provider - not even to learn a model's vector width.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from zemble.chunking.capsule import CapsuleOptions, embedding_text
from zemble.embedding.base import declared_dimensions, is_remote
from zemble.embedding.cache import EmbeddingCache, text_hash
from zemble.embedding.pricing import (
    budget_tokens,
    estimate_cost,
    estimate_tokens,
    exceeds_budget,
    format_cost,
    price_per_million,
)
from zemble.embedding.registry import build_embedder, caching_enabled, resolve_embedder_spec
from zemble.types import ContentType


@dataclass(frozen=True)
class EmbedStatus:
    """The pre-flight numbers for one root and one embedder."""

    path: str
    embedder: str
    family: str
    remote: bool
    dimensions: int | None
    content: list[str]
    chunks_total: int
    reusable: int
    cached: int
    uncached: int
    estimated_tokens: int
    price_per_million_usd: float | None
    estimated_usd: float | None
    cache_path: str | None
    budget_tokens: int
    would_refuse: bool
    chunk_seconds: float
    cache_lookup_seconds: float

    def to_dict(self) -> dict:
        """Return the JSON shape."""
        return asdict(self)

    def to_text(self) -> str:
        """Render the report for a human."""
        price = self.price_per_million_usd
        lines = [
            f"root       {self.path} [{','.join(self.content)}]",
            f"embedder   {self.embedder}{'' if self.remote else '  (local, never billed)'}",
            f"dimensions {self.dimensions if self.dimensions is not None else 'unknown without a provider request'}",
            f"chunks     {self.chunks_total} total, {self.reusable} reusable from the previous index, "
            f"{self.cached} cached, {self.uncached} uncached",
            f"tokens     ~{self.estimated_tokens:,} for the uncached chunks (estimated at chars / 3.6)",
            f"cost       ~{format_cost(self.estimated_tokens, price)}"
            + (f" at ${price:.2f} per million tokens" if price else ""),
            f"cache      {self.cache_path or 'not used by this embedder'}",
            f"budget     {self.budget_tokens:,} tokens; a build would be "
            f"{'REFUSED' if self.would_refuse else 'allowed'}",
            f"timing     {self.chunk_seconds:.1f}s chunking, {self.cache_lookup_seconds:.1f}s cache lookup",
        ]
        return "\n".join(lines)


def embed_status(
    path: Path | str,
    content: Sequence[ContentType] = (ContentType.CODE,),
    embedder_spec: str | None = None,
    capsules: CapsuleOptions | None = None,
) -> EmbedStatus:
    """Report what building an index over a root would embed and what it would cost.

    :param path: The root to inspect.
    :param content: Content types a build would index.
    :param embedder_spec: An explicit embedder spec, or None for the environment default.
    :param capsules: Context-capsule knobs; None resolves the environment override.
    :return: The pre-flight numbers.
    :raises FileNotFoundError: If the root does not exist.
    :raises EmbedderSpecError: If the spec cannot be parsed.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Path does not exist: {root}")
    root = root.resolve()

    from zemble.cache import load_manifest_for_incremental
    from zemble.index.create import plan_files

    spec = resolve_embedder_spec(embedder_spec)
    resolved = build_embedder(spec)
    remote = is_remote(resolved.embedder)
    dimensions = declared_dimensions(resolved.embedder)
    resolved_capsules = CapsuleOptions.resolve(capsules)
    # A remote model whose width is only knowable from a probe request has no model_id we may
    # ask for, and without a width there is nothing to look up in the cache either.
    model_id = resolved.embedder.model_id if not remote or dimensions is not None else None

    manifest = (
        load_manifest_for_incremental(str(root), model_id, content, resolved_capsules) if model_id is not None else None
    )

    started = time.monotonic()
    reusable = 0
    texts: list[str] = []
    for planned in plan_files(root, content, display_root=root, previous_manifest=manifest, capsules=resolved_capsules):
        if planned.reused:
            reusable += planned.count
            continue
        texts.extend(embedding_text(chunk) for chunk in planned.chunks)
    chunk_seconds = time.monotonic() - started

    cache_path: str | None = None
    covered: set[str] = set()
    lookup_seconds = 0.0
    digests = [text_hash(text) for text in texts]
    if remote and dimensions is not None and caching_enabled():
        started = time.monotonic()
        cache = EmbeddingCache(resolved.family)
        cache_path = str(cache.path)
        try:
            covered = cache.covered(digests, dimensions)
        finally:
            cache.close()
        lookup_seconds = time.monotonic() - started

    uncached_texts = [text for text, digest in zip(texts, digests, strict=True) if digest not in covered]
    cached = len(texts) - len(uncached_texts)
    tokens = estimate_tokens(uncached_texts)
    price = price_per_million(resolved.family)

    return EmbedStatus(
        path=str(root),
        embedder=model_id or spec,
        family=resolved.family,
        remote=remote,
        dimensions=dimensions,
        content=[item.value for item in content],
        chunks_total=reusable + len(texts),
        reusable=reusable,
        cached=cached,
        uncached=len(uncached_texts),
        estimated_tokens=tokens,
        price_per_million_usd=price,
        estimated_usd=estimate_cost(tokens, price),
        cache_path=cache_path,
        budget_tokens=budget_tokens(remote),
        would_refuse=exceeds_budget(tokens, remote),
        chunk_seconds=round(chunk_seconds, 2),
        cache_lookup_seconds=round(lookup_seconds, 2),
    )
