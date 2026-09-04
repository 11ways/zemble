"""What a build would do, what it would cost, and whether it would be refused - embedding nothing.

The report chunks the tree through the same walker, capsules and mtime-based reuse rule a
build uses, then asks the sqlite cache which of the would-be-embedded texts are already paid
for. Nothing here ever contacts a provider - not even to learn a model's vector width.

It reports BOTH ceilings a build passes, because reporting only one is what let this report
call a build affordable while the pre-parse guard refused the very same tree.
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
    applicable_budget_usd,
    bill_refusal,
    budget_tokens,
    estimate_cost,
    estimate_tokens,
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
    source_bytes: int
    work_limit_bytes: int
    budget_usd: float | None
    budget_tokens: int | None
    would_refuse: bool
    chunk_seconds: float
    cache_lookup_seconds: float

    def to_dict(self) -> dict:
        """Return the JSON shape."""
        return asdict(self)

    def _spending_ceilings(self) -> str:
        """Render the ceilings the BILL is judged against, in the unit each one is set in."""
        parts = []
        if self.budget_usd is not None:
            parts.append(f"${self.budget_usd:.2f}")
        if self.budget_tokens is not None and self.budget_tokens > 0:
            parts.append(f"{self.budget_tokens:,} tokens")
        return " and ".join(parts) if parts else "nothing to spend, so no ceiling"

    def to_text(self) -> str:
        """Render the report for a human, naming both ceilings a build has to pass."""
        from zemble.index.scope import megabytes

        price = self.price_per_million_usd
        ceiling = f"a {megabytes(self.work_limit_bytes)}" if self.work_limit_bytes > 0 else "no"
        return "\n".join(
            [
                f"root       {self.path} [{','.join(self.content)}]",
                f"embedder   {self.embedder}{'' if self.remote else '  (local, never billed)'}",
                f"dimensions {self.dimensions if self.dimensions is not None else 'unknown without a request'}",
                f"chunks     {self.chunks_total} total, {self.reusable} reusable from the previous index, "
                f"{self.cached} cached, {self.uncached} uncached",
                f"tokens     ~{self.estimated_tokens:,} for the uncached chunks (estimated at chars / 3.6)",
                f"cost       ~{format_cost(self.estimated_tokens, price)}"
                + (f" at ${price:.2f} per million tokens" if price else ""),
                f"cache      {self.cache_path or 'not used by this embedder'}",
                f"work       {megabytes(self.source_bytes)} of source to chunk, against {ceiling} ceiling",
                f"budget     {self._spending_ceilings()}",
                f"verdict    a build would be {'REFUSED' if self.would_refuse else 'allowed'}",
                f"timing     {self.chunk_seconds:.1f}s chunking, {self.cache_lookup_seconds:.1f}s cache lookup",
            ]
        )


def embed_status(
    path: Path | str,
    content: Sequence[ContentType] = (ContentType.CODE,),
    embedder_spec: str | None = None,
    capsules: CapsuleOptions | None = None,
) -> EmbedStatus:
    """Report what building an index over a root would chunk, embed, cost, and whether it is refused.

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
    from zemble.index.scope import estimate_tree, exceeds_work_limit, work_limit_bytes

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

    # AIDEV-NOTE: a build buys each DISTINCT text once - `CachingEmbedder.embed_documents` gives
    # duplicate texts one provider slot - so the bill is counted over the distinct set while the
    # chunk counts stay per chunk. Counting a repeated chunk twice reported a bill nobody pays,
    # and on a tree of near-identical vendored copies that gap is an order of magnitude.
    uncached = 0
    billed: list[str] = []
    seen: set[str] = set()
    for text, digest in zip(texts, digests, strict=True):
        if digest in covered:
            continue
        uncached += 1
        if digest not in seen:
            seen.add(digest)
            billed.append(text)
    cached = len(texts) - uncached
    tokens = estimate_tokens(billed)
    price = price_per_million(resolved.family)

    # AIDEV-NOTE: the verdict covers BOTH guards. Reporting only the bill is what let this
    # report say "allowed" while the pre-parse work guard refused the very same tree, which is
    # the bug the two-guard split fixes; the extra walk is the cheap half of what already ran.
    source_bytes = estimate_tree(root, content, (), manifest).bytes
    refused = exceeds_work_limit(source_bytes) or bill_refusal(tokens, resolved.family) is not None

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
        uncached=uncached,
        estimated_tokens=tokens,
        price_per_million_usd=price,
        estimated_usd=estimate_cost(tokens, price),
        cache_path=cache_path,
        source_bytes=source_bytes,
        work_limit_bytes=work_limit_bytes(),
        budget_usd=applicable_budget_usd(price),
        budget_tokens=budget_tokens(price),
        would_refuse=refused,
        chunk_seconds=round(chunk_seconds, 2),
        cache_lookup_seconds=round(lookup_seconds, 2),
    )
