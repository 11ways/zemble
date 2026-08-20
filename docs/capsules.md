# Context capsules

A chunk is a slice of a file. Read on its own it rarely says where it lives: the
package, the class it belongs to, the method it sits inside and the annotations on
that method are all lines the chunker cut away. A **context capsule** is a short
description of that location, built from the same tree-sitter tree the chunk
boundaries came from, stored on the chunk as `Chunk.context`.

The capsule is **embedding text, never output**. `search` and `find_related` return
`chunk.content` unchanged, and the result shape is untouched.

## What a capsule contains

For Java, in this order, joined by ` | `:

1. The repo-relative path, followed by its own segments as words.
2. The `package` declaration.
3. The enclosing type chain, with kind and inheritance:
   `class EntityTags extends BaseTags implements Comparable<EntityTags> > class Inner`.
4. The annotations on the enclosing member.
5. The enclosing member's signature, annotations stripped, body excluded.
6. The imports whose simple name literally appears inside the chunk, capped at 10.

Example, from a chunk in the middle of a method body:

```
zenit/src/common/java/EntityTags.java zenit src common java EntityTags | package be.elevenways.zenit.common.http | class EntityTags extends BaseTags implements Comparable<EntityTags>, Cloneable | @Override @NonNull | public static boolean matchesIfNoneMatch(String ifNoneMatch, String quotedEtag) | uses List
```

Other tree-sitter languages get the generic subset: path plus the enclosing
definition chain (`class Config > function load`), derived from node types rather
than from a per-language rule set. A file with no parse tree - line chunking, or a
grammar the platform does not bundle - gets the path segment and nothing else.

Which chunk owns which context is decided by the chunk's **start**, never its span:
a chunk running out of one method and into the next is credited to the one it opens
in, a chunk starting mid-body still names that body's member, and a chunk opening on
a doc comment is credited to what the comment documents.

## Configuration

`CapsuleOptions(level, in_bm25)`, passed to `ZembleIndex.from_path(...,
capsules=...)` or `create_index_from_path(..., capsules=...)`. Levels are `full`
(everything above), `lite` (path and type chain only) and `off` (no capsule at all;
byte-for-byte the pre-capsule behaviour). `in_bm25` additionally appends the capsule,
minus its path segment, to the BM25 document - BM25 already enriches with path
tokens, so that part would be duplicate mass.

`ZEMBLE_CAPSULE=off|lite|full` and `ZEMBLE_CAPSULE_BM25=0|1` override the defaults;
explicit options beat the environment. The configuration is part of the index cache
identity (`capsules` in `metadata.json`), so a cache built under one setting is never
reused under another. The persisted index format is `12`.

**Default: `full` with `in_bm25=True`** - the measured winner below.

## Experiments

Every variant was run on both evaluation sets with the default embedder
(`potion-code-16M`). E0 reproduced both published baselines exactly
(javaweb 0.5321, upstream 0.8517), which is what makes the rest comparable.

| Exp | Variant | javaweb NDCG@10 | upstream NDCG@10 |
| --- | ------- | --------------: | ---------------: |
| E0 | no capsule (baseline) | 0.5321 | 0.8517 |
| E1 | capsule prefixed to the dense document | 0.5435 | 0.8642 |
| **E2** | **E1 + capsule tokens in BM25** | **0.5669** | **0.8643** |
| E3 | capsule-lite (path + type chain), dense only | 0.5416 | not run |
| E4 | E2 + `find_related` embeds its seed with its capsule | not measurable | not measurable |

### javaweb, by query kind (NDCG@10)

| kind | E0 | E1 | E2 | E3 |
| ---- | -: | -: | -: | -: |
| symbol | 0.9403 | 0.9710 | **0.9775** | 0.9710 |
| behavioural | 0.4993 | 0.5121 | **0.5703** | 0.5124 |
| architecture | 0.4163 | 0.4163 | 0.4163 | **0.4286** |
| bug-report | **0.3832** | 0.3591 | **0.3832** | 0.3818 |
| consumer | 0.1207 | 0.1422 | **0.1467** | 0.0850 |
| NDCG@5 | 0.5213 | 0.5264 | **0.5425** | 0.5281 |

### Upstream 63 repos, by language (NDCG@10)

| language | E0 | E1 | E2 | | language | E0 | E1 | E2 |
| -------- | -: | -: | -: |-| -------- | -: | -: | -: |
| bash | 0.840 | 0.873 | 0.867 | | php | 0.864 | 0.888 | 0.885 |
| c | 0.756 | 0.774 | 0.774 | | python | 0.875 | 0.881 | 0.875 |
| cpp | 0.888 | 0.905 | 0.911 | | ruby | 0.909 | 0.892 | 0.898 |
| csharp | 0.872 | 0.887 | 0.885 | | rust | 0.820 | 0.838 | 0.856 |
| elixir | 0.906 | 0.915 | 0.915 | | scala | 0.921 | 0.915 | 0.915 |
| go | 0.910 | 0.910 | 0.910 | | swift | 0.848 | 0.854 | 0.858 |
| haskell | 0.773 | 0.769 | 0.769 | | typescript | 0.703 | 0.712 | 0.713 |
| java | 0.803 | 0.833 | 0.831 | | zig | 0.899 | 0.923 | 0.919 |
| javascript | 0.911 | 0.915 | 0.915 | | kotlin | 0.803 | 0.835 | 0.836 |
| lua | 0.835 | 0.870 | 0.870 | | | | | |

By category, upstream: architecture 0.8061 -> 0.8236, semantic 0.8428 -> 0.8571,
symbol 0.9470 -> 0.9524.

### Cost

| | E0 | E2 |
| - | -: | -: |
| javaweb index (73.9k chunks) | 42.8 s | 52.1 s |
| javaweb query p50 | 132 ms | 143 ms |
| upstream index, mean per repo | 2.15 s | 2.71 s |
| upstream retrieved tokens per query | 1747 | 1703 |

## Decision

**E2 ships as the default.** It is the best javaweb variant (+0.0348) and it does not
regress the upstream set - it improves it by +0.0126, far outside the -0.003
tolerance. No per-kind trade-off had to be accepted: against the baseline, E2 wins on
symbol, behavioural and consumer queries, ties on architecture and bug-report, and
loses on nothing.

Notes worth keeping:

- **The dense-only gain is real but small; the BM25 half is where the javaweb win
  comes from** (0.5435 -> 0.5669). That is the expected shape: the default embedder is
  a static, order-free model, so a capsule prefix only shifts token mass, whereas BM25
  gains genuinely new terms - a chunk body that mentions no package, no class name and
  no imported type is now findable by all three.
- **The lite capsule is not a cheap version of the full one.** E3 is roughly E1 on
  behavioural and symbol queries, is the only variant that improves architecture
  (0.4286), and is the only variant that makes `consumer` queries *worse* than
  baseline (0.0850 vs 0.1207) - a type chain without signatures apparently pulls
  declaration-shaped chunks over the call sites those queries want.
- **E1 alone hurts bug-report queries** (0.3832 -> 0.3591) and E2 restores them
  exactly. Dense-only capsules are not a safe subset of E2.
- **E4 is unfalsifiable with the current harness.** `run_benchmark` only calls
  `index.search`, so nothing measures `find_related`. The change ships anyway on a
  consistency argument, not a measured one: with capsules on, every indexed vector is
  a capsule-plus-body vector, so embedding the seed chunk without its capsule would
  compare two different text conventions. It is inert when capsules are off.
- The javaweb set is pinned to a live working tree, not a SHA: chunk counts moved from
  73,906 to 73,961 across the runs. The deltas are far larger than that drift, but a
  0.001-scale difference on this set is not meaningful.
