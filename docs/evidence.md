# Evidence bundles

`zemble explain` answers a question with a fixed number of tokens. It searches,
follows the Java symbol graph one hop out of what it found, and packs the result
under a budget with a one-line reason per item. Two smaller surfaces come with
it: `zemble outline` (what a file or type declares, signatures only) and
`zemble signatures` (a declaration plus the call sites the graph resolved
exactly).

The measurement below is the honest headline: **bundles do not beat plain search
on hit rate, and they cost more tokens.** What they add is structure - reasons,
outlines, callers, tests, and an explicit list of what did not fit. Read the
[measurement](#measurement) section before reaching for them as a retrieval win.

## The bundle

A bundle is an ordered list of items. Each item is a source region with a kind,
a reason, its own token cost, and how much of it survived the budget.

    ## zenit/src/common/java/.../data/PageWindow.java:19-33  (search hit #1 inside PageWindow)
    ```java
    public record PageWindow(int page, int pageCount, int offset, int pageSize) {

        /** What a page number outside {@code [1, pageCount]} resolves to. */
        public enum OutOfRange {
    ...
    ```

### Tiers

Packing order. Tier 0 is what search found; every later tier is one graph hop
away from it, ordered by how often it turns out to be the thing you needed.

| Tier | Kind | Where it comes from |
| --- | --- | --- |
| 0 | `chunk` | The search hits in the primary files, verbatim. |
| 1 | `outline` | The anchor's enclosing type, from the graph: signatures only, no file text. |
| 2 | `test` | `tests_of` the anchor: naming matches (`TESTS`) before incidental use (`EXERCISES`). |
| 3 | `caller` | `callers` of the anchor, `EXACT` before `UNIQUE_NAME`. |
| 3 | `note` | One line counting the call sites that were ambiguous or unresolved. |
| 3 | `implementation`, `supertype` | Only for an interface or an abstract type. |
| 4 | `callee` | What the anchor calls, outside its own file. |
| 4 | `doc` | A `.md` the same query matched, or one sitting beside a primary file. |

The **primary files** are the five files with the best single search hit behind
them (best chunk, not the sum of a file's chunks: summing rewards a fragmented
file over the one holding the actual answer). At most three chunks per primary
file become tier 0 items.

An **anchor** is the nearest enclosing callable of a primary chunk, or its
enclosing type when the chunk sits between declarations. It is found with
`GraphProvider.symbols_at(file, start, end)` - containment first, overlap as the
fallback. At most six anchors per bundle, at most three items per tier per
anchor, deduplicated across anchors by source region so the same lines never
arrive twice under two different reasons.

### Packing and degrade rules

1. Candidates are sorted by tier, then fused search score, then discovery order.
2. Each tier packs against the budget **minus a reserve**: what it would cost to
   name every later-tier candidate as a location line, capped at 40 percent of
   the budget. Without it a handful of long primary chunks eats everything and
   the graph expansion - the whole point of a bundle - never appears.
3. An item that does not fit as content **degrades**, it is not dropped: a
   primary chunk is truncated to the largest line count that fits (bisected, and
   marked `... (truncated, N more lines)`), anything else falls back to a
   one-line location plus its signature. The reserve limits how much *content* a
   tier may take; it never stops an item from being *named*.
4. What still does not fit is listed under `## Not included (locations only)`,
   together with the search hits that fell outside the primary files.
5. The budget covers the **whole rendered answer**, headings and that footer
   included. If the footer pushes it over, footer entries are dropped first
   (counted as `... and N more, not listed for budget`), evidence last.

`bundle.total_tokens` is the item sum; `bundle.rendered_tokens` is the whole
markdown answer and is what the budget is enforced against.

### Token estimate

`zemble.evidence.tokens.estimate_tokens` divides the character count by 3.6. It
is an estimate, not a tokenizer: no model, no network, cheap enough for the
thousands of calls a bisecting packer makes. Expect roughly 10 percent error on
source text, worse on text that is mostly punctuation.

## CLI

```
zemble explain <path> "<query>" [--budget 3000] [--top-k 20] [--json]
zemble outline <path> <file-or-type> [--members PATTERN] [--json]
zemble signatures <path> <symbol> [--json]
```

All three surfaces ask the warm daemon first and answer in-process only when it
cannot be reached (see `docs/daemon.md`), so they share one RAM copy of the index
and the graph with `search`. Both the index and the symbol graph are built on
demand and refreshed once per process, the same way `zemble graph` does it. `outline` takes a workspace
relative file path or a simple or qualified type name; `--members` matches a
plain word as `*word*` and prunes the types left with nothing under them. Exit
codes match the graph subcommands: `0` answered, `1` nothing found, `2` the name
was ambiguous, with the candidates on stderr.

```
$ zemble outline ~/projects/javaweb PageWindow --members page
package be.elevenways.zenit.common.data
# zenit/src/common/java/be/elevenways/zenit/common/data/PageWindow.java

record PageWindow(int page, int pageCount, int offset, int pageSize)  L19-85
  field int page  L19
  field int pageCount  L19
  field int pageSize  L19
  method int pageCountFor(long total, int pageSize)  L52-61
  method PageWindow withPageCountCappedAt(int maxPageCount)  L80-84  [@NonNull]
```

## MCP

Three tools are registered on the existing server: `explain` (returns the
markdown bundle, default budget 2500), `outline` and `signatures` (both return
JSON). All three go through the daemon first; when there is none, `explain`
receives the index from the server's own cache, so it still shares the warm index
with `search`.

## Measurement

`benchmarks/evidence_eval.py` over the javaweb local set: 80 hand-verified
queries against a 40-repo Java workspace (73,957 chunks, 101,166 symbols,
922,803 edges). For every query it builds a bundle at 1500, 3000 and 6000 tokens
and records whether an annotated file appears as content, as any item, or as a
named location, beside what plain `search` costs and hits.

```bash
uv run python -m benchmarks.evidence_eval
```

Result file: `benchmarks/results/evidence-javaweb-5cdabc0880b2.json`.

Reference points on the same 80 queries:

| Reference | Hit rate | Tokens |
| --- | ---: | ---: |
| `search` top-5 (full chunks) | 0.588 | 1015 |
| Reading the annotated files outright | 1.000 | 3867 |

Bundles:

| Budget | content | any item | named | tier-0 | expansion | expansion only | tokens | items | ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | 0.562 | 0.575 | 0.575 | 0.575 | 0.450 | 0.000 | 1472 | 13.4 | 186 |
| 3000 | 0.575 | 0.588 | 0.625 | 0.575 | 0.475 | 0.013 | 2967 | 18.7 | 163 |
| 6000 | 0.575 | 0.588 | 0.650 | 0.575 | 0.475 | 0.013 | 5750 | 19.2 | 159 |

- **content**: an annotated file is in the bundle as code (full or truncated).
- **any item**: as any item, location-only included.
- **named**: as an item or in the omission list.
- **tier-0 / expansion**: which half of the bundle the annotated file came from.
- **expansion only**: the graph hop found it and the search chunks did not.

By kind, at budget 1500 (`n` queries, `search@5` for comparison):

| kind | n | content | any | search@5 | bundle tokens | full-file tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| symbol | 20 | 0.950 | 0.950 | 1.000 | 1473 | 3063 |
| behavioural | 25 | 0.560 | 0.600 | 0.600 | 1468 | 3978 |
| architecture | 15 | 0.467 | 0.467 | 0.467 | 1467 | 4874 |
| bug-report | 10 | 0.400 | 0.400 | 0.300 | 1479 | 4958 |
| consumer | 10 | 0.100 | 0.100 | 0.200 | 1476 | 2599 |

At 3000 the picture is the same to within one query per kind; the extra budget
buys content where 1500 bought location lines, not new files.

### The verdict, and what is responsible

**Bundles do not beat plain search on hit rate at equal or lower tokens.** At
1500 tokens (1.45x what `search` top-5 costs) they land marginally *below* it
(0.575 versus 0.588 any-hit); at 3000 they draw level. The one number that
improves monotonically is `named`, 0.575 to 0.650: with more budget the bundle
mentions more of what exists.

The cause is structural, not a tuning miss:

- **Tier 0 is the search result set.** A bundle's recall over annotated files is
  the recall of `index.search`, plus whatever the graph reaches from it. The
  `expansion only` column measures exactly that surplus: **one query in eighty**.
  The graph hop reaches files search already found (`expansion` 0.45-0.48,
  almost entirely overlapping tier 0) far more often than it reaches new ones.
- **The reserve trades content for coverage on purpose.** At 1500 it holds back
  up to 600 tokens so tiers 1-4 can appear at all, which is why `content` at
  1500 (0.562) sits below `tier-0` (0.575): the annotated file is in the bundle,
  but as a location line rather than as code. Removing the reserve would buy
  back that one query and turn the bundle back into fenced search results.
- **`consumer` queries are the worst kind (0.1 against search's 0.2)** and they
  are the ones bundles should have owned: their annotated answer is a test or a
  caller, which is tier 2 or 3 - exactly the tiers that degrade to location
  lines first. A bundle that packed by "which tier answers this question" rather
  than by fixed tier order would do better here, but that requires classifying
  the query, which this design deliberately does not do.
- **The budget is always spent.** There are always more candidates than budget,
  so a bundle costs what it is given. The "compression ratio" against reading
  the annotated files outright (2.6x at 1500, 0.67x at 6000) says more about the
  budget chosen than about the packer.

No per-query tuning was done, and nothing in the packer looks at the annotation
set. The two changes made after seeing aggregate numbers were both principled
and are documented above: ranking primary files on their best chunk rather than
the sum (no measurable effect: identical numbers), and letting the location form
bypass the reserve (symbol content 0.90 to 0.95 at 1500).

### Where they are worth it anyway

The hit-rate result says a bundle is not a better *finder*. It is a better
*answer shape* once something is found: every item says why it is there, tests
and callers arrive without a second round trip, and the omission list tells an
agent what exists without paying for it. The cheapest surfaces here are
`outline` (a whole class for 150-300 tokens) and `signatures`, and neither
depends on the packing result above.

## Limits

- Java only, because the expansion is the Java symbol graph. A non-Java primary
  chunk still becomes a tier 0 item; it simply has no anchor and no expansion.
- Every graph limit applies (see `docs/graph.md`): `UNIQUE_NAME` is a guess, and
  a bundle's reasons say so rather than hiding it.
- Tier-3 ambiguous callers are counted in a note, never listed. Use
  `zemble graph callers` when the count matters.
- The doc tier only sees `.md` files that the index actually holds (so
  `--content docs` or `all` for the search-matched half) plus one sibling
  markdown file per primary directory.
- A bundle over a git-URL index has no readable workspace root, so tiers 2-4
  degrade to location lines: their text is read from files, not from the index.
