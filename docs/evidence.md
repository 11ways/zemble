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

The default packing order, and the one every bundle ships with. Tier 0 is what search found; every later tier is one graph hop
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

An intent-chosen order is available on request; see [query intent](#query-intent).

### Query intent

`zemble.evidence.intent.classify` decides, from the query text alone, which of
six intents a question has, and says which rule decided it. The rules are
ordered and the first to fire wins:

| Rule | Intent | Fires on |
| --- | --- | --- |
| `identifier` | symbol | The whole query is one code name: `PageWindow`, `SessionIds.forToken`, `zc-inbox`. |
| `who-uses` | consumer | "who uses/calls/consumes X", "what code calls X". |
| `callers-of` | consumer | "callers of X", "consumers of X", "used by". |
| `where-used` | consumer | "where is X used/called/referenced". |
| `that-uses` | consumer | Code described by what it uses: "templates that hand a picker a provider". |
| `tests-for` | consumer | The subject is a test: "tests covering X". A test is a consumer. |
| `how-does` | architecture | "how does/is/are/can ... ". |
| `wiring` | architecture | "wired", "work together", "flows through". |
| `symptom` | bug | "instead of", "should be", "keeps ...ing", "shows nothing", "fails". |
| `default` | behaviour | Everything else. |

Each intent names a `TierPlan`: the tier of every item kind plus its caps, in one
table (`PLANS` in `bundle.py`). `symbol`, `behaviour` and `unknown` are the
default order above. The three that differ:

| Intent | Order | Caps |
| --- | --- | --- |
| consumer | chunk, then **caller/implementation/test/note**, then outline/supertype, then callee/doc | 8 per anchor instead of 3; primary chunks cut to 20 lines; exact resolutions first. |
| architecture | **outline/supertype/implementation**, then chunk, then test, then caller, then callee/doc | At most 3 types outlined. |
| bug | chunk, then **test**, then outline, then the rest | Default caps. |

A consumer plan also **seeds**: every identifier-looking word in the query is
resolved through `GraphProvider.definition`, and the exact callers of a method,
or the implementations of a type, become candidates whether or not search
returned them. This is the one hop that can put a file in a bundle that search
never retrieved.

The detected intent is always printed in the bundle header, together with the
order that actually packed it:

    intent: consumer (rule: who-uses; order: default)

**Detection does not reorder anything by itself.** The measurement below says the
intent orders do not beat the fixed one on the javaweb set, so the default stays
the fixed order and an intent order is opt-in:

```
zemble explain <path> "<query>" --intent consumer
```

`--intent` answers in-process rather than through the daemon: the daemon protocol
carries no intent argument, and asking it would silently ignore the override.

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
zemble explain <path> "<query>" [--budget 3000] [--top-k 20] [--intent NAME] [--json]
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

`benchmarks/evidence_eval.py` over the javaweb local set: 90 hand-verified
queries against a 40-repo Java workspace (77,102 chunks, 102,376 symbols,
958,269 edges). For every query it builds a bundle at 1500, 3000 and 6000 tokens,
in each tier order under test, and records whether an annotated file appears as
content, as any item, or as a named location, beside what plain `search` costs
and hits.

```bash
uv run python -m benchmarks.evidence_eval                    # both orders
uv run python -m benchmarks.evidence_eval --variants base    # the shipped one only
```

Result file: `benchmarks/results/evidence-javaweb-d56f2e1683da.json`, measured
2026-08-21 with the local embedder. The numbers below replace an earlier run over
80 queries; the set has since grown by the ten template queries, so the two are
not comparable and the older table is gone rather than kept beside this one.

Reference points on the same 90 queries:

| Reference | Hit rate | Tokens |
| --- | ---: | ---: |
| `search` top-5 (full chunks) | 0.600 | 1015 |
| Reading the annotated files outright | 1.000 | 3846 |

Bundles, in the shipped default order:

| Budget | content | any item | named | tier-0 | expansion | expansion only | tokens | items | ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | 0.567 | 0.611 | 0.611 | 0.611 | 0.422 | 0.000 | 1476 | 12.8 | 173 |
| 3000 | 0.611 | 0.622 | 0.678 | 0.611 | 0.433 | 0.011 | 2931 | 17.9 | 172 |
| 6000 | 0.611 | 0.622 | 0.700 | 0.611 | 0.433 | 0.011 | 5499 | 18.3 | 167 |

- **content**: an annotated file is in the bundle as code (full or truncated).
- **any item**: as any item, location-only included.
- **named**: as an item or in the omission list.
- **tier-0 / expansion**: which half of the bundle the annotated file came from.
- **expansion only**: the graph hop found it and the search chunks did not.

By kind, at budget 1500 (`n` queries, `search@5` for comparison):

| kind | n | content | any | search@5 | bundle tokens | full-file tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| symbol | 24 | 0.958 | 1.000 | 1.000 | 1477 | 2962 |
| behavioural | 29 | 0.552 | 0.621 | 0.621 | 1475 | 4028 |
| architecture | 16 | 0.375 | 0.438 | 0.438 | 1474 | 5022 |
| bug-report | 10 | 0.400 | 0.400 | 0.300 | 1481 | 4958 |
| consumer | 11 | 0.182 | 0.182 | 0.182 | 1475 | 2572 |

At 3000 the picture is the same to within one query per kind; the extra budget
buys content where 1500 bought location lines, not new files.

### Intent-aware tier order, measured

The classifier agrees with the eval set's `kind` labels on **0.944** of the 90
queries (`architecture` 1.00, `symbol` 1.00, `behavioural` 0.966, `bug-report`
0.80, `consumer` 0.818). The labels are ground truth for this diagnostic only;
nothing in the packer reads them.

Default order versus the intent-chosen order, same run, same tree:

| Budget | Kind | content | any item |
| ---: | --- | ---: | ---: |
| 1500 | overall | 0.567 -> 0.556 | 0.611 -> 0.611 |
| 1500 | symbol | 0.958 -> 0.958 | 1.000 -> 1.000 |
| 1500 | behavioural | 0.552 -> 0.552 | 0.621 -> 0.621 |
| 1500 | architecture | 0.375 -> 0.312 | 0.438 -> 0.438 |
| 1500 | bug-report | 0.400 -> 0.400 | 0.400 -> 0.400 |
| 1500 | consumer | 0.182 -> 0.182 | 0.182 -> 0.182 |
| 3000 | overall | 0.611 -> 0.600 | 0.622 -> 0.611 |
| 3000 | architecture | 0.438 -> 0.375 | 0.438 -> 0.438 |
| 3000 | consumer | 0.182 -> 0.182 | 0.273 -> 0.182 |
| 6000 | overall | 0.611 -> 0.611 | 0.622 -> 0.611 |

The one column that improves is `expansion only`, 0.000 -> 0.056 at every
budget: the intent order pulls five annotated files out of the graph hop that the
search chunks did not carry. It pays for them with content elsewhere.

**Decision: the fixed order stays the default.** The rule set before the run was
"ship it only if overall content-hit and any-hit at 1500 and 3000 do not drop and
consumer and architecture improve". Overall content drops (0.567 -> 0.556 and
0.611 -> 0.600), architecture drops, and consumer does not move at 1500 and loses
one query at 3000. So `explain` keeps packing the fixed order, the detected
intent is reported in the header, and `--intent` applies an order on request.

Why it did not help, in the terms the rest of this document uses:

- **Reordering cannot add a file.** Consumer answers in this set are whole test
  files that search never returned; moving tier 3 in front of tier 1 rearranges
  what was already found. `consumer` any-hit at 3000 even falls, because the
  outline that used to name the annotated file is now packed after eight callers.
- **The seed, which CAN add a file, never fires here.** It needs the query to
  name a symbol the graph resolves; **0 of the 11 consumer queries do** - they
  are prose ("admin list pages that consume the paging window helper"). The
  mechanism is exercised by its own tests, not by this eval.
- **Architecture pays for outlines it did not need.** Outlining the top three
  types before any chunk spends 200-400 tokens on signatures, and at 1500 that is
  exactly the room the annotated chunk needed.

### The verdict, and what is responsible

**Bundles do not beat plain search on hit rate at equal or lower tokens.** At
1500 tokens (1.45x what `search` top-5 costs) they draw level with it (0.611
versus 0.600 any-hit); at 3000 they are one query ahead. The one number that
improves monotonically is `named`, 0.611 to 0.700: with more budget the bundle
mentions more of what exists.

The cause is structural, not a tuning miss:

- **Tier 0 is the search result set.** A bundle's recall over annotated files is
  the recall of `index.search`, plus whatever the graph reaches from it. The
  `expansion only` column measures exactly that surplus: **one query in ninety**
  in the default order, five in the intent order. The graph hop reaches files
  search already found (`expansion` 0.42-0.46, almost entirely overlapping tier
  0) far more often than it reaches new ones.
- **The reserve trades content for coverage on purpose.** At 1500 it holds back
  up to 600 tokens so tiers 1-4 can appear at all, which is why `content` at
  1500 (0.567) sits below `tier-0` (0.611): the annotated file is in the bundle,
  but as a location line rather than as code. Removing the reserve would buy
  back that one query and turn the bundle back into fenced search results.
- **`consumer` queries are the worst kind (0.182, level with search)** and they
  are the ones bundles should have owned: their annotated answer is a test or a
  caller, which is tier 2 or 3 - exactly the tiers that degrade to location
  lines first. Packing by "which tier answers this question" was the obvious
  follow-up; it was built, measured, and did not help (see
  [intent-aware tier order](#intent-aware-tier-order-measured)). What is missing
  is recall, not order.
- **The budget is always spent.** There are always more candidates than budget,
  so a bundle costs what it is given. The "compression ratio" against reading
  the annotated files outright (2.6x at 1500, 0.67x at 6000) says more about the
  budget chosen than about the packer.

No per-query tuning was done, and nothing in the packer looks at the annotation
set. The changes made after seeing aggregate numbers were all principled and are
documented above: ranking primary files on their best chunk rather than the sum
(no measurable effect: identical numbers), letting the location form bypass the
reserve (symbol content 0.90 to 0.95 at 1500), and keeping the fixed order as the
default after the intent orders lost.

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
- The intent classifier reads English phrasing and Java-shaped names. A query in
  another language, or one that describes its subject without naming it, falls to
  `behaviour` - which is the default order anyway, so a miss costs nothing but
  the header line.
- The consumer seed needs a name it can resolve. A prose question about a subject
  it never names ("the pages that consume the paging helper") seeds nothing, and
  that is the majority of real consumer questions in the eval set.
- `--intent` is a CLI argument only. The daemon protocol and the MCP `explain`
  tool have no intent parameter, so both always answer in the default order.
