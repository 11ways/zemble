# Where does this belong?

`zemble home` answers the question a new capability starts with: **does this
already exist, and if not, which module should own it?**

```bash
zemble home /home/skerit/projects/javaweb "let a user schedule a recurring action on one record"
```

It searches, asks the symbol graph who consumes what it found, reads the module
map the workspace declares about itself, and returns four sections: the existing
mechanisms that look like the description, the candidate homes ranked with
reasons, a verdict, and the rules, forbidden dependencies and skills that apply
to the candidates.

Everything it knows about a workspace comes from `<root>/.zemble/home.toml`.
Nothing about any particular workspace lives in zemble.

## The answer

### Existing mechanisms

The distinct symbols behind the best code hits, at most eight. For each: its
signature, its module, and its **consumer spread** - the distinct modules that
call, import, subclass or otherwise point at it, its own module excluded. A type
is consumed by every incoming edge (`references`), a method by its callers.

A mechanism is marked a **strong match** when it is within 15% of the best hit's
score AND carries the shape of a mechanism rather than of one caller's private
helper: consumers in two or more modules, a position closer to the core than every
one of its consumers, or - the case consumer spread cannot see - **being the symbol
a matched declared row names**.

That last lane exists because a mechanism's wrappers often all live inside its own
module by design. `PreferenceCookie` is the reference case: `Themes`, `Timezones`
and `Disclosures` wrap it without leaving `zenit`, so its cross-module consumer
spread is zero and the first two rules read the declared mechanism as a private
helper. A row that writes a name in backticks - `PreferenceCookie.named(...)`,
`Brand`, `RecordSource.accessCriteria(AccessContext -> Criteria)` - is naming that
symbol as the mechanism, and a hit on it is "this already exists".

Only `Class` and `Class.member` are read as names: argument lists and generic
parameters are dropped first, the `Class.a/b/c` shorthand expands to one name per
member, the owner of a `Class.member` name counts as named too, and a backticked
path, package or setting key (`common/holder`, `comms.channels.*`) names nothing.
And only rows within 15% of the best-matching row may name anything - the same
near-the-top rule the hits themselves are held to, because every row of such a table
names classes, and letting a faintly matching row speak turns its neighbour's
mechanism into "this already exists" (measured below).

Three more lanes hang off this section: the declared-home rows the description
resembles, `find_related` on the single best hit ("also similar"), and the
documentation chunks the same description matched (only when the index covers the
docs lane, which the `home` surfaces always request).

### Candidate homes

Every module the code hits touched is scored:

| Term | Value | When |
| --- | --- | --- |
| relevance share | `module mass / total mass` | always; the module's share of the summed hit scores |
| declared bonus | up to `+0.8` | a matched declared-home row names this module, scaled by how well the row matched (word overlap weighted by how many rows use each word, so "record" or "user" cannot carry a match) |
| core proximity | `+0.35` | the module already holds this family AND at least one other DECLARED hit module sits after it in `order` |
| forbidden | `-1.0` each | another module the description touched may not depend on this one |

The top three are reported, each with its reasons as sentences ("3 of 5 hits live
here", "declared home for 'Pagination arithmetic' (row match 43%)", "closer to
the core than its 2 consumer module(s): zenit-cms, quirkyquarters", "would make
zenit-widget depend on zenit-cms: widget never depends on cms").

The forbidden check deliberately looks at **every** other module in the hits, not
only the ones further from the core: the module that may not depend on a
candidate is usually the one closer in, and scoping the check by order would make
the rule unreachable.

### Verdict

| Verdict | When |
| --- | --- |
| `EXTEND_EXISTING` | a strong match was found - by consumer spread, by core position or by declaration; it is named, with "wire or extend it; do not duplicate it" |
| `NEW_MECHANISM` | no strong match, and one candidate leads clearly; its module is the home |
| `UNCERTAIN` | the top two candidates are within 15% of each other, or nothing matched |

Confidence is `high` / `medium` / `low`, and a lead of less than 40% of the
leader's score is never called high. A workspace without a config never gets more
than medium.

### Checklist

The workspace's own rules whose scope covers the candidate modules, the forbidden
dependencies that touch them, and the skills to read before designing there.

## Configuration

`<root>/.zemble/home.toml`. A missing file is **generic mode**: modules are the
first path segment, there are no declared homes, rules or skills, and the answer
says so in a note. A file that is present but malformed is a loud error, never a
silent downgrade.

```toml
# Closer to the core first: this is the ranking.
order = ["protoblast", "hawkeye", "zenit", "plumage", "zenit-cms"]

# module name -> glob or list of globs. A path matching none of them falls back
# to its first path segment.
[modules]
protoblast = "protoblast/**"
hawkeye = ["hawkeye/**", "hawkeye-core/**"]

# Dependencies the workspace refuses. A candidate home that would create one loses.
[[forbidden]]
from = "zenit-widget"
to = "zenit-cms"
why = "zenit-widget never depends on zenit-cms"

# Markdown tables that already DECLARE homes. Backticked names in the home column
# that resolve to a declared module are read as the home; the rest stays prose.
[[tables]]
file = "CLAUDE.md"
capability = "Capability"
home = "Mechanism home"
consumers = "Consumers (thin wiring)"

# Read before designing in a module.
[skills]
zenit-forms = ["zenit-forms-editing"]

# Echoed in every answer whose candidates the rule covers. `modules` is optional;
# without it the rule is unscoped and always applies.
[[rules]]
text = "Nothing lands without at least one wired consumer and a test"

[[rules]]
text = "zenit core owns no visual components beyond Icon"
modules = ["zenit"]
```

`order` may also be written inside `[modules]`, which is how the design note
spelled it; a workspace that does that cannot also have a module named `order`.
Reading the file needs Python 3.11 (`tomllib`).

## Surfaces

| Surface | Call |
| --- | --- |
| CLI | `zemble home <path> "<description>" [-k 40] [--json] [--no-daemon]` |
| MCP | tool `home(description, repo, top_k)` |
| daemon | command `home` with `{path, description, top_k, content}` |

All three are daemon-first with an in-process fallback, and all three ask for the
`code` and `docs` content lanes: a design note naming a module is evidence too.
The CLI exits 1 when nothing matched at all.

## Measurement

`benchmarks/home_eval.py`, against the javaweb workspace. Ground truth is the
capability table in its `CLAUDE.md`: the 61 rows whose "Mechanism home" cell names
exactly one module. The queries are hand-written paraphrases in
`benchmarks/local/home_queries.json` - "I need one abstraction whose
implementation differs on the JVM and in the browser" for the row that declares
`protoblast` the home of platform seams - so a row's own words are not what finds
it.

Every query is answered twice: once with the declared-table lane **disabled**,
which measures what search, the graph and the module order can do alone, and once
with it enabled.

61 queries, 80946 chunks, 101166 symbols / 922803 edges, 0.2 s per answer:

| Lane | hit@1 | hit@3 | verdict names the home | EXTEND_EXISTING | UNCERTAIN |
| --- | --- | --- | --- | --- | --- |
| search + graph only | 0.770 | 0.918 | 0.738 | 0.59 | 0.08 |
| plus the declared table | 0.902 | 0.934 | 0.787 | 0.62 | 0.07 |

"hit@1" is the declared home ranked first among the candidates; "verdict names the
home" is the module the verdict itself named, which is `null` whenever the answer
is `UNCERTAIN`. `EXTEND_EXISTING` is the right verdict for all 61 - every one of
these capabilities exists - and it is reached for 62% of them; the rest report the
right module but do not find a symbol the graph or the table proves is the
mechanism.

The by-declaration lane is what moved that last figure from 0.59 to 0.623, at no
cost anywhere else: hit@1, hit@3, "verdict names the home" and the `UNCERTAIN` rate
are unchanged, no candidate ranking moved, and the two answers that changed are
`Dev tunnel` and `Trusted-proxy client IP`, both `NEW_MECHANISM -> EXTEND_EXISTING`
naming the same, correct module.

The six remaining rank-1 misses with the table lane on are nearly all the same
shape: the core-proximity bonus pulls the answer one step towards the core.

| Row | Declared home | Answer |
| --- | --- | --- |
| Record-scoped grants | zenit-auth | zenit led (13 of 33 hits); the grant tables read as ORM |
| Field-level access decisions | zenit | hawkeye led on one hit; zenit is #2 |
| Widget trees | zenit-widget | hawkeye led (15 of 27 hits); templates dominate, zenit-widget is #2 |
| Surface actions | zenit-widget | hawkeye led, same reason |
| Pages + public page routing | zenit-pages | zenit led on routing |
| Microcopy catalogs | zenit-microcopy | zenit led; `zenit-microcopy` depends on `zenit`, so the bonus points the wrong way |

That bias is deliberate - "the mechanism lives as close to core as its mechanics
allow" is the rule it encodes - and it is why the answer is a ranked list with
reasons rather than a single module. hit@3 is 0.918 either way: what the answer
misses at rank 1 it almost always still shows.

`EXTEND_EXISTING` is reached for 62% of the rows at the 15% strong-match window.
Widening that window to 30% raised it to 80% when it was measured, but "the verdict
names the declared home" fell from 0.787 to 0.721: the extra matches are mechanisms in a *different*
module, and an `EXTEND_EXISTING` pointing at the wrong module is a worse answer
than a `NEW_MECHANISM` pointing at the right one. The narrow window is kept.

A second measured non-improvement, of the by-declaration lane: letting *every*
matched row name symbols, rather than only the rows near the best match, reaches
`EXTEND_EXISTING` for 0.656 of the rows but drops "verdict names the home" from
0.787 to 0.770. It buys `Outbound message shapes` (`ChannelMessage`, named by a row
matching at 0.26 while the leading row matched at 0.72) and loses `Event delegation`
to `KeyedListenerTable` in protoblast, named by a row matching at 0.26 while the
leading rows matched at 0.57. The two are the same shape - a faint row naming a
neighbouring capability's class - and nothing in the numbers separates them, so the
narrow rule is kept.

One measured non-improvement of the row scoring itself, recorded so it is not tried
again: discounting a
declared row by how little of ITSELF the overlap covers - the obvious fix for a
row whose capability text is a paragraph and therefore shares a word with
everything - drops hit@1 from 0.902 to 0.869, because the genuine rows in that
table are long paragraphs too. The row-frequency weighting stays; the length
penalty does not.

Three limits worth stating: a module the search does not touch at all is never a
candidate, however core it is; the declared-table match is a token overlap
(weighted by how many rows use a word, so that "record" and "user" cannot carry a
match on their own), so a row phrased in vocabulary the description does not share
is not found; and the answer only knows the modules that are checked out - the
javaweb `CLAUDE.md` names consumers such as `hohenheim` that live in no directory
here, and those names simply do not resolve.
