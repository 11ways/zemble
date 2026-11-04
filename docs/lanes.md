# Lanes: worktrees of an indexed workspace

A lane is a folder whose checkouts are linked worktrees (`git worktree add`) of an indexed
workspace's checkouts at the same relative paths, like the zenit workspace's build lanes. The
lane root itself need not be a repository. `zemble.lane` is the one home of the identity rule
and of the per-file "same as the base" decision.

## Identity

Walking up from the requested path to its checkout and that checkout's main worktree (read from
the `.git` file and `commondir`, no git process), every level where the two paths still differ
is a candidate (lane, base) pair; for a folder that is no checkout, its checkouts up to two
levels down vote for a base. The highest pair whose base holds a compatible index wins, so a
sub-path of a lane is a sub-tree of the lane's overlay. No indexed base: no lane, the path is
indexed on its own (fail closed). A clone (own object store) is never a lane.

## Index overlay

`zemble.index.overlay` walks the lane as its own build would and keeps a base row for every
file the base indexed, whose stored copy is still the base's (manifest mtime), and whose lane
copy is byte-for-byte the base's: git answers that for tracked files (`git diff <base HEAD>` in
the lane, `git diff HEAD` in the base), bytes for everything else. The rest goes into a small
delta store in the lane's cache folder (`overlay-<content>/`), embedded through the shared
content-addressed embedding cache. Chunks, BM25, vectors and symbol definitions are presented as
one row space; BM25 scores both segments against one corpus. Results carry lane paths.

## Graph

A lane's graph is an overlay too (`zemble.graph.overlay`): a small store holding only the rows of
the files the lane owns, over the base's graph version, which a connection attaches as `base`.
TEMP views named like every graph table present base rows minus masks plus the overlay's rows,
so every reader, and the ordinary incremental build, sees one graph. The base's files outside
the checkouts the lane holds (`zemble.lane.LaneScope`: a file's innermost base checkout is one
the lane has a worktree of) are never copied, extracted or removed: a build keeps them as the
world the lane's edits resolve against, a reader hides them. Its first build walks the lane,
stamps every file the comparer calls the base's with the base's stamp, and runs the ordinary
build in two passes. The first extracts and resolves only the files the lane changed or added
(removing the ones it deleted), from their source alone, and publishes: the lane answers with its
edits from then on. The second re-resolves the files that wrote a name those moved and maps
compiler facts; what it owes is recorded in the overlay first (`overlay_pending_names`), so a
worker stopped between the passes leaves the next refresh to finish it. A re-resolved base file
whose edges come out exactly as the base's is left to the base (`restore_unchanged`), so the
overlay keeps only rows that differ. What it then answers for the lane's checkouts is what a whole
graph of the workspace with the lane's edits applied answers (`tests/test_graph_lane_overlay.py`).
Facts files the lane lacks are read from the base, against the lane's sources, so unchanged
sources keep their compiler facts. Out-of-scope callers are hidden, so a tally of a symbol's
sources counts the lane's own.

An overlay speaks the base version's `refs` keys, so it is valid for one base version and one
build of it. Once the base moved, the overlay still answers (or, its version retired, the base's
current graph does) and the daemon rebuilds it from scratch on the lane's next graph question.
Until a lane's first overlay is published, the base's graph answers for the lane.

A lane parked inside its own base (`<workspace>/lanes/<name>`) is the base's checkouts again: the
base's walk, watcher and facts discovery skip every linked worktree of a checkout the base holds
(`zemble.index.file_walker.is_duplicate_worktree`), so it is indexed only as its own root.

## Daemon

An overlay takes no `ZEMBLE_DAEMON_MAX_INDEXES` slot, is re-planned when its lane changes or its
base is rebuilt, and is evicted with its base. A root that no longer exists is evicted on its next
rebuild, and the daemon prunes the caches of gone roots on start and every
`ZEMBLE_DAEMON_PRUNE_HOURS` (6); `zemble clear gone --dry-run` lists the same set.

## Measured (archdev, 2026-10-25, 16-checkout lane of the zenit workspace)

| | before | after |
| --- | --- | --- |
| first `zemble search` in a fresh lane (daemon, base resident) | 206 s (in-process build) | 7.5 s |
| first `zemble home` in a fresh lane (daemon) | 451 s (in-process, graph from scratch) | 8.2 s |
| first `zemble search`, in-process (`ZEMBLE_DAEMON=0`) | 206 s | 33 s |
| lane resident cost | a full 145k-chunk index and a slot | 36 delta chunks |
| daemon with main + 3 lanes | thrashed 4 slots at the 4 GiB cap | rss 2.3 GB, private 0.94 GB |

## Measured: lane graph as an overlay (this host, 2026-11-04)

Base: the zenit workspace graph built fresh without its lanes (19,426 files, 343,001 symbols,
4,232,747 edges, 1,737 MB), in an isolated cache; one extraction worker, as the daemon's graph
worker runs. Before is a8afb53 (copy the base whole, then the walking refresh).

| | before | after |
| --- | --- | --- |
| `semantic-search` (36 of 49 checkouts, 154 changed, 38 deleted): the lane answers from its own graph | 426.9 s wall, 408.5 s CPU | 14.8 s into the build; 21.5 s wall, 19.8 s CPU end to end in a fresh process |
| `semantic-search`: build complete | 426.9 s (3,543 files removed, 13,685 re-resolved) | 463.2 s wall, 458.4 s CPU (11,141 dependents, 7,348 left to the base) |
| `semantic-search`: lane store | 1,737 MB | 575 MB |
| `toolchain-waste` (48 of 49 checkouts, 60 changed): the lane answers from its own graph | 443.5 s wall, 425.1 s CPU | 9.2 s into the build; 13.8 s wall, 13.4 s CPU end to end |
| `toolchain-waste`: build complete | 443.5 s | 205.4 s wall, 201.7 s CPU (2,815 dependents, 1,385 left to the base) |
| `toolchain-waste`: lane store | 1,741 MB | 330 MB |

The second pass costs what the base's own incremental refresh pays for the same edits: a lane
that adds or removes a common name (`get`, `of`) makes thousands of files dependents. It runs after
the lane already answers.
