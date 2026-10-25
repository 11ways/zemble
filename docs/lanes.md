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

A lane's first graph build copies the base's current graph (sqlite online backup), stamps every
unchanged file with the lane's mtime and size, and runs the ordinary walking refresh, so only the
lane's edits are extracted and the files it lacks removed. Until it is published, the base's graph
answers for the lane. A lane holding a subset of the workspace's checkouts pays for removing the
rest: measured 225 s for a 16-of-45-checkout lane, in the daemon's graph worker, never in a query.

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
