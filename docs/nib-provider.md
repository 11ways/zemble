# nib provider mode

[nib](https://github.com/11ways/nib) is an MCP server with line-anchored file tools. It can host
other MCP programs as *providers*: nib starts the program privately and offers its tools as its own
(zemble's `search` becomes `zemble_search`), so a client configures only nib. nib, never the
provider, prints file lines: zemble says where the code is, and nib prints those lines from the file
as it is now, as `[PATH#TAG]` + `N:TEXT` rows the agent can edit from.

```
nib --provider zemble=zemble          # or: nib setup --provider zemble=zemble
```

`zemble install` does this for you when nib is on `PATH` (see
[installation](installation.md#hosted-by-nib)); `zemble uninstall` undoes it with
`nib setup --user --provider zemble=`.

nib runs `zemble nib-provider` with its project directory as the working directory. That is the same
stdio server as `zemble` / `zemble mcp`, with the same tools, the same daemon routing and the same
`--content` and `--no-daemon` options; a tool's default `repo` is that directory, as it is today.
`zemble` and `zemble mcp` are unchanged.

## What changes in nib mode

Every tool result keeps its normal text `content`, byte for byte, and adds `structuredContent`
`{"nib": 1, "parts": [...]}`. Each part is either `{"text": "..."}` (prose, never file lines) or
`{"path": "/absolute/file", "start": N, "end": M, "indexed": "..."}`: lines N..M of a file, with
`indexed` (when zemble has it) the text of those lines as zemble indexed them, so nib can tell
when the file changed since. Locations of one file with no text between them print as one block.

| Tools | Layout |
| --- | --- |
| `search`, `find_related` | A header, then per hit a text line (rank, path:range, score) and the snippet's location: the first `max_snippet_lines` lines, the whole chunk for `None`, none for `0`. |
| `graph_*` | The queried symbol, then per hit a text line (name, kind, edge, resolution, depth, source, reason) and the one line it points at; a footer with `total`, `truncated`, `note`. |
| `outline` | The package and file, then each declaration's first line, as one block. |
| `signatures` | The signature's label and line, then per exact call site its text and line. |
| `explain` | The bundle header, then per item its heading and the lines it shows; outline-style items and bare locations stay prose. |
| `dupes`, `home`, `health`, `status` | One text part holding the plain text. |

Only lines zemble knows are located: a graph hit or declaration gets its one line, never a guessed
span. A chunk is a byte range, so its first line may start after the indentation; when it still sits
there in the file, `indexed` is widened to the whole lines, otherwise zemble's own text is sent and
nib reports the index as older than the file. A git URL repo answers without `structuredContent`
(its files live in a clone nib cannot show), and nib then shows the text as prose.

The protocol itself is nib's (`docs/design-providers.md` in the nib repository); the part
vocabulary lives in `zemble.nib.parts`.
