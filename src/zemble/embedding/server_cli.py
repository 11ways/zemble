"""Command-line surface for `zemble embed-server`: run the shared embedding server, and talk to it.

`run`, `add-key` and `gc` are for the machine that hosts the server; `check` and `push` are for
every machine that uses it.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import threading
from pathlib import Path

from zemble.embedding.cache import cache_root, family_slug, read_rows
from zemble.embedding.gc import day_text
from zemble.embedding.http import EmbeddingRequestError
from zemble.embedding.served import ServerClient
from zemble.embedding.wire import ROWS_PER_REQUEST, SERVER_ENV, SERVER_KEY_ENV, server_settings

EMBED_SERVER_COMMANDS = ("embed-server",)

EXIT_ERROR = 1

#: Days the server keeps a vector nobody asked for. Far longer than a client's grace period: the
#: indexes that reference a vector live on other machines, and an index that is only ever loaded
#: never asks for its vectors again, so a use stamp is the server's only evidence and a weak one.
DEFAULT_SERVER_GRACE_DAYS = 180

logger = logging.getLogger(__name__)


def add_embed_server_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `embed-server` subcommand and its actions."""
    from zemble.cli import declare_no_root
    from zemble.embedding.server import DEFAULT_HOST, DEFAULT_PORT

    parser = declare_no_root(
        sub.add_parser("embed-server", help="Run or use the shared embedding server that caches paid vectors.")
    )
    actions = parser.add_subparsers(dest="embed_server_action", required=True)

    run = declare_no_root(
        actions.add_parser("run", help="Serve vectors over HTTP, buying each distinct text from the provider once.")
    )
    run.add_argument("--host", default=DEFAULT_HOST, help=f"Address to bind (default: {DEFAULT_HOST}).")
    run.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT}).")
    run.add_argument("--data-dir", type=Path, default=None, help="Where the cache files live.")
    run.add_argument("--keys-file", type=Path, default=None, help="The API key file (see `add-key`).")
    run.add_argument(
        "--embedder",
        action="append",
        default=None,
        metavar="SPEC",
        help="An embedder whose family is served; repeatable (default: ZEMBLE_EMBEDDER).",
    )
    run.add_argument(
        "--reranker",
        action="append",
        default=None,
        metavar="SPEC",
        help="A hosted reranker that is served; repeatable (default: ZEMBLE_RERANKER when hosted).",
    )
    run.add_argument("--certfile", type=Path, default=None, help="TLS certificate chain; serves HTTPS when given.")
    run.add_argument("--keyfile", type=Path, default=None, help="TLS private key for --certfile.")

    key = declare_no_root(actions.add_parser("add-key", help="Create an API key for one client machine and print it."))
    key.add_argument("--keys-file", type=Path, default=None, help="The API key file to append to.")
    key.add_argument("--label", default=socket.gethostname(), help="A name for the key in the server log.")

    declare_no_root(actions.add_parser("check", help=f"Ask the server named by {SERVER_ENV} what it holds."))

    push = declare_no_root(
        actions.add_parser("push", help="Upload vectors this machine already paid for to the server.")
    )
    push.add_argument(
        "files", nargs="*", type=Path, help="Cache files to upload (default: this machine's, for served families)."
    )
    push.add_argument("--family", default=None, help="The embedder family the files hold, e.g. voyage:voyage-4-lite.")
    push.add_argument(
        "--remove-local", action="store_true", help="Delete each uploaded file once the server holds all of it."
    )

    gc = declare_no_root(
        actions.add_parser("gc", help="Sweep vectors nobody asked for within the grace period, then VACUUM.")
    )
    gc.add_argument("--data-dir", type=Path, default=None, help="Where the cache files live.")
    gc.add_argument(
        "--grace-days",
        type=int,
        default=DEFAULT_SERVER_GRACE_DAYS,
        help=f"Keep vectors stored or served within this many days (default: {DEFAULT_SERVER_GRACE_DAYS}).",
    )
    gc.add_argument("--dry-run", action="store_true", help="Report what would go, writing nothing.")


def run_embed_server(args: argparse.Namespace) -> int:
    """Run one `zemble embed-server` action and return its exit code."""
    action = args.embed_server_action
    try:
        if action == "run":
            return _run(args)
        if action == "add-key":
            return _add_key(args)
        if action == "check":
            return _check()
        if action == "push":
            return _push(args)
        if action == "gc":
            return _gc(args)
    except (EmbeddingRequestError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    raise AssertionError(f"unhandled embed-server action {action!r}")  # argparse restricts the choices


def _run(args: argparse.Namespace) -> int:
    """Serve until interrupted or terminated."""
    from zemble.embedding.registry import cached_family, resolve_embedder_spec
    from zemble.embedding.server import KeyRing, default_data_dir, default_keys_file, make_server
    from zemble.embedding.service import EmbeddingService
    from zemble.rerank.registry import HOSTED_SCHEMES, resolve_reranker_spec, split_reranker_spec

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    embedders = args.embedder or [resolve_embedder_spec()]
    embedders = [spec for spec in embedders if cached_family(spec) is not None]
    if not embedders:
        print("Nothing to serve: name a paid embedder with --embedder or ZEMBLE_EMBEDDER.", file=sys.stderr)
        return EXIT_ERROR
    rerankers = args.reranker
    if rerankers is None:
        configured = split_reranker_spec(resolve_reranker_spec())
        rerankers = [f"{configured[0]}:{configured[1]}"] if configured and configured[0] in HOSTED_SCHEMES else []
    directory = args.data_dir or default_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    service = EmbeddingService(directory, embedders, rerankers)
    server = make_server(
        service, KeyRing(args.keys_file or default_keys_file()), args.host, args.port, args.certfile, args.keyfile
    )
    scheme = "https" if args.certfile else "http"
    logger.info(
        "embedding server on %s://%s:%d serving %s, rerankers %s, cache in %s",
        scheme,
        args.host,
        server.server_address[1],
        ", ".join(service.families),
        ", ".join(service.rerankers) or "none",
        directory,
    )
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _add_key(args: argparse.Namespace) -> int:
    """Append a new key and print what a client puts in its env file."""
    from zemble.embedding.server import add_key, default_keys_file

    path = args.keys_file or default_keys_file()
    key = add_key(path, args.label)
    print(f"Added a key labelled {args.label!r} to {path}. On the client, add to ~/.config/zemble/env:")
    print(f"{SERVER_ENV}=http://<this host>:<port>")
    print(f"{SERVER_KEY_ENV}={key}")
    return 0


def _client() -> ServerClient:
    """Return a client for the configured server, refusing when none is configured."""
    settings = server_settings()
    if settings is None:
        raise ValueError(f"{SERVER_ENV} is not set; name the embedding server in ~/.config/zemble/env")
    return ServerClient(settings)


def _check() -> int:
    """Print the server's health and status."""
    client = _client()
    health = client.health()
    status = client.status()
    print(f"{client.settings.url}: up, zemble {health.get('version')}, for {status['uptime_seconds']} s")
    for family in status["families"]:
        print(f"  {family['family']}: {family['vectors']} vectors, {family['bytes'] / 1e6:.1f} MB")
    print(f"  rerankers: {', '.join(status['rerankers']) or 'none'}")
    print(
        f"  since start: {status['documents_requested']} document texts asked for, "
        f"{status['documents_missed']} missed, {status['provider_tokens']} provider tokens, "
        f"{status['queries']} queries, {status['rerank_passages']} reranked passages, "
        f"{status['vectors_pushed']} vectors pushed"
    )
    return 0


def _push(args: argparse.Namespace) -> int:
    """Upload local cache files the server serves the family of, optionally deleting them afterwards."""
    from zemble.openfiles import held_open

    client = _client()
    served = [family["family"] for family in client.status()["families"]]
    by_slug = {family_slug(family): family for family in served}
    files: list[Path] = args.files or sorted(cache_root().glob("*.sqlite"))
    failed = False
    for path in files:
        family = args.family or by_slug.get(path.stem)
        if family is None:
            print(f"Skipped {path}: the server serves no family named like it ({', '.join(served)}); pass --family")
            continue
        if family not in served:
            print(f"Skipped {path}: the server does not serve {family}")
            continue
        sent = added = 0
        for batch in read_rows(path, ROWS_PER_REQUEST):
            added += client.store(family, batch)
            sent += len(batch)
        print(f"{path}: {sent} vectors sent, {added} new to the server")
        if not args.remove_local:
            continue
        # Every row was accepted, new or already there, so the server now holds all of this file.
        present = [candidate for candidate in _sqlite_files(path) if candidate.exists()]
        if any(held_open(candidate) for candidate in present):
            print(f"  kept {path}: a process holds it open (`zemble daemon restart` first)")
            failed = True
            continue
        for candidate in present:
            candidate.unlink()
        print(f"  removed {path}")
    return EXIT_ERROR if failed else 0


def _sqlite_files(path: Path) -> list[Path]:
    """Return a sqlite file and its WAL and shared-memory companions."""
    return [path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")]


def _gc(args: argparse.Namespace) -> int:
    """Sweep the server's files of vectors nobody asked for lately."""
    from zemble.embedding.gc import collect_unused
    from zemble.embedding.server import default_data_dir
    from zemble.index.scope import megabytes

    directory = args.data_dir or default_data_dir()
    reports = collect_unused(directory, grace_days=args.grace_days, dry_run=args.dry_run)
    if not reports:
        print(f"No embedding cache found in `{directory}`")
        return 0
    for report in reports:
        if report.refused is not None:
            print(f"Skipped {report.path.name}: {report.refused}")
            continue
        verb = "would sweep" if args.dry_run else "swept"
        print(
            f"{report.path.name}: {report.rows} vectors, {verb} {report.swept} last used on "
            f"{day_text(report.cutoff_day)} or earlier ({megabytes(report.swept_bytes)})"
        )
        if not args.dry_run:
            after = report.size_after if report.size_after is not None else report.size_before
            print(f"  {megabytes(report.size_before)} -> {megabytes(after)}")
    return 0


__all__ = ["DEFAULT_SERVER_GRACE_DAYS", "EMBED_SERVER_COMMANDS", "add_embed_server_parser", "run_embed_server"]
