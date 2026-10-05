"""The embedding server's HTTP face: bearer-key auth, JSON in and out, one thread per request.

Plain HTTP by default; pass a certificate and key to serve HTTPS directly, or put it behind a
reverse proxy that terminates TLS. Without TLS the API key crosses the network in the clear.
"""

from __future__ import annotations

import base64
import hmac
import logging
import os
import secrets
import ssl
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import orjson

from zemble.embedding.http import EmbeddingRequestError
from zemble.embedding.service import EmbeddingService, SpecNotServed
from zemble.embedding.wire import (
    AUTH_HEADER,
    COVERED_ROUTE,
    DESCRIBE_ROUTE,
    DOCUMENTS_ROUTE,
    HEALTH_ROUTE,
    MAX_BODY_BYTES,
    PROVIDER_FAILED_STATUS,
    QUERIES_ROUTE,
    RERANK_ROUTE,
    STATUS_ROUTE,
    STORE_ROUTE,
    encode_vectors,
)
from zemble.version import __version__

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def default_data_dir() -> Path:
    """Return where the server keeps its sqlite files: outside the client cache, so no client sweep removes them."""
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return root / "zemble" / "embed-server"


def default_keys_file() -> Path:
    """Return the file the server reads its API keys from."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "zemble" / "server-keys"


class KeyRing:
    """The API keys a server accepts, re-read whenever the file changes, so adding one needs no restart.

    One key per line, optionally preceded by a label (``<label> <key>``); ``#`` starts a comment.
    """

    def __init__(self, path: Path) -> None:
        """Remember the file; it is read on first use."""
        self.path = path
        self._stamp: int | None = None
        self._keys: dict[str, str] = {}
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        """Re-read the file when its modification time moved."""
        try:
            stamp = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            self._stamp, self._keys = None, {}
            return
        if stamp == self._stamp:
            return
        keys: dict[str, str] = {}
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            keys[parts[-1]] = " ".join(parts[:-1]) or "unlabelled"
        self._stamp, self._keys = stamp, keys

    def count(self) -> int:
        """Return how many keys the file holds now."""
        with self._lock:
            self._refresh()
            return len(self._keys)

    def label_for(self, presented: str) -> str | None:
        """Return the label of the key presented, or None when it is not one of them.

        Every key is compared in constant time, so a timing difference cannot tell a caller how
        much of a guess was right.
        """
        with self._lock:
            self._refresh()
            keys = dict(self._keys)
        found = None
        for key, label in keys.items():
            if hmac.compare_digest(key.encode("utf-8"), presented.encode("utf-8")):
                found = label
        return found


def add_key(path: Path, label: str) -> str:
    """Generate a new API key, append it to the key file (mode 600) and return it."""
    key = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).date().isoformat()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(f"# added {stamp}\n{label.replace(' ', '-') or 'unlabelled'} {key}\n")
    os.chmod(path, 0o600)
    return key


class _HttpError(Exception):
    """A request the handler answers with a status and a message instead of a result."""

    def __init__(self, status: int, message: str) -> None:
        """Carry the status beside the message."""
        super().__init__(message)
        self.status = status


def _texts(payload: dict[str, Any], field: str = "texts") -> list[str]:
    """Read a list of strings out of a request, refusing anything else."""
    value = payload.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"'{field}' must be a list of strings")
    return value


def _spec(payload: dict[str, Any]) -> str:
    """Read the spec a request names."""
    spec = payload.get("spec")
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("'spec' must name an embedder or reranker")
    return spec


def _rows(payload: dict[str, Any]) -> list[tuple[str, int, bytes]]:
    """Read store rows: ``[text hash, dims, base64 float32 bytes]`` triples."""
    raw = payload.get("rows")
    if not isinstance(raw, list):
        raise ValueError("'rows' must be a list")
    rows = []
    for entry in raw:
        if not (
            isinstance(entry, list) and len(entry) == 3 and isinstance(entry[0], str) and isinstance(entry[1], int)
        ):
            raise ValueError("each row must be [text hash, dims, base64 vector]")
        rows.append((entry[0], entry[1], base64.b64decode(str(entry[2]).encode("ascii"), validate=True)))
    return rows


def _route(service: EmbeddingService, route: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Answer one authenticated POST."""
    if route == DOCUMENTS_ROUTE:
        texts = _texts(payload)
        vectors = service.documents(_spec(payload), texts)
        return {"rows": len(texts), "dimensions": int(vectors.shape[1]), "vectors": encode_vectors(vectors)}
    if route == QUERIES_ROUTE:
        texts = _texts(payload)
        vectors = service.queries(_spec(payload), texts)
        return {"rows": len(texts), "dimensions": int(vectors.shape[1]), "vectors": encode_vectors(vectors)}
    if route == COVERED_ROUTE:
        covered, width = service.covered(_spec(payload), _texts(payload, "digests"), bool(payload.get("probe")))
        return {"covered": sorted(covered), "dimensions": width}
    if route == DESCRIBE_ROUTE:
        return service.describe(_spec(payload))
    if route == STORE_ROUTE:
        return {"added": service.store(_spec(payload), _rows(payload))}
    if route == RERANK_ROUTE:
        query = payload.get("query")
        if not isinstance(query, str):
            raise ValueError("'query' must be a string")
        return {"scores": service.rerank(_spec(payload), query, _texts(payload, "passages"))}
    raise _HttpError(HTTPStatus.NOT_FOUND, f"unknown route {route}")


class EmbeddingHttpServer(ThreadingHTTPServer):
    """A threading HTTP server that carries the service and the key ring its handlers use."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: EmbeddingService, keys: KeyRing) -> None:
        """Bind the address."""
        super().__init__(address, _Handler)
        self.service = service
        self.keys = keys


class _Handler(BaseHTTPRequestHandler):
    """One request: authenticate, decode, route, answer in JSON."""

    protocol_version = "HTTP/1.1"
    server: EmbeddingHttpServer
    label = "-"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
        """Route the access log through logging, naming the key's label rather than only an address."""
        logger.info("%s [%s] %s", self.address_string(), self.label, format % args)

    def do_GET(self) -> None:  # noqa: N802 - the base class's name
        """Answer health without a key, status with one."""
        try:
            if self.path == HEALTH_ROUTE:
                self._answer(HTTPStatus.OK, {"ok": True, "version": __version__})
                return
            self._authenticate()
            if self.path == STATUS_ROUTE:
                self._answer(HTTPStatus.OK, {"version": __version__, **self.server.service.status()})
                return
            raise _HttpError(HTTPStatus.NOT_FOUND, f"unknown route {self.path}")
        except _HttpError as exc:
            self._answer(exc.status, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - the base class's name
        """Answer one embedding, coverage, store or rerank request."""
        try:
            self._authenticate()
            payload = self._payload()
            self._answer(HTTPStatus.OK, _route(self.server.service, self.path, payload))
        except _HttpError as exc:
            self._answer(exc.status, {"error": str(exc)})
        except SpecNotServed as exc:
            self._answer(HTTPStatus.FORBIDDEN, {"error": str(exc)})
        except EmbeddingRequestError as exc:
            # The provider refused or gave up after its own retries; say so as the provider said it.
            logger.warning("provider failed for %s: %s", self.path, exc)
            self._answer(PROVIDER_FAILED_STATUS, {"error": f"provider failed: {exc}"})
        except (ValueError, TypeError) as exc:
            self._answer(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            logger.exception("request %s failed", self.path)
            self._answer(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

    def _authenticate(self) -> None:
        """Accept a request only with a bearer key the key file lists."""
        header = self.headers.get(AUTH_HEADER, "")
        scheme, _, presented = header.partition(" ")
        label = self.server.keys.label_for(presented.strip()) if scheme.lower() == "bearer" else None
        if label is None:
            raise _HttpError(HTTPStatus.UNAUTHORIZED, "missing or unknown API key")
        self.label = label

    def _payload(self) -> dict[str, Any]:
        """Read and decode the JSON body, refusing one past the size limit."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise _HttpError(HTTPStatus.BAD_REQUEST, "bad Content-Length") from None
        if length > MAX_BODY_BYTES:
            raise _HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body over {MAX_BODY_BYTES} bytes")
        try:
            payload = orjson.loads(self.rfile.read(length))
        except orjson.JSONDecodeError:
            raise _HttpError(HTTPStatus.BAD_REQUEST, "body is not JSON") from None
        if not isinstance(payload, dict):
            raise _HttpError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
        return payload

    def _answer(self, status: int, body: dict[str, Any]) -> None:
        """Write one JSON response; an error also ends the connection.

        A request refused before its body was read leaves that body in the stream, where the
        next request on a kept-alive connection would be parsed out of it.
        """
        data = orjson.dumps(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if status != HTTPStatus.OK:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(data)


def make_server(
    service: EmbeddingService,
    keys: KeyRing,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    certfile: Path | None = None,
    keyfile: Path | None = None,
) -> EmbeddingHttpServer:
    """Bind the server, refusing to start without a single key to check requests against.

    :param service: What answers the requests.
    :param keys: The keys requests are checked against.
    :param host: The address to bind.
    :param port: The port to bind; 0 picks a free one.
    :param certfile: A TLS certificate chain, to serve HTTPS.
    :param keyfile: The private key for `certfile`.
    :return: The bound, not yet serving, server.
    :raises ValueError: If the key file lists no key.
    """
    if keys.count() == 0:
        raise ValueError(f"{keys.path} lists no API key; add one with `zemble embed-server add-key`")
    server = EmbeddingHttpServer((host, port), service, keys)
    if certfile is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile, keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "EmbeddingHttpServer",
    "KeyRing",
    "add_key",
    "default_data_dir",
    "default_keys_file",
    "make_server",
]
