"""Local web server: serves the dashboard and streams live state over Server-Sent Events."""

from __future__ import annotations

import json
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

from .hub import Hub

_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


def _asset(name: str) -> bytes | None:
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        return (resources.files("visualagents") / "web" / name).read_bytes()
    except (FileNotFoundError, OSError):
        return None


class _Handler(BaseHTTPRequestHandler):
    hub: Hub
    server_version = "VisualAgents"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal clean for the agent's output
        pass

    # -- helpers -----------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str = "application/json", extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or urlparse(origin).netloc == self.headers.get("Host")

    # -- routes ------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = url.path
        if path in ("/", "/index.html"):
            body = _asset("index.html") or b"missing index.html"
            self._send(200, body, _TYPES[".html"])
        elif path.startswith("/assets/"):
            name = path[len("/assets/"):]
            body = _asset(name)
            if body is None:
                self._send(404, b"not found", "text/plain")
            else:
                ext = name[name.rfind("."):]
                self._send(200, body, _TYPES.get(ext, "application/octet-stream"))
        elif path == "/api/state":
            self._json(self.hub.snapshot())
        elif path == "/api/blueprint":
            self._json(self.hub.blueprint)
        elif path == "/api/stream":
            self._stream(url)
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(min(length, 8_000_000)) if length else b""
        if path == "/api/ingest":
            if self.headers.get("X-VisualAgents-Token") != self.hub.token:
                self._json({"error": "bad token"}, HTTPStatus.FORBIDDEN)
                return
            try:
                payload = json.loads(raw or b"null")
            except ValueError:
                self._json({"error": "invalid json"}, HTTPStatus.BAD_REQUEST)
                return
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "line":
                    self.hub.feed_line(str(item.get("text", "")), str(item.get("stream", "stdout")))
                else:
                    self.hub.feed_structured(item)
            self._json({"ok": True, "count": len(items)})
        elif path == "/api/reset":
            if not self._same_origin():
                self._json({"error": "cross-origin"}, HTTPStatus.FORBIDDEN)
                return
            self.hub.reset_view()
            self._json({"ok": True})
        else:
            self._send(404, b"not found", "text/plain")

    def _stream(self, url) -> None:
        q = parse_qs(url.query)
        try:
            last_seq = int(self.headers.get("Last-Event-ID") or q.get("since", ["0"])[0])
        except ValueError:
            last_seq = 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        version = -1
        try:
            self.wfile.write(b"retry: 1500\n\n")
            while True:
                events, state, version = self.hub.wait(last_seq, version, timeout=15.0)
                chunks = []
                for ev in events:
                    last_seq = ev["seq"]
                    chunks.append(f"id: {ev['seq']}\nevent: ev\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n")
                if state is not None:
                    chunks.append(f"event: state\ndata: {json.dumps(state, ensure_ascii=False)}\n\n")
                if not chunks:
                    chunks.append(": ping\n\n")
                self.wfile.write("".join(chunks).encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return


class Server:
    def __init__(self, hub: Hub, host: str = "127.0.0.1", port: int = 7420):
        handler = type("Handler", (_Handler,), {"hub": hub})
        self.httpd = _bind(host, port, handler)
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address[:2]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="visualagents-http", daemon=True)

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> "Server":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _bind(host: str, port: int, handler) -> ThreadingHTTPServer:
    last: OSError | None = None
    for p in [port + i for i in range(20)] if port else [0]:
        try:
            return ThreadingHTTPServer((host, p), handler)
        except OSError as exc:  # port in use - try the next one
            last = exc
            if getattr(exc, "errno", None) not in (48, 98, 10048) and not isinstance(exc, socket.error):
                raise
    raise last or OSError("could not bind a port")
