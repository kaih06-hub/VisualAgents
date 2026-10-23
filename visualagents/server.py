"""Local web server: serves the dashboard and streams live state over Server-Sent Events.

Bound to this computer (the default), it answers only requests addressed to 127.0.0.1 / localhost / [::1], so a web
page that re-points its own domain at 127.0.0.1 (DNS rebinding) cannot read the live output or press Start. Start and
reset also need the per-run session token, which is written into the page only for a browser on this computer.
"""

from __future__ import annotations

import hmac
import json
import secrets
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

from .hub import Hub
from .projects import Projects

_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOOPBACK_CLIENTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
SESSION_HEADER = "X-VisualAgents-Session"


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def exposure_warning(host: str) -> str | None:
    """What to tell the user when the dashboard is bound to the network rather than this computer."""
    if is_loopback(host):
        return None
    return (f"The dashboard listens on {host or 'every interface'}: anyone who can reach this computer on the network can "
            "watch your agents' live output and prompts. Start and reset still work only from this computer.")


def _host_name(header: str) -> str:
    """'127.0.0.1:7420' -> '127.0.0.1', '[::1]:7420' -> '::1', 'localhost' -> 'localhost'."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1:header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def _asset(name: str) -> bytes | None:
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        return (resources.files("visualagents") / "web" / name).read_bytes()
    except (FileNotFoundError, OSError):
        return None


class _Handler(BaseHTTPRequestHandler):
    projects: Projects
    session: str
    local_only: bool  # bound to this computer: only requests addressed to a loopback name are answered
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

    def _hub(self, url) -> Hub:
        return self.projects.resolve(parse_qs(url.query).get("project", [None])[0])

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or urlparse(origin).netloc == self.headers.get("Host")

    def _host_ok(self) -> bool:
        host = self.headers.get("Host")
        if not self.local_only or host is None:  # bound to the network on purpose (--host), or a bare HTTP/1.0 client
            return True
        if is_loopback(_host_name(host)):
            return True
        self._send(HTTPStatus.MISDIRECTED_REQUEST, b"this dashboard only answers 127.0.0.1 / localhost", "text/plain")
        return False

    def _may_control(self) -> bool:
        """Start / reset: same origin, and the session token the page was served with."""
        token = self.headers.get(SESSION_HEADER) or ""
        if self._same_origin() and hmac.compare_digest(token.encode(), self.session.encode()):
            return True
        self._json({"ok": False, "error": "forbidden",
                    "message": "Start and reset work only from the dashboard on this computer."}, HTTPStatus.FORBIDDEN)
        return False

    def _index(self) -> bytes:
        body = _asset("index.html") or b"missing index.html"
        local = self.client_address[0] in LOOPBACK_CLIENTS
        return body.replace(b"{{VA_SESSION}}", self.session.encode() if local else b"")

    # -- routes ------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return
        url = urlparse(self.path)
        path = url.path
        if path in ("/", "/index.html"):
            self._send(200, self._index(), _TYPES[".html"])
        elif path.startswith("/assets/"):
            name = path[len("/assets/"):]
            body = _asset(name)
            if body is None:
                self._send(404, b"not found", "text/plain")
            else:
                ext = name[name.rfind("."):]
                self._send(200, body, _TYPES.get(ext, "application/octet-stream"))
        elif path == "/api/state":
            hub = self._hub(url)
            snap = hub.snapshot()
            snap["meta"]["daemon"] = self.projects.daemon
            self._json(snap)
        elif path == "/api/projects":
            self._json({"daemon": self.projects.daemon, "projects": self.projects.listing()})
        elif path == "/api/blueprint":
            self._json(self._hub(url).blueprint)
        elif path == "/api/daemon":  # proof that daemon.json still names this process (see home.read_daemon)
            if self.headers.get("X-VisualAgents-Token") != self.projects.token:
                self._json({"error": "bad token"}, HTTPStatus.FORBIDDEN)
            else:
                self._json({"pid": self.projects.pid})
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
        if not self._host_ok():
            return
        if path == "/api/ingest":
            if self.headers.get("X-VisualAgents-Token") != self.projects.token:
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
                hub = self.projects.for_item(item)
                if item.get("type") == "hello":  # a project was just hooked: the hub now exists
                    continue
                if item.get("type") == "line":
                    hub.feed_line(str(item.get("text", "")), str(item.get("stream", "stdout")))
                else:
                    hub.feed_structured(item)
            self._json({"ok": True, "count": len(items)})
        elif path == "/api/start":
            if not self._may_control():
                return
            hub = self._hub(urlparse(self.path))
            if hub.launcher is None:
                self._json({"ok": False, "message": "Start it the way you usually do - it shows up here."})
                return
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            task = str(body.get("task") or "").strip() or None
            ok, message = hub.launcher.start(task)
            self._json({"ok": ok, "message": message, "launch": hub.launcher.info()})
        elif path == "/api/reset":
            if not self._may_control():
                return
            self._hub(urlparse(self.path)).reset_view()
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
        hub = self._hub(url)
        try:
            self.wfile.write(b"retry: 1500\n\n")
            while True:
                events, state, version = hub.wait(last_seq, version, timeout=15.0)
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
    def __init__(self, target: Hub | Projects, host: str = "127.0.0.1", port: int = 7420):
        self.projects = target if isinstance(target, Projects) else Projects.single(target)
        self.session = secrets.token_urlsafe(16)
        handler = type("Handler", (_Handler,), {"projects": self.projects, "session": self.session,
                                                "local_only": is_loopback(host)})
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
            if getattr(exc, "errno", None) not in (48, 98, 10048):  # EADDRINUSE on macOS, Linux, Windows
                raise  # a bad --host (EADDRNOTAVAIL, an unknown name) fails the same on every port
    raise last or OSError("could not bind a port")
