"""Hubs keyed by project. `run`/`pipe`/`watch` serve one; the `up` daemon serves every hooked project."""

from __future__ import annotations

import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from . import blueprint as bpmod
from . import home
from .hub import Hub


class Projects:
    def __init__(self, *, token: str | None = None, daemon: bool = False):
        self.token = token or secrets.token_urlsafe(16)
        self.daemon = daemon
        self.pid = os.getpid()  # what /api/daemon reports, so a stale daemon.json is never trusted
        self.hubs: dict[str, Hub] = {}
        self.paths: dict[str, str] = {}
        self.order: list[str] = []
        self.lock = threading.RLock()
        self._placeholder: Hub | None = None
        if daemon:
            threading.Thread(target=self._reap, name="visualagents-reaper", daemon=True).start()

    @classmethod
    def single(cls, hub: Hub, project_id: str = "") -> "Projects":
        p = cls(token=hub.token)
        pid = project_id or bpmod.slugify(hub.blueprint.get("project") or "project").replace("_", "-")
        p.add(pid, hub, hub.meta.get("project", ""))
        return p

    def add(self, pid: str, hub: Hub, path: str = "") -> Hub:
        with self.lock:
            hub.token = self.token
            hub.meta.setdefault("project_id", pid)
            self.hubs[pid] = hub
            self.paths[pid] = path
            if pid not in self.order:
                self.order.append(pid)
        return hub

    # -- lookup ------------------------------------------------------------------
    def resolve(self, key: str | None) -> Hub:
        with self.lock:
            if key and key in self.hubs:
                return self.hubs[key]
            if key:
                for pid, path in self.paths.items():
                    if path and os.path.realpath(key) == os.path.realpath(path):
                        return self.hubs[pid]
            if self.hubs:
                return self.hubs[self._most_active()]
            return self.placeholder()

    def for_item(self, item: dict[str, Any]) -> Hub:
        path = item.get("project")
        if not self.daemon or not path:
            return self.resolve(None)
        with self.lock:
            for pid, p in self.paths.items():
                if p and os.path.realpath(p) == os.path.realpath(str(path)):
                    return self.hubs[pid]
        return self.open(Path(str(path)))

    def open(self, path: Path) -> Hub:
        """Hub for a project folder, created on first use (blueprint from disk, or a quick static scan)."""
        path = Path(path).resolve()
        with self.lock:
            hub = self._opened(path)
            if hub is not None:
                return hub
        # outside the lock: a first scan can take a while, and every request (state, ingest, listing) needs the lock
        bp = _load_or_scan(path)
        record = bpmod.state_dir(path) / "runs"
        try:
            record.mkdir(parents=True, exist_ok=True)
        except OSError:
            record = None
        with self.lock:
            hub = self._opened(path)  # another request may have opened it meanwhile
            if hub is not None:
                return hub
            registered = next((k for k, v in home.registry().items() if v.get("path") == str(path)), None)
            pid = registered if registered and registered not in self.hubs else \
                home.project_id(path, {k: {"path": v} for k, v in self.paths.items()})
            hub = Hub(bp, record_dir=record, meta={"mode": "hook", "project": str(path), "project_id": pid})
            return self.add(pid, hub, str(path))

    def _opened(self, path: Path) -> Hub | None:
        for pid, p in self.paths.items():
            if p == str(path):
                return self.hubs[pid]
        return None

    def placeholder(self) -> Hub:
        if self._placeholder is None:
            self._placeholder = Hub(bpmod.empty("VisualAgents"), meta={"mode": "daemon-empty"})
            self._placeholder.token = self.token
        return self._placeholder

    def _most_active(self) -> str:
        def last(pid: str) -> float:
            evs = self.hubs[pid].events
            return evs[-1]["ts"] if evs else 0.0
        return max(self.order, key=last)

    def listing(self) -> list[dict[str, Any]]:
        with self.lock:
            out = []
            for pid in self.order:
                hub = self.hubs[pid]
                run = hub.tracker.run
                out.append({
                    "id": pid, "name": hub.blueprint.get("project") or pid, "path": self.paths.get(pid, ""),
                    "status": run["status"], "agents": len(hub.blueprint["agents"]),
                    "last": hub.events[-1]["ts"] if hub.events else None,
                })
            return out

    # -- process bookkeeping (daemon) ----------------------------------------------
    def _reap(self) -> None:
        """Processes killed without a goodbye still end their run."""
        while True:
            time.sleep(5)
            with self.lock:
                hubs = list(self.hubs.values())
            for hub in hubs:
                for pid in list(hub.procs):
                    if not home._alive(pid):
                        hub.feed_structured({"type": "process_end", "pid": pid, "reaped": True})


def _load_or_scan(path: Path) -> dict[str, Any]:
    bp_path = bpmod.blueprint_path(path)
    if bp_path.exists():
        try:
            return bpmod.load(bp_path)
        except (OSError, ValueError):
            pass
    from .static_scan import scan

    bp = scan(path)
    try:
        bpmod.save(bp, bp_path)
    except OSError:
        pass
    return bp
