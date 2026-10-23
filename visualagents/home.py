"""User-level VisualAgents state: ~/.visualagents (override with VISUALAGENTS_HOME).

    daemon.json     where the background dashboard listens (url, token, pid)
    registry.json   hooked projects: id -> {path, envs}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def home() -> Path:
    return Path(os.environ.get("VISUALAGENTS_HOME") or Path.home() / ".visualagents")


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # private from the moment it exists (keys.json holds API keys), not only after a chmod
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        os.fchmod(fd, 0o600)  # a tmp left behind by an older version keeps its mode under O_CREAT
        fh.write(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


# -- daemon ----------------------------------------------------------------------

def daemon_file() -> Path:
    return home() / "daemon.json"


def read_daemon() -> dict[str, Any]:
    """The running daemon's {url, token, pid}, or {} when none is running.

    daemon.json outlives a crash or a reboot, and its pid may since belong to another program: the daemon counts as
    running only when one of this user's processes has that pid and the dashboard at its url, asked with its token,
    answers with that same pid."""
    info = _read(daemon_file())
    try:
        pid = int(info.get("pid") or 0)
    except (TypeError, ValueError):
        return {}
    if pid <= 0 or not _alive(pid, ours=True) or _daemon_pid(info) != pid:
        return {}
    return info


def _daemon_pid(info: dict[str, Any]) -> int | None:
    """The pid the dashboard at info's url reports, given info's token; None when nothing (of ours) answers."""
    import urllib.request

    url, token = info.get("url"), info.get("token")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")) or not isinstance(token, str):
        return None
    req = urllib.request.Request(url + "/api/daemon", headers={"X-VisualAgents-Token": token})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=2) as res:
            pid = json.loads(res.read(4096)).get("pid")
    except Exception:
        return None
    return pid if isinstance(pid, int) else None


def write_daemon(url: str, token: str, pid: int) -> None:
    _write(daemon_file(), {"url": url, "token": token, "pid": pid})


def clear_daemon(pid: int | None = None) -> None:
    info = _read(daemon_file())
    if pid is None or info.get("pid") == pid:
        try:
            daemon_file().unlink()
        except OSError:
            pass


def _alive(pid: int, *, ours: bool = False) -> bool:
    """Whether a process has this pid; with ours=True, one of this user's (another user's process is never ours)."""
    try:
        os.kill(pid, 0)
    except PermissionError:
        return not ours
    except OSError:
        return False
    return True


# -- registry of hooked projects ----------------------------------------------

def registry() -> dict[str, dict[str, Any]]:
    return _read(home() / "registry.json").get("projects", {})


def save_registry(projects: dict[str, dict[str, Any]]) -> None:
    _write(home() / "registry.json", {"projects": projects})


def project_id(path: Path, projects: dict[str, dict[str, Any]] | None = None) -> str:
    """Short, URL-friendly id for a project folder (stable across calls)."""
    path = Path(path).resolve()
    projects = registry() if projects is None else projects
    for pid, info in projects.items():
        if info.get("path") == str(path):
            return pid
    base = re.sub(r"[^a-z0-9]+", "-", path.name.lower()).strip("-") or "project"
    pid, n = base, 2
    while pid in projects and projects[pid].get("path") != str(path):
        pid, n = f"{base}-{n}", n + 1
    return pid


# -- saved settings and keys (set by the setup wizard) ------------------------------

def settings() -> dict[str, Any]:
    return _read(home() / "settings.json")


def save_settings(data: dict[str, Any]) -> None:
    current = settings()
    current.update(data)
    _write(home() / "settings.json", current)


def remember_project(path: Path) -> None:
    recent = [p for p in settings().get("recent", []) if p != str(path)]
    save_settings({"recent": [str(path), *recent][:8]})


def saved_keys() -> dict[str, str]:
    return {k: v for k, v in _read(home() / "keys.json").items() if isinstance(v, str) and v}


def save_key(var: str, value: str) -> Path:
    keys = saved_keys()
    keys[var] = value
    path = home() / "keys.json"
    _write(path, keys)
    return path


def apply_saved() -> None:
    """Fill in saved keys / provider / models for anything the environment and project .env left unset."""
    for var, value in saved_keys().items():
        if not os.environ.get(var):
            os.environ[var] = value
    s = settings()
    provider = s.get("provider")
    if provider and not os.environ.get("VISUALAGENTS_PROVIDER"):
        os.environ["VISUALAGENTS_PROVIDER"] = provider
    models = s.get("models", {})
    if provider and provider in models and not os.environ.get("VISUALAGENTS_MODEL"):
        os.environ["VISUALAGENTS_MODEL"] = models[provider]
