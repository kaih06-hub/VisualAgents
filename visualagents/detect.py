"""Works out how to start a project, so `visualagents run <folder>` needs no command.

Looks at the README's shell snippets, start scripts, Makefile targets, Python entry points (FastAPI,
Gradio, Streamlit, Chainlit, `__main__` scripts), pyproject console scripts and package.json, then
points `python` / `uvicorn` / ... at the project's own virtualenv. The chosen command is saved to
`.visualagents/config.json`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import blueprint as bpmod

_SKIP = {".git", "node_modules", "__pycache__", ".visualagents", "dist", "build", ".tox", ".mypy_cache",
         ".pytest_cache", "site-packages", ".next", ".cache"}
_RUN_LINE = re.compile(
    r"^(?:\./[\w.\-/]+\.sh|bash\s+[\w.\-/]+\.sh|sh\s+[\w.\-/]+\.sh|python3?\s+(?!-m\s+(?:venv|pip)\b)\S+"
    r"|uv\s+run\s+\S+|poetry\s+run\s+\S+|streamlit\s+run|chainlit\s+run|uvicorn\s+\S+|fastapi\s+(?:dev|run)|"
    r"gradio\s+\S+|npm\s+run\s+(?:dev|start)|npm\s+start|pnpm\s+(?:run\s+)?(?:dev|start)|"
    r"yarn\s+(?:dev|start)|make\s+(?:run|start|dev|serve)|docker\s+compose\s+up|docker-compose\s+up)\b"
)
_NOT_RUN = re.compile(r"\b(pip|install|venv|virtualenv|setup\.py|pytest|test|lint|build|format|mypy|ruff|black|"
                      r"stop|kill|clean|uninstall|teardown|down|reset)\b")
_VENV_TOOLS = {"python", "python3", "uvicorn", "streamlit", "chainlit", "gradio", "fastapi", "flask", "jupyter",
               "hypercorn", "gunicorn", "celery", "panel"}
_ENTRY_NAMES = ("main.py", "app.py", "run.py", "server.py", "cli.py", "agent.py", "agents.py", "chat.py",
                "webui.py", "demo.py", "start.py", "launch.py")
_ENTRY_DIRS = (".", "src", "backend", "app", "server", "api")


@dataclass
class Candidate:
    command: list[str]
    cwd: str = "."
    source: str = ""
    note: str = ""
    score: int = 0
    env: dict[str, str] = field(default_factory=dict)

    def label(self) -> str:
        where = f"  (in {self.cwd}/)" if self.cwd not in (".", "") else ""
        return f"{shlex.join(self.command)}{where}"

    def to_json(self) -> dict[str, Any]:
        return {"command": self.command, "cwd": self.cwd}


# -- saved choice ------------------------------------------------------------------

def config_path(project: Path) -> Path:
    return bpmod.state_dir(project) / "config.json"


def load_saved(project: Path) -> Candidate | None:
    try:
        data = json.loads(config_path(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cmd = data.get("command")
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    if not cmd:
        return None
    return Candidate(list(cmd), data.get("cwd") or ".", "saved")


def save_choice(project: Path, cand: Candidate) -> Path:
    path = config_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.update(cand.to_json())
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    ignore = path.parent / ".gitignore"
    if not ignore.exists():
        ignore.write_text("runs/\n", encoding="utf-8")
    return path


# -- virtualenvs --------------------------------------------------------------------

def find_venvs(project: Path, max_depth: int = 3) -> list[Path]:
    """Virtualenvs inside the project (anything with a pyvenv.cfg), shallowest first."""
    project = Path(project).resolve()
    found: list[Path] = []
    for dirpath, dirnames, _ in os.walk(project):
        here = Path(dirpath)
        depth = len(here.relative_to(project).parts)
        if (here / "pyvenv.cfg").is_file():
            found.append(here)
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP)
    found.sort(key=lambda p: (len(p.relative_to(project).parts), p.name not in (".venv", "venv")))
    return found


def venv_bin(venv: Path, name: str) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / (name + (".exe" if os.name == "nt" else ""))


def best_venv(project: Path, cwd: str = ".") -> Path | None:
    """The virtualenv closest to where the command runs (backend/venv for a command run in backend/)."""
    project = Path(project).resolve()
    venvs = find_venvs(project)
    if not venvs:
        return None
    here = (project / cwd).resolve()

    def distance(v: Path) -> tuple[int, int]:
        parent = v.parent
        if here == parent or parent in here.parents:
            return (0, len(here.relative_to(parent).parts))
        if here in parent.parents:
            return (1, len(parent.relative_to(here).parts))
        return (2, len(v.relative_to(project).parts))

    return sorted(venvs, key=distance)[0]


def resolve(project: Path, cand: Candidate) -> tuple[list[str], list[str]]:
    """Point the command at the project's environment. Returns (argv, notes for the user)."""
    project = Path(project).resolve()
    argv = list(cand.command)
    notes: list[str] = []
    if not argv:
        return argv, notes
    head = Path(argv[0]).name
    venv = best_venv(project, cand.cwd)
    if head in _VENV_TOOLS and "/" not in argv[0]:
        if venv is not None and venv_bin(venv, head if head != "python3" else "python").exists():
            argv[0] = str(venv_bin(venv, head if head != "python3" else "python"))
            notes.append(f"using the project's virtualenv: {venv.relative_to(project)}")
        elif (project / "uv.lock").exists():
            argv = ["uv", "run", *argv]
            notes.append("using uv (uv.lock found)")
        elif (project / "poetry.lock").exists():
            argv = ["poetry", "run", *argv]
            notes.append("using poetry (poetry.lock found)")
        else:
            if head in ("python", "python3"):
                argv[0] = working_python(head)
            if (project / "requirements.txt").exists() or (project / "pyproject.toml").exists():
                notes.append("no virtualenv found - create one if imports fail: "
                             "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt")
    elif argv[0].endswith(".py") and (project / cand.cwd / argv[0]).exists():
        py = venv_bin(venv, "python") if venv else None
        argv = [str(py) if py and py.exists() else working_python("python3"), *argv]
    return argv, notes


def working_python(preferred: str = "python3") -> str:
    """First of python3 / python on PATH that actually starts (pyenv shims can point at missing versions)."""
    import shutil
    import subprocess
    import sys

    for name in dict.fromkeys([preferred, "python3", "python"]):
        exe = shutil.which(name)
        if not exe:
            continue
        try:
            if subprocess.run([exe, "-c", "import sys"], capture_output=True, timeout=10).returncode == 0:
                return name if exe == shutil.which(name) else exe
        except (OSError, subprocess.SubprocessError):
            continue
    return sys.executable


# -- detection ------------------------------------------------------------------------

def detect(project: Path) -> list[Candidate]:
    project = Path(project).resolve()
    cands: list[Candidate] = []
    cands += _from_readme(project)
    cands += _from_scripts(project)
    cands += _from_python_entries(project)
    cands += _from_pyproject(project)
    cands += _from_makefile(project)
    cands += _from_package_json(project)
    seen: set[tuple[str, str]] = set()
    out = []
    for c in sorted(cands, key=lambda c: -c.score):
        key = (shlex.join(c.command), os.path.normpath(c.cwd))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _from_readme(project: Path) -> list[Candidate]:
    out: list[Candidate] = []
    for name in ("README.md", "readme.md", "README.MD", "README.rst", "README"):
        readme = project / name
        if readme.is_file():
            break
    else:
        return out
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    shells = {"", "bash", "sh", "shell", "console", "zsh", "text", "cmd", "terminal", "powershell"}
    blocks = [body for lang, body in re.findall(r"^[ \t]*```[ \t]*([\w+-]*)[^\n]*\n(.*?)^[ \t]*```", text, re.S | re.M)
              if lang.lower() in shells]
    order = 0
    for block in blocks:
        cwd = "."
        for raw in block.splitlines():
            line = raw.strip()
            line = re.sub(r"^(\$|>|%)\s+", "", line)
            if not line or line.startswith("#"):
                continue
            for part in [p.strip() for p in re.split(r"\s*&&\s*", line)]:
                m_cd = re.match(r"^cd\s+(\S+)$", part)
                if m_cd:
                    target = m_cd.group(1).strip("'\"")
                    if not target.startswith(("/", "~", "$")) and "<" not in target:
                        cwd = os.path.normpath(os.path.join(cwd, target))
                    continue
                if not _RUN_LINE.match(part) or _NOT_RUN.search(part.split(" #")[0]):
                    continue
                cmd_text = part.split(" #")[0].strip()
                try:
                    argv = shlex.split(cmd_text)
                except ValueError:
                    continue
                if not (project / cwd).is_dir() or not _files_exist(project / cwd, argv):
                    continue
                order += 1
                out.append(Candidate(argv, cwd, "README", score=100 - order))
    return out


def _files_exist(base: Path, argv: list[str]) -> bool:
    for tok in argv[:3]:
        if tok.endswith((".py", ".sh")) and not (base / tok).exists():
            return False
    return True


def _from_scripts(project: Path) -> list[Candidate]:
    out = []
    for name in ("start.sh", "run.sh", "dev.sh", "launch.sh", "serve.sh", "main.sh", "start", "run"):  # never stop.sh etc.
        p = project / name
        if p.is_file() and (name.endswith(".sh") or os.access(p, os.X_OK)):
            cmd = [f"./{name}"] if os.access(p, os.X_OK) else ["bash", name]
            out.append(Candidate(cmd, ".", "start script", score=90 if name.startswith("start") else 80))
    return out


def _from_python_entries(project: Path) -> list[Candidate]:
    out = []
    for d in _ENTRY_DIRS:
        base = project / d
        if not base.is_dir():
            continue
        files = [base / n for n in _ENTRY_NAMES] + sorted(base.glob("*_app.py")) + sorted(base.glob("app_*.py"))
        for f in files:
            if not f.is_file():
                continue
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel_dir = os.path.relpath(f.parent, project)
            if re.search(r"^\s*(import streamlit|from streamlit)", src, re.M):
                out.append(Candidate(["streamlit", "run", f.name], rel_dir, "Streamlit app", score=70))
            elif re.search(r"^\s*(import chainlit|from chainlit)", src, re.M):
                out.append(Candidate(["chainlit", "run", f.name], rel_dir, "Chainlit app", score=70))
            elif "__main__" in src:
                kind = "Gradio app" if "gradio" in src else "script"
                out.append(Candidate(["python", f.name], rel_dir, kind, score=65 if kind != "script" else 60))
            else:
                m = re.search(r"^(\w+)\s*=\s*FastAPI\(", src, re.M)
                if m:
                    out.append(Candidate(["python", "-m", "uvicorn", f"{f.stem}:{m.group(1)}", "--port", "8000"],
                                         rel_dir, "FastAPI app", score=68))
    return out


def _from_pyproject(project: Path) -> list[Candidate]:
    p = project / "pyproject.toml"
    if not p.is_file():
        return []
    try:
        import tomllib  # type: ignore

        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    scripts = (data.get("project") or {}).get("scripts") or {}
    scripts.update(((data.get("tool") or {}).get("poetry") or {}).get("scripts") or {})
    return [Candidate([name], ".", "pyproject script", score=55) for name in list(scripts)[:4]]


def _from_makefile(project: Path) -> list[Candidate]:
    p = project / "Makefile"
    if not p.is_file():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [Candidate(["make", t], ".", "Makefile", score=50)
            for t in ("run", "start", "dev", "serve") if re.search(rf"^{t}\s*:", text, re.M)]


def _from_package_json(project: Path) -> list[Candidate]:
    out = []
    for d in (".", "frontend", "web", "app", "ui"):
        p = project / d / "package.json"
        if not p.is_file():
            continue
        try:
            scripts = json.loads(p.read_text(encoding="utf-8")).get("scripts") or {}
        except (OSError, ValueError):
            continue
        for name in ("dev", "start"):
            if name in scripts:
                out.append(Candidate(["npm", "run", name], os.path.normpath(d), "package.json",
                                     score=45 if d == "." else 30))
                break
    return out
