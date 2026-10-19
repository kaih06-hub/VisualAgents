"""Install / remove the startup hook that lets a project's Python processes report to the dashboard.

`visualagents hook <folder>` writes `visualagents_hook.pth` into the site-packages of each virtualenv in
the folder (or an interpreter you name). Python runs .pth "import" lines at startup, so every process from
that environment - started by a script, an IDE, a notebook kernel - loads VisualAgents' runtime, which stays
silent until the process makes an LLM / agent call.
"""

from __future__ import annotations

import subprocess
from importlib import resources
from pathlib import Path

PTH_NAME = "visualagents_hook.pth"


def boot_path() -> Path:
    return Path(str(resources.files("visualagents") / "_hook" / "visualagents_boot.py")).resolve()


def site_packages(python: Path) -> Path:
    out = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return Path(out.stdout.strip())


def pth_line(project: Path) -> str:
    boot = str(boot_path())
    return (
        f"import os; _va = {boot!r}; os.path.exists(_va) and exec(compile(open(_va, encoding='utf-8').read(), _va, "
        f"'exec'), {{'__name__': 'visualagents_boot', '__file__': _va, 'VISUALAGENTS_PROJECT': {str(project)!r}}})"
    )


def install(python: Path, project: Path) -> Path:
    site = site_packages(python)
    site.mkdir(parents=True, exist_ok=True)
    target = site / PTH_NAME
    target.write_text(
        f"# VisualAgents hook for {project} - remove with: visualagents unhook {project}\n{pth_line(project)}\n",
        encoding="utf-8",
    )
    return target


def uninstall(site: Path) -> bool:
    target = Path(site) / PTH_NAME
    if target.exists():
        target.unlink()
        return True
    return False


def hooked_project(site: Path) -> str | None:
    """Which project an installed hook points at (None when not hooked)."""
    try:
        first = (Path(site) / PTH_NAME).read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    return first.split("hook for ", 1)[-1].split(" - remove", 1)[0] if "hook for " in first else ""
