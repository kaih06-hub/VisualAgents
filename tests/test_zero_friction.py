"""`visualagents run <folder>` with no command, background processes, and `visualagents hook`."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from visualagents import blueprint as bpm
from visualagents import cli, detect, home, hooks
from visualagents.hub import Hub
from visualagents.projects import Projects
from visualagents.runner import run_command
from visualagents.server import Server

BP = bpm.normalize({"project": "t", "agents": [{"name": "planner"}, {"name": "writer"}]})

APP = """
import time, visualagents_runtime as rt
rt.agent_start("planner", "make a plan")
print("planning...", flush=True)
rt.agent_end("planner")
rt.agent_start("writer")
rt.emit("tool_start", "writer", "draft.md", tool="write_file")
rt.agent_end("writer")
"""


def _wait(pred, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_detects_how_a_lambda_like_project_starts(tmp_path):
    (tmp_path / "README.md").write_text(
        "# App\n\n```env\nOPENAI_API_KEY=x\n```\n\n```bash\ncp backend/.env.example backend/.env\n./start.sh\n```\n\n"
        "```bash\n./stop.sh\n```\n\n```bash\ncd backend\npython3 -m venv venv\npip install -r requirements.txt\n"
        "uvicorn main:app --port 8000 --reload\n```\n"
    )
    (tmp_path / "start.sh").write_text("#!/bin/sh\n")
    (tmp_path / "stop.sh").write_text("#!/bin/sh\n")
    os.chmod(tmp_path / "start.sh", 0o755)
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
    venv = tmp_path / "backend" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    for tool in ("python", "uvicorn"):
        (venv / "bin" / tool).write_text("")
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text(json.dumps({"scripts": {"dev": "vite"}}))

    cands = detect.detect(tmp_path)
    labels = [c.label() for c in cands]
    assert labels[0] == "./start.sh"
    assert "uvicorn main:app --port 8000 --reload  (in backend/)" in labels
    assert not any("stop.sh" in label for label in labels)
    assert "npm run dev  (in frontend/)" in labels
    uvicorn = next(c for c in cands if c.command[0] == "uvicorn")
    argv, notes = detect.resolve(tmp_path, uvicorn)
    assert argv[0] == str(venv / "bin" / "uvicorn")
    assert "backend/venv" in notes[0]


def test_a_script_that_backgrounds_the_app_still_reports(tmp_path):
    """start.sh puts the app in the background with its output in a log file - events arrive over HTTP."""
    (tmp_path / "app.py").write_text(APP)
    (tmp_path / "start.sh").write_text(
        f"mkdir -p .logs\n{sys.executable} app.py > .logs/backend.log 2>&1 &\necho started\n")
    hub = Hub(BP)
    server = Server(hub, port=0).start()
    try:
        code = run_command(["sh", "start.sh"], hub, cwd=tmp_path, server_url=server.url, instrument=True,
                           use_pty=False, echo=False)
        assert code == 0
        assert _wait(lambda: [p["agent"] for p in hub.tracker.snapshot()["path"]] == ["planner", "writer"])
        assert hub.tracker.snapshot()["agents"]["writer"]["tools"] == 1
        assert any(e["text"] == "started" for e in hub.tracker.events)
        assert "planning..." in (tmp_path / ".logs" / "backend.log").read_text()  # its own output is untouched
    finally:
        server.stop()


def test_hooked_virtualenv_reports_without_a_wrapper(tmp_path, monkeypatch):
    monkeypatch.setenv("VISUALAGENTS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VISUALAGENTS_URL", raising=False)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "app.py").write_text(APP)
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(project / ".venv")], check=True)
    py = detect.venv_bin(project / ".venv", "python")
    pth = hooks.install(py, project)
    assert pth.exists() and hooks.hooked_project(pth.parent) == str(project)

    projects = Projects(daemon=True)
    server = Server(projects, port=0).start()
    home.write_daemon(server.url, projects.token, os.getpid())
    env = {k: v for k, v in os.environ.items() if not k.startswith("VISUALAGENTS_") or k == "VISUALAGENTS_HOME"}
    try:
        subprocess.run([str(py), "-c", "print('not an agent')"], cwd=project, env=env, check=True)
        out = subprocess.run([str(py), "app.py"], cwd=project, env=env, check=True, capture_output=True, text=True)
        assert out.stdout.strip() == "planning..."  # the app's own output is unchanged
        assert _wait(lambda: projects.listing() and projects.listing()[0]["status"] == "finished")
        hub = projects.resolve(str(project))
        s = hub.tracker.snapshot()
        assert [p["agent"] for p in s["path"]] == ["planner", "writer"]
        assert any(e["text"] == "planning..." and e["agent"] == "planner" for e in hub.tracker.events)
        attached = [e["text"] for e in hub.tracker.events if e["text"].startswith("Process ")]
        assert len(attached) == 1 and "app.py" in attached[0]  # the print-only process never showed up
    finally:
        server.stop()
        hooks.uninstall(pth.parent)


def test_run_without_a_command_picks_and_remembers(tmp_path, monkeypatch):
    monkeypatch.setenv("VISUALAGENTS_NO_BROWSER", "1")
    (tmp_path / "main.py").write_text('if __name__ == "__main__":\n    print("# Agent: planner")\n')
    args = ["run", str(tmp_path), "--no-browser", "--exit", "--port", "0", "--no-record", "--static", "--no-pty"]
    assert cli.main(args) == 0
    saved = json.loads((tmp_path / ".visualagents" / "config.json").read_text())
    assert saved == {"command": ["python", "main.py"], "cwd": "."}
    assert cli.main(args) == 0
