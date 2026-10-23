"""The dashboard's ▶ Start button (and its task box)."""

import json
import re
import sys
import time
import urllib.request

from visualagents import blueprint as bpm
from visualagents import detect
from visualagents.hub import Hub
from visualagents.launcher import Launcher
from visualagents.server import Server

BP = bpm.normalize({"project": "p", "agents": [{"name": "Writer"}]})
APP = 'import sys, time\nprint("# Agent: Writer", flush=True)\ntime.sleep(0.5)\nprint("Final Answer: " + " ".join(sys.argv[1:]), flush=True)\n'


def _post(url, body):
    """Like the dashboard page: take the session token the page was served with, then post with it."""
    page = urllib.request.urlopen(url.split("/api/")[0] + "/", timeout=5).read().decode()
    token = re.search(r'name="va-session" content="([^"]+)"', page).group(1)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-VisualAgents-Session": token})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def _wait(pred, timeout=10):
    end = time.time() + timeout
    while time.time() < end and not pred():
        time.sleep(0.1)
    return pred()


def _setup(tmp_path, command=None):
    (tmp_path / "main.py").write_text('if __name__ == "__main__":\n' + "".join("    " + l + "\n" for l in APP.splitlines()))
    if command:
        detect.save_choice(tmp_path, detect.Candidate(command, "."))
    hub = Hub(BP, meta={"mode": "blueprint", "project": str(tmp_path)})
    server = Server(hub, port=0).start()
    Launcher(tmp_path, hub, server.url, instrument=False)
    return hub, server


def test_start_button_runs_the_detected_command(tmp_path):
    hub, server = _setup(tmp_path)
    try:
        state = json.loads(urllib.request.urlopen(server.url + "/api/state").read())
        assert state["meta"]["launch"] == {"command": "python main.py", "running": False, "takes_task": False}
        out = _post(server.url + "/api/start", {})
        assert out["ok"], out
        assert _wait(lambda: hub.tracker.run["status"] == "finished")
        assert [p["agent"] for p in hub.tracker.snapshot()["path"]] == ["writer"]
        assert detect.load_saved(tmp_path).command == ["python", "main.py"]
    finally:
        server.stop()


def test_task_box_fills_the_placeholder(tmp_path):
    hub, server = _setup(tmp_path, [sys.executable, "main.py", "{task}"])
    try:
        assert hub.launcher.info()["takes_task"] is True
        assert _post(server.url + "/api/start", {})["ok"] is False  # a task is required
        assert _post(server.url + "/api/start", {"task": "summarise Q3 sales"})["ok"]
        assert _post(server.url + "/api/start", {"task": "again"})["ok"] is False  # already running
        assert _wait(lambda: hub.tracker.run["status"] == "finished")
        assert any(e["text"] == "Final Answer: summarise Q3 sales" for e in hub.tracker.events)
        assert detect.load_saved(tmp_path).command[-1] == "{task}"  # the template is kept
    finally:
        server.stop()


def test_hooked_projects_have_no_start_button(tmp_path):
    hub = Hub(BP, meta={"mode": "hook"})
    server = Server(hub, port=0).start()
    try:
        out = _post(server.url + "/api/start", {})
        assert out["ok"] is False and "usually" in out["message"]
    finally:
        server.stop()
