import json
import sys
import threading
import urllib.request
from pathlib import Path

from visualagents import blueprint as bpm
from visualagents.hub import Hub
from visualagents.runner import LineSplitter, replay, run_command
from visualagents.server import Server

BP = bpm.normalize({
    "project": "t",
    "agents": [{"name": "planner"}, {"name": "writer"}],
    "edges": [{"from": "planner", "to": "writer"}],
})


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **({"X-VisualAgents-Token": token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def test_line_splitter_hides_markers_from_echo():
    lines, shown = [], []
    sp = LineSplitter(lines.append, shown.append)
    sp.feed(b"hello\n@@visualagents {\"type\": \"log\"}\nwor")
    sp.feed(b"ld\n")
    sp.close()
    assert lines == ["hello", '@@visualagents {"type": "log"}', "world"]
    assert "".join(shown) == "hello\nworld\n"


def test_server_state_ingest_and_stream():
    hub = Hub(BP)
    server = Server(hub, port=0).start()
    try:
        state = _get(server.url + "/api/state")
        assert state["blueprint"]["project"] == "t"
        assert _post(server.url + "/api/ingest", {"type": "agent_start", "agent": "planner"})[0] == 403
        status, body = _post(server.url + "/api/ingest", [
            {"type": "agent_start", "agent": "planner"},
            {"type": "line", "text": "Thought: plan it"},
        ], token=hub.token)
        assert status == 200 and body["count"] == 2
        assert _get(server.url + "/api/state")["state"]["active"] == "planner"

        got = []

        def read_stream():
            with urllib.request.urlopen(server.url + "/api/stream?since=0", timeout=5) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith("event: state"):
                        got.append("state")
                        return

        t = threading.Thread(target=read_stream, daemon=True)
        t.start()
        t.join(5)
        assert got == ["state"]
        index = urllib.request.urlopen(server.url + "/").read().decode()
        assert "VisualAgents" in index
        assert urllib.request.urlopen(server.url + "/assets/app.js").status == 200
    finally:
        server.stop()


def test_run_command_with_markers_and_recording(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text(
        "import json, os\n"
        "print('# Agent: planner', flush=True)\n"
        "print('Thought: outline first', flush=True)\n"
        "os.write(2, b'@@visualagents ' + json.dumps({'type': 'tool_start', 'agent': 'planner', 'tool': 'search'}).encode() + b'\\n')\n"
        "print('Handing off to writer', flush=True)\n"
        "print('Final Answer: done', flush=True)\n"
    )
    hub = Hub(BP, record_dir=tmp_path / "runs")
    code = run_command([sys.executable, str(script)], hub, cwd=tmp_path, server_url="http://127.0.0.1:9",
                       instrument=False, use_pty=False, echo=False)
    hub.close()
    assert code == 0
    s = hub.tracker.snapshot()
    assert [p["agent"] for p in s["path"]] == ["planner", "writer"]
    assert s["agents"]["planner"]["tools"] == 1
    assert s["run"]["status"] == "finished"

    # the recording replays to the same end state
    rec = next((tmp_path / "runs").glob("*.jsonl"))
    hub2 = Hub(BP)
    replay(rec, hub2, instant=True)
    s2 = hub2.tracker.snapshot()
    assert [p["agent"] for p in s2["path"]] == ["planner", "writer"]
    assert s2["run"]["status"] == "finished"


def test_failed_command_is_reported(tmp_path):
    hub = Hub(BP)
    code = run_command([sys.executable, "-c", "import sys; print('# Agent: writer'); sys.exit(3)"], hub, cwd=tmp_path,
                       server_url="http://127.0.0.1:9", instrument=False, use_pty=False, echo=False)
    assert code == 3
    s = hub.tracker.snapshot()
    assert s["run"]["status"] == "failed"
    assert s["agents"]["writer"]["status"] == "error"


def test_autoinstrument_runtime_emits_markers(tmp_path):
    script = tmp_path / "manual.py"
    script.write_text(
        "from visualagents_runtime import agent, emit\n"
        "with agent('planner', task='make a plan'):\n"
        "    emit('thought', 'planner', 'thinking')\n"
        "with agent('writer'):\n"
        "    print('writing', flush=True)\n"
    )
    hub = Hub(BP)
    code = run_command([sys.executable, str(script)], hub, cwd=tmp_path, server_url="http://127.0.0.1:9",
                       instrument=True, use_pty=False, echo=False)
    assert code == 0
    s = hub.tracker.snapshot()
    assert s["precise"]
    assert [p["agent"] for p in s["path"]] == ["planner", "writer"]
    writing = [e for e in hub.tracker.events if e["text"] == "writing"]
    assert writing and writing[0]["agent"] == "writer"


def test_pty_mode_child_sees_a_terminal(tmp_path):
    if sys.platform == "win32":
        return
    hub = Hub(BP)
    code = run_command(
        [sys.executable, "-c", "import sys; print('# Agent: planner'); print('tty' if sys.stdout.isatty() else 'pipe')"],
        hub, cwd=tmp_path, server_url="http://127.0.0.1:9", instrument=False, use_pty=True, echo=False,
    )
    assert code == 0
    texts = [e["text"] for e in hub.tracker.events]
    assert "tty" in texts
    assert hub.tracker.snapshot()["path"][0]["agent"] == "planner"
