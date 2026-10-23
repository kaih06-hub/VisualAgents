"""What a scanned repository and a web page must not be able to do.

- A project's .env may not choose where model requests go (that would send your saved key to its server).
- A link in a project may not make the scan read files outside it.
- Private keys are redacted like API keys.
- The dashboard answers only requests addressed to this computer, and Start / reset need the page's session token.
"""

import http.client
import json
import os
import re
import subprocess

from visualagents import blueprint as bpm
from visualagents import llm, scanner
from visualagents import server as server_mod
from visualagents.collector import collect, pack, redact
from visualagents.hub import Hub
from visualagents.server import Server, exposure_warning

ROUTING = ("VISUALAGENTS_PROVIDER", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI",
           "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")
PEM = "-----BEGIN OPENSSH PRIVATE KEY-----\nDUMMY-PRIVATE-KEY-MATERIAL-0123456789\n-----END OPENSSH PRIVATE KEY-----\n"


def _clean_env(monkeypatch):
    for k in (*ROUTING, "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "VISUALAGENTS_MODEL"):
        monkeypatch.delenv(k, raising=False)


def test_a_scanned_projects_env_cannot_choose_where_requests_go(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text(
        "VISUALAGENTS_PROVIDER=anthropic\nANTHROPIC_BASE_URL=http://evil.example\nOPENAI_BASE_URL=http://evil.example\n"
        "GOOGLE_CLOUD_PROJECT=their-project\nANTHROPIC_API_KEY=sk-ant-the-repos-own-key-000\nVISUALAGENTS_MODEL=m1\n")
    monkeypatch.chdir(repo)  # `visualagents scan .` inside the clone: its .env is still the project's
    ignored = llm.load_dotenv_keys(repo)
    assert set(ignored) == {"VISUALAGENTS_PROVIDER", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "GOOGLE_CLOUD_PROJECT"}
    assert not any(os.environ.get(k) for k in ROUTING)
    # keys and model names still come from the project: they cannot send anything anywhere new
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-the-repos-own-key-000" and os.environ["VISUALAGENTS_MODEL"] == "m1"


def test_a_env_in_your_own_folder_outside_the_project_still_sets_the_endpoint(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    mine, repo = tmp_path / "me", tmp_path / "repo"
    mine.mkdir()
    repo.mkdir()
    (mine / ".env").write_text("OPENAI_BASE_URL=http://localhost:11434/v1\n")
    monkeypatch.chdir(mine)
    assert llm.load_dotenv_keys(repo) == []
    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:11434/v1"


def test_the_scan_says_what_it_ignored(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    (tmp_path / ".env").write_text("ANTHROPIC_BASE_URL=http://evil.example\n")
    (tmp_path / "app.py").write_text("from agents import Agent\nwriter = Agent(name='Writer', instructions='w')\n")
    lines = []
    scanner.scan_project(tmp_path, mode="static", log=lines.append)
    assert any("Ignored ANTHROPIC_BASE_URL" in line for line in lines)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True,
                   capture_output=True)


def test_links_out_of_the_project_are_never_read(tmp_path):
    outside = tmp_path / "home" / ".ssh"
    outside.mkdir(parents=True)
    (outside / "id_ed25519").write_text(PEM)
    repo = tmp_path / "home" / "agents" / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "agent.py").write_text("import openai\nclient = openai.OpenAI()\n")
    (repo / "README.md").symlink_to("../../.ssh/id_ed25519")  # always packed when it is the root README
    (repo / "docs" / "agent.md").symlink_to("../agent.py")  # a link inside the project is fine
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "x")
    files = collect(repo)
    rels = {f.rel for f in files}
    assert "README.md" not in rels and "docs/agent.md" in rels
    assert "DUMMY-PRIVATE-KEY-MATERIAL" not in pack(files)[0]


def test_private_keys_are_redacted():
    assert "DUMMY-PRIVATE-KEY-MATERIAL" not in redact("config:\n" + PEM + "done")
    assert "DUMMY-PRIVATE-KEY-MATERIAL" not in redact(PEM.split("-----END")[0])  # cut off before its END line
    assert redact("-----BEGIN PUBLIC KEY-----\nabc\n") == "-----BEGIN PUBLIC KEY-----\nabc\n"


class _Launcher:
    def __init__(self):
        self.calls = []

    def start(self, task):
        self.calls.append(task)
        return True, "started"

    def info(self):
        return {"command": "python main.py {task}", "running": False}


def _server():
    hub = Hub(bpm.normalize({"project": "p", "agents": [{"name": "Writer"}]}))
    hub.launcher = _Launcher()
    return hub, Server(hub, port=0).start()


def _req(server, method, path, host=None, headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    conn.request(method, path, body=body, headers={"Host": host or f"127.0.0.1:{server.port}", **(headers or {})})
    resp = conn.getresponse()
    return resp.status, resp.read().decode()


def test_the_dashboard_answers_only_requests_addressed_to_this_computer():
    hub, server = _server()
    try:
        for host in (f"127.0.0.1:{server.port}", f"localhost:{server.port}", f"[::1]:{server.port}"):
            assert _req(server, "GET", "/api/state", host)[0] == 200
        # a page whose own domain now points at 127.0.0.1 (DNS rebinding) is turned away, reads and writes alike
        evil = f"evil.example:{server.port}"
        assert _req(server, "GET", "/api/state", evil)[0] == 421
        assert _req(server, "GET", "/", evil)[0] == 421
        status, _ = _req(server, "POST", "/api/start", evil, {"Origin": f"http://{evil}"}, '{"task": "x"}')
        assert status == 421 and hub.launcher.calls == []
    finally:
        server.stop()


def test_start_and_reset_need_the_session_token_from_the_page():
    hub, server = _server()
    try:
        page = _req(server, "GET", "/")[1]
        token = re.search(r'name="va-session" content="([^"]+)"', page).group(1)
        assert token == server.session
        assert _req(server, "POST", "/api/start", body='{"task": "a"}')[0] == 403  # no token
        assert _req(server, "POST", "/api/start", headers={"X-VisualAgents-Session": "guess"}, body="{}")[0] == 403
        assert _req(server, "POST", "/api/reset")[0] == 403
        assert hub.launcher.calls == []
        ok = {"X-VisualAgents-Session": token}
        status, body = _req(server, "POST", "/api/start", headers=ok, body='{"task": "summarise"}')
        assert status == 200 and json.loads(body)["ok"] and hub.launcher.calls == ["summarise"]
        assert _req(server, "POST", "/api/reset", headers=ok)[0] == 200
        cross = {**ok, "Origin": "https://evil.example"}  # the token alone is not enough from another site
        assert _req(server, "POST", "/api/start", headers=cross, body="{}")[0] == 403
    finally:
        server.stop()


def test_a_page_served_to_another_computer_carries_no_token():
    handler = type("H", (server_mod._Handler,), {"session": "tok-123", "local_only": False})
    h = handler.__new__(handler)
    h.client_address = ("192.168.1.20", 50000)
    assert b"tok-123" not in h._index()
    h.client_address = ("127.0.0.1", 50000)
    assert b"tok-123" in h._index()


def test_binding_to_the_network_is_announced():
    assert exposure_warning("127.0.0.1") is None and exposure_warning("localhost") is None
    assert "network" in exposure_warning("0.0.0.0") and "every interface" in exposure_warning("")
