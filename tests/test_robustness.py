"""Failure paths: a stale daemon file, a generated source file, a failed LLM scan, and the edges of running a command."""

import errno
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from visualagents import blueprint as bpm
from visualagents import cli, home, llm, projects, runner, scanner, server
from visualagents.hub import Hub
from visualagents.projects import Projects
from visualagents.server import Server

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "visualagents" / "_autoinstrument"
BP = bpm.normalize({"project": "t", "agents": [{"name": "planner"}]})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# -- daemon.json outlives the daemon ---------------------------------------------------------------------------

@pytest.fixture
def bystander():
    """An unrelated process of this user's that now has the pid a stale daemon.json names."""
    proc = subprocess.Popen(["sleep", "60"])
    yield proc
    proc.kill()
    proc.wait()


def test_down_never_signals_the_process_a_stale_daemon_file_names(bystander, capsys):
    home.write_daemon(f"http://127.0.0.1:{_free_port()}", "old-token", bystander.pid)
    assert cli.main(["down"]) == 0
    time.sleep(0.2)
    assert bystander.poll() is None  # still running: nothing was sent to it
    assert "No background dashboard is running" in capsys.readouterr().err
    assert home.read_daemon() == {}


def test_status_and_up_do_not_trust_a_stale_daemon_file(bystander, monkeypatch, capsys):
    home.write_daemon(f"http://127.0.0.1:{_free_port()}", "old-token", bystander.pid)
    assert cli.main(["status"]) == 0
    assert "not running" in capsys.readouterr().out
    started = []
    monkeypatch.setattr(cli, "_ensure_daemon", lambda port: started.append(port) or {"url": "http://127.0.0.1:7400"})
    assert cli.main(["up", "--no-browser"]) == 0
    assert started == [cli.DAEMON_PORT]  # a new dashboard is started, not "already running"


def test_a_dashboard_answering_with_another_token_or_pid_is_not_the_daemon(bystander):
    srv = Server(Projects(token="live-token"), port=0).start()
    try:
        home.write_daemon(srv.url, "live-token", os.getpid())
        assert home.read_daemon()["url"] == srv.url  # the real daemon is still found
        home.write_daemon(srv.url, "old-token", os.getpid())
        assert home.read_daemon() == {}
        home.write_daemon(srv.url, "live-token", bystander.pid)  # the url answers, but for another pid
        assert home.read_daemon() == {}
    finally:
        srv.stop()


def test_another_users_process_is_never_the_daemon(monkeypatch):
    def kill(pid, sig):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(home.os, "kill", kill)
    assert home._alive(1) is True
    assert home._alive(1, ours=True) is False


def test_down_tolerates_a_daemon_that_exits_just_before_the_signal(monkeypatch, capsys):
    answers = iter([{"url": "http://127.0.0.1:9", "token": "t", "pid": 4242}])
    monkeypatch.setattr(home, "read_daemon", lambda: next(answers, {}))

    def kill(pid, sig):
        raise ProcessLookupError(errno.ESRCH, "No such process")

    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(home, "clear_daemon", lambda pid=None: None)
    assert cli.main(["down"]) == 0
    assert "Stopped the background dashboard" in capsys.readouterr().err


# -- a generated source file -----------------------------------------------------------------------------------

def test_a_generated_table_of_ten_thousand_terms_does_not_crash_the_scan(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "agent.py").write_text(
        "import anthropic\nclient = anthropic.Anthropic()\n"
        "def planner(q):\n    return client.messages.create(model='m', max_tokens=9, messages=[{'role': 'user', 'content': q}])\n")
    (project / "table.py").write_text("TABLE = " + " + ".join(f"'row{i}'" for i in range(10_000)) + "\n")  # ~119 KB
    assert cli.main(["scan", str(project), "--static"]) == 0
    saved = bpm.load(project / ".visualagents" / "blueprint.json")
    assert [a["id"] for a in saved["agents"]] == ["planner"]  # the rest of the project is still mapped


def test_a_table_that_parses_but_nests_too_deep_to_walk_is_skipped(tmp_path):
    """3,000 terms parse fine, but the analysers recurse once per level and would exhaust the stack."""
    table = "TABLE = " + " + ".join(f"'row{i}'" for i in range(3_000)) + "\n"
    (tmp_path / "table.py").write_text(table)
    assert cli.main(["scan", str(tmp_path), "--static"]) == 0
    from visualagents.collector import parse_python

    assert parse_python(table) is None
    assert parse_python("x = " + "(1 + " * 50 + "1" + ")" * 50) is not None  # real code nests far less


# -- the static blueprint is the LLM scan's fallback -----------------------------------------------------------

@pytest.mark.parametrize("answer", ["fails", "no agents"])
def test_the_fallback_blueprint_keeps_its_keywords(monkeypatch, answer):
    monkeypatch.setattr(llm, "detect_provider", lambda preferred="auto": "anthropic")

    def complete_json(**kw):
        if answer == "fails":
            raise llm.LLMError("quota exceeded")
        return {"project": "demo", "agents": []}

    monkeypatch.setattr(llm, "complete_json", complete_json)
    bp = scanner.scan_project(ROOT / "visualagents" / "demo", mode="auto", log=lambda m: None)
    assert bp["generated_by"] == "static"
    assert all(a["keywords"] for a in bp["agents"])  # what log lines are matched to agents by


# -- `run` and the command it runs -----------------------------------------------------------------------------

def test_a_double_dash_inside_the_command_is_the_commands_own(tmp_path):
    assert cli._split_run_argv(["run", str(tmp_path), "npm", "run", "dev", "--", "--port", "3000"]) == (
        ["run", str(tmp_path)], ["npm", "run", "dev", "--", "--port", "3000"])
    assert cli._split_run_argv(["run", str(tmp_path), "--", "npm", "run", "dev", "--", "--port", "3000"]) == (
        ["run", str(tmp_path)], ["npm", "run", "dev", "--", "--port", "3000"])
    assert cli._split_run_argv(["run", "--port", "--", "node", "a.js"]) == (["run", "--port"], ["node", "a.js"])


def test_the_pipe_stays_open_while_a_background_grandchild_still_writes_to_it(monkeypatch):
    """The command exits at once but leaves a child holding its stdout: join() times out with the read still
    blocked, and closing the fd then would let its number be reused - and read - by an unrelated file."""
    pipes = []
    real_pipe = os.pipe

    def pipe():
        r, w = real_pipe()
        pipes.append((r, os.fstat(r).st_ino))
        return r, w

    monkeypatch.setattr(os, "pipe", pipe)
    hub = Hub(BP)
    lines = []
    hub.feed_line = lambda line, stream="stdout": lines.append(line)
    late = "import time; time.sleep(3.5); print('late', flush=True)"
    command = [sys.executable, "-c", f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {late!r}])"]
    runner._run_pipes(command, hub, None, dict(os.environ), None)
    fd, inode = pipes[0]
    assert os.fstat(fd).st_ino == inode  # still the pipe: the pump owns it until it reads EOF
    deadline = time.time() + 6
    while "late" not in lines and time.time() < deadline:
        time.sleep(0.05)
    assert lines == ["late"]


def test_the_log_follower_survives_a_log_it_cannot_open_yet(tmp_path):
    log = tmp_path / "app.log"
    log.write_text("first\n")
    log.chmod(0)
    if os.access(log, os.R_OK):
        pytest.skip("running as a user who can read anything")
    hub = Hub(BP)
    lines = []
    hub.feed_line = lambda line, stream="stdout": lines.append(line)
    stop = threading.Event()
    t = threading.Thread(target=runner.follow_file, args=(log, hub), daemon=True,
                         kwargs={"from_start": True, "stop": stop, "announce": False})
    t.start()
    try:
        time.sleep(0.5)
        assert t.is_alive()
        log.chmod(0o644)
        deadline = time.time() + 5
        while not lines and time.time() < deadline:
            time.sleep(0.05)
        assert lines == ["first"]
    finally:
        stop.set()
        t.join(timeout=2)


def test_the_runtime_never_raises_into_the_program_even_from_a_deleted_folder(tmp_path):
    gone = tmp_path / "gone"
    gone.mkdir()
    script = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(RUNTIME_DIR)!r})\n"
        "import visualagents_runtime as rt\n"
        "rt.configure(url='http://127.0.0.1:9', token='t', lazy=True)\n"
        f"os.chdir({str(gone)!r}); os.rmdir({str(gone)!r})\n"
        "rt.emit('thought', 'planner', 'first agent-like event: arms the process')\n"
        "rt._enqueue = None  # anything else going wrong inside emit\n"
        "rt.emit('thought', 'planner', 'again')\n"
        "print('host program carried on')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
                         env={**os.environ, "PYTHONPATH": ""})
    assert out.returncode == 0, out.stderr
    assert "host program carried on" in out.stdout


# -- the daemon and its files ----------------------------------------------------------------------------------

def test_saved_keys_are_private_from_the_moment_the_file_exists(monkeypatch):
    """Not 0644 first and chmod-ed later: keys.json.tmp is never readable by other users."""
    monkeypatch.setattr(os, "chmod", lambda *a, **kw: None)  # the moment between creating the file and a chmod
    old = os.umask(0o022)
    try:
        path = home.save_key("ANTHROPIC_API_KEY", "sk-ant-test")
    finally:
        os.umask(old)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_first_scan_does_not_hold_up_every_other_request(tmp_path, monkeypatch):
    scanning, done = threading.Event(), threading.Event()

    def slow_scan(path):
        scanning.set()
        done.wait(10)
        return bpm.normalize({"project": "slow", "agents": [{"name": "planner"}]})

    monkeypatch.setattr(projects, "_load_or_scan", slow_scan)
    p = Projects(daemon=True)
    p.add("other", Hub(BP), str(tmp_path / "other"))
    opener = threading.Thread(target=p.for_item, args=({"type": "hello", "project": str(tmp_path)},), daemon=True)
    opener.start()
    assert scanning.wait(5)
    lister = threading.Thread(target=p.listing, daemon=True)
    lister.start()
    lister.join(timeout=2)
    try:
        assert not lister.is_alive()  # answered while the scan is still running
    finally:
        done.set()
        opener.join(timeout=5)
    assert str(tmp_path.resolve()) in p.paths.values()
    assert p.open(tmp_path) is p.for_item({"project": str(tmp_path)})  # opened once


def test_bind_gives_up_at_once_on_an_error_other_than_port_in_use(monkeypatch):
    tried = []

    def busy_then_bad(addr, handler):
        tried.append(addr[1])
        if len(tried) == 1:
            raise OSError(errno.EADDRINUSE, "Address already in use")
        raise OSError(errno.EADDRNOTAVAIL, "Can't assign requested address")

    monkeypatch.setattr(server, "ThreadingHTTPServer", busy_then_bad)
    with pytest.raises(OSError) as exc:
        server._bind("10.255.255.1", 7420, object)
    assert exc.value.errno == errno.EADDRNOTAVAIL
    assert tried == [7420, 7421]  # in use: the next port; anything else: no 18 more tries
