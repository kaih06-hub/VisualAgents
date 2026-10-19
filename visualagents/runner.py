"""Input sources: run a command (tee its output), tail a log file, read stdin, or replay a recording."""

from __future__ import annotations

import codecs
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from importlib import resources
from pathlib import Path
from typing import Callable, Iterable

from .hub import Hub
from .patterns import MARKER

_MARKER_CORE = MARKER.strip()


class LineSplitter:
    """Incrementally decodes output chunks, echoes them to the terminal, and yields complete lines.

    Marker lines (structured events printed by any language) are hidden from the echo.
    """

    def __init__(self, on_line: Callable[[str], None], echo: Callable[[str], None] | None):
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.on_line = on_line
        self.echo = echo
        self.buf = ""
        self.disp = ""

    def feed(self, data: bytes) -> None:
        text = self.decoder.decode(data)
        if not text:
            return
        if self.echo is not None:
            self._echo(text)
        self.buf += text.replace("\r\n", "\n")
        while True:
            i = self.buf.find("\n")
            if i < 0:
                break
            line, self.buf = self.buf[:i], self.buf[i + 1:]
            self.on_line(line)
        if len(self.buf) > 20000:  # a very long line without a newline (progress bars)
            self.on_line(self.buf)
            self.buf = ""

    def _echo(self, text: str) -> None:
        self.disp += text
        out = []
        while True:
            i = self.disp.find("\n")
            if i < 0:
                break
            line, self.disp = self.disp[: i + 1], self.disp[i + 1:]
            if _MARKER_CORE not in line:
                out.append(line)
        rem = self.disp
        if rem and _MARKER_CORE not in rem and "@" not in rem[-len(_MARKER_CORE):]:
            out.append(rem)
            self.disp = ""
        if out:
            self.echo("".join(out))

    def close(self) -> None:
        tail = self.decoder.decode(b"", final=True)
        if tail:
            self.buf += tail
        if self.echo is not None and self.disp and _MARKER_CORE not in self.disp:
            self.echo(self.disp)
        self.disp = ""
        if self.buf.strip():
            self.on_line(self.buf)
        self.buf = ""


def _terminal_echo(text: str) -> None:
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


def autoinstrument_dir() -> str:
    return str(resources.files("visualagents") / "_autoinstrument")


def build_env(server_url: str, token: str, *, instrument: bool, use_pty: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = env.get("PYTHONIOENCODING", "utf-8")
    env["VISUALAGENTS_URL"] = server_url
    env["VISUALAGENTS_TOKEN"] = token
    env.pop("VISUALAGENTS_STREAM_ID", None)  # set per child once its pipe / pty exists
    if instrument:
        parts = [autoinstrument_dir()] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join(parts)
    if not use_pty:
        env.setdefault("FORCE_COLOR", "0")
    return env


def run_command(
    command: list[str],
    hub: Hub,
    *,
    cwd: Path,
    server_url: str,
    instrument: bool = True,
    use_pty: bool | None = None,
    echo: bool = True,
) -> int:
    """Run `command`, mirror its output to this terminal, and feed every line to the hub."""
    if use_pty is None:
        use_pty = os.name == "posix" and sys.stdout.isatty()
    env = build_env(server_url, hub.token, instrument=instrument, use_pty=use_pty)
    hub.run_started(shlex.join(command))
    echo_fn = _terminal_echo if echo else None
    try:
        if use_pty:
            code = _run_pty(command, hub, cwd, env, echo_fn)
        else:
            code = _run_pipes(command, hub, cwd, env, echo_fn)
    except FileNotFoundError:
        hub.system(f"Command not found: {command[0]}")
        code = 127
    hub.run_finished(code)
    return code


def _wait_interruptibly(proc: subprocess.Popen) -> int:
    """Wait for the child; Ctrl+C reaches it directly (same process group), a second one kills it."""
    interrupts = 0
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts >= 2:
                proc.kill()


def _run_pty(command, hub, cwd, env, echo_fn) -> int:
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    env["VISUALAGENTS_STREAM_ID"] = _stream_id(slave)
    try:
        size = os.get_terminal_size(sys.stdout.fileno())
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", size.lines, size.columns, 0, 0))
    except OSError:
        pass
    # stdin stays the real terminal so prompts like input("Query: ") work as usual.
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    splitter = LineSplitter(lambda line: hub.feed_line(line, "stdout"), echo_fn)

    def pump():
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:  # EIO once every writer has closed the pty
                break
            if not data:
                break
            splitter.feed(data)
        splitter.close()

    t = threading.Thread(target=pump, name="visualagents-pty", daemon=True)
    t.start()
    code = _wait_interruptibly(proc)
    t.join(timeout=2.0)
    try:
        os.close(master)
    except OSError:
        pass
    return code


def _stream_id(fd: int) -> str:
    st = os.fstat(fd)
    return f"{st.st_dev}:{st.st_ino}"


def _run_pipes(command, hub, cwd, env, echo_fn) -> int:
    # stderr is merged into stdout (as on a terminal) so lines and event markers stay in order.
    read_fd, write_fd = os.pipe()
    env["VISUALAGENTS_STREAM_ID"] = _stream_id(write_fd)
    try:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=write_fd, stderr=write_fd)
    finally:
        os.close(write_fd)
    splitter = LineSplitter(lambda line: hub.feed_line(line, "stdout"), echo_fn)

    def pump():
        fd = read_fd
        while True:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            splitter.feed(data)
        splitter.close()

    t = threading.Thread(target=pump, name="visualagents-pipe", daemon=True)
    t.start()
    code = _wait_interruptibly(proc)
    t.join(timeout=2.0)
    try:
        os.close(read_fd)
    except OSError:
        pass
    return code


def follow_file(path: Path, hub: Hub, *, from_start: bool = False, echo: bool = False,
                stop: threading.Event | None = None, announce: bool = True, stream: str = "stdout") -> None:
    """`tail -F` a log file into the hub until `stop` is set (or Ctrl+C)."""
    stop = stop or threading.Event()
    pos = 0
    inode = None
    first = True
    buf = ""
    if announce:
        hub.run_started(f"tail {path}")
    while not stop.is_set():
        try:
            st = path.stat()
        except FileNotFoundError:
            time.sleep(0.3)
            continue
        if inode != st.st_ino or st.st_size < pos:  # new or rotated / truncated file
            inode, pos = st.st_ino, 0 if (from_start or not first) else st.st_size
            first = False
        if st.st_size > pos:
            with open(path, "rb") as fh:
                fh.seek(pos)
                data = fh.read(st.st_size - pos)
            pos += len(data)
            buf += data.decode("utf-8", errors="replace")
            *lines, buf = buf.split("\n")
            for line in lines:
                if echo:
                    _terminal_echo(line + "\n")
                hub.feed_line(line, stream)
        else:
            stop.wait(0.2)


_LOG_SKIP = {".git", "node_modules", "__pycache__", ".visualagents", "venv", ".venv", "site-packages", "dist", "build"}


def follow_new_logs(project: Path, hub: Hub, *, since: float, stop: threading.Event, max_depth: int = 3) -> None:
    """Follow *.log files the app writes during the run (e.g. a start script sending the backend to .logs/)."""
    project = Path(project).resolve()
    following: set[Path] = set()
    while not stop.wait(1.0):
        for dirpath, dirnames, filenames in os.walk(project):
            depth = len(Path(dirpath).relative_to(project).parts)
            dirnames[:] = [] if depth >= max_depth else [d for d in dirnames if d not in _LOG_SKIP]
            for name in filenames:
                if not name.endswith((".log", ".out")):
                    continue
                p = Path(dirpath) / name
                try:
                    if p in following or p.stat().st_mtime < since:
                        continue
                except OSError:
                    continue
                following.add(p)
                rel = str(p.relative_to(project))
                hub.system(f"Following {rel}")
                threading.Thread(target=follow_file, args=(p, hub), daemon=True,
                                 kwargs={"from_start": True, "stop": stop, "announce": False, "stream": rel}).start()


def read_stream(stream: Iterable[bytes] | None, hub: Hub, *, echo: bool = True) -> None:
    """Consume stdin (e.g. `python agent.py 2>&1 | visualagents pipe`)."""
    hub.run_started("stdin")
    src = sys.stdin.buffer if stream is None else stream
    splitter = LineSplitter(hub.feed_line, _terminal_echo if echo else None)
    fd = src.fileno() if hasattr(src, "fileno") else None
    try:
        if fd is not None:
            while True:
                data = os.read(fd, 65536)
                if not data:
                    break
                splitter.feed(data)
        else:
            for chunk in src:
                splitter.feed(chunk)
    except KeyboardInterrupt:
        pass
    splitter.close()
    hub.run_finished(0)


def replay(path: Path, hub: Hub, *, speed: float = 1.0, instant: bool = False, stop: threading.Event | None = None) -> None:
    """Re-feed a recorded run through the tracker at (scaled) original timing."""
    stop = stop or threading.Event()
    started = time.time()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if stop.is_set():
                return
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if not instant and speed > 0:
                due = started + float(rec.get("t", 0)) / speed
                delay = due - time.time()
                if delay > 0 and stop.wait(min(delay, 10.0)):
                    return
            typ = rec.get("type")
            if typ == "line":
                hub.feed_line(rec.get("text", ""), rec.get("stream", "stdout"))
            elif typ == "structured":
                hub.feed_structured(rec.get("payload") or {})
            elif typ == "narration":
                hub.narrate(rec.get("agent"), rec.get("activity", ""))
            elif typ == "run_start":
                hub.run_started(rec.get("command", ""))
            elif typ == "run_end":
                hub.run_finished(rec.get("code"))


def read_recording_meta(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        first = json.loads(fh.readline() or "{}")
    return first if first.get("type") == "meta" else {}


def install_sigterm_handler() -> None:
    def _term(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _term)
    except (ValueError, OSError):
        pass
