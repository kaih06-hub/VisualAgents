"""Tiny dependency-free terminal widgets: arrow-key menus, prompts with path completion, hidden input,
and a spinner. Falls back to numbered prompts when there is no interactive terminal."""

from __future__ import annotations

import glob
import os
import sys
import threading
import time
from typing import Callable, Sequence

from . import termui

OUT = sys.stderr


def interactive() -> bool:
    return sys.stdin.isatty() and OUT.isatty()


def _w(text: str) -> None:
    OUT.write(text)
    OUT.flush()


def heading(step: str, title: str) -> None:
    _w(f"\n{termui.accent(step)} {termui.bold(title)}\n")


def select(title: str, options: Sequence[tuple[str, str]], default: int = 0) -> int:
    """Pick one of (label, hint) options. Returns its index."""
    default = max(0, min(default, len(options) - 1))
    if not (interactive() and os.name == "posix"):
        return _select_numbered(title, options, default)
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    idx = default
    lines = len(options)

    def render(first: bool = False) -> None:
        if not first:
            _w(f"\x1b[{lines}A")
        for i, (label, hint) in enumerate(options):
            mark = termui.accent("❯") if i == idx else " "
            text = termui.bold(label) if i == idx else label
            _w(f"\x1b[2K  {mark} {text}" + (f"  {termui.dim(hint)}" if hint else "") + "\n")

    if title:
        _w(f"  {termui.dim(title)}\n")
    _w("\x1b[?25l")
    try:
        render(first=True)
        tty.setcbreak(fd)
        while True:
            key = os.read(fd, 8)
            if key in (b"\r", b"\n"):
                break
            if key in (b"\x1b[A", b"k", b"\x1bOA"):
                idx = (idx - 1) % len(options)
            elif key in (b"\x1b[B", b"j", b"\x1bOB"):
                idx = (idx + 1) % len(options)
            elif key.isdigit() and 1 <= int(key) <= len(options):
                idx = int(key) - 1
            elif key in (b"\x03",):
                raise KeyboardInterrupt
            elif key == b"\x04":
                raise EOFError
            render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        _w("\x1b[?25h")
    # collapse the menu to the chosen line
    _w(f"\x1b[{lines}A")
    for _ in range(lines):
        _w("\x1b[2K\n")
    _w(f"\x1b[{lines}A\x1b[2K  {termui.good('✔')} {options[idx][0]}\n")
    return idx


def _select_numbered(title: str, options: Sequence[tuple[str, str]], default: int) -> int:
    if title:
        _w(f"  {termui.dim(title)}\n")
    for i, (label, hint) in enumerate(options, 1):
        _w(f"  {termui.accent(str(i))}. {label}" + (f"  {termui.dim(hint)}" if hint else "") + "\n")
    while True:
        _w(f"  Choose [{default + 1}]: ")
        raw = sys.stdin.readline()
        if not raw:
            raise EOFError
        raw = raw.strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1


def ask(prompt: str, default: str = "", *, paths: bool = False, validate: Callable[[str], str | None] | None = None) -> str:
    """Free-text answer. With paths=True, Tab completes file names. validate returns an error or None."""
    restore = _path_completion() if paths and interactive() else None
    try:
        while True:
            suffix = f" {termui.dim('[' + default + ']')}" if default else ""
            if interactive():
                raw = input(f"  {prompt}{suffix}: ")
            else:
                _w(f"  {prompt}{suffix}: ")
                raw = sys.stdin.readline()
                if not raw:
                    raise EOFError
            value = raw.strip() or default
            problem = validate(value) if validate else None
            if not problem:
                return value
            _w(f"  {termui.bad('✖')} {problem}\n")
    finally:
        if restore:
            restore()


def secret(prompt: str) -> str:
    if interactive():
        import getpass

        return getpass.getpass(f"  {prompt}: ", stream=OUT).strip()
    _w(f"  {prompt}: ")
    raw = sys.stdin.readline()
    if not raw:
        raise EOFError
    return raw.strip()


def confirm(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    answer = ask(f"{prompt} {termui.dim('(' + hint + ')')}", "").lower()
    return default if not answer else answer.startswith("y")


def note(text: str) -> None:
    _w(f"  {text}\n")


def ok(text: str) -> None:
    _w(f"  {termui.good('✔')} {text}\n")


def warn(text: str) -> None:
    _w(f"  {termui.warn('!')} {text}\n")


class Spinner:
    """`with Spinner("Reading the code") as sp: sp.update("detail")`"""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, text: str):
        self.text = text
        self.detail = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self.live = interactive()

    def update(self, detail: str) -> None:
        self.detail = detail.replace("\n", " ")
        if not self.live:
            note(termui.dim(self.detail))

    def _spin(self) -> None:
        i = 0
        while not self._stop.wait(0.08):
            width = max(20, (os.get_terminal_size(OUT.fileno()).columns if OUT.isatty() else 80) - 6)
            line = f"{self.text}  {termui.dim(self.detail)}" if self.detail else self.text
            if len(self.text) + len(self.detail) + 2 > width:
                line = (f"{self.text}  " + termui.dim(self.detail[: max(0, width - len(self.text) - 3)] + "…"))
            _w(f"\r\x1b[2K  {termui.accent(self.FRAMES[i % len(self.FRAMES)])} {line}")
            i += 1

    def __enter__(self) -> "Spinner":
        if self.live:
            self._thread.start()
        else:
            note(self.text + "…")
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self.live:
            self._thread.join(timeout=1)
            _w("\r\x1b[2K")


def _path_completion():
    try:
        import readline
    except ImportError:
        return None
    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()

    def complete(text: str, state: int):
        expanded = os.path.expanduser(text)
        matches = [m + ("/" if os.path.isdir(m) else "") for m in glob.glob(expanded + "*")]
        if text.startswith("~"):
            home = os.path.expanduser("~")
            matches = ["~" + m[len(home):] if m.startswith(home) else m for m in matches]
        matches.sort()
        return matches[state] if state < len(matches) else None

    readline.set_completer_delims(" \t\n;")
    readline.set_completer(complete)
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

    def restore() -> None:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)

    return restore


def clean_path(raw: str) -> str:
    """Accept drag-and-dropped paths: quotes and backslash-escaped spaces."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1]
    return os.path.expanduser(raw.replace("\\ ", " "))


def short(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home) else path


def pause(seconds: float) -> None:
    time.sleep(seconds)
