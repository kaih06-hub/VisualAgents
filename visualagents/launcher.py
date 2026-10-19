"""Lets the dashboard start the project (the "▶ Start" button) when nothing is running yet.

The agent runs in the terminal VisualAgents was started from, so it can ask its questions there.
If the saved start command contains `{task}` (e.g. `python main.py "{task}"`), the dashboard also
shows a box to type the task in.
"""

from __future__ import annotations

import shlex
import threading
import time
from pathlib import Path
from typing import Any

from . import detect, termui
from .hub import Hub

PLACEHOLDERS = ("{task}", "{prompt}")


class Launcher:
    def __init__(self, project: Path, hub: Hub, server_url: str, *, instrument: bool = True,
                 fixed: detect.Candidate | None = None):
        self.project = Path(project).resolve()
        self.fixed = fixed  # an exact command given on the command line wins over saved / detected ones
        self.hub = hub
        self.server_url = server_url
        self.instrument = instrument
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        hub.launcher = self

    def candidate(self) -> detect.Candidate | None:
        return self.fixed or detect.load_saved(self.project) or next(iter(detect.detect(self.project)), None)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def info(self) -> dict[str, Any]:
        cand = self.candidate()
        if cand is None:
            return {"command": None, "running": self.running,
                    "hint": f"visualagents run {self.project} -- <the command that starts it>"}
        return {"command": cand.label(), "running": self.running,
                "takes_task": any(ph in tok for tok in cand.command for ph in PLACEHOLDERS)}

    def start(self, task: str | None = None) -> tuple[bool, str]:
        with self.lock:
            if self.running or self.hub.tracker.run["status"] == "running":
                return False, "It's already running."
            cand = template = self.candidate()
            if cand is None:
                return False, "VisualAgents doesn't know how to start this project yet."
            argv = list(cand.command)
            if any(ph in tok for tok in argv for ph in PLACEHOLDERS):
                if not task:
                    return False, "Type what the agent should do first."
                for ph in PLACEHOLDERS:
                    argv = [tok.replace(ph, task) for tok in argv]
                cand = detect.Candidate(argv, cand.cwd, cand.source)
            if self.fixed is None and detect.load_saved(self.project) is None:
                detect.save_choice(self.project, template)  # remember the command (with its {task} slot)
            command, notes = detect.resolve(self.project, cand)
            cwd = (self.project / cand.cwd).resolve()
            self.thread = threading.Thread(target=self._run, args=(command, cwd, notes), name="visualagents-launch",
                                           daemon=True)
            self.thread.start()
            return True, "Starting in your terminal - give it its task there."

    def _run(self, command: list[str], cwd: Path, notes: list[str]) -> None:
        from .runner import follow_new_logs, run_command

        termui.info(f"Starting from the dashboard: {shlex.join(command)}")
        for note in notes:
            termui.info(note)
        termui.info("The map comes alive as it works - answer its questions here in the terminal.")
        stop = threading.Event()
        threading.Thread(target=follow_new_logs, args=(self.project, self.hub), daemon=True,
                         kwargs={"since": time.time() - 1, "stop": stop}).start()
        try:
            code = run_command(command, self.hub, cwd=cwd, server_url=self.server_url, instrument=self.instrument)
            termui.info(f"{self.project.name} {'finished' if code == 0 else f'exited with code {code}'}. "
                        "Press ▶ in the dashboard to run it again, or Ctrl+C here to close.")
        finally:
            stop.set()
