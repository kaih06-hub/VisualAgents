"""Thread-safe fan-out between inputs (process output, SDK events, LLM narration) and UI clients.

Raw inputs are also recorded to `.visualagents/runs/<timestamp>.jsonl` so any run can be replayed
later - with a newer blueprint or smarter patterns - via `visualagents replay`.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .tracker import Tracker


class Hub:
    def __init__(self, blueprint: dict[str, Any], *, record_dir: Path | None = None, meta: dict[str, Any] | None = None):
        self.blueprint = blueprint
        self.tracker = Tracker(blueprint)
        self.meta = meta or {}
        self.token = secrets.token_urlsafe(16)
        self.cond = threading.Condition()
        self.events: deque[dict[str, Any]] = deque(maxlen=20000)
        self.state_version = 0
        self.recent_lines: deque[str] = deque(maxlen=60)
        self.lines_seen = 0
        self.started_at = time.time()
        self._record = None
        self.record_path: Path | None = None
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)
            self.record_path = record_dir / time.strftime("%Y%m%d-%H%M%S.jsonl")
            self._record = open(self.record_path, "a", encoding="utf-8")
            self._write_record({"type": "meta", "blueprint": blueprint, "meta": self.meta})

    # -- inputs --------------------------------------------------------------
    def feed_line(self, line: str, stream: str = "stdout") -> None:
        with self.cond:
            self._write_record({"type": "line", "stream": stream, "text": line})
            self.lines_seen += 1
            self.recent_lines.append(line)
            self._publish(self.tracker.ingest_line(line, stream))

    def feed_structured(self, payload: dict[str, Any]) -> None:
        with self.cond:
            self._write_record({"type": "structured", "payload": payload})
            self._publish(self.tracker.ingest_structured(payload))

    def narrate(self, agent: str | None, activity: str) -> None:
        with self.cond:
            self._write_record({"type": "narration", "agent": agent, "activity": activity})
            self._publish(self.tracker.apply_narration(agent, activity))

    def run_started(self, command: str = "") -> None:
        with self.cond:
            self._write_record({"type": "run_start", "command": command})
            self._publish(self.tracker.run_started(command))

    def run_finished(self, code: int | None) -> None:
        with self.cond:
            self._write_record({"type": "run_end", "code": code})
            self._publish(self.tracker.run_finished(code))

    def system(self, text: str) -> None:
        with self.cond:
            self._publish([self.tracker._event("status", None, text, source="system")])

    def reset_view(self) -> None:
        with self.cond:
            run = dict(self.tracker.run)
            self.tracker.reset()
            self.tracker.run.update(run)
            self._publish([self.tracker._event("status", None, "View reset", source="system")])

    # -- outputs -------------------------------------------------------------
    def _publish(self, events: list[dict[str, Any]]) -> None:
        for ev in events:
            self.events.append(ev)
        self.state_version += 1
        self.cond.notify_all()

    def snapshot(self, max_events: int = 1500) -> dict[str, Any]:
        with self.cond:
            return {
                "blueprint": self.blueprint,
                "state": self.tracker.snapshot(),
                "events": list(self.events)[-max_events:],
                "meta": {**self.meta, "lines": self.lines_seen, "record": str(self.record_path or "")},
                "version": self.state_version,
            }

    def wait(self, last_seq: int, last_version: int, timeout: float = 15.0):
        """Block until something newer than (last_seq, last_version) exists. Returns (events, state, version)."""
        with self.cond:
            if self.state_version == last_version:
                self.cond.wait(timeout)
            events = [e for e in self.events if e["seq"] > last_seq] if self.events and self.events[-1]["seq"] > last_seq else []
            state = self.tracker.snapshot() if self.state_version != last_version else None
            return events, state, self.state_version

    def _write_record(self, obj: dict[str, Any]) -> None:
        if self._record is None:
            return
        obj["t"] = round(time.time() - self.started_at, 3)
        try:
            self._record.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._record.flush()
        except (OSError, ValueError):
            self._record = None

    def close(self) -> None:
        if self._record is not None:
            try:
                self._record.close()
            except OSError:
                pass
            self._record = None
