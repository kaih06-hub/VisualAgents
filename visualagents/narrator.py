"""Smart mode: periodically asks an LLM which agent is working and what it is doing."""

from __future__ import annotations

import threading
import time

from . import llm
from .hub import Hub
from .patterns import clean


class Narrator:
    def __init__(self, hub: Hub, *, provider: str, model: str | None, interval: float = 4.0):
        self.hub = hub
        self.provider = provider
        self.model = model
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="visualagents-narrator", daemon=True)
        self._seen = 0
        self._failures = 0

    def start(self) -> "Narrator":
        self.thread.start()
        return self

    def _loop(self) -> None:
        agents = "\n".join(
            f"- {a['id']}: {a['name']} - {a['role']}" for a in self.hub.blueprint["agents"]
        ) or "- (no agents in the blueprint)"
        while not self.stop.wait(self.interval * (1 + min(self._failures, 5))):
            if self.hub.lines_seen == self._seen:
                continue
            self._seen = self.hub.lines_seen
            lines = [clean(x) for x in list(self.hub.recent_lines)[-40:]]
            lines = [x for x in lines if x.strip()]
            if not lines:
                continue
            active = self.hub.tracker.active or "none yet"
            prompt = (
                f"Agents:\n{agents}\n\nCurrently highlighted agent: {active}\n\n"
                "Most recent log lines (oldest first):\n<log>\n" + "\n".join(lines) + "\n</log>"
            )
            try:
                started = time.time()
                result = llm.complete_json(
                    provider=self.provider, model=self.model, system=llm.NARRATE_SYSTEM, prompt=prompt,
                    schema=llm.NARRATE_SCHEMA, effort="low", max_tokens=4000,
                )
                self._failures = 0
                self.hub.narrate(result.get("agent") or None, str(result.get("activity") or "").strip())
                if time.time() - started > self.interval * 3:
                    self.interval *= 1.5
            except llm.LLMError as exc:
                self._failures += 1
                if self._failures in (1, 5):
                    self.hub.system(f"Smart narration paused: {exc}")
