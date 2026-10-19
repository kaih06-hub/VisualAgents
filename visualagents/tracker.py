"""The attribution engine: turns raw console lines and structured events into
"which agent is working, on what, and where are we in the workflow".

The tracker is single-threaded pure logic; `hub.Hub` serialises access to it.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from collections import deque
from typing import Any

from . import blueprint as bpmod
from .patterns import (
    ACTIVITY_KINDS,
    CONTINUES,
    KIND_PATTERNS,
    MARKER,
    SWITCH_PATTERNS,
    clean,
    strip_symbols,
)

START, END = "__start__", "__end__"
_SUFFIXES = ("agent", "node", "bot", "team", "crew", "assistant")
_TEMPLATE = re.compile(r"\{[^}]*\}")


class NameMatcher:
    """Resolves free-form names seen in logs ("Senior Data Researcher", "research_node") to agent ids."""

    def __init__(self, agents: list[dict[str, Any]]):
        self.keys: dict[str, set[str]] = {}
        self.prefixes: list[tuple[str, str]] = []
        for a in agents:
            self.add(a)

    def add(self, a: dict[str, Any]) -> None:
        names = [a["id"], a["name"], *a.get("aliases", [])]
        if a.get("role") and len(a["role"]) <= 60:
            names.append(a["role"])
        for n in names:
            n = _TEMPLATE.sub("", str(n))
            for k in {bpmod.norm_key(n), _strip_suffix(bpmod.norm_key(n))}:
                if len(k) >= 2:
                    self.keys.setdefault(k, set()).add(a["id"])
            disp = n.strip().lower()
            if len(disp) >= 4:
                self.prefixes.append((disp, a["id"]))
        self.prefixes.sort(key=lambda p: -len(p[0]))

    def resolve(self, text: str | None) -> str | None:
        if not text:
            return None
        k = bpmod.norm_key(_TEMPLATE.sub("", text))
        if not k:
            return None
        for cand in (k, _strip_suffix(k)):
            ids = self.keys.get(cand)
            if ids and len(ids) == 1:
                return next(iter(ids))
        # containment either way, longest key wins
        best: tuple[int, set[str]] = (0, set())
        for key, ids in self.keys.items():
            if len(key) >= 4 and key in k or len(k) >= 5 and k in key:
                score = len(key) if key in k else len(k)
                if score > best[0]:
                    best = (score, set(ids))
                elif score == best[0]:
                    best[1].update(ids)
        if best[0] and len(best[1]) == 1:
            return next(iter(best[1]))
        close = difflib.get_close_matches(k, list(self.keys), n=2, cutoff=0.86)
        if close and len(self.keys[close[0]]) == 1:
            return next(iter(self.keys[close[0]]))
        return None

    def prefix(self, line: str) -> str | None:
        low = line.lower()
        for disp, aid in self.prefixes:
            if low.startswith(disp) and (len(low) == len(disp) or not low[len(disp)].isalnum()):
                return aid
        return None


def _strip_suffix(k: str) -> str:
    for s in _SUFFIXES:
        if k.endswith(s) and len(k) > len(s) + 2:
            return k[: -len(s)]
        if k.startswith(s) and len(k) > len(s) + 2:
            return k[len(s):]
    return k


def _short(text: str, n: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


class Tracker:
    def __init__(self, blueprint: dict[str, Any], *, clock=time.time):
        self.clock = clock
        self.bp = blueprint
        self.matcher = NameMatcher(blueprint["agents"])
        self.tool_owner: dict[str, set[str]] = {}
        for a in blueprint["agents"]:
            for t in a["tools"]:
                self.tool_owner.setdefault(bpmod.norm_key(t), set()).add(a["id"])
        self.blueprint_edges = {(e["from"], e["to"]) for e in blueprint["edges"]}
        self.keyword_index = [
            (a["id"], [k.lower() for k in a.get("keywords", []) if len(k) >= 4]) for a in blueprint["agents"]
        ]
        self.seq = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=5000)
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self) -> None:
        self.agents: dict[str, dict[str, Any]] = {
            a["id"]: self._agent_state() for a in self.bp["agents"]
        }
        self.extra_agents: list[dict[str, Any]] = getattr(self, "extra_agents", [])
        for a in self.extra_agents:
            self.agents[a["id"]] = self._agent_state()
        self.active: str | None = None
        self.stack: list[str] = []
        self.path: list[dict[str, Any]] = []
        self.edge_counts: dict[str, int] = {}
        self.dynamic_edges: set[tuple[str, str]] = set()
        self.step_index = -1
        self.step_seen: set[str] = set()
        self.precise = False
        self.last_kind: str | None = None
        self.last_kind_at = 0.0
        self.run: dict[str, Any] = {"status": "idle", "started": None, "ended": None, "exit_code": None, "command": ""}

    @staticmethod
    def _agent_state() -> dict[str, Any]:
        return {"status": "idle", "activity": "", "activity_kind": "", "activity_at": 0.0, "events": 0,
                "tools": 0, "visits": 0, "errors": 0, "first_seen": None, "last_seen": None}

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "agents": self.agents,
            "extra_agents": self.extra_agents,
            "path": self.path[-300:],
            "path_total": len(self.path),
            "edge_counts": self.edge_counts,
            "dynamic_edges": [list(e) for e in sorted(self.dynamic_edges)],
            "step_index": self.step_index,
            "run": self.run,
            "precise": self.precise,
            "seq": self.seq,
        }

    # -- run lifecycle -----------------------------------------------------
    def run_started(self, command: str = "") -> list[dict[str, Any]]:
        self.reset()
        now = self.clock()
        self.run.update(status="running", started=now, command=command)
        return [self._event("status", None, f"Run started: {command}" if command else "Run started", source="system")]

    def run_finished(self, code: int | None = 0) -> list[dict[str, Any]]:
        now = self.clock()
        ok = code in (0, None)
        self.run.update(status="finished" if ok else "failed", ended=now, exit_code=code)
        out = []
        if self.active:
            st = self.agents[self.active]
            st["status"] = "done" if ok else "error"
            self._count_edge(self.active, END)
            if self.path:
                self.path[-1]["end"] = now
            self.active = None
        for st in self.agents.values():
            if st["status"] in {"active", "waiting"}:
                st["status"] = "done"
        self.stack = []
        out.append(self._event("status", None, "Run finished" if ok else f"Run failed (exit code {code})", source="system"))
        return out

    # -- input -------------------------------------------------------------
    def ingest_line(self, raw: str, stream: str = "stdout") -> list[dict[str, Any]]:
        if MARKER.strip() in raw:
            idx = raw.find(MARKER.strip())
            try:
                payload = json.loads(raw[idx + len(MARKER.strip()):].strip())
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                return self.ingest_structured(payload)
        line = clean(raw)
        if not line.strip():
            return []
        if self.run["status"] == "idle":
            self.run.update(status="running", started=self.clock())
        out: list[dict[str, Any]] = []
        text = strip_symbols(line.strip())
        kind = "log"
        target: str | None = None
        tool_name: str | None = None
        continuation = False
        now = self.clock()

        # 1. explicit agent switches ("# Agent: X", "researcher (to writer):", "Handing off to X", ...)
        for pat, strength, pkind in SWITCH_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            name = m.group("agent").strip()
            aid = self.matcher.resolve(name)
            if aid is None and strength == "strong" and not self.precise:
                aid = self._discover(name)
            if aid is None:
                continue
            target, kind = aid, pkind
            to = m.groupdict().get("to")
            if pkind == "message" and to:
                text = f"→ {to.strip()}"
            elif pkind == "log":
                text = text[m.end("agent"):].lstrip(" :>|»-]").strip() or text
            break

        # 2. what kind of line is it? (Thought / Action / Observation / Final Answer / error ...)
        if kind in {"log", "status"}:
            for pat, pkind in KIND_PATTERNS:
                m = pat.search(text)
                if m:
                    if kind == "log" or pkind == "error":
                        kind = pkind
                    tool_name = m.groupdict().get("tool")
                    break
            else:
                if target is None and self.last_kind in CONTINUES and now - self.last_kind_at < 3.0:
                    kind, continuation = self.last_kind, True
                elif target is None and not self.precise:
                    target = self.matcher.prefix(text)
        if tool_name and target is None and not self.precise:
            target = self._tool_agent(tool_name)
        if target is None and self.active is None and not self.precise:
            target = self._keyword_guess(text)

        # 3. apply
        if target and target != self.active and not self.precise:
            out += self._activate(target, source="log")
        agent = self.active
        if kind == "handoff":  # the switch itself was emitted by _activate; keep the raw line as context
            kind = "status"
        out.append(self._event(kind, agent, line.strip(), source="log", stream=stream, tool=tool_name))
        if not continuation:
            self._touch(agent, kind, line.strip(), tool_name)
            self.last_kind = kind
        elif agent in self.agents:
            self.agents[agent]["events"] += 1
        self.last_kind_at = now
        return out

    def ingest_structured(self, p: dict[str, Any]) -> list[dict[str, Any]]:
        """Precise events from the SDK / auto-instrumentation (see visualagents_runtime)."""
        typ = str(p.get("type") or p.get("event") or "log")
        if self.run["status"] == "idle":
            self.run.update(status="running", started=self.clock())
        soft = bool(p.get("soft"))
        name = p.get("agent")
        aid = self._lookup(name, create=not soft) if name else None
        out: list[dict[str, Any]] = []
        text = str(p.get("text") or p.get("output") or p.get("input") or "")

        if typ == "run_start":
            return self.run_started(str(p.get("command") or text))
        if typ == "run_end":
            return self.run_finished(p.get("code", 0))
        if typ == "agent_start":
            if aid is None:
                return []
            if not self.precise:
                self._rebase_for_precise()
            self.precise = True
            parent = self.active if self.active and self.active != aid and self.agents[self.active]["status"] == "active" else None
            out += self._activate(aid, source="sdk")
            if parent:  # nested call (agent-as-tool, supervisor sub-call): parent waits for us
                self.stack.append(parent)
                self.agents[parent]["status"] = "waiting"
            if text:
                out.append(self._event("task", aid, text, source="sdk"))
                self._touch(aid, "task", text)
            return out
        if typ == "agent_end":
            if aid is None:
                return []
            st = self.agents[aid]
            if text:
                out.append(self._event("output", aid, text, source="sdk"))
                self._touch(aid, "output", text)
            self.stack = [s for s in self.stack if s != aid]
            if self.active == aid:
                parent = self.stack.pop() if self.stack else None
                if parent:
                    out += self._activate(parent, source="sdk", returning=True)
                else:
                    st["status"] = "done"
            elif st["status"] in {"active", "waiting"}:
                st["status"] = "done"
            return out
        if typ == "handoff":
            dst = self._lookup(p.get("to"), create=True)
            src = self._lookup(p.get("from"), create=True)
            if src and src != self.active:
                out += self._activate(src, source="sdk")
            if dst:
                out += self._activate(dst, source="sdk", reason=text)
            return out

        if aid and aid != self.active and not soft:
            out += self._activate(aid, source="sdk")
        agent = aid or self.active
        kind = {
            "thought": "thought", "llm_end": "thought", "tool_start": "tool", "tool_end": "observation",
            "output": "output", "error": "error", "log": "log", "task": "task", "message": "message",
        }.get(typ, "log")
        tool = p.get("tool")
        if kind == "tool" and tool:
            text = f"{tool}({_short(text, 300)})" if text else str(tool)
        if not text:
            return out
        out.append(self._event(kind, agent, text, source="sdk", tool=tool))
        self._touch(agent, kind, text, tool if kind == "tool" else None)
        return out

    def apply_narration(self, agent: str | None, activity: str) -> list[dict[str, Any]]:
        """An LLM summary of recent output (smart mode)."""
        aid = self.matcher.resolve(agent) if agent else None
        out: list[dict[str, Any]] = []
        if aid and aid != self.active and not self.precise:
            out += self._activate(aid, source="llm")
        target = aid or self.active
        if target and activity:
            out.append(self._event("narration", target, activity, source="llm"))
            st = self.agents[target]
            st["activity"], st["activity_kind"], st["activity_at"] = _short(activity), "narration", self.clock()
        return out

    # -- internals ---------------------------------------------------------
    def _rebase_for_precise(self) -> None:
        """Instrumented events just started: drop whatever the log heuristics guessed before them."""
        for st in self.agents.values():
            st.update(status="idle", visits=0)
        self.active = None
        self.stack = []
        self.path = []
        self.edge_counts = {}
        self.dynamic_edges = set()
        self.step_index = -1
        self.step_seen = set()

    def _lookup(self, name: Any, create: bool) -> str | None:
        if not name:
            return None
        aid = self.matcher.resolve(str(name))
        if aid is None and create:
            aid = self._discover(str(name))
        return aid

    def _discover(self, name: str) -> str | None:
        """An agent the blueprint missed: add it so it still shows up on the map."""
        name = _TEMPLATE.sub("", name).strip(" :#*-")
        if not name or len(name) > 60:
            return None
        aid = bpmod.slugify(name)
        existing = {a["id"] for a in self.bp["agents"]} | {a["id"] for a in self.extra_agents}
        if aid in existing:
            return aid
        agent = {"id": aid, "name": name, "role": "discovered at runtime", "description": "", "tools": [],
                 "file": "", "line": 0, "aliases": [name], "keywords": [], "dynamic": True}
        self.extra_agents.append(agent)
        self.agents[aid] = self._agent_state()
        self.matcher.add(agent)
        return aid

    def _tool_agent(self, tool: str) -> str | None:
        owners = self.tool_owner.get(bpmod.norm_key(tool))
        if not owners or (self.active in owners):
            return None
        return next(iter(owners)) if len(owners) == 1 else None

    def _keyword_guess(self, text: str) -> str | None:
        low = text.lower()
        scores = [(sum(1 for k in kws if k in low), aid) for aid, kws in self.keyword_index]
        scores = [s for s in scores if s[0] > 0]
        if not scores:
            return None
        scores.sort(reverse=True)
        if len(scores) > 1 and scores[0][0] == scores[1][0]:
            return None
        return scores[0][1] if scores[0][0] >= 2 else None

    def _activate(self, aid: str, *, source: str, reason: str = "", returning: bool = False) -> list[dict[str, Any]]:
        if aid == self.active:
            return []
        now = self.clock()
        prev = self.active
        if prev:
            pst = self.agents[prev]
            if pst["status"] == "active":
                pst["status"] = "done"
            if self.path:
                self.path[-1]["end"] = now
        self._count_edge(prev or START, aid)
        st = self.agents.setdefault(aid, self._agent_state())
        st["status"] = "active"
        st["visits"] += 1
        st["first_seen"] = st["first_seen"] or now
        st["last_seen"] = now
        self.active = aid
        self.path.append({"agent": aid, "start": now, "end": None, "from": prev})
        self._advance_step(aid)
        frm = self._name(prev) if prev else "start"
        text = f"{frm} → {self._name(aid)}" + (" (returns)" if returning else "")
        if reason and reason not in text:
            text += f" · {_short(reason, 120)}"
        return [self._event("handoff", aid, text, source=source, frm=prev or START, to=aid)]

    def _count_edge(self, src: str, dst: str) -> None:
        key = f"{src}->{dst}"
        self.edge_counts[key] = self.edge_counts.get(key, 0) + 1
        if src not in (START,) and dst not in (END,) and (src, dst) not in self.blueprint_edges:
            self.dynamic_edges.add((src, dst))

    def _advance_step(self, aid: str) -> None:
        steps = self.bp["steps"]
        cands = [i for i, s in enumerate(steps) if aid in s["agents"]]
        if not cands:
            return
        cur = self.step_index
        target = None
        if cur in cands:
            pass
        elif cur < 0 or len(cands) == 1:
            target = cands[0]  # first step, a forward jump, or looping back to an earlier phase
        elif cur + 1 in cands and set(steps[cur]["agents"]) <= self.step_seen:
            # agents that recur across steps (supervisors, routers, revisers) only move on once
            # everyone in the current step has had their turn
            target = cur + 1
        if target is None:
            self.step_seen.add(aid)
        else:
            self.step_index = target
            self.step_seen = {aid}

    def _touch(self, aid: str | None, kind: str, text: str, tool: str | None = None) -> None:
        if not aid or aid not in self.agents:
            return
        st = self.agents[aid]
        now = self.clock()
        st["events"] += 1
        st["last_seen"] = now
        if kind == "tool":
            st["tools"] += 1
        if kind == "error":
            st["errors"] += 1
        rank = ACTIVITY_KINDS.get(kind, 0)
        cur = ACTIVITY_KINDS.get(st["activity_kind"], 0)
        if rank and (rank >= cur or now - st["activity_at"] > 2.5):
            label = text
            if kind == "tool" and tool:
                label = f"Using {tool}"
            label = re.sub(r"^#+\s*", "", label)
            label = re.sub(r"^(?:Thought|Thinking|Final Answer|Answer|Task|Output)\s*:\s*", "", label, flags=re.I)
            st["activity"], st["activity_kind"], st["activity_at"] = _short(label), kind, now

    def _name(self, aid: str | None) -> str:
        for a in self.bp["agents"] + self.extra_agents:
            if a["id"] == aid:
                return a["name"]
        return str(aid)

    def _event(self, kind: str, agent: str | None, text: str, *, source: str, **data: Any) -> dict[str, Any]:
        self.seq += 1
        ev = {"seq": self.seq, "ts": self.clock(), "kind": kind, "agent": agent, "text": text[:4000], "source": source}
        extra = {k: v for k, v in data.items() if v is not None}
        if extra:
            ev["data"] = extra
        self.events.append(ev)
        if agent:
            self.agents.setdefault(agent, self._agent_state())
        return ev
