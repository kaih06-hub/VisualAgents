"""The blueprint: a static map of the agents in a project and how work flows between them.

A blueprint is plain JSON so it can be hand-edited, committed, or produced by any tool:

    {
      "version": 1,
      "project": "research-crew",
      "framework": "crewai",
      "summary": "...",
      "agents": [{"id": "researcher", "name": "Researcher", "role": "...", ...}],
      "edges":  [{"from": "planner", "to": "researcher", "label": "plan", "conditional": false}],
      "entry":  ["planner"],
      "exits":  ["writer"],
      "steps":  [{"order": 1, "title": "Plan", "agents": ["planner"], "description": "..."}]
    }
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATE_DIR = ".visualagents"

# JSON schema handed to the LLM as a structured-output format. Every property is
# required (structured outputs reject optional keys); "unknown" values are empty strings.
_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "short snake_case identifier, e.g. 'researcher'"},
        "name": {"type": "string", "description": "display name, e.g. 'Researcher'"},
        "role": {"type": "string", "description": "one-line role, max ~8 words"},
        "description": {"type": "string", "description": "1-2 sentences: what this agent does and when"},
        "tools": {"type": "array", "items": {"type": "string"}, "description": "tool / function names it can call"},
        "file": {"type": "string", "description": "relative path where the agent is defined"},
        "line": {"type": "integer", "description": "1-based line of the definition, 0 if unknown"},
        "aliases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "other strings that identify this agent in logs: variable names, node names, role titles",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "distinctive words/phrases this agent is likely to print while working",
        },
    },
    "required": ["id", "name", "role", "description", "tools", "file", "line", "aliases", "keywords"],
    "additionalProperties": False,
}

BLUEPRINT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "project": {"type": "string"},
        "framework": {
            "type": "string",
            "description": "langgraph, langchain, crewai, autogen, openai-agents, swarm, llamaindex, smolagents, custom, ...",
        },
        "summary": {"type": "string", "description": "2-4 sentences describing the overall workflow"},
        "agents": {"type": "array", "items": _AGENT_SCHEMA},
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "agent id"},
                    "to": {"type": "string", "description": "agent id"},
                    "label": {"type": "string", "description": "what is handed over, max ~5 words"},
                    "conditional": {"type": "boolean"},
                },
                "required": ["from", "to", "label", "conditional"],
                "additionalProperties": False,
            },
        },
        "entry": {"type": "array", "items": {"type": "string"}, "description": "agent ids that receive the user query first"},
        "exits": {"type": "array", "items": {"type": "string"}, "description": "agent ids that produce the final answer"},
        "steps": {
            "type": "array",
            "description": "the general blueprint as ordered phases",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "title": {"type": "string", "description": "2-4 words"},
                    "agents": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["order", "title", "agents", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["project", "framework", "summary", "agents", "edges", "entry", "exits", "steps"],
    "additionalProperties": False,
}


def slugify(text: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(text).strip())
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "agent"


def norm_key(text: str) -> str:
    """Loose key for matching names in logs: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def state_dir(project: Path) -> Path:
    return Path(project) / STATE_DIR


def blueprint_path(project: Path) -> Path:
    return state_dir(project) / "blueprint.json"


def load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return normalize(json.load(fh))


def save(bp: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ignore = path.parent / ".gitignore"
    if path.parent.name == STATE_DIR and not ignore.exists():
        ignore.write_text("runs/\n", encoding="utf-8")
    path.write_text(json.dumps(bp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def empty(project: str = "project") -> dict[str, Any]:
    return normalize({"project": project, "framework": "unknown", "agents": []})


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Make any blueprint-ish dict safe to consume: unique ids, resolved references, ordered steps."""
    bp: dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "project": str(raw.get("project") or "project"),
        "framework": str(raw.get("framework") or "unknown"),
        "summary": str(raw.get("summary") or ""),
        "generated_by": str(raw.get("generated_by") or ""),
        "generated_at": str(raw.get("generated_at") or ""),
        "agents": [],
        "edges": [],
        "entry": [],
        "exits": [],
        "steps": [],
    }

    lookup: dict[str, str] = {}  # norm_key(any name) -> id
    used: set[str] = set()
    for a in raw.get("agents") or []:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name") or a.get("id") or "Agent").strip()
        aid = slugify(a.get("id") or name)
        base, n = aid, 2
        while aid in used:
            aid, n = f"{base}_{n}", n + 1
        used.add(aid)
        source = a.get("source") if isinstance(a.get("source"), dict) else {}
        agent = {
            "id": aid,
            "name": name,
            "role": str(a.get("role") or ""),
            "description": str(a.get("description") or ""),
            "tools": _str_list(a.get("tools")),
            "file": str(a.get("file") or source.get("file") or ""),
            "line": _int(a.get("line") or source.get("line")),
            "aliases": _str_list(a.get("aliases")),
            "keywords": _str_list(a.get("keywords")),
        }
        bp["agents"].append(agent)
        for key in [aid, name, *agent["aliases"]]:
            k = norm_key(key)
            if k and k not in lookup:
                lookup[k] = aid

    def resolve(ref: Any) -> str | None:
        return lookup.get(norm_key(ref)) if ref is not None else None

    seen_edges: set[tuple[str, str]] = set()
    for e in raw.get("edges") or []:
        if not isinstance(e, dict):
            continue
        src, dst = resolve(e.get("from") or e.get("source")), resolve(e.get("to") or e.get("target"))
        if not src or not dst or (src, dst) in seen_edges:
            continue
        seen_edges.add((src, dst))
        bp["edges"].append(
            {"from": src, "to": dst, "label": str(e.get("label") or ""), "conditional": bool(e.get("conditional"))}
        )

    ids = [a["id"] for a in bp["agents"]]
    bp["entry"] = _dedupe(r for r in map(resolve, _as_list(raw.get("entry"))) if r)
    bp["exits"] = _dedupe(r for r in map(resolve, _as_list(raw.get("exits"))) if r)
    if not bp["entry"] and ids:
        targets = {e["to"] for e in bp["edges"]}
        bp["entry"] = [i for i in ids if i not in targets][:1] or ids[:1]
    if not bp["exits"] and ids:
        sources = {e["from"] for e in bp["edges"]}
        bp["exits"] = [i for i in ids if i not in sources][-1:] or ids[-1:]

    steps = []
    for i, s in enumerate(raw.get("steps") or []):
        if not isinstance(s, dict):
            continue
        agents = _dedupe(r for r in map(resolve, _as_list(s.get("agents"))) if r)
        steps.append(
            {
                "order": _int(s.get("order")) or i + 1,
                "title": str(s.get("title") or f"Step {i + 1}"),
                "agents": agents,
                "description": str(s.get("description") or ""),
            }
        )
    steps.sort(key=lambda s: s["order"])
    if not steps and ids:
        steps = [
            {"order": i + 1, "title": a["name"], "agents": [a["id"]], "description": a["role"]}
            for i, a in enumerate(_topo_order(bp))
        ]
    for i, s in enumerate(steps):
        s["order"] = i + 1
    bp["steps"] = steps
    return bp


def _topo_order(bp: dict[str, Any]) -> list[dict[str, Any]]:
    """Agents in BFS order from the entry points; unreachable agents appended."""
    by_id = {a["id"]: a for a in bp["agents"]}
    out: dict[str, None] = {}
    queue = list(bp["entry"])
    while queue:
        cur = queue.pop(0)
        if cur in out or cur not in by_id:
            continue
        out[cur] = None
        queue.extend(e["to"] for e in bp["edges"] if e["from"] == cur)
    for aid in by_id:
        out.setdefault(aid, None)
    return [by_id[i] for i in out]


def stamp(bp: dict[str, Any], generated_by: str) -> dict[str, Any]:
    bp["generated_by"] = generated_by
    bp["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return bp


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _str_list(v: Any) -> list[str]:
    return _dedupe(str(x).strip() for x in _as_list(v) if str(x).strip())


def _dedupe(items: Any) -> list[Any]:
    return list(dict.fromkeys(items))


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0
