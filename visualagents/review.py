"""The LLM pass of a scan: check the static map against the source and return changes to it, not a new map.

The static map stays the base. The model may name and describe blocks, merge duplicates, drop blocks that are not
agents, and add the agents and handoffs static analysis missed - but every addition has to quote the line of code it
comes from, and blocks of a graph the code wires up itself stay on the map. So a model can fill gaps without
inventing agents, and cannot lose what the code states outright (the steps of a LangGraph graph, say).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from . import blueprint as bpmod
from . import llm
from .collector import CONFIG_EXTS, SourceFile, pack, redact

REMOVE_REASONS = ("plumbing", "alternative", "not_part_of_run", "duplicate")
FIXED_REMOVE_REASONS = {"alternative", "not_part_of_run"}  # a wired graph node is a workflow step, never plumbing
MAX_READS = 8
MAX_READ_LINES = 400
QUOTE_WINDOW = 25  # lines either side of the cited line; a quote found further away moves the line there

_STR_LIST = {"type": "array", "items": {"type": "string"}}
_EVIDENCE = {
    "file": {"type": "string", "description": "path of the file, as shown in the source"},
    "line": {"type": "integer", "description": "1-based line number"},
    "evidence": {"type": "string", "description": "a short exact quote of the code at that line, without the number"},
}
_DESCRIBED = {
    "id": {"type": "string"},
    "name": {"type": "string", "description": "display name"},
    "role": {"type": "string", "description": "one-line role, max ~8 words"},
    "description": {"type": "string", "description": "1-2 sentences: what this agent does and when"},
    "tools": _STR_LIST,
    "aliases": _STR_LIST,
    "keywords": _STR_LIST,
}


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


REVIEW_SCHEMA: dict[str, Any] = _obj({
    "read": {"type": "array", "items": _obj({"path": {"type": "string"}, "start": {"type": "integer"},
                                             "end": {"type": "integer"}})},
    "update": {"type": "array", "items": _obj(_DESCRIBED)},
    "remove": {"type": "array", "items": _obj({
        "id": {"type": "string"},
        "reason": {"type": "string", "enum": list(REMOVE_REASONS)},
        "merge_into": {"type": "string", "description": "for duplicate: the id of the block that stays, else ''"},
        "why": {"type": "string"},
    })},
    "add": {"type": "array", "items": _obj({**_DESCRIBED, **_EVIDENCE})},
    "add_edges": {"type": "array", "items": _obj({
        "from": {"type": "string"}, "to": {"type": "string"},
        "label": {"type": "string", "description": "what is handed over, max ~5 words"},
        "conditional": {"type": "boolean"}, **_EVIDENCE,
    })},
    "remove_edges": {"type": "array", "items": _obj({"from": {"type": "string"}, "to": {"type": "string"},
                                                     "why": {"type": "string"}})},
    "framework": {"type": "string", "description": "langgraph, crewai, autogen, openai-agents, google-adk, custom, ..."},
    "summary": {"type": "string", "description": "2-4 sentences describing the overall workflow"},
    "entry": {**_STR_LIST, "description": "ids of the agents that receive the user's query first"},
    "exits": {**_STR_LIST, "description": "ids of the agents that produce the final answer"},
    "steps": {"type": "array", "items": _obj({
        "order": {"type": "integer"}, "title": {"type": "string", "description": "2-4 words"},
        "agents": _STR_LIST, "description": {"type": "string"},
    })},
})


def check_and_add(
    static: dict[str, Any],
    files: list[SourceFile],
    facts: dict[str, Any],
    *,
    provider: str,
    model: str,
    max_chars: int,
    log: Callable[[str], None],
    rounds: int = 2,
) -> dict[str, Any]:
    """Ask the model to check `static` against the source; returns the changed map (raises llm.LLMError)."""
    fixed = set(facts.get("graph_nodes", []))
    facts = {k: v for k, v in facts.items() if k != "graph_nodes"}
    current = _current_map(static, fixed)
    head = f"Project name: {static['project']}\n\n<current_map>\n{current}\n</current_map>\n\n"
    if facts:
        head += f"<static_analysis_facts>\n{json.dumps(facts, indent=1)}\n</static_analysis_facts>\n\n"
    prefer = {a["file"] for a in static["agents"] if a.get("file")}
    budget = max(20_000, max_chars - len(head))
    source, included = pack(files, budget_chars=budget, prefer=prefer)
    cut = sum(1 for i in included if i.endswith(" (excerpt)"))
    log(f"Asking {provider}:{model} to check the map ({len(static['agents'])} block(s)) against the source "
        f"(outline + {len(included)} file(s)" + (f", {cut} cut to excerpts" if cut else "") + f", {len(source):,} chars)…")
    patch = _ask(provider, model, head + f"<source>\n{source}</source>\n\nReturn your changes to the map.")

    if rounds > 1 and patch.get("read"):
        extra, shown = serve_reads(patch["read"], files, budget=max_chars // 3)
        if extra:
            log(f"It asked to read {len(shown)} more range(s): {', '.join(shown[:6])}" + ("…" if len(shown) > 6 else ""))
            source, _ = pack(files, budget_chars=max(20_000, budget - len(extra)), prefer=prefer)
            prompt = (head + f"<source>\n{source}</source>\n\n<requested_code>\n{extra}</requested_code>\n\n"
                      "This is your last chance to read code: leave read empty. Return your changes to the map.")
            try:
                patch = _ask(provider, model, prompt)
            except llm.LLMError as exc:  # the first answer is still a complete set of changes
                log(f"The second round failed ({exc}) - using its first answer.")
    bp, notes = apply_changes(static, patch, fixed, files)
    for note in notes:
        log(note)
    return bp


def _ask(provider: str, model: str, prompt: str) -> dict[str, Any]:
    raw = llm.complete_json(provider=provider, model=model, system=llm.SCAN_SYSTEM, prompt=prompt, schema=REVIEW_SCHEMA)
    if not isinstance(raw, dict):
        raise llm.LLMError("the model did not return an object")
    return raw


def _current_map(static: dict[str, Any], fixed: set[str]) -> str:
    agents = [{"id": a["id"], "name": a["name"], "role": a["role"], "file": a["file"], "line": a["line"],
               "aliases": a["aliases"][:6], "fixed": a["id"] in fixed} for a in static["agents"]]
    edges = [{"from": e["from"], "to": e["to"], "label": e["label"], "conditional": e["conditional"],
              "fixed": e["from"] in fixed and e["to"] in fixed} for e in static["edges"]]
    return json.dumps({"framework": static["framework"], "agents": agents, "edges": edges,
                       "entry": static["entry"], "exits": static["exits"]}, indent=1)


# -- reading more -----------------------------------------------------------

def _find_file(path: str, files: list[SourceFile]) -> SourceFile | None:
    path = str(path or "").strip().lstrip("./").replace("\\", "/")
    if not path:
        return None
    for f in files:
        if f.rel.replace("\\", "/") == path:
            return f
    ends = [f for f in files if f.rel.replace("\\", "/").endswith("/" + path)]
    return ends[0] if len(ends) == 1 else None


def serve_reads(reads: list[Any], files: list[SourceFile], budget: int) -> tuple[str, list[str]]:
    """The line ranges the model asked for - only from files the scan collected (inside the project, not ignored,
    not secrets), redacted like the rest of the source and within `budget` characters."""
    parts, shown, used = [], [], 0
    for r in reads[:MAX_READS]:
        if not isinstance(r, dict):
            continue
        f = _find_file(r.get("path", ""), files)
        if f is None:
            continue
        lines = redact(f.text, config=f.path.suffix.lower() in CONFIG_EXTS).splitlines()
        start = max(1, bpmod._int(r.get("start")) or 1)
        end = min(len(lines), max(start, bpmod._int(r.get("end")) or start + MAX_READ_LINES - 1),
                  start + MAX_READ_LINES - 1)
        if start > len(lines):
            continue
        body = "\n".join(f"{i:>4}| {lines[i - 1][:1000]}" for i in range(start, end + 1))
        chunk = f'<file path="{f.rel}" lines="{start}-{end}">\n{body}\n</file>\n'
        if used + len(chunk) > budget:
            break
        parts.append(chunk)
        shown.append(f"{f.rel}:{start}-{end}")
        used += len(chunk)
    return "".join(parts), shown


# -- applying the changes ---------------------------------------------------

_SPACE = re.compile(r"[\s,]+")  # what formatters add and quotes drop: line breaks, indents, trailing commas
MIN_QUOTE = 6  # characters, spaces left out: enough to name something, not a lone bracket or keyword


def find_evidence(item: dict[str, Any], files: list[SourceFile]) -> tuple[str, int] | None:
    """(file, line) when the quoted code is really in that file - at the cited line or, failing that, anywhere in it
    (the line moves to the nearest place it appears). Spacing, line breaks and commas do not count, so a statement
    a formatter spread over several lines matches when quoted on one. None when the file or quote cannot be found."""
    f = _find_file(item.get("file", ""), files)
    quote = str(item.get("evidence") or "").strip().strip("`")
    quote = re.sub(r"(?m)^\s*\d+\s*\|\s?", "", quote)  # line numbers copied along with the code
    quote = _SPACE.sub("", quote)
    if f is None or len(quote) < MIN_QUOTE:
        return None
    flat, starts = [], []  # the file without whitespace, and where each line begins in it
    for line in redact(f.text, config=f.path.suffix.lower() in CONFIG_EXTS).splitlines():
        starts.append(sum(len(x) for x in flat))
        flat.append(_SPACE.sub("", line))
    text = "".join(flat)
    hits, at = [], text.find(quote)
    while at != -1:
        hits.append(_line_at(starts, at))
        at = text.find(quote, at + 1)
    if not hits:
        return None
    cited = bpmod._int(item.get("line"))
    best = min(hits, key=lambda n: abs(n - cited))
    return f.rel, best if abs(best - cited) > QUOTE_WINDOW or cited <= 0 else cited


def _line_at(starts: list[int], offset: int) -> int:
    """1-based line holding `offset` of the whitespace-free text (the last line that starts at or before it)."""
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def apply_changes(static: dict[str, Any], patch: dict[str, Any], fixed: set[str],
                  files: list[SourceFile]) -> tuple[dict[str, Any], list[str]]:
    """The static map with the model's changes, each checked; returns (blueprint, notes on what was refused)."""
    notes: list[str] = []
    agents: dict[str, dict[str, Any]] = {a["id"]: json.loads(json.dumps(a)) for a in static["agents"]}
    edges = [dict(e) for e in static["edges"]]

    def lookup(ref: Any) -> str | None:
        k = bpmod.norm_key(ref)
        if not k:
            return None
        for aid, a in agents.items():
            if k in {bpmod.norm_key(x) for x in (aid, a["name"], *a["aliases"])}:
                return aid
        return None

    def describe(a: dict[str, Any], d: dict[str, Any]) -> None:
        name = str(d.get("name") or "").strip()
        if name and name != a["name"]:
            if a["name"] not in a["aliases"]:
                a["aliases"].append(a["name"])
            a["name"] = name
        for k in ("role", "description"):
            if str(d.get(k) or "").strip():
                a[k] = str(d[k]).strip()
        for k in ("tools", "aliases"):
            for x in d.get(k) or []:
                if str(x).strip() and str(x).strip() not in a[k]:
                    a[k].append(str(x).strip())
        new_kw = [str(x).strip() for x in d.get("keywords") or [] if str(x).strip()]
        a["keywords"] = list(dict.fromkeys(new_kw + a["keywords"]))[:10]

    # agents the model adds - only with a quote of the code that defines them
    for d in patch.get("add") or []:
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or d.get("id") or "").strip()
        same = lookup(d.get("id")) or lookup(name)
        if same:  # already on the map: take it as a description of that block
            describe(agents[same], d)
            continue
        where = find_evidence(d, files)
        if where is None:
            notes.append(f"Left out {name or '?'}: the code it quotes was not found in {d.get('file') or '?'}.")
            continue
        aid = bpmod.slugify(d.get("id") or name)
        agents[aid] = {"id": aid, "name": name or aid, "role": "", "description": "", "tools": [], "file": where[0],
                       "line": where[1], "aliases": [], "keywords": []}
        describe(agents[aid], d)

    # descriptions of blocks already on the map
    for d in patch.get("update") or []:
        aid = lookup(d.get("id")) if isinstance(d, dict) else None
        if aid:
            describe(agents[aid], d)

    # blocks the model removes - never a wired graph node as plumbing, never the whole map
    removed: dict[str, str | None] = {}
    for d in patch.get("remove") or []:
        aid = lookup(d.get("id")) if isinstance(d, dict) else None
        if not aid or aid in removed:
            continue
        reason = str(d.get("reason") or "")
        target = lookup(d.get("merge_into")) if reason == "duplicate" else None
        if aid in fixed and reason not in FIXED_REMOVE_REASONS and not (target in fixed and target != aid):
            notes.append(f"Kept {agents[aid]['name']}: it is a node of the graph the code wires up.")
            continue
        if reason == "duplicate" and (not target or target == aid):
            continue  # nothing to merge into
        if reason not in REMOVE_REASONS:
            continue
        removed[aid] = target
    if removed and len(removed) >= len(agents):
        notes.append("Ignored the removals: they would have left no agents on the map.")
        removed = {}
    for aid, target in removed.items():
        seen = {aid}
        while target in removed and target not in seen:  # merged into a block that is itself merged away
            seen.add(target)
            target = removed[target]
        if target in seen:  # merges that go round in a circle: nothing to keep them in
            target = None
        gone = agents.pop(aid)
        if target and target in agents:
            t = agents[target]
            for x in [gone["name"], gone["id"], *gone["aliases"]]:
                if x not in t["aliases"] and x != t["name"]:
                    t["aliases"].append(x)
            t["tools"] += [x for x in gone["tools"] if x not in t["tools"]]
        for e in edges:
            for end in ("from", "to"):
                if e[end] == aid:
                    e[end] = target or ""
    edges = [e for e in edges if e["from"] and e["to"] and e["from"] != e["to"]]

    # arrows the model removes - not the graph's own wiring
    for d in patch.get("remove_edges") or []:
        if not isinstance(d, dict):
            continue
        src, dst = lookup(d.get("from")), lookup(d.get("to"))
        if src in fixed and dst in fixed:
            continue
        edges = [e for e in edges if not (e["from"] == src and e["to"] == dst)]

    # arrows the model adds - only with a quote of the code where the handoff happens
    for d in patch.get("add_edges") or []:
        if not isinstance(d, dict):
            continue
        src, dst = lookup(d.get("from")), lookup(d.get("to"))
        if not src or not dst or src == dst or any(e["from"] == src and e["to"] == dst for e in edges):
            continue
        if find_evidence(d, files) is None:
            notes.append(f"Left out the arrow {src} -> {dst}: the code it quotes was not found.")
            continue
        edges.append({"from": src, "to": dst, "label": str(d.get("label") or ""),
                      "conditional": bool(d.get("conditional"))})

    def ids(refs: Any) -> list[str]:
        return [x for x in dict.fromkeys(lookup(r) for r in refs or []) if x]

    raw = {
        "project": static["project"],
        "framework": str(patch.get("framework") or "").strip() or static["framework"],
        "summary": str(patch.get("summary") or "").strip() or static["summary"],
        "agents": list(agents.values()),
        "edges": edges,
        "entry": ids(patch.get("entry")) or [x for x in static["entry"] if x in agents],
        "exits": ids(patch.get("exits")) or [x for x in static["exits"] if x in agents],
        "steps": [{**s, "agents": ids(s.get("agents"))} for s in patch.get("steps") or [] if isinstance(s, dict)]
        or [{**s, "agents": [x for x in s["agents"] if x in agents]} for s in static["steps"]],
    }
    return bpmod.normalize(raw), notes
