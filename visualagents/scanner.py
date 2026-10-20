"""Builds a blueprint for a project: static analysis first, then (optionally) an LLM pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from . import blueprint as bpmod
from . import llm
from .collector import collect, pack
from .static_scan import scan as static_scan


def scan_project(
    project: Path,
    *,
    mode: str = "auto",
    provider: str = "auto",
    model: str | None = None,
    max_chars: int = 180_000,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """mode: 'static' (offline), 'llm' (require an LLM), or 'auto' (LLM when a key is configured)."""
    project = Path(project).resolve()
    llm.load_dotenv_keys(project if project.is_dir() else project.parent)
    files = collect(project)
    log(f"Read {len(files)} candidate file(s) from {project}")
    facts: dict[str, Any] = {}
    static = static_scan(project, files, facts)
    log(f"Static analysis: {len(static['agents'])} agent(s), {len(static['edges'])} edge(s), framework={static['framework']}")
    if mode == "static":
        return static

    chosen = llm.detect_provider(provider)
    if chosen is None:
        if mode == "llm":
            raise llm.LLMError("no LLM credentials found - set ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY")
        log("No ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY found - using the static blueprint "
            "(add a key for a richer map; see `visualagents models`).")
        return static

    # files static analysis found agents in are sent in full first, whatever their keyword score
    source, included = pack(files, budget_chars=max_chars, prefer={a["file"] for a in static["agents"] if a.get("file")})
    hints = {k: static[k] for k in ("framework", "agents", "edges", "entry", "exits", "steps")}
    for a in hints["agents"]:
        a.pop("keywords", None)
    if facts:
        hints["facts"] = facts
    prompt = (
        f"Project name: {static['project']}\n\n"
        f"<static_analysis_hints>\n{json.dumps(hints, indent=1)}\n</static_analysis_hints>\n\n"
        f"<source>\n{source}</source>\n\n"
        "Return the blueprint for this project."
    )
    used_model = model or llm.default_model(chosen)
    cut = sum(1 for i in included if i.endswith(" (excerpt)"))
    log(f"Asking {chosen}:{used_model} to map the agents (outline of the project + {len(included)} file(s)"
        + (f", {cut} cut to excerpts" if cut else "") + f", {len(source):,} chars)…")
    try:
        raw = llm.complete_json(
            provider=chosen, model=used_model, system=llm.SCAN_SYSTEM, prompt=prompt, schema=bpmod.BLUEPRINT_JSON_SCHEMA
        )
    except llm.LLMError as exc:
        if mode == "llm":
            raise
        log(f"LLM scan failed ({exc}) - falling back to the static blueprint.")
        return static

    raw["project"] = raw.get("project") or static["project"]
    bp = bpmod.normalize(raw)
    if not bp["agents"]:
        log("The LLM found no agents - keeping the static blueprint.")
        return static
    _merge_static_aliases(bp, static)
    return bpmod.stamp(bp, f"{chosen}:{used_model}")


def _merge_static_aliases(bp: dict[str, Any], static: dict[str, Any]) -> None:
    """Static analysis knows exact node/variable names; keep them as aliases so log matching stays precise."""
    by_key = {}
    for a in bp["agents"]:
        for k in [a["id"], a["name"], *a["aliases"]]:
            by_key.setdefault(bpmod.norm_key(k), a)
    for s in static["agents"]:
        target = None
        for k in [s["id"], s["name"], *s["aliases"]]:
            target = by_key.get(bpmod.norm_key(k))
            if target:
                break
        if not target:
            continue
        for alias in [s["id"], s["name"], *s["aliases"]]:
            if alias not in target["aliases"] and alias != target["name"]:
                target["aliases"].append(alias)
        if not target["file"]:
            target["file"], target["line"] = s["file"], s["line"]
