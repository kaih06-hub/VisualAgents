"""Builds a blueprint for a project: static analysis first, then (optionally) an LLM pass that checks it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import blueprint as bpmod
from . import llm, review
from .collector import collect
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
    folder = project if project.is_dir() else project.parent
    ignored = llm.load_dotenv_keys(folder)
    if ignored:
        log(f"Ignored {', '.join(ignored)} in {folder / '.env'}: where requests go and which provider is used come "
            "from your environment or `visualagents setup`, never from the project being scanned.")
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

    # The model checks the static map and returns changes to it (review.py): the map stays the base, additions need
    # a quote of the code they come from, and a graph's own nodes stay. The static map is the fallback.
    used_model = model or llm.default_model(chosen)
    try:
        bp = review.check_and_add(static, files, facts, provider=chosen, model=used_model, max_chars=max_chars, log=log)
    except llm.LLMError as exc:
        if mode == "llm":
            raise
        log(f"LLM scan failed ({exc}) - showing the static map instead.")
        return static
    if not bp["agents"]:
        log("The checked map has no agents - keeping the static map.")
        return static
    return bpmod.stamp(bp, f"{chosen}:{used_model}")
