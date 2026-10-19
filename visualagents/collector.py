"""Collects the source files of a project that are most likely to define agents.

Secrets never leave the machine: env/key files are skipped and anything that looks
like an API key is redacted before file contents are handed to an LLM.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", ".env", ".tox",
    ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", "site-packages",
    ".idea", ".vscode", ".next", ".turbo", "coverage", ".visualagents", ".ipynb_checkpoints",
    "target", "vendor", ".cache", "htmlcov", "migrations",
}
SOURCE_EXTS = {".py", ".ts", ".tsx", ".js", ".mjs", ".cjs", ".jsx", ".yaml", ".yml", ".toml", ".json", ".ipynb", ".md"}
SECRET_NAMES = re.compile(r"(^\.env)|(\.(pem|key|p12|pfx|crt)$)|(secret|credential|password|token)", re.I)
MAX_FILE_BYTES = 400_000

AGENT_HINTS = re.compile(
    r"\b(agent|agents|crew|crewai|autogen|langgraph|langchain|StateGraph|add_node|add_edge|handoff|"
    r"supervisor|swarm|tool|tools|llm|chat_model|ChatOpenAI|ChatAnthropic|prompt|system_message|"
    r"instructions|role|backstory|goal|task|workflow|orchestrat\w*|planner|executor|router|"
    r"GroupChat|AssistantAgent|Runner\.run|create_react_agent|function_call)\b",
    re.I,
)
NAME_HINTS = re.compile(r"(agent|crew|graph|workflow|flow|team|main|app|orchestr|pipeline|node|task|config)", re.I)

SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"""(?i)((?:api[_-]?key|secret|token|password)\s*[:=]\s*["'])([^"']{8,})(["'])"""),
]


@dataclass
class SourceFile:
    path: Path
    rel: str
    text: str
    score: float


def redact(text: str) -> str:
    for pat in SECRET_PATTERNS:
        if pat.groups >= 3:
            text = pat.sub(lambda m: f"{m.group(1)}<redacted>{m.group(3)}", text)
        else:
            text = pat.sub("<redacted>", text)
    return text


def _read(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(raw)
            cells = [
                "".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"
            ]
            raw = "\n\n# %% cell\n".join(cells)
        except (ValueError, AttributeError):
            return None
    return raw


def iter_files(root: Path):
    root = Path(root)
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".") and not d.endswith(".egg-info")
        )
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix.lower() not in SOURCE_EXTS or SECRET_NAMES.search(name):
                continue
            if name in {"package-lock.json", "poetry.lock", "uv.lock", "yarn.lock", "pnpm-lock.yaml"}:
                continue
            yield p


def collect(root: Path, max_files: int = 400) -> list[SourceFile]:
    """All candidate files, ranked by how agent-related they look (highest first)."""
    root = Path(root).resolve()
    base = root if root.is_dir() else root.parent
    out: list[SourceFile] = []
    for p in iter_files(root):
        text = _read(p)
        if not text or not text.strip():
            continue
        rel = str(p.relative_to(base))
        hits = len(AGENT_HINTS.findall(text))
        score = hits / (1 + len(text) / 4000) + hits * 0.05
        if NAME_HINTS.search(p.stem):
            score += 3
        if p.suffix == ".md":
            score = 5 if p.stem.lower() == "readme" and rel.count(os.sep) == 0 else score * 0.2
        if p.suffix in {".json", ".toml"}:
            score *= 0.3
        if "test" in rel.lower():
            score *= 0.3
        out.append(SourceFile(p, rel, text, score))
        if len(out) >= max_files * 4:
            break
    out.sort(key=lambda f: (-f.score, f.rel))
    return out[:max_files]


def pack(files: list[SourceFile], budget_chars: int = 180_000) -> tuple[str, list[str]]:
    """Concatenate the best files into one prompt-sized block. Returns (text, included rel paths)."""
    parts: list[str] = []
    included: list[str] = []
    used = 0
    for f in files:
        if f.score <= 0 and included:
            continue
        body = redact(f.text)
        numbered = "\n".join(f"{i:>4}| {line}" for i, line in enumerate(body.splitlines(), 1))
        chunk = f'<file path="{f.rel}">\n{numbered}\n</file>\n'
        if used + len(chunk) > budget_chars:
            remaining = budget_chars - used
            if remaining > 2000:
                parts.append(chunk[: remaining - 40] + "\n... (truncated)\n</file>\n")
                included.append(f.rel + " (truncated)")
            break
        parts.append(chunk)
        included.append(f.rel)
        used += len(chunk)
    return "".join(parts), included
