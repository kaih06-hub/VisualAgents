"""Collects the source files of a project that are most likely to define agents.

Secrets never leave the machine: files git ignores (local config, data, user documents)
and env/key files are skipped, and anything that looks like an API key is redacted
before file contents are handed to an LLM.

What the LLM sees (pack): an outline of every source file first - its classes, functions and prompt
personas with line numbers, a few percent of the code's size - then as many files as fit in full, agent
definitions first, with big files cut down to the lines around their classes, prompts and model calls.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
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
# Build-tool configs say nothing about agents but would win ranking slots for "config" in their name.
TOOL_CONFIGS = re.compile(
    r"^(\.?(eslint|prettier|postcss|tailwind|next|vite|vitest|webpack|babel|jest|rollup|svelte|nuxt|"
    r"astro|stylelint|commitlint|lint-staged)(\.config)?\.(js|cjs|mjs|ts|json|ya?ml)|"
    r"tsconfig(\.[\w-]+)?\.json|components\.json)$",
    re.I,
)
CONFIG_EXTS = {".yaml", ".yml", ".toml", ".json", ".md"}
MAX_FILE_BYTES = 400_000

AGENT_HINTS = re.compile(
    r"\b(agent|agents|crew|crewai|autogen|langgraph|langchain|StateGraph|add_node|add_edge|handoff|"
    r"supervisor|swarm|tool|tools|llm|chat_model|ChatOpenAI|ChatAnthropic|prompt|system_message|"
    r"instructions|role|backstory|goal|task|workflow|orchestrat\w*|planner|executor|router|"
    r"GroupChat|AssistantAgent|Runner\.run|create_react_agent|function_call)\b",
    re.I,
)
NAME_HINTS = re.compile(r"(agent|crew|graph|workflow|flow|team|main|app|orchestr|pipeline|node|task|config)", re.I)
# Tests restate agents with mocks and fixtures: never worth budget the real definitions could use.
TEST_PATH = re.compile(r"(^|/)(tests?|testing|__tests__)/|(^|/)(test_[^/]*|[^/]*_test\.py|conftest\.py|"
                       r"[^/]*\.(test|spec)\.[jt]sx?)$")
# Classes that are agents by name, or by what they inherit from (directly or through the project's own classes).
AGENT_CLASS_NAME = re.compile(r"(Agent|Expert|Persona)$")
AGENT_BASES = {"Agent", "ChatAgent", "ConversableAgent", "AssistantAgent", "UserProxyAgent", "Role",
               "dspy.Module", "dspy.Signature"}
PERSONA = re.compile(r"\bYou are (?:an? |the )?[^\n.!?\"']{3,100}")
AGENT_FILE_BONUS = 8  # a file defining agent classes - and so each sibling subclass of the same base - outranks helpers

SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),  # JWTs (Qdrant, Supabase...)
    # quoted values of anything *named* like a secret: KEY = "...", "api_key": "...", password='...'
    re.compile(
        r"""(?i)(\b[\w.\-]*(?:key|secret|token|passw(?:or)?d|pwd|credential)s?\b["']?\s*[:=]\s*["'])"""
        r"""([^"'\n]{6,})(["'])"""
    ),
]
# Unquoted `name: value` / `NAME=value` lines; only in config-like files, where code would be misread.
CONFIG_SECRET = re.compile(
    r"""(?im)^(\s*[\w.\-]*(?:key|secret|token|passw(?:or)?d|pwd|credential)s?\s*[:=][ \t]*)([^\s"'#]{6,})[ \t]*$"""
)


@dataclass
class SourceFile:
    path: Path
    rel: str
    text: str
    score: float
    defines_agent: bool = False


def redact(text: str, config: bool = False) -> str:
    for pat in SECRET_PATTERNS:
        if pat.groups >= 3:
            text = pat.sub(lambda m: f"{m.group(1)}<redacted>{m.group(3)}", text)
        else:
            text = pat.sub("<redacted>", text)
    if config:
        text = CONFIG_SECRET.sub(lambda m: f"{m.group(1)}<redacted>", text)
    return text


def _git_visible(root: Path) -> set[Path] | None:
    """Files git would not ignore under root (tracked + untracked), or None outside a git repo."""
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return {(root / rel).resolve() for rel in res.stdout.decode("utf-8", "replace").split("\0") if rel}


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
    visible = _git_visible(root.resolve())
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".") and not d.endswith(".egg-info")
        )
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix.lower() not in SOURCE_EXTS or SECRET_NAMES.search(name) or TOOL_CONFIGS.match(name):
                continue
            if name in {"package-lock.json", "poetry.lock", "uv.lock", "yarn.lock", "pnpm-lock.yaml"}:
                continue
            if visible is not None and p.resolve() not in visible:
                continue  # gitignored: local config with credentials, data, the user's own documents
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
    for f in _mark_agent_files(out):
        f.score += AGENT_FILE_BONUS
    out.sort(key=lambda f: (-f.score, f.rel))
    return out[:max_files]


def _classes(f: SourceFile) -> list[ast.ClassDef]:
    if f.path.suffix.lower() not in (".py", ".ipynb"):
        return []
    try:
        tree = ast.parse(f.text)
    except (SyntaxError, ValueError):
        return []
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def _base_names(cls: ast.ClassDef) -> list[str]:
    out = []
    for b in cls.bases:
        node = b.value if isinstance(b, ast.Subscript) else b  # Generic[T], BaseAgent[State]
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        out.append(".".join(reversed(parts)))
    return [b for b in out if b]


def _mark_agent_files(files: list[SourceFile]) -> list[SourceFile]:
    """Files that define an agent class: one named like an agent, or one inheriting - through the project's own
    classes too - from an agent base (MungerAgent(LLMAgent), Planner(Role), TopicExpert(dspy.Module))."""
    bases: dict[str, set[str]] = {}
    per_file: list[tuple[SourceFile, list[str]]] = []
    for f in files:
        names = []
        for cls in _classes(f):
            bases.setdefault(cls.name, set()).update(_base_names(cls))
            names.append(cls.name)
        per_file.append((f, names))

    memo: dict[str, bool] = {}

    def agentish(name: str, seen: frozenset = frozenset()) -> bool:
        if name in memo:
            return memo[name]
        if name in seen:
            return False
        hit = bool(AGENT_CLASS_NAME.search(name)) or any(
            b in AGENT_BASES or b.split(".")[-1] in AGENT_BASES or agentish(b.split(".")[-1], seen | {name})
            for b in bases.get(name, ()))
        memo[name] = hit
        return hit

    marked = []
    for f, names in per_file:
        if any(agentish(n) for n in names) and not TEST_PATH.search(f.rel.replace(os.sep, "/")):
            f.defines_agent = True
            marked.append(f)
    return marked


def _outline_lines(f: SourceFile) -> list[str]:
    """Classes (with bases and methods), top-level functions and prompt personas, each with its line number."""
    lines: list[str] = []
    suffix = f.path.suffix.lower()
    if suffix in (".py", ".ipynb"):
        try:
            tree = ast.parse(f.text)
        except (SyntaxError, ValueError):
            tree = None
        for node in tree.body if tree else []:
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                bases = ", ".join(_base_names(node))
                lines.append(f"{node.lineno:>5} class {node.name}({bases})" + (f": {', '.join(methods[:12])}" if methods else ""))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.append(f"{node.lineno:>5} def {node.name}")
    elif suffix in (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"):
        for i, line in enumerate(f.text.splitlines(), 1):
            m = re.match(r"\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(class|function)\s+(\w+)", line)
            if m:
                lines.append(f"{i:>5} {m.group(1)} {m.group(2)}")
    elif suffix in (".yaml", ".yml"):
        for i, line in enumerate(f.text.splitlines(), 1):
            m = re.match(r"([A-Za-z_][\w-]*):\s*(#.*)?$", line)
            if m:
                lines.append(f"{i:>5} {m.group(1)}:")
    if suffix in (".py", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".yaml", ".yml"):
        personas = 0
        for i, line in enumerate(f.text.splitlines(), 1):
            m = PERSONA.search(line)
            if m and personas < 3:
                personas += 1
                lines.append(f"{i:>5} prompt: \"{redact(m.group(0))}\"")
    return lines


def outline(files: list[SourceFile], budget_chars: int) -> str:
    """The shape of the whole project in a few percent of its size, so agents whose code does not fit in full -
    another subclass of the same base, another module of the same pipeline - are still visible. Files are listed
    by path so siblings sit together; when the budget is short, the lowest-ranked files are left out."""
    blocks = []
    for rank, f in enumerate(files):
        if TEST_PATH.search(f.rel.replace(os.sep, "/")) or f.path.suffix.lower() in CONFIG_EXTS - {".yaml", ".yml"}:
            continue
        body = _outline_lines(f)
        if len(body) > OUTLINE_LINES:
            body = body[:OUTLINE_LINES] + [f"      … {len(body) - OUTLINE_LINES} more"]
        blocks.append((rank, f.rel, f.rel + ("\n" + "\n".join(body) if body else "") + "\n"))
    kept, used = [], 0
    for rank, rel, block in sorted(blocks):  # best-ranked first, so they win when the budget is short
        if used + len(block) > budget_chars:
            block = rel + "\n"  # no room for its contents: still show the file exists
            if used + len(block) > budget_chars:
                continue
        kept.append((rel, block))
        used += len(block)
    return "".join(block for _, block in sorted(kept))


# Lines worth keeping when a big file is cut down: definitions, prompts, model calls and graph wiring.
EXCERPT_LINES = re.compile(r"^\s*(class |def |async def )|You are|system|prompt|completions\.create|messages\.create|"
                           r"generate_content|\.a?invoke\(|add_node|add_edge|add_conditional_edges|Agent\(|tools?\s*=", re.I)
MIN_EXCERPT = 4000
MAX_LINE = 1000
OUTLINE_LINES = 40  # per file


def _excerpt(lines: list[str], limit: int, context: int = 3) -> tuple[str, bool]:
    """lines are numbered ('  12| code'). Whole file when it fits in `limit` characters; otherwise its head plus
    the windows around its definitions, prompts and model calls, in file order, with '...' marking each gap."""
    if sum(len(line) + 1 for line in lines) <= limit:
        return "\n".join(lines), False
    hits = []
    for i, line in enumerate(lines):
        code = line.split("| ", 1)[-1]
        weight = len(AGENT_HINTS.findall(code)) + 3 * len(EXCERPT_LINES.findall(code))
        if weight:
            hits.append((-weight, i))
    keep, size = set(), 0
    for i in range(min(12, len(lines))):  # the head: imports say which framework this is
        if size + len(lines[i]) + 1 > limit // 4:
            break
        keep.add(i)
        size += len(lines[i]) + 1
    for _, i in sorted(hits):
        window = [j for j in range(max(0, i - context), min(len(lines), i + context + 1)) if j not in keep]
        add = sum(len(lines[j]) + 1 for j in window) + 8  # + the '...' marker it may need
        if size + add <= limit:
            keep.update(window)
            size += add
    out, prev = [], -1
    for i in sorted(keep):
        if i != prev + 1:
            out.append("   ...")
        out.append(lines[i])
        prev = i
    if prev != len(lines) - 1:
        out.append("   ...")
    return "\n".join(out), True


def _spread(files: list[SourceFile], prefer: set[str]) -> list[SourceFile]:
    """Best first, but each further file from an already-picked folder counts for less, so one subsystem (fifteen
    co_storm modules) cannot crowd out another (the storm_wiki pipeline) that is just as much a part of the app."""
    pool = sorted(files, key=lambda f: f.rel)  # max() keeps the first of equals, so ties go alphabetically
    per_dir: dict[str, int] = {}
    order = []

    def worth(i: int) -> tuple[bool, float]:
        f = pool[i]  # files static analysis found agents in come first, whatever their keyword score
        return f.rel in prefer, f.score / (1 + 0.35 * per_dir.get(os.path.dirname(f.rel), 0))

    while pool:
        best = pool.pop(max(range(len(pool)), key=worth))
        order.append(best)
        d = os.path.dirname(best.rel)
        per_dir[d] = per_dir.get(d, 0) + 1
    return order


def pack(files: list[SourceFile], budget_chars: int = 180_000, prefer: set[str] | list[str] = ()) -> tuple[str, list[str]]:
    """The LLM's view of the project, within budget_chars: an outline of every source file, then the best files
    (agent definitions and the files in `prefer` first). A file too big for its share is cut to the lines around
    its classes, prompts and model calls; one that does not fit is skipped, and smaller ones after it still get in.
    Returns (text, included rel paths, marked " (excerpt)" when cut)."""
    prefer = set(prefer)
    source = [f for f in files if not TEST_PATH.search(f.rel.replace(os.sep, "/"))] or files
    shape = outline(source, budget_chars // 4)
    parts = [f"<repo_outline>\n{shape}</repo_outline>\n"] if shape else []
    used = sum(len(p) for p in parts)
    included: list[str] = []
    for f in _spread(source, prefer):
        remaining = budget_chars - used
        if remaining < MIN_EXCERPT // 2:
            break
        readme = f.path.stem.lower() == "readme" and os.sep not in f.rel
        important = f.defines_agent or f.rel in prefer or readme
        if f.score <= 0 and included and not important:
            continue
        # no single file takes more than its share: agent definitions a third, the README and helpers less
        cap = max(MIN_EXCERPT * 2, budget_chars // (15 if readme else 3 if important else 9))
        body = redact(f.text, config=f.path.suffix.lower() in CONFIG_EXTS)
        lines = [f"{i:>4}| {line if len(line) <= MAX_LINE else line[:MAX_LINE] + ' …'}"
                 for i, line in enumerate(body.splitlines(), 1)]  # one-line JSON / minified code stays short
        head = f'<file path="{f.rel}"'
        full = sum(len(line) + 1 for line in lines) + len(head) + 12
        room = min(cap, remaining) - len(head) - 30
        if full > min(cap, remaining) and not (full > cap or important):
            continue  # a mid-sized helper that no longer fits: smaller files after it still can
        if room < MIN_EXCERPT and full > min(cap, remaining):
            continue
        text, cut = _excerpt(lines, room)
        mark = ' excerpt="selected lines"' if cut else ""
        chunk = f"{head}{mark}>\n{text}\n</file>\n"
        parts.append(chunk)
        included.append(f.rel + (" (excerpt)" if cut else ""))
        used += len(chunk)
    return "".join(parts), included
