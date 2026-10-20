"""Offline agent detection: reads Python (AST) and JS/TS/YAML (regex) without calling an LLM.

Understands the common shapes of LangGraph, LangChain, CrewAI (code + YAML), AutoGen
(0.2 and 0.4), the OpenAI Agents SDK, Swarm, Google ADK, smolagents, LlamaIndex, CAMEL and DSPy,
plus agent classes a project defines itself. The result is a raw blueprint dict; `blueprint.normalize`
resolves names to ids.
"""

from __future__ import annotations

import ast
import json
import os
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import Any

from . import blueprint as bpmod
from . import call_sites
from .collector import TEST_PATH, SourceFile, collect

AGENT_CLASSES = {
    "Agent", "AssistantAgent", "UserProxyAgent", "ConversableAgent", "CodeExecutorAgent",
    "SocietyOfMindAgent", "GPTAssistantAgent", "RetrieveAssistantAgent", "ReActAgent", "FunctionAgent",
    "CodeAgent", "ToolCallingAgent", "MultiStepAgent", "LlmAgent", "SequentialAgent", "ParallelAgent",
    "LoopAgent", "ChatAgent", "BaseChatAgent",
}
AGENT_FACTORIES = {
    "create_react_agent", "create_agent", "create_tool_calling_agent", "create_openai_functions_agent",
    "create_openai_tools_agent", "create_structured_chat_agent", "initialize_agent", "create_supervisor",
    "create_swarm", "create_handoff_tool",
}
GROUP_CLASSES = {"GroupChat", "RoundRobinGroupChat", "SelectorGroupChat", "Swarm", "MagenticOneGroupChat"}
SEQUENTIAL_CLASSES = {"SequentialAgent", "LoopAgent"}
START_NAMES = {"START", "__start__"}
END_NAMES = {"END", "__end__"}
DESC_KWARGS = ("goal", "description", "instructions", "system_message", "system_prompt", "prompt", "backstory")
ROLE_KWARGS = ("role", "description", "goal")
HANDOFF_KWARGS = ("handoffs", "sub_agents", "managed_agents", "can_handoff_to", "agents")
# What a file is built with, read from its imports.
FRAMEWORK_IMPORTS = (
    ("langgraph", ("langgraph",)),
    ("crewai", ("crewai",)),
    ("autogen", ("autogen", "autogen_agentchat", "autogen_core", "autogen_ext", "ag2")),
    ("camel", ("camel",)),
    ("openai-agents", ("agents",)),
    ("swarm", ("swarm",)),
    ("google-adk", ("google.adk",)),
    ("smolagents", ("smolagents",)),
    ("llamaindex", ("llama_index",)),
    ("dspy", ("dspy",)),
    ("langchain", ("langchain",)),  # and langchain_core, langchain_openai, ...
)
# Model calls through LangChain, or no framework at all: other frameworks run their agents on LangChain chat
# models, so these name the framework only when nothing else does.
WEAK_FRAMEWORKS = {"langchain", "custom"}
DSPY_PREDICTORS = {"Predict", "ChainOfThought", "ChainOfThoughtWithHint", "ReAct", "ProgramOfThought",
                   "MultiChainComparison", "TypedPredictor", "TypedChainOfThought", "Refine", "BestOfN", "CodeAct"}
# config keys that list the roles a run uses: session.roles, default_agents, WORKERS
ROLE_LIST_KEY = re.compile(r"(^|[._])(roles|agents|workers|members)$", re.I)


class _Collector:
    def __init__(self) -> None:
        self.agents: dict[str, dict[str, Any]] = {}  # key -> agent dict (key = norm name)
        self.edges: list[dict[str, Any]] = []
        self.entry: list[str] = []
        self.exits: list[str] = []
        self.steps: list[dict[str, Any]] = []
        self.frameworks: dict[str, int] = {}
        self.graph_nodes: dict[str, dict[str, Any]] = {}  # node name -> agent dict
        self.node_funcs: dict[str, str] = {}  # function name -> node name
        self.var_to_agent: dict[str, str] = {}  # variable / method name -> agent key
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.yaml_agents: dict[str, dict[str, Any]] = {}
        self.yaml_tasks: list[tuple[str, dict[str, Any]]] = []
        self.tool_funcs: set[str] = set()
        self.instances: dict[str, set[str]] = {}  # class name -> variable/attribute names holding instances
        self.root = Path(".")
        self.file: str | None = None  # the file being scanned
        self.file_votes: dict[str, dict[str, int]] = {}  # file -> framework -> evidence
        self.sites: dict[str, list[tuple[str, int]]] = {}  # agent key -> every (file, line) that defines it
        self.entry_src: list[str | None] = []  # the file each entry / exit came from, by position
        self.exit_src: list[str | None] = []
        self.role_lists: list[tuple[str, int, list[str]]] = []  # (file, line, names) of default roles lists

    # -- helpers ---------------------------------------------------------
    def vote(self, fw: str, n: int = 1) -> None:
        self.frameworks[fw] = self.frameworks.get(fw, 0) + n
        if self.file:
            votes = self.file_votes.setdefault(self.file, {})
            votes[fw] = votes.get(fw, 0) + n

    def add_agent(self, name: str, *, file: str, line: int, role: str = "", description: str = "",
                  tools: list[str] | None = None, aliases: list[str] | None = None) -> dict[str, Any]:
        key = bpmod.norm_key(name)
        agent = self.agents.get(key)
        if agent is None:
            agent = {"name": _pretty(name), "role": "", "description": "", "tools": [], "file": file,
                     "line": line, "aliases": [name], "keywords": []}
            self.agents[key] = agent
        self.sites.setdefault(key, []).append((file, line))
        agent["role"] = agent["role"] or _short(role, 80)
        agent["description"] = agent["description"] or _short(description, 280)
        for t in tools or []:
            if t not in agent["tools"]:
                agent["tools"].append(t)
        for a in aliases or []:
            if a and a not in agent["aliases"]:
                agent["aliases"].append(a)
                self.var_to_agent[a] = key
        self.var_to_agent[name] = key
        return agent

    def edge(self, src: str, dst: str, label: str = "", conditional: bool = False) -> None:
        if src in START_NAMES:
            if dst not in END_NAMES:
                self.entry.append(dst)
            return
        if dst in END_NAMES:
            self.exits.append(src)
            return
        self.edges.append({"from": src, "to": dst, "label": label, "conditional": conditional})


def _pretty(name: str) -> str:
    if re.search(r"[A-Z ]", name):
        return name.strip()
    return " ".join(w.capitalize() for w in re.split(r"[_\-]+", name) if w) or name


def _short(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _lit(node: ast.AST | None) -> str | None:
    """String value of a literal-ish node; START/END names map to their sentinel."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{…}" for v in node.values)
    if isinstance(node, ast.Name) and node.id in START_NAMES | END_NAMES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in START_NAMES | END_NAMES:
        return node.attr
    return None


def _ref(node: ast.AST | None) -> str | None:
    """A string literal, variable name, attribute name or called-function name."""
    if node is None:
        return None
    s = _lit(node)
    if s is not None:
        return s
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _ref(node.func)
    return None


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _kw(node: ast.Call, *names: str) -> ast.AST | None:
    for k in node.keywords:
        if k.arg in names:
            return k.value
    return None


def _list_refs(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out = []
        for elt in node.elts:
            if isinstance(elt, ast.Call) and _call_name(elt) == "handoff":
                elt = _kw(elt, "agent") or (elt.args[0] if elt.args else elt)
            if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) and elt.func.attr == "as_tool":
                elt = elt.func.value
            r = _ref(elt)
            if r:
                out.append(r)
        return out
    if isinstance(node, ast.Dict):
        return [r for r in (_ref(v) for v in node.values) if r]
    r = _ref(node)
    return [r] if r else []


def _assigned_names(parent: ast.AST) -> list[str]:
    if isinstance(parent, ast.Assign):
        return [t.id if isinstance(t, ast.Name) else t.attr for t in parent.targets if isinstance(t, (ast.Name, ast.Attribute))]
    if isinstance(parent, (ast.AnnAssign, ast.AugAssign)) and isinstance(parent.target, (ast.Name, ast.Attribute)):
        t = parent.target
        return [t.id if isinstance(t, ast.Name) else t.attr]
    return []


def _imported_modules(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module)
    return out


def _frameworks_imported(modules: set[str], src: SourceFile, root: Path) -> list[str]:
    """Frameworks a file imports. A string in the code (a "crewai.com" User-Agent) is not an import, and
    `from agents import ...` is the OpenAI Agents SDK only when the project has no agents module of its own."""
    found = []
    for fw, prefixes in FRAMEWORK_IMPORTS:
        if any(m == p or m.startswith(p + ".") or (p == "langchain" and m.startswith("langchain_"))
               for m in modules for p in prefixes):
            if fw == "openai-agents" and _local_module(src, root, "agents"):
                continue
            found.append(fw)
    return found


def _local_module(src: SourceFile, root: Path, name: str) -> bool:
    d = src.path.parent
    while d == root or root in d.parents:
        if (d / f"{name}.py").is_file() or (d / name).is_dir():
            return True
        d = d.parent
    return False


def _str_items(node: ast.AST | None) -> list[str]:
    """["planner", "code_interpreter"] -> its names, when it is a list of two or more identifiers."""
    if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) < 2:
        return []
    names = [_lit(e) for e in node.elts]
    return names if all(n and n.isidentifier() for n in names) else []  # type: ignore[misc]


def _role_lists(tree: ast.AST, rel: str, c: _Collector) -> None:
    """Default roles lists: roles = [...], self.roles = cfg.get_list("roles", [...]), {"roles": [...]}."""
    for node in ast.walk(tree):
        found: list[str] = []
        if isinstance(node, ast.Call) and len(node.args) >= 2:
            key = _lit(node.args[0])
            if key and ROLE_LIST_KEY.search(key):
                found = _str_items(node.args[1])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            if any(ROLE_LIST_KEY.search(n) for n in _assigned_names(node)):
                found = _str_items(node.value)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                key = _lit(k)
                if key and ROLE_LIST_KEY.search(key) and _str_items(v):
                    c.role_lists.append((rel, v.lineno, _str_items(v)))
        if found:
            c.role_lists.append((rel, node.lineno, found))


def _scan_python(src: SourceFile, c: _Collector) -> None:
    try:
        tree = ast.parse(src.text)
    except (SyntaxError, ValueError):
        return
    text = src.text
    imported = _frameworks_imported(_imported_modules(tree), src, c.root)
    for fw in imported:
        c.vote(fw, 2)
    _role_lists(tree, src.rel, c)

    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            c.functions[node.name] = node

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            cls_name = _call_name(node.value)
            if cls_name and cls_name[:1].isupper():
                for t in node.targets:
                    name = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else None
                    if name:
                        c.instances.setdefault(cls_name, set()).add(name)

    # CrewAI @CrewBase classes: @agent / @task decorated methods
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decos = {_ref(d) for d in node.decorator_list}
            if "tool" in decos or "function_tool" in decos or "beta_tool" in decos:
                c.tool_funcs.add(node.name)
            if "agent" in decos:
                c.vote("crewai", 3)
                cfg = re.search(r"agents_config\[['\"]([\w\-]+)['\"]\]", ast.get_source_segment(text, node) or "")
                name = cfg.group(1) if cfg else node.name
                c.add_agent(name, file=src.rel, line=node.lineno, aliases=[node.name])
            if "task" in decos:
                seg = ast.get_source_segment(text, node) or ""
                cfg = re.search(r"tasks_config\[['\"]([\w\-]+)['\"]\]", seg)
                ag = re.search(r"agent\s*=\s*self\.(\w+)\(", seg)
                c.steps.append({"title": _pretty(cfg.group(1) if cfg else node.name),
                                "agents": [ag.group(1)] if ag else [], "description": "",
                                "_task": cfg.group(1) if cfg else node.name})

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        parent = parents.get(node)
        targets = _assigned_names(parent) if parent is not None else []

        # -- graph builders (LangGraph) ------------------------------------
        if name == "add_node" and isinstance(node.func, ast.Attribute):
            c.vote("langgraph")
            first = node.args[0] if node.args else _kw(node, "node")
            second = node.args[1] if len(node.args) > 1 else _kw(node, "action")
            node_name = _lit(first) or _ref(first)
            fn_name = _ref(second) if second is not None else (_ref(first) if _lit(first) is None else None)
            if not node_name:
                continue
            aliases = [fn_name] if fn_name and fn_name != node_name else []
            ag = c.add_agent(node_name, file=src.rel, line=node.lineno, aliases=aliases)
            c.graph_nodes[node_name] = ag
            if fn_name:
                c.node_funcs[fn_name] = node_name
            continue
        if name == "add_edge" and isinstance(node.func, ast.Attribute) and len(node.args) >= 2:
            srcs = _loop_values(node.args[0], node, parents) or (
                _list_refs(node.args[0]) if isinstance(node.args[0], (ast.List, ast.Tuple)) else [_ref(node.args[0])])
            dsts = _loop_values(node.args[1], node, parents) or [_ref(node.args[1])]
            for a in srcs:
                for b in dsts:
                    if a and b:
                        c.edge(a, b)
            continue
        if name == "add_conditional_edges" and node.args:
            srcn = _ref(node.args[0])
            pm = node.args[2] if len(node.args) > 2 else _kw(node, "path_map")
            router = _ref(node.args[1]) if len(node.args) > 1 else None
            targets_ = _list_refs(pm) if pm is not None else []
            if not targets_ and router and router in c.functions:
                targets_ = _returned_strings(c.functions[router])
            for t in targets_:
                if srcn:
                    c.edge(srcn, t, label=router or "", conditional=True)
            continue
        if name in {"set_entry_point", "set_finish_point"} and node.args:
            r = _ref(node.args[0])
            if r:
                (c.entry if name == "set_entry_point" else c.exits).append(r)
            continue
        if name == "Command":
            goto = _kw(node, "goto")
            fn = _enclosing_function(node, parents)
            if goto is not None and fn is not None:
                for t in _list_refs(goto):
                    c.edge(f"@fn:{fn}", t, conditional=True)
            continue

        # -- agent constructors ------------------------------------------
        if name in AGENT_CLASSES or name in AGENT_FACTORIES or (name.endswith("Agent") and node.keywords):
            if name in {"create_handoff_tool"}:
                continue
            agent_name = _lit(_kw(node, "name", "agent_name"))
            role = _lit(_kw(node, *ROLE_KWARGS)) or ""
            if not agent_name and name == "Agent" and _kw(node, "role") is not None:
                agent_name = role  # CrewAI: the role is the identity
                c.vote("crewai")
            if not agent_name and node.args and _lit(node.args[0]) and name not in AGENT_FACTORIES:
                agent_name = _lit(node.args[0])
            if not agent_name and targets:
                agent_name = targets[0]
            if not agent_name:
                continue
            desc = next((v for v in (_lit(_kw(node, k)) for k in DESC_KWARGS) if v), "")
            tools_node = _kw(node, "tools", "functions")
            if tools_node is None and name in AGENT_FACTORIES and len(node.args) > 1:
                tools_node = node.args[1]
            tools = [t for t in _list_refs(tools_node) if t]
            if name in {"AssistantAgent", "UserProxyAgent", "ConversableAgent"}:
                c.vote("autogen")
            ag = c.add_agent(agent_name, file=src.rel, line=node.lineno, role=role if role != agent_name else "",
                             description=desc, tools=tools, aliases=targets)
            # Swarm-style transfer functions and agent-as-tool are handoffs
            for t in tools:
                if t.startswith(("transfer_to_", "handoff_to_", "transfer_back_to_")):
                    fn = c.functions.get(t)
                    dst = _returned_name(fn) if fn else None
                    c.edges.append({"from": agent_name, "to": dst or t.split("_to_", 1)[1], "label": "handoff",
                                    "conditional": True})
            if tools_node is not None and isinstance(tools_node, (ast.List, ast.Tuple)):
                for elt in tools_node.elts:
                    if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) and elt.func.attr == "as_tool":
                        r = _ref(elt.func.value)
                        if r:
                            c.edges.append({"from": agent_name, "to": r, "label": "as tool", "conditional": True})
                            ag["tools"] = [x for x in ag["tools"] if x != r]
            for kw in HANDOFF_KWARGS:
                sub = _kw(node, kw)
                if sub is None or (kw == "agents" and name not in {"AgentWorkflow"}):
                    continue
                refs = _list_refs(sub)
                if name in SEQUENTIAL_CLASSES and kw == "sub_agents":
                    for x, y in zip(refs, refs[1:]):
                        c.edges.append({"from": x, "to": y, "label": "", "conditional": False})
                    if refs:
                        c.edges.append({"from": agent_name, "to": refs[0], "label": "starts", "conditional": False})
                else:
                    for r in refs:
                        c.edges.append({"from": agent_name, "to": r, "label": "handoff" if kw != "managed_agents" else "delegates",
                                        "conditional": True})
            if name == "create_supervisor":
                for r in _list_refs(node.args[0] if node.args else _kw(node, "agents")):
                    c.edges.append({"from": agent_name, "to": r, "label": "delegates", "conditional": True})
                    c.edges.append({"from": r, "to": agent_name, "label": "reports", "conditional": True})
            continue

        # -- teams / crews -----------------------------------------------
        if name in GROUP_CLASSES or name == "AgentWorkflow":
            members = _list_refs(_kw(node, "agents", "participants") or (node.args[0] if node.args else None))
            c.vote("autogen" if name != "AgentWorkflow" else "llamaindex")
            ring = members + members[:1] if len(members) > 2 else members
            for x, y in zip(ring, ring[1:]):
                c.edges.append({"from": x, "to": y, "label": "turn", "conditional": True})
            root = _lit(_kw(node, "root_agent"))
            if root:
                c.entry.append(root)
            continue
        if name == "initiate_chat" and isinstance(node.func, ast.Attribute):
            a = _ref(node.func.value)
            b = _ref(node.args[0]) if node.args else _ref(_kw(node, "recipient"))
            if a and b:
                c.edges.append({"from": a, "to": b, "label": "chat", "conditional": False})
                c.entry.append(a)
            continue
        if name == "Task":
            if _kw(node, "config") is not None:  # CrewAI @task methods - the YAML describes it
                continue
            if "crewai" not in imported and _kw(node, "agent", "expected_output") is None:
                continue  # CAMEL and others have a Task too: Task(content=question, id="0")
            ag = _ref(_kw(node, "agent"))
            desc = _lit(_kw(node, "description")) or ""
            title = targets[0] if targets else (desc[:40] or "Task")
            c.steps.append({"title": _pretty(title), "agents": [ag] if ag else [], "description": _short(desc, 200),
                            "_task": title})
            c.vote("crewai")
            continue
        if name == "Crew":
            c.vote("crewai", 2)
            process = ast.get_source_segment(text, _kw(node, "process")) if _kw(node, "process") is not None else ""
            mgr = _ref(_kw(node, "manager_agent"))
            if process and "hierarchical" in process:
                members = _list_refs(_kw(node, "agents"))
                hub = mgr or "manager"
                if not mgr:
                    c.add_agent("manager", file=src.rel, line=node.lineno, role="Crew manager (auto-created)")
                for m in members:
                    c.edges.append({"from": hub, "to": m, "label": "delegates", "conditional": True})
                    c.edges.append({"from": m, "to": hub, "label": "reports", "conditional": True})
                c.entry.append(hub)


def _loop_values(arg: ast.AST, node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    """`for w in ("a", "b"): g.add_edge(w, "c")` -> ["a", "b"] for the loop variable."""
    if not isinstance(arg, ast.Name):
        return []
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.For) and isinstance(cur.target, ast.Name) and cur.target.id == arg.id:
            if isinstance(cur.iter, (ast.List, ast.Tuple, ast.Set)):
                return [v for v in (_lit(e) for e in cur.iter.elts) if v]
            return []
        cur = parents.get(cur)
    return []


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parents.get(cur)
    return None


def _returned_strings(fn: ast.AST) -> list[str]:
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and n.value is not None:
            s = _lit(n.value)
            if s:
                out.append(s)
    return list(dict.fromkeys(out))


def _returned_name(fn: ast.AST) -> str | None:
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and n.value is not None:
            r = _ref(n.value)
            if r:
                return r
    return None


# -- non-Python sources ---------------------------------------------------

_JS_NODE = re.compile(r"\.addNode\(\s*['\"`]([\w\- ]+)['\"`]")
_JS_EDGE = re.compile(r"\.addEdge\(\s*([\w'\"`\-]+)\s*,\s*([\w'\"`\-]+)")
_JS_COND = re.compile(r"\.addConditionalEdges\(\s*['\"`]([\w\-]+)['\"`]\s*,\s*\w+\s*(?:,\s*(\{[^}]*\}|\[[^\]]*\]))?")
_JS_AGENT = re.compile(r"new\s+(\w*Agent)\s*\(\s*\{(.{0,600}?)\}\s*\)", re.S)
_JS_NAME = re.compile(r"name\s*:\s*['\"`]([^'\"`]+)['\"`]")
_JS_INSTR = re.compile(r"(?:instructions|description|systemPrompt|role)\s*:\s*['\"`]([^'\"`]{1,300})")
_JS_HANDOFFS = re.compile(r"handoffs\s*:\s*\[([^\]]*)\]")


def _unq(s: str) -> str:
    return s.strip().strip("'\"`")


def _scan_js(src: SourceFile, c: _Collector) -> None:
    t = src.text
    if "@langchain/langgraph" in t:
        c.vote("langgraph", 2)
    if "@openai/agents" in t:
        c.vote("openai-agents", 2)
    for m in _JS_NODE.finditer(t):
        line = t.count("\n", 0, m.start()) + 1
        c.graph_nodes[m.group(1)] = c.add_agent(m.group(1), file=src.rel, line=line)
    for m in _JS_EDGE.finditer(t):
        c.edge(_unq(m.group(1)), _unq(m.group(2)))
    for m in _JS_COND.finditer(t):
        targets = re.findall(r"['\"`]([\w\-]+)['\"`]", m.group(2) or "")
        for tgt in targets:
            c.edge(m.group(1), tgt, conditional=True)
    for m in _JS_AGENT.finditer(t):
        body = m.group(2)
        nm = _JS_NAME.search(body)
        if not nm:
            continue
        line = t.count("\n", 0, m.start()) + 1
        before = t[max(0, m.start() - 80): m.start()]
        var = re.search(r"(?:const|let|var)\s+(\w+)\s*=\s*$", before)
        desc = _JS_INSTR.search(body)
        c.add_agent(nm.group(1), file=src.rel, line=line, description=desc.group(1) if desc else "",
                    aliases=[var.group(1)] if var else [])
        ho = _JS_HANDOFFS.search(body)
        if ho:
            for ref in re.findall(r"[\w]+", ho.group(1)):
                c.edges.append({"from": nm.group(1), "to": ref, "label": "handoff", "conditional": True})


def _parse_simple_yaml(text: str) -> dict[str, dict[str, str]]:
    """Two-level YAML mappings (CrewAI agents.yaml / tasks.yaml) without requiring PyYAML."""
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return {str(k): {str(kk): str(vv) for kk, vv in v.items()} for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    out: dict[str, dict[str, str]] = {}
    cur: dict[str, str] | None = None
    key: str | None = None
    kv_indent = 0
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        m_top = re.match(r"^([\w\-]+)\s*:\s*$", raw)
        if indent == 0 and m_top:
            cur = out.setdefault(m_top.group(1), {})
            key = None
            kv_indent = 0
            continue
        if cur is None:
            continue
        m_kv = re.match(r"^\s+([\w\-]+)\s*:\s*(.*)$", raw)
        if m_kv and (kv_indent == 0 or indent == kv_indent):
            kv_indent = indent
            key = m_kv.group(1)
            val = m_kv.group(2).strip()
            cur[key] = "" if val in {">", "|", ">-", "|-"} else val.strip("'\"")
        elif key:
            cur[key] = (cur[key] + " " + raw.strip()).strip()
    return out


def _scan_yaml(src: SourceFile, c: _Collector) -> None:
    stem = Path(src.rel).stem.lower()
    if stem not in {"agents", "tasks"}:
        return
    data = _parse_simple_yaml(src.text)
    if stem == "agents":
        for name, fields in data.items():
            if "role" in fields or "goal" in fields:
                c.vote("crewai", 2)
                c.add_agent(name, file=src.rel, line=_line_of(src.text, name + ":"), role=fields.get("role", ""),
                            description=fields.get("goal", "") or fields.get("backstory", ""))
    else:
        for name, fields in data.items():
            if "description" in fields:
                c.yaml_tasks.append((name, fields))


def _scan_json(src: SourceFile, c: _Collector) -> None:
    """Config files can set the default roles list too: {"session.roles": ["planner", "code_interpreter"]}."""
    try:
        data = json.loads(src.text)
    except ValueError:
        return
    stack = [data]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if ROLE_LIST_KEY.search(str(k)) and isinstance(v, list) and len(v) >= 2 \
                        and all(isinstance(x, str) and x.isidentifier() for x in v):
                    c.role_lists.append((src.rel, _line_of(src.text, f'"{k}"'), list(v)))
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)


def _line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    return text.count("\n", 0, idx) + 1 if idx >= 0 else 0


# -- agent classes ----------------------------------------------------------


class _ClassDef:
    def __init__(self, node: ast.ClassDef, rel: str, dspy_mods: set[str], dspy_names: dict[str, str]) -> None:
        self.node, self.name, self.rel = node, node.name, rel
        self.bases = [_dotted(b) for b in node.bases]
        self.dspy_mods, self.dspy_names = dspy_mods, dspy_names  # `import dspy as d`; `from dspy import Predict`
        self.doc = (ast.get_docstring(node) or "").strip().split("\n")[0]

    def calls(self) -> list[ast.Call]:
        return [n for n in ast.walk(self.node) if isinstance(n, ast.Call)]

    def dspy_attr(self, func: ast.AST) -> str | None:
        """dspy.Predict / d.ChainOfThought / Predict (imported from dspy) -> the dspy name."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in self.dspy_mods:
            return func.attr
        if isinstance(func, ast.Name):
            return self.dspy_names.get(func.id)
        return None


def _dotted(node: ast.AST) -> str:
    node = node.value if isinstance(node, ast.Subscript) else node  # Generic[T]
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _class_defs(files: list[SourceFile]) -> tuple[list[_ClassDef], set[str]]:
    """Every class, and the names of everything the project calls (to tell which classes it ever creates)."""
    out, called = [], set()
    for f in files:
        if f.path.suffix.lower() not in (".py", ".ipynb") or TEST_PATH.search(f.rel.replace(os.sep, "/")):
            continue
        try:
            tree = ast.parse(f.text)
        except (SyntaxError, ValueError):
            continue
        mods, names = set(), {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.asname or a.name for a in node.names if a.name == "dspy")
            elif isinstance(node, ast.ImportFrom) and node.module == "dspy":
                names.update({a.asname or a.name: a.name for a in node.names})
            elif isinstance(node, ast.Call):
                called.add(_dotted(node.func).split(".")[-1])
        out += [_ClassDef(n, f.rel, mods, names) for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    return out, called


def _scan_classes(files: list[SourceFile], c: _Collector) -> None:
    """Agents a project defines as classes rather than builds from a framework's constructors: subclasses of an
    agent base (CoStormExpert(Agent), through the project's own classes too) and DSPy programs - dspy.Module
    subclasses that run predictors of their own (WikiWriter, TopicExpert). A program an agent class holds is that
    agent's tool; a module that calls programs in turn (a writer asking an expert, round after round) is how work
    flows between them. Where a folder defines agent classes (Co-STORM's modules), the programs next to them are
    what those agents are built from, not agents of their own; a program nothing creates is not part of a run."""
    defs, called = _class_defs(files)
    by_name: dict[str, list[_ClassDef]] = {}
    for d in defs:
        by_name.setdefault(d.name, []).append(d)
    subclassed = {b.split(".")[-1] for d in defs for b in d.bases}

    def inherits(d: _ClassDef, hit: Any, seen: frozenset = frozenset()) -> bool:
        for b in d.bases:
            last = b.split(".")[-1]
            if hit(d, b, last):
                return True
            if last not in seen and any(inherits(x, hit, seen | {d.name}) for x in by_name.get(last, ()) if x is not d):
                return True
        return False

    def is_module(d: _ClassDef, b: str, last: str) -> bool:
        return (b.count(".") == 1 and b.split(".")[0] in d.dspy_mods and last == "Module") or d.dspy_names.get(b) == "Module"

    agent_classes = [d for d in defs if d.name not in subclassed and not d.name.startswith(("Base", "Abstract"))
                     and inherits(d, lambda d, b, last: last in AGENT_CLASSES)]
    modules = [d for d in defs if inherits(d, is_module)]
    programs = {d.name: d for d in modules if any(d.dspy_attr(n.func) in DSPY_PREDICTORS for n in d.calls())}

    held_by: dict[str, set[str]] = {}  # class name -> names of the classes that create one
    for d in defs:
        for call in d.calls():
            if isinstance(call.func, ast.Name) and call.func.id in by_name and call.func.id != d.name:
                held_by.setdefault(call.func.id, set()).add(d.name)

    agent_keys: dict[str, str] = {}  # agent class name -> its agent's key
    for d in agent_classes:
        existing = next((c.var_to_agent[v] for v in sorted(c.instances.get(d.name, ())) if v in c.var_to_agent), None)
        if existing and existing in c.agents:  # researcher = ResearchAgent(...) was already found: same agent
            if d.name not in c.agents[existing]["aliases"]:
                c.agents[existing]["aliases"].append(d.name)
            agent_keys[d.name] = existing
            continue
        c.add_agent(d.name, file=d.rel, line=d.node.lineno, description=d.doc)
        agent_keys[d.name] = bpmod.norm_key(d.name)

    def owner(name: str) -> str | None:
        seen, queue = {name}, sorted(held_by.get(name, ()))
        while queue:
            h = queue.pop(0)
            if h in agent_keys:
                return h
            if h not in seen:
                seen.add(h)
                queue += sorted(held_by.get(h, set()) - seen)
        return None

    agent_dirs = {str(_posix(d.rel).parent) for d in agent_classes}
    created = {n for n in programs if n in called} or set(programs)
    standalone = set()
    for name, d in programs.items():
        boss = owner(name)
        agent = c.agents.get(agent_keys[boss]) if boss else None
        if agent is not None:
            if name not in agent["tools"]:
                agent["tools"].append(name)
            continue
        if name not in created or str(_posix(d.rel).parent) in agent_dirs:
            continue
        c.add_agent(name, file=d.rel, line=d.node.lineno, description=d.doc)
        standalone.add(name)

    for d in modules:  # ConvSimulator.forward: the writer asks, the expert answers, for max_turn rounds
        attrs = {}
        for node in ast.walk(d.node):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        attrs[t.attr] = node.value.func.id
        fwd = next((m for m in d.node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and m.name in ("forward", "aforward", "__call__")), None)
        if fwd is None:
            continue
        order, looped = [], False
        parents = {ch: n for n in ast.walk(fwd) for ch in ast.iter_child_nodes(n)}
        for call in sorted((n for n in ast.walk(fwd) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)):
            f = call.func
            if isinstance(f, ast.Attribute) and f.attr in ("forward", "aforward", "acall"):
                f = f.value
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self"):
                continue
            target = attrs.get(f.attr)
            if target in standalone:
                if not order or order[-1] != target:
                    order.append(target)
                cur = parents.get(call)
                while cur is not None and cur is not fwd:
                    looped = looped or isinstance(cur, (ast.For, ast.AsyncFor, ast.While))
                    cur = parents.get(cur)
        for a, b in zip(order, order[1:]):
            c.edges.append({"from": a, "to": b, "label": "", "conditional": False, "_file": d.rel})
        if looped and len(set(order)) > 1:
            c.edges.append({"from": order[-1], "to": order[0], "label": "next round", "conditional": True, "_file": d.rel})


# -- alternatives -----------------------------------------------------------


def _strongest(votes: dict[str, int]) -> str:
    """The framework with the most evidence; LangChain only when nothing else has any."""
    return min(votes, key=lambda fw: (fw in WEAK_FRAMEWORKS, -votes[fw], fw)) if votes else "custom"


def _framework(c: _Collector) -> str:
    """What the agents are built with: each agent counts for the framework of the file that defines it, and one
    on LangChain chat models or no framework counts half - so CAMEL agents outvote framework-free examples, and
    a LangGraph team outvotes forty files that import LangChain for model calls."""
    weight: dict[str, float] = {}
    for a in c.agents.values():
        fw = _strongest(c.file_votes.get(a["file"], {}))
        weight[fw] = weight.get(fw, 0) + (0.5 if fw in WEAK_FRAMEWORKS else 1)
    if weight:
        return max(sorted(weight), key=lambda fw: (weight[fw], fw not in WEAK_FRAMEWORKS, c.frameworks.get(fw, 0)))
    return _strongest(c.frameworks)


def _posix(rel: str) -> PurePosixPath:
    return PurePosixPath(rel.replace(os.sep, "/"))


def _own_dir(files: set[str], others: set[str]) -> str:
    """The highest folder holding `files` and none of `others`: multi_agents/ag2 next to multi_agents/agents."""
    parts = PurePosixPath(posixpath.commonpath([str(_posix(f).parent) for f in files])).parts
    for i in range(1, len(parts) + 1):
        d = "/".join(parts[:i])
        if not any(o.startswith(d + "/") for o in others):
            return d
    return "/".join(parts)


def _drop_ports(c: _Collector) -> list[dict[str, Any]]:
    """The same team again on another framework (an AG2 copy of a LangGraph team) is an alternative, not more
    agents. When most of a smaller framework's agents share names with a bigger one's, keep the bigger team -
    its files, lines and edges - and report the other."""
    groups: dict[str, set[str]] = {}
    for key in c.agents:
        for file, _ in c.sites.get(key, []):
            fw = _strongest(c.file_votes.get(file, {}))
            if fw not in WEAK_FRAMEWORKS:
                groups.setdefault(fw, set()).add(key)
    def own_files(fw: str) -> list[PurePosixPath]:
        return [_posix(f) for k in groups[fw] for f, _ in c.sites[k] if _strongest(c.file_votes.get(f, {})) == fw]

    def port_like(fw: str) -> tuple[bool, int]:
        """A port usually sits in a folder named after its framework (multi_agents/ag2), a level below the original."""
        names = {fw, fw.replace("-", "_"), *dict(FRAMEWORK_IMPORTS).get(fw, ())} - {"agents"}
        return any(n in names for f in own_files(fw) for n in f.parent.parts), min(len(f.parts) for f in own_files(fw))

    ranked = sorted(groups, key=lambda fw: (port_like(fw)[0], -len(groups[fw]), port_like(fw)[1], fw))
    out: list[dict[str, Any]] = []
    for fw in ranked[1:]:
        team, main = groups[fw] & set(c.agents), groups[ranked[0]]
        shared = team & main
        if len(shared) < 2 or 2 * len(shared) < len(team):
            continue

        def files_of(keys: set[str], framework: str) -> set[str]:
            return {f for k in keys for f, _ in c.sites.get(k, []) if _strongest(c.file_votes.get(f, {})) == framework}

        port, kept = files_of(team, fw), files_of(main, ranked[0])
        names = [c.agents[k]["name"] for k in sorted(team, key=list(c.agents).index)]
        for key in team - shared:
            del c.agents[key]
        for key in shared:
            a = c.agents[key]
            if a["file"] in port:
                a["file"], a["line"] = next((f, n) for f, n in c.sites[key] if f not in port)
        c.edges = [e for e in c.edges if e.get("_file") not in port]
        c.entry = [x for x, src in zip(c.entry, c.entry_src + [None] * len(c.entry)) if src not in port]
        c.exits = [x for x, src in zip(c.exits, c.exit_src + [None] * len(c.exits)) if src not in port]
        out.append({"kind": "port", "paths": [_own_dir(port, kept)], "framework": fw, "agents": names,
                    "note": f"the same agents on {fw}, next to the {ranked[0]} version that is mapped"})
    return out


def _drop_unselected_roles(c: _Collector, files: list[SourceFile]) -> list[dict[str, Any]]:
    """Roles a configuration picks between are not parallel workers. When a default roles list names role
    folders (session.roles = ["planner", "code_interpreter"]), an agent in a variant of one of them - a sibling
    folder named after it, code_interpreter_cli_only next to code_interpreter - runs instead of it, not with it."""
    dirs: dict[str, set[str]] = {}  # folder name -> its paths
    for f in files:
        for d in _posix(f.rel).parents:
            if str(d) != ".":
                dirs.setdefault(d.name.lower(), set()).add(str(d))
    for rel, line, names in c.role_lists:
        chosen = {n.lower() for n in names}
        if sum(1 for n in chosen if n in dirs) < 2:
            continue
        role_dirs = {d for n in chosen for d in dirs.get(n, ())}
        dropped: dict[str, list[str]] = {}
        for key, a in list(c.agents.items()):
            folder = _posix(a["file"]).parent
            fname = folder.name.lower()
            if fname in chosen or not any(str(folder.parent / n) in role_dirs and fname.startswith(n + "_") for n in chosen):
                continue
            dropped.setdefault(str(folder), []).append(a["name"])
            del c.agents[key]
        if dropped:
            return [{"kind": "config", "paths": sorted(dropped), "file": rel, "line": line, "default": names,
                     "agents": [n for v in dropped.values() for n in v],
                     "note": f"variants of the default roles {', '.join(names)} set at {rel}:{line}; "
                             "configuration picks one of them"}]
    return []


# -- assembly -------------------------------------------------------------


def scan(root: Path, files: list[SourceFile] | None = None,
         alternatives: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Pass a list as `alternatives` to get the implementations left off the map: ports of the same agents to
    another framework, and role variants a config picks between."""
    root = Path(root).resolve()
    files = files if files is not None else collect(root, max_files=600)
    c = _Collector()
    c.root = root if root.is_dir() else root.parent
    for f in files:
        ext = f.path.suffix.lower()
        c.file, n_edges, n_entry, n_exits = f.rel, len(c.edges), len(c.entry), len(c.exits)
        if ext in {".py", ".ipynb"}:
            _scan_python(f, c)
        elif ext in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}:
            _scan_js(f, c)
        elif ext in {".yaml", ".yml"}:
            _scan_yaml(f, c)
        elif ext == ".json":
            _scan_json(f, c)
        for e in c.edges[n_edges:]:
            e.setdefault("_file", f.rel)
        c.entry_src += [f.rel] * (len(c.entry) - n_entry)
        c.exit_src += [f.rel] * (len(c.exits) - n_exits)
    c.file = None

    # tasks from YAML fill in CrewAI steps that were declared via config keys
    yaml_tasks = dict(c.yaml_tasks)
    for step in c.steps:
        fields = yaml_tasks.pop(step.get("_task", ""), None)
        if fields:
            step["description"] = step["description"] or _short(fields.get("description", ""), 200)
            if not step["agents"] and fields.get("agent"):
                step["agents"] = [fields["agent"]]
    for name, fields in c.yaml_tasks:
        if name in yaml_tasks:
            c.steps.append({"title": _pretty(name), "agents": [fields["agent"]] if fields.get("agent") else [],
                            "description": _short(fields.get("description", ""), 200)})

    # Command(goto=...) edges come from the function that implements a node
    def _unfn(ref: str) -> str:
        return c.node_funcs.get(ref[4:], ref[4:]) if ref.startswith("@fn:") else ref

    for e in c.edges:
        e["from"] = _unfn(e["from"])
    c.exits = [_unfn(x) for x in c.exits]

    # A LangGraph node usually wraps an agent object: fold the agent into the node
    if c.graph_nodes:
        for node_name, node_agent in c.graph_nodes.items():
            fn_names = [a for a in node_agent["aliases"] if a in c.functions]
            referenced: set[str] = set()
            for fn in fn_names:
                referenced |= {n.id for n in ast.walk(c.functions[fn]) if isinstance(n, ast.Name)}
            for t in sorted(referenced & c.tool_funcs):
                if t not in node_agent["tools"]:
                    node_agent["tools"].append(t)
            referenced |= set(node_agent["aliases"])
            for var in list(referenced):
                key = c.var_to_agent.get(var)
                inner = c.agents.get(key) if key else None
                if inner is None or inner is node_agent or any(inner is g for g in c.graph_nodes.values()):
                    continue
                node_agent["role"] = node_agent["role"] or inner["role"]
                node_agent["description"] = node_agent["description"] or inner["description"]
                node_agent["tools"] += [t for t in inner["tools"] if t not in node_agent["tools"]]
                node_agent["aliases"] += [a for a in inner["aliases"] if a not in node_agent["aliases"]]
                if not node_agent["file"]:
                    node_agent["file"], node_agent["line"] = inner["file"], inner["line"]
                del c.agents[key]

    # CrewAI sequential crews: tasks run in order, so their agents hand off in order
    if c.steps and not c.edges:
        seq = [s["agents"][0] for s in c.steps if s["agents"]]
        for a, b in zip(seq, seq[1:]):
            if bpmod.norm_key(a) != bpmod.norm_key(b):
                c.edges.append({"from": a, "to": b, "label": "next task", "conditional": False})
        if seq:
            c.entry.append(seq[0])
            c.exits.append(seq[-1])

    _scan_classes(files, c)

    # no framework: every place that calls an LLM is an agent, even when there is only one
    if not c.agents:
        found = call_sites.find(files)
        for a in found["agents"]:
            c.add_agent(a["name"], file=a["file"], line=a["line"], role=a["role"], description=a["description"],
                        tools=a["tools"], aliases=a["aliases"] + sorted(c.instances.get(a["name"], ())))
        c.edges += found["edges"]
        if not found["edges"]:  # independent features (chat, export, titles): each starts and ends on its own
            c.entry += [a["name"] for a in found["agents"]]
            c.exits += [a["name"] for a in found["agents"]]
        if found["agents"]:
            c.vote("custom", 1)

    # generic fallback: classes named *Agent with a run-ish method
    if not c.agents:
        for f in files:
            if f.path.suffix != ".py":
                continue
            for m in re.finditer(r"^class\s+(\w*Agent)\b[^\n]*:", f.text, re.M):
                if m.group(1) not in {"Agent", "BaseAgent", "AbstractAgent"}:
                    c.add_agent(m.group(1), file=f.rel, line=f.text.count("\n", 0, m.start()) + 1)
            for m in re.finditer(r"^(?:async\s+)?def\s+(\w+_agent)\s*\(", f.text, re.M):
                c.add_agent(m.group(1), file=f.rel, line=f.text.count("\n", 0, m.start()) + 1)

    for e in c.edges:
        e["from"] = c.node_funcs.get(e["from"], e["from"])

    left_out = _drop_ports(c) + _drop_unselected_roles(c, files)
    if alternatives is not None:
        alternatives += left_out
    framework = _framework(c)
    names = [a["name"] for a in c.agents.values()]
    not_drawn = "".join(f" Not drawn: {', '.join(alt['paths'])} ({alt['note']})." for alt in left_out)
    raw = {
        "project": root.name if root.is_dir() else root.stem,
        "framework": framework,
        "summary": (f"Static analysis found {len(names)} agent(s): {', '.join(names[:8])}"
                    + ("…" if len(names) > 8 else "") + "." + not_drawn
                    + " Run `visualagents scan --llm` for a richer description.")
        if names else "No agents were detected statically. Run `visualagents scan --llm` to analyse the code with an LLM.",
        "agents": list(c.agents.values()),
        "edges": c.edges,
        "entry": c.entry,
        "exits": c.exits,
        "steps": [{k: v for k, v in s.items() if not k.startswith("_")} for s in c.steps],
    }
    for a in raw["agents"]:
        a["keywords"] = _keywords_for(a)
    return bpmod.stamp(bpmod.normalize(raw), "static")


def _keywords_for(agent: dict[str, Any]) -> list[str]:
    words = re.findall(r"[A-Za-z]{5,}", f"{agent['role']} {agent['description']}")
    stop = {"agent", "about", "their", "which", "should", "using", "based", "provide", "information", "would",
            "these", "there", "other", "every", "where", "while", "given", "helpful", "assistant", "expert",
            "senior", "responsible", "following", "always", "possible"}
    out = []
    for w in words:
        lw = w.lower()
        if lw not in stop and lw not in out:
            out.append(lw)
    return out[:8]
