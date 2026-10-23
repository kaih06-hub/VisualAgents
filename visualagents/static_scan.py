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
from . import call_sites, run_order
from .collector import SKIP_DIRS, SourceFile, collect, is_test_file, parse_python

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
# graph-building calls: LangGraph's where it is imported, a project's own workflow API (literal node names) where no
# graph library is, and nothing in files that build networkx / graphviz graphs
GRAPH_CALLS = {"add_node", "add_edge", "add_conditional_edges", "set_entry_point", "set_finish_point", "Command"}
OTHER_GRAPH_LIBS = ("networkx", "pyvis", "graphviz", "pygraphviz", "pydot", "igraph", "rustworkx", "dgl", "torch_geometric")
# a project's own agent base class: BaseAgent, LLMAgent, AbstractAgent, AgentBase, Role
OWN_AGENT_BASE = re.compile(r"(Agent|AgentBase|Role)$")
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
        self.role_lists: list[tuple[str, int, list[str]]] = []  # (file, line, names) of default roles lists
        self.trees: dict[str, ast.Module] = {}  # file -> its parsed Python, parsed once
        self.project_frameworks: set[str] = set()  # frameworks any file of the project imports
        self.agent_classes: set[str] = set()  # names of the classes found to be agents or agent programs
        self.test_files: set[str] = set()  # the project's tests: they restate agents with stubs and fixtures
        self.node_keys: dict[str, set[str]] = {}  # graph node -> names its action is built from ("coder_agent")
        self.builder_ends: dict[str, tuple[list[str], list[str]]] = {}  # "file:builder" -> (entry nodes, exit nodes)
        self.compiled: dict[str, str] = {}  # variable -> the graph builder it holds compiled: sub = builder.compile()
        self.subgraphs: list[tuple[str, str]] = []  # (node, builder) for a node that runs a whole compiled graph
        self.on_their_own: set[str] = set()  # model callers with no known caller: each its own start and end
        self.unused_classes: list[dict[str, str]] = []  # agent classes nothing in the project creates or names
        self.class_defs: tuple[list[Any], set[str]] | None = None  # every class of the project, parsed once
        self.own_defs: dict[str, ast.AST] = {}  # the functions the project defines at module level, by name
        self.node_args: dict[str, set[str]] = {}  # graph node -> names handed to the factory that builds its action
        self.edge_tables: set[tuple[str, str]] = set()  # add_edge(e["from"], e["to"]): the keys of its edge dicts
        self.group_chats: list[dict[str, Any]] = []  # members and speaker selection of each group chat
        self.humans: set[str] = set()  # agents that stand for the person (UserProxyAgent, a group chat's admin)
        self.builder_nodes: dict[str, set[str]] = {}  # "file:builder" -> the nodes added to it
        self.templates: dict[str, set[str]] = {}  # "file:builder" built by a function per node -> those outer nodes
        self.llm_methods: dict[str, set[str]] = {}  # project class -> its methods that reach a model
        self.llm_functions: set[str] = set()  # the project's module-level functions that reach a model
        self.callbacks: list[tuple[str, str, str]] = []  # (agent, function, callback keyword) given to an agent
        self.record_agents: list[str] = []  # agents written as data records: AgentConfig(name=..., system_prompt=...)

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
    f = node.func.value if isinstance(node.func, ast.Subscript) else node.func  # Agent[Context](...)
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
    if isinstance(node, ast.Dict):  # {**{f"speech_{p}": f"speech_{p}" for p in players}, "end": END} unpacks a comprehension
        return [r for r in (_ref(v.value if k is None and isinstance(v, ast.DictComp) else v)
                            for k, v in zip(node.keys, node.values)) if r]
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):  # [f"vote_{p}" for p in alive]
        return _list_refs(node.elt) if not isinstance(node.elt, ast.Call) else _send_targets(node.elt)
    r = _ref(node)
    return [r] if r else []


_AGENT_TOOL = re.compile(r"(^|_)?Agent_?Tool$|^agent_?tool$|^AgentAsTool$", re.I)


def _agent_tool_target(elt: ast.AST) -> str | None:
    """AgentTool(agent=writer) / AgentTool(writer): a tool that runs another agent - that agent's name."""
    if not isinstance(elt, ast.Call) or not _AGENT_TOOL.search(_call_name(elt) or ""):
        return None
    arg = _kw(elt, "agent") or (elt.args[0] if elt.args else None)
    return _ref(arg) if isinstance(arg, (ast.Name, ast.Attribute)) else None


_RECORD_CLS = re.compile(r"Agent(Config|Spec|Definition|Def|Settings|Profile|Card)$|^(Agent|Role)(Config|Spec)")
_PROMPT_KWARGS = ("system_prompt", "instructions", "instruction", "system_message", "prompt", "persona")
_HUB = re.compile(r"(^|_)(orchestrat\w*|router|route|supervis\w*|dispatch\w*|triage|concierge|coordinat\w*)($|_)", re.I)


def _agent_record(cls: str, call: ast.Call) -> str | None:
    """The name of an agent described as a record - AgentConfig(name="Billing", system_prompt="You ...") - or None."""
    if not cls or not _RECORD_CLS.search(cls) or cls in AGENT_CLASSES:
        return None
    name = _lit(_kw(call, "name", "agent_name"))
    return name if name and _kw(call, *_PROMPT_KWARGS) is not None else None


def _record_hub(c: _Collector) -> None:
    """Agents written as records are run by the code that picks among them: a project function or method named as
    an orchestrator / router / supervisor / dispatcher / triage step. It hands each its turn and gets the user back."""
    recs = [r for r in dict.fromkeys(c.record_agents) if bpmod.norm_key(r) in c.agents]
    if len(recs) < 2:
        return
    keys = {bpmod.norm_key(r) for r in recs}
    if any(bpmod.norm_key(e["from"]) in keys or bpmod.norm_key(e["to"]) in keys for e in c.edges):
        return
    found: dict[str, set[str]] = {}  # hub function -> the classes it is a method of
    for rel, t in c.trees.items():
        if rel in c.test_files:
            continue
        for cls in [None, *(n for n in ast.walk(t) if isinstance(n, ast.ClassDef))]:
            for n in (cls.body if cls is not None else t.body):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _HUB.search(n.name) \
                        and not n.name.startswith("test"):
                    found.setdefault(n.name, set()).update([cls.name] if cls is not None else [])
    if len(found) != 1:
        return
    hub, owners = next(iter(found.items()))
    for cls in owners:  # workflow = ConciergeAgent(...): the object that runs the hub is the hub, not another agent
        for var in c.instances.get(cls, ()):
            a = c.agents.get(bpmod.norm_key(var))
            if a is not None and set(a["aliases"]) <= {var, cls}:
                del c.agents[bpmod.norm_key(var)]
    if bpmod.norm_key(hub) not in c.agents:
        c.add_agent(hub, file="", line=0, role="Picks which agent handles the request")
    for r in recs:
        c.edges.append({"from": hub, "to": r, "label": "routes", "conditional": True})
        c.edges.append({"from": r, "to": hub, "label": "hands back", "conditional": True})
    c.entry.append(hub)


_LEAD = re.compile(r"(^|_)(planner|orchestrator|supervisor|manager|coordinator|dispatcher|router|leader|boss|chief)($|_)")


def _lone_hub(c: _Collector) -> None:
    """Roles that never name each other in code (a framework session passes their messages) still answer to the one
    role that plans the work: with no edges at all and exactly one planner / orchestrator / supervisor among three or
    more agents, it hands work to each and gets the results back."""
    if c.edges or len(c.agents) < 3:
        return
    leads = [k for k in c.agents if _LEAD.search(k)]
    if len(leads) != 1:
        return
    hub = c.agents[leads[0]]["name"]
    for k, a in c.agents.items():
        if k != leads[0]:
            c.edges.append({"from": hub, "to": a["name"], "label": "assigns", "conditional": True})
            c.edges.append({"from": a["name"], "to": hub, "label": "reports", "conditional": True})
    c.entry = [hub]
    c.exits = [hub]
    c.on_their_own = set()


def _send_targets(node: ast.AST) -> list[str]:
    """Send("search", state) / [Send("search", s) for s in queries]: the nodes a router fans out to."""
    return [t for t in (_lit(n.args[0]) for n in ast.walk(node)
                        if isinstance(n, ast.Call) and _call_name(n) == "Send" and n.args) if t]


def _assigned_names(parent: ast.AST) -> list[str]:
    if isinstance(parent, ast.Assign):
        return [t.id if isinstance(t, ast.Name) else t.attr for t in parent.targets if isinstance(t, (ast.Name, ast.Attribute))]
    if isinstance(parent, (ast.AnnAssign, ast.AugAssign)) and isinstance(parent.target, (ast.Name, ast.Attribute)):
        t = parent.target
        return [t.id if isinstance(t, ast.Name) else t.attr]
    return []


def _imported_names(tree: ast.AST) -> set[str]:
    """Names brought in by `from x import name` from a framework (not from the project's own modules)."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if any(node.module == p or node.module.startswith(p + ".") or node.module.startswith("langchain_")
                   for _, prefixes in FRAMEWORK_IMPORTS for p in prefixes):
                out |= {a.asname or a.name for a in node.names}
    return out


def _imported_modules(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module)
    return out


# parts of a framework that only prepare data (split text, load files) - importing one builds no agent
UTILITY_IMPORTS = ("langchain_text_splitters", "langchain.text_splitter", "langchain.document_loaders",
                   "langchain_community.document_loaders", "llama_index.core.node_parser", "llama_index.readers")


def _frameworks_imported(modules: set[str], src: SourceFile, root: Path) -> list[str]:
    """Frameworks a file imports. A framework's name in a URL or a comment is not an import, and
    `from agents import ...` is the OpenAI Agents SDK only when the project has no agents module of its own."""
    modules = {m for m in modules if not any(m == u or m.startswith(u + ".") for u in UTILITY_IMPORTS)}
    found = []
    for fw, prefixes in FRAMEWORK_IMPORTS:
        if any(m == p or m.startswith(p + ".") or (p == "langchain" and m.startswith("langchain_"))
               for m in modules for p in prefixes):
            if fw == "openai-agents" and _local_module(src, root, "agents"):
                continue
            found.append(fw)
    return found


def _local_module(src: SourceFile, root: Path, name: str) -> bool:
    """A sibling agents.py shadows the SDK only from a folder Python runs scripts in - not from inside a package,
    where `from agents import` is absolute."""
    d = src.path.parent
    while d == root or root in d.parents:
        if not (d / "__init__.py").is_file() and ((d / f"{name}.py").is_file() or (d / name).is_dir()):
            return True
        d = d.parent
    return False


def _str_items(node: ast.AST | None) -> list[str]:
    """["planner", "coder"] -> its names, when it is a list of two or more identifiers."""
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


def _parse(src: SourceFile) -> ast.Module | None:
    return parse_python(src.text)


def _scan_python(src: SourceFile, c: _Collector) -> None:
    tree = c.trees.get(src.rel) or _parse(src)
    if tree is None:
        return
    text = src.text
    modules = _imported_modules(tree)
    imported = _frameworks_imported(modules, src, c.root)
    for fw in imported:
        c.vote(fw, 2)
    _role_lists(tree, src.rel, c)
    # add_node / add_edge are LangGraph's where this file imports it, or where the project does (a builder split
    # from the module that creates the StateGraph); without it they are the project's own workflow API, read only
    # with literal node names; a file that builds networkx / graphviz graphs has no agent graph at all
    other_graph = any(m == lib or m.startswith(lib + ".") for m in modules for lib in OTHER_GRAPH_LIBS)
    langgraph = "langgraph" in imported or ("langgraph" in c.project_frameworks and not other_graph)

    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            c.functions[node.name] = node

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if langgraph and _call_name(node.value) == "compile" and isinstance(node.value.func, ast.Attribute):
                for t in _assigned_names(node):
                    c.compiled[t] = f"{src.rel}:{_ref(node.value.func.value) or ''}"
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

    for node in ast.walk(tree):  # handoffs wired after the agents exist: a.handoffs = [b, c]; a.handoffs.append(d)
        target, value = None, None
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            value = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("append", "extend") \
                and node.args:
            target, value = node.func.value, node.args[0] if node.func.attr == "extend" else ast.List(elts=node.args)
        if isinstance(target, ast.Attribute) and target.attr == "handoffs" and value is not None:
            owner = _ref(target.value)
            for ref in _list_refs(value) if owner else []:
                c.edges.append({"from": owner, "to": ref, "label": "handoff", "conditional": True})

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        parent = parents.get(node)
        targets = _assigned_names(parent) if parent is not None else []
        if langgraph and name in ("add_edge", "set_entry_point", "set_finish_point") and \
                isinstance(node.func, ast.Attribute) and node.args:
            ends = c.builder_ends.setdefault(f"{src.rel}:{_ref(node.func.value) or ''}", ([], []))
            first, last = _ref(node.args[0]), _ref(node.args[-1])
            if name == "set_entry_point" or (name == "add_edge" and first in START_NAMES):
                ends[0].append(last or "")
            if name == "set_finish_point" or (name == "add_edge" and last in END_NAMES and len(node.args) > 1):
                ends[1].append(first or "")
        if name in GRAPH_CALLS and not langgraph and (other_graph or name == "Command" or not node.args or not all(
                _lit(a) for a in node.args[:1 if name in ("add_node", "add_conditional_edges") else 2])):
            continue

        # -- graph builders (LangGraph) ------------------------------------
        if name == "add_node" and isinstance(node.func, ast.Attribute):
            if langgraph:
                c.vote("langgraph")
            first = node.args[0] if node.args else _kw(node, "node")
            second = node.args[1] if len(node.args) > 1 else _kw(node, "action")
            made = _node_names(first, second, node, parents, tree, c)
            c.builder_nodes.setdefault(f"{src.rel}:{_ref(node.func.value) or ''}", set()).update(n for n, _ in made)
            tmpl = _template_builder(second, tree, src.rel)
            if tmpl:  # add_node(name, _analyst_graph(spec, ...)): a function that builds and compiles a graph per node
                c.templates.setdefault(tmpl, set()).update(n for n, _ in made)
            for node_name, fn_name in made:
                aliases = [fn_name] if fn_name and fn_name != node_name else []
                ag = c.add_agent(node_name, file=src.rel, line=node.lineno, aliases=aliases)
                c.graph_nodes[node_name] = ag
                if fn_name:
                    c.node_funcs[fn_name] = node_name
                # an agent built for the node by a wrapper: add_node("coder", wrap(agents["coder_agent"], "coder_agent"))
                if isinstance(second, (ast.Call, ast.Subscript)):
                    c.node_keys.setdefault(node_name, set()).update(
                        v for v in (_lit(n) for n in ast.walk(second) if isinstance(n, ast.Constant)) if v and v.isidentifier())
                # ...or handed to the factory that builds the action: make_node("search", web_scraper, policy, ...)
                if isinstance(second, ast.Call):
                    c.node_args.setdefault(node_name, set()).update(
                        a.id for a in [*second.args, *(k.value for k in second.keywords)] if isinstance(a, ast.Name))
                sub = None  # add_node("write", writing_graph.compile()) / add_node("write", compiled_writer)
                if isinstance(second, ast.Call) and isinstance(second.func, ast.Attribute) and second.func.attr == "compile":
                    sub = f"{src.rel}:{_ref(second.func.value) or ''}"
                elif isinstance(second, ast.Name):
                    sub = c.compiled.get(second.id)
                if sub:
                    c.subgraphs.append((node_name, sub))
            continue
        if name == "add_edge" and isinstance(node.func, ast.Attribute) and len(node.args) >= 2 and all(
                _subscript_key(a) for a in node.args[:2]):
            # add_edge(edge["from"], edge["to"]) over a list of edge dicts: the dicts that use those keys are the edges
            c.edge_tables.add((_subscript_key(node.args[0]), _subscript_key(node.args[1])))
            continue
        if name == "add_edge" and isinstance(node.func, ast.Attribute) and len(node.args) >= 2:
            srcs = _node_refs(node.args[0], node, parents, tree, c)
            dsts = _node_refs(node.args[1], node, parents, tree, c)
            for a in srcs:
                for b in dsts:
                    if a and b:
                        c.edge(a, b)
            continue
        if name == "add_conditional_edges" and node.args:
            srcs = _node_refs(node.args[0], node, parents, tree, c)
            pm = node.args[2] if len(node.args) > 2 else _kw(node, "path_map")
            router = _ref(node.args[1]) if len(node.args) > 1 else None
            if isinstance(pm, ast.Name):  # a path map kept in a constant: add_conditional_edges(n, route, PATH_MAP)
                pm = _assigned_value(pm, node, parents, tree) or _module_value(pm.id, tree) or pm
            targets_ = _list_refs(pm) if pm is not None else []
            if not targets_ and router and router in c.functions:
                targets_ = _returned_strings(c.functions[router])
            for t in targets_:
                for srcn in srcs:
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
        record = _agent_record(name, node)
        if record:  # AgentConfig(name="Stock Lookup Agent", system_prompt=..., tools=...): an agent written as data
            c.record_agents.append(record)
        if name in AGENT_CLASSES or name in AGENT_FACTORIES or (name.endswith("Agent") and node.keywords) or record:
            if name in {"create_handoff_tool"}:
                continue
            own = c.own_defs.get(name) if isinstance(node.func, ast.Name) else None
            if name in AGENT_FACTORIES and own is not None and name not in _imported_names(tree) and not any(
                    isinstance(n, ast.Call) and _call_name(n) in AGENT_FACTORIES | AGENT_CLASSES
                    for n in ast.walk(own)):
                continue  # the project's own create_agent() that compiles its graph - not a wrapper of a factory
            agent_name = _agent_name_kw(node, tree, c)
            role = _lit(_kw(node, *ROLE_KWARGS)) or ""
            if not agent_name and name == "Agent" and _kw(node, "role") is not None:
                agent_name = role  # CrewAI: the role is the identity
                c.vote("crewai")
            if not agent_name and node.args and _lit(node.args[0]) and name not in AGENT_FACTORIES:
                agent_name = _lit(node.args[0])
            if not agent_name and targets:
                agent_name = targets[0]
            if not agent_name and isinstance(parent, ast.Return) and name in AGENT_CLASSES:
                agent_name = _enclosing_function(node, parents)  # return CodeAgent(...): the factory names it
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
            if name == "UserProxyAgent":  # the person in the chat, not a speaker the manager schedules
                c.humans.update([agent_name, *targets])
            # Swarm-style transfer functions and agent-as-tool are handoffs
            for t in tools:
                if t.startswith(("transfer_to_", "handoff_to_", "transfer_back_to_")):
                    fn = c.functions.get(t)
                    dst = _returned_name(fn) if fn else None
                    c.edges.append({"from": agent_name, "to": dst or t.split("_to_", 1)[1], "label": "handoff",
                                    "conditional": True})
            if tools_node is not None and isinstance(tools_node, (ast.List, ast.Tuple)):
                for elt in tools_node.elts:
                    wrapped = _agent_tool_target(elt)
                    if wrapped:  # ADK AgentTool(agent=x) / AgentTool(x): the agent is called as a tool
                        c.edges.append({"from": agent_name, "to": wrapped, "label": "as tool", "conditional": True})
                        ag["tools"] = [x for x in ag["tools"] if x != _call_name(elt)]
                        continue
                    if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) and elt.func.attr == "as_tool":
                        r = _ref(elt.func.value)
                        if r:
                            c.edges.append({"from": agent_name, "to": r, "label": "as tool", "conditional": True})
                            ag["tools"] = [x for x in ag["tools"] if x != r]
            for k in node.keywords:  # after_agent_callback=llm_judge_gate, output_guardrails=[check]
                if k.arg and _CALLBACK_KW.search(k.arg):
                    for fn_name in _list_refs(k.value) if isinstance(k.value, (ast.Name, ast.List, ast.Tuple)) else []:
                        c.callbacks.append((agent_name, fn_name, k.arg))
            for kw in HANDOFF_KWARGS:
                sub = _kw(node, kw)
                if sub is None or (kw == "agents" and name not in {"AgentWorkflow"}):
                    continue
                refs = _list_refs(sub)
                if isinstance(sub, ast.Name):  # managed_agents=team, with team.append(CodeAgent(name=...)) above
                    refs = _appended_agents(sub.id, node, parents, tree, c) or refs
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
            c.group_chats.append({"cls": name, "members": members, "admin": _lit(_kw(node, "admin_name")) or "",
                                  "method": _lit(_kw(node, "speaker_selection_method")) or ""})
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
            proc = _kw(node, "process")
            if isinstance(proc, ast.Name):  # process=process_type, set a few lines up
                proc = _assigned_value(proc, node, parents, tree) or proc
            process = ast.get_source_segment(text, proc) if proc is not None else ""
            mgr = _ref(_kw(node, "manager_agent"))
            if process and "hierarchical" in process:
                members = _members(_kw(node, "agents"), node, parents, tree)
                hub = mgr or "manager"
                if not mgr:
                    c.add_agent("manager", file=src.rel, line=node.lineno, role="Crew manager (auto-created)")
                for m in members:
                    c.edges.append({"from": hub, "to": m, "label": "delegates", "conditional": True})
                    c.edges.append({"from": m, "to": hub, "label": "reports", "conditional": True})
                c.entry.append(hub)


def _scope(node: ast.AST, parents: dict[ast.AST, ast.AST], kinds: tuple) -> ast.AST | None:
    cur = parents.get(node)
    while cur is not None and not isinstance(cur, kinds):
        cur = parents.get(cur)
    return cur


def _assigned_value(target: ast.AST, node: ast.AST, parents: dict[ast.AST, ast.AST], tree: ast.AST) -> ast.AST | None:
    """What a variable (in the enclosing function, else the module) or a self.attribute (in the enclosing class) was
    last set to before `node`."""
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
        scope, want = _scope(node, parents, (ast.ClassDef,)), target.attr
    elif isinstance(target, ast.Name):
        scope, want = _scope(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef)) or tree, target.id
    else:
        return None
    found = None
    for n in ast.walk(scope) if scope is not None else []:
        if isinstance(n, ast.Assign) and any(_ref(t) == want and isinstance(t, type(target)) for t in n.targets):
            if scope is not tree or getattr(n, "lineno", 0) <= getattr(node, "lineno", 0):
                found = n.value
    return found


def _members(expr: ast.AST | None, node: ast.AST, parents: dict[ast.AST, ast.AST], tree: ast.AST,
             depth: int = 0) -> list[str]:
    """The agents a crew is given, however the list is built: [a, b], workers, list(self.agents.values()), or what a
    builder method returns (agents = {}; agents["writer"] = Agent(...); return agents) - dict keys name the agents."""
    if expr is None or depth > 4:
        return []
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        return _list_refs(expr)
    if isinstance(expr, ast.Dict):
        return [r for k, v in zip(expr.keys, expr.values) for r in (_lit(k), _ref(v) if isinstance(v, ast.Name) else None)
                if r]
    if isinstance(expr, ast.Call):
        f = expr.func
        if isinstance(f, ast.Name) and f.id in ("list", "tuple", "set", "sorted") and expr.args:
            return _members(expr.args[0], node, parents, tree, depth + 1)
        if isinstance(f, ast.Attribute) and f.attr in ("values", "keys"):
            return _members(f.value, node, parents, tree, depth + 1)
        cls = _scope(node, parents, (ast.ClassDef,))
        name = f.attr if isinstance(f, ast.Attribute) and _ref(f.value) == "self" else f.id if isinstance(f, ast.Name) else None
        defs = [d for d in ast.walk(cls if isinstance(f, ast.Attribute) and cls is not None else tree)
                if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)) and d.name == name] if name else []
        for d in defs[:1]:  # a builder: what it returns
            out: list[str] = []
            for r in (n for n in ast.walk(d) if isinstance(n, ast.Return) and n.value is not None):
                if isinstance(r.value, ast.Name):
                    out += _built(r.value.id, d)
                else:
                    out += _members(r.value, r, parents, tree, depth + 1)
            return list(dict.fromkeys(out))
        return []
    if isinstance(expr, (ast.Name, ast.Attribute)):
        value = _assigned_value(expr, node, parents, tree)
        if value is not None:
            return _members(value, node, parents, tree, depth + 1)
        fn = _scope(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
        if isinstance(expr, ast.Name) and fn is not None:
            return _built(expr.id, fn)
    return []


def _built(var: str, fn: ast.AST) -> list[str]:
    """Names a function puts into a dict or list it fills: var["writer"] = ..., var.append(writer), var = [a, b]."""
    out: list[str] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and _ref(t.value) == var and isinstance(t.slice, ast.Constant):
                    out.append(_lit(t.slice))
                elif isinstance(t, ast.Name) and t.id == var and isinstance(n.value, (ast.List, ast.Tuple, ast.Dict)):
                    out += _list_refs(n.value) if not isinstance(n.value, ast.Dict) else [
                        _lit(k) for k in n.value.keys if k is not None and _lit(k)]
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append" and \
                _ref(n.func.value) == var and n.args and isinstance(n.args[0], ast.Name):
            out.append(n.args[0].id)
    return [o for o in dict.fromkeys(out) if o]


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


def _module_value(name: str, tree: ast.AST) -> ast.AST | None:
    """What a module-level name is set to: PATH_MAP = {...}."""
    for st in getattr(tree, "body", []):
        if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in st.targets):
            return st.value
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == name:
            return st.value
    return None


def _loop_of(var: str, node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.For | None:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.For, ast.AsyncFor)) and isinstance(cur.target, ast.Name) and cur.target.id == var:
            return cur
        if isinstance(cur, (ast.ListComp, ast.GeneratorExp, ast.SetComp)) and any(
                isinstance(g.target, ast.Name) and g.target.id == var for g in cur.generators):
            return cur
        cur = parents.get(cur)
    return None


def _record_values(field: str, c: Any) -> list[str]:
    """The node names a table of records gives a field: AnalystSpec(key="market", agent_node="Market Analyst"), ...
    - the literal values of `field=` in the constructor calls of the one class that sets it most often (two or more)."""
    by_cls: dict[str, list[str]] = {}
    for rel, t in c.trees.items():
        if rel in c.test_files:
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.Call):
                v = _lit(_kw(n, field))
                if v and "{" not in v:
                    by_cls.setdefault(_call_name(n), []).append(v)
    if not by_cls:
        return []
    vals = max(by_cls.values(), key=len)
    return list(dict.fromkeys(vals)) if len(vals) >= 2 else []


def _node_refs(arg: ast.AST, node: ast.AST, parents: dict[ast.AST, ast.AST], tree: ast.AST, c: Any,
               depth: int = 0) -> list[str]:
    """The node names an add_edge / add_conditional_edges argument stands for: a literal, a list of them, a loop
    variable over either, a list built from a table of records ([s.agent_node for s in specs]) - otherwise its name."""
    if depth > 3:
        return []
    if isinstance(arg, (ast.List, ast.Tuple)):
        return [r for e in arg.elts for r in _node_refs(e, node, parents, tree, c, depth + 1)]
    if isinstance(arg, (ast.ListComp, ast.GeneratorExp)) and isinstance(arg.elt, ast.Attribute) and \
            isinstance(arg.elt.value, ast.Name):
        return _record_values(arg.elt.attr, c)
    if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and _loop_of(arg.value.id, node, parents):
        return _record_values(arg.attr, c)
    if isinstance(arg, ast.Name) and _lit(arg) is None:
        looped = _loop_values(arg, node, parents)
        if looped:
            return looped
        loop = _loop_of(arg.id, node, parents)
        it = loop.iter if isinstance(loop, (ast.For, ast.AsyncFor)) else None
        if isinstance(it, ast.Name):  # for n in first_steps: ... with first_steps built above
            value = _assigned_value(it, node, parents, tree)
            if value is not None:
                return _node_refs(value, node, parents, tree, c, depth + 1)
        value = _assigned_value(arg, node, parents, tree)
        if isinstance(value, (ast.List, ast.Tuple, ast.ListComp)):
            got = _node_refs(value, node, parents, tree, c, depth + 1)
            if got:
                return got
    r = _ref(arg)
    return [r] if r else []


def _node_names(first: ast.AST | None, second: ast.AST | None, node: ast.AST, parents: dict[ast.AST, ast.AST],
                tree: ast.AST, c: Any) -> list[tuple[str, str | None]]:
    """(node name, action name) for one add_node call. A name held in a variable is never the node's name: it is
    the values a loop gives it, the string a constant holds, or the keys of the table its action is looked up in
    (add_node(key, self.node_functions[key])); otherwise the node is left off rather than named `agent_key`."""
    if first is None:
        return []
    action = _ref(second) if second is not None else None
    lit = _lit(first)
    if lit:
        return [(lit, action)]
    if second is None:  # add_node(write_draft) is named after the function - add_node(new_node) holds an object
        if isinstance(first, ast.Name):
            fn_like = first.id in c.own_defs or first.id in _imported_names(tree) or \
                _defined_function(first.id, node, parents)
            return [(first.id, first.id)] if fn_like else []
        r = _ref(first)
        return [(r, r)] if r else []
    if isinstance(first, ast.Name):
        looped = _loop_values(first, node, parents)
        if looped:
            return [(v, action) for v in looped]
        const = _const_value(first.id, tree)
        if const:
            return [(const, action)]
    if isinstance(first, ast.Attribute) and isinstance(first.value, ast.Name) and \
            _loop_of(first.value.id, node, parents):  # for spec in specs: add_node(spec.agent_node, ...)
        recs = _record_values(first.attr, c)
        if recs:
            return [(v, action) for v in recs]
    table = _node_table(second, node, parents, tree)
    if table:
        return table
    # an enum member or a module constant used as the name: Nodes.RESEARCHER, RESEARCHER
    tail = first.value if isinstance(first, ast.Attribute) and first.attr == "value" else first
    r = _ref(tail)
    return [(r, action)] if r and r.isupper() else []


def _agent_name_kw(call: ast.Call, tree: ast.AST, c: Any) -> str | None:
    """name="writer", or name=WRITER_NAME with the constant defined in the module or elsewhere in the project."""
    kw = _kw(call, "name", "agent_name")
    lit = _lit(kw)
    if lit or not isinstance(kw, (ast.Name, ast.Attribute)):
        return lit
    const = kw.id if isinstance(kw, ast.Name) else kw.attr
    return _const_value(const, tree) or _project_const(const, c)


def _project_const(name: str, c: Any) -> str | None:
    """The string a module-level constant holds in exactly one of the project's files."""
    found = {v for t in c.trees.values() for v in [_const_value(name, t)] if v}
    return next(iter(found)) if len(found) == 1 else None


def _appended_agents(var: str, node: ast.AST, parents: dict[ast.AST, ast.AST], tree: ast.AST, c: Any) -> list[str]:
    """The agents a function puts in a list it hands over: team = [a]; team.append(CodeAgent(name=...)); team += [b]."""
    fn = parents.get(node)
    while fn is not None and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
        fn = parents.get(fn)
    if fn is None:
        return []
    items: list[ast.AST] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("append", "extend") and \
                isinstance(n.func.value, ast.Name) and n.func.value.id == var and n.args:
            arg = n.args[0]
            items += list(arg.elts) if n.func.attr == "extend" and isinstance(arg, (ast.List, ast.Tuple)) else [arg]
        elif isinstance(n, (ast.Assign, ast.AugAssign)) and isinstance(n.value, (ast.List, ast.Tuple)) and any(
                isinstance(t, ast.Name) and t.id == var for t in (n.targets if isinstance(n, ast.Assign) else [n.target])):
            items += list(n.value.elts)
    out = []
    for it in sorted(items, key=lambda x: (getattr(x, "lineno", 0), getattr(x, "col_offset", 0))):
        name = _agent_name_kw(it, tree, c) if isinstance(it, ast.Call) else _ref(it)
        if name and name not in out:
            out.append(name)
    return out


def _defined_function(name: str, node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Is `name` a function defined in a scope enclosing `node` (a nested def handed to add_node)?"""
    cur = parents.get(node)
    while cur is not None:
        for st in getattr(cur, "body", []):
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name == name:
                return True
        cur = parents.get(cur)
    return False


def _const_value(name: str, tree: ast.AST) -> str | None:
    """The string a module-level constant holds: RESEARCHER = "researcher"."""
    for st in getattr(tree, "body", []):
        if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in st.targets):
            return _lit(st.value) if isinstance(st.value, ast.Constant) else None
    return None


def _subscript_key(node: ast.AST) -> str | None:
    """edge["from"] -> "from" (also inside `edge["to"] if edge["to"] != "END" else END`)."""
    if _lit(node) is not None:
        return None
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and _lit(n.slice):
            return _lit(n.slice)
    return None


def _node_table(action: ast.AST | None, node: ast.AST, parents: dict[ast.AST, ast.AST],
                tree: ast.AST) -> list[tuple[str, str | None]]:
    """add_node(key, table[key]) or `fn = table.get(key); add_node(key, fn)` with `table = {"refiner": refine, ...}`
    written as a dict literal in the file: one node per key."""
    lookup = action
    if isinstance(action, ast.Name):  # the action was looked up a line earlier
        fn = parents.get(node)
        while fn is not None and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            fn = parents.get(fn)
        lookup = next((st.value for st in ast.walk(fn) if isinstance(st, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == action.id for t in st.targets)), None) if fn is not None else None
    if isinstance(lookup, ast.Subscript):
        table = lookup.value
    elif isinstance(lookup, ast.Call) and isinstance(lookup.func, ast.Attribute) and lookup.func.attr == "get":
        table = lookup.func.value
    else:
        return []
    table_name = _ref(table)
    if not table_name:
        return []
    for st in ast.walk(tree):
        if isinstance(st, (ast.Assign, ast.AnnAssign)) and isinstance(st.value, ast.Dict):
            targets = st.targets if isinstance(st, ast.Assign) else [st.target]
            if any(_ref(t) == table_name for t in targets) and st.value.keys and \
                    all(k is not None and _lit(k) for k in st.value.keys):
                return [(_lit(k), _ref(v)) for k, v in zip(st.value.keys, st.value.values)]
    return []


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parents.get(cur)
    return None


def _returned_strings(fn: ast.AST) -> list[str]:
    """The node names a router returns: "write", f"speech_{pid}", [f"vote_{p}" for p in alive], [Send("search", s)]."""
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and n.value is not None:
            s = _lit(n.value)
            if s:
                out.append(s)
            elif isinstance(n.value, (ast.List, ast.Tuple, ast.ListComp, ast.GeneratorExp)):
                out += _send_targets(n.value) or (_list_refs(n.value) if _lit_like(n.value) else [])
            elif isinstance(n.value, ast.Call):
                out += _send_targets(n.value)
    return list(dict.fromkeys(out))


def _lit_like(node: ast.AST) -> bool:
    """A list of string literals / f-strings (not of variables)."""
    elts = [node.elt] if isinstance(node, (ast.ListComp, ast.GeneratorExp)) else getattr(node, "elts", [])
    return bool(elts) and all(_lit(e) is not None for e in elts)


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
    """Config files can set the default roles list too: {"app.roles": ["planner", "coder"]}."""
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


def _class_defs(files: list[SourceFile], trees: dict[str, ast.Module]) -> tuple[list[_ClassDef], set[str]]:
    """Every class, and the names of everything the project calls (to tell which classes it ever creates)."""
    out, called = [], set()
    for f in files:
        tree = trees.get(f.rel)
        if tree is None or is_test_file(f.rel, f.text):
            continue
        mods, names = set(), {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.asname or a.name for a in node.names if a.name == "dspy")
            elif isinstance(node, ast.ImportFrom) and node.module == "dspy":
                names.update({a.asname or a.name: a.name for a in node.names})
            elif isinstance(node, ast.Call):  # Agent(...), Agent.from_config(...), Agent[Ctx](...)
                parts = _dotted(node.func).split(".")
                called.update({parts[0], parts[-1]})
        out += [_ClassDef(n, f.rel, mods, names) for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    return out, called


def _cached_defs(files: list[SourceFile], c: _Collector) -> tuple[list[_ClassDef], set[str]]:
    if c.class_defs is None:
        c.class_defs = _class_defs(files, c.trees)
    return c.class_defs


def _scan_classes(files: list[SourceFile], c: _Collector) -> None:
    """Agents a project defines as classes rather than builds from a framework's constructors: subclasses of an
    agent base (Reviewer(Agent), through the project's own classes too) and DSPy programs - dspy.Module
    subclasses that run predictors of their own (an asker, an answerer). A program an agent class holds is that
    agent's tool; a module that calls programs in turn (a writer asking an expert, round after round) is how work
    flows between them. A program nothing in the project creates is not part of a run; the others say which
    classes create them, so the model can tell a pipeline step from a part of something bigger."""
    defs, called = _cached_defs(files, c)
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

    def abstract(d: _ClassDef) -> bool:  # a base others build on - unless the project also creates it directly
        return d.name.startswith(("Base", "Abstract")) or d.name in ("Agent", "Role", "AgentBase") \
            or any(b.split(".")[-1] in ("ABC", "ABCMeta") for b in d.bases) or (d.name in subclassed and d.name not in called)

    def agent_base(d: _ClassDef, b: str, last: str) -> bool:  # a framework's agent class, or the project's own
        return last in AGENT_CLASSES or (last in by_name and bool(OWN_AGENT_BASE.search(last)))

    agent_classes = [d for d in defs if not abstract(d) and inherits(d, agent_base)]
    # Fact: agent classes the project defines but never creates, names or subclasses (roles only re-exported by an
    # __init__, an earlier design kept in the tree) - most likely not part of a run, when other agent classes are
    used = {n.id if isinstance(n, ast.Name) else n.attr for rel, tree in c.trees.items() if rel not in c.test_files
            for n in ast.walk(tree) if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(n.ctx, ast.Load)}
    if any(d.name in used for d in agent_classes):
        # ...and that nothing names in text either: a class loaded by its module or folder name from configuration
        # ("roles: [planner, code_interpreter]") is used
        words = {w.lower() for f in files if f.rel not in c.test_files and f.path.suffix.lower() not in (".py", ".ipynb")
                 for w in re.findall(r"\w+", f.text)}
        words |= {w.lower() for rel, tree in c.trees.items() if rel not in c.test_files for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str) for w in re.findall(r"\w+", n.value)}
        for d in agent_classes:
            path = _posix(d.rel)
            if d.name not in used and not {d.name.lower(), path.stem.lower(), path.parent.name.lower()} & words:
                c.unused_classes.append({"class": d.name, "file": d.rel})
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

    c.agent_classes |= {d.name for d in agent_classes} | set(programs)
    created = {n for n in programs if n in called} or set(programs)
    standalone = set()
    for name, d in programs.items():
        boss = owner(name)
        agent = c.agents.get(agent_keys[boss]) if boss else None
        if agent is not None:
            if name not in agent["tools"]:
                agent["tools"].append(name)
            continue
        if name not in created:
            continue
        makers = sorted(held_by.get(name, ()))
        c.add_agent(name, file=d.rel, line=d.node.lineno, description=d.doc,
                    role="DSPy program" + (f", created by {', '.join(makers)}" if makers else ""))
        standalone.add(name)

    for d in modules:  # forward(): one program asks, the next answers, for so many rounds
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
            c.edges.append({"from": a, "to": b, "label": "", "conditional": False})
        if looped and len(set(order)) > 1:
            c.edges.append({"from": order[-1], "to": order[0], "label": "next round", "conditional": True})

    for name in sorted(standalone):
        _split_dspy_program(c, programs[name], name)


def _split_dspy_program(c: _Collector, d: Any, name: str) -> None:
    """A DSPy program whose forward() runs three or more predictors of its own with different signatures
    (self.guard = dspy.Predict(InputGuard), self.answer = dspy.ChainOfThought(Answer), ...) is drawn as those steps, in
    the order forward() runs them; hand-offs to the program start at its first step and leave from its last."""
    preds: dict[str, tuple[str, str]] = {}  # attribute -> (predictor kind, signature)
    for node in ast.walk(d.node):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and \
                d.dspy_attr(node.value.func) in DSPY_PREDICTORS:
            sig = node.value.args[0] if node.value.args else _kw(node.value, "signature")
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    preds[t.attr] = (d.dspy_attr(node.value.func), _ref(sig) or "")
    fwd = next((m for m in d.node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and m.name in ("forward", "aforward", "__call__")), None)
    if fwd is None or len({sig for _, sig in preds.values()}) < 3:
        return
    order: list[str] = []
    for call in _calls_in_order(fwd):
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self" and f.attr in preds \
                and f.attr not in order:
            order.append(f.attr)
    if len({preds[a][1] for a in order}) < 3:
        return
    key = bpmod.norm_key(name)
    old = c.agents.pop(key, None)
    if old is None:
        return
    for i, attr in enumerate(order):
        kind, sig = preds[attr]
        c.add_agent(attr, file=d.rel, line=d.node.lineno, role=f"{kind} step of {name}",
                    description=old["description"] if i == 0 else "",
                    aliases=[a for a in (sig, f"{name}.{attr}", name if i == 0 else "") if a])
    for a, b in zip(order, order[1:]):
        c.edges.append({"from": a, "to": b, "label": "", "conditional": False})
    olds = {key, *(bpmod.norm_key(a) for a in old["aliases"])}
    for e in c.edges:
        if bpmod.norm_key(e["from"]) in olds:
            e["from"] = order[-1]
        if bpmod.norm_key(e["to"]) in olds:
            e["to"] = order[0]


# a role's actions and the actions it watches for: roles hand work on by publishing what the next one watches
ACTION_SETTERS = ("_init_actions", "set_actions", "init_actions", "_set_actions")
WATCHERS = ("_watch", "watch", "subscribe", "_subscribe")


def _message_routes(files: list[SourceFile], c: _Collector) -> None:
    """Publish/subscribe teams: a role class declares the actions it runs (self.set_actions([WriteSpec])) and the
    actions it reacts to (self._watch([WriteSpec])). A role whose action another watches hands work to it; a role
    watching something no role produces (the user's requirement) is where a run starts."""
    defs, _ = _cached_defs(files, c)
    by_class: dict[str, str] = {}
    for key, a in c.agents.items():
        for n in [a["name"], *a["aliases"]]:
            by_class.setdefault(n, key)
    outs: dict[str, set[str]] = {}
    watches: dict[str, set[str]] = {}
    for d in defs:
        if d.name not in by_class:
            continue
        for call in d.calls():
            f = call.func
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self"
                    and call.args):
                continue
            names = {r for r in _list_refs(call.args[0]) if r[:1].isupper()} \
                if isinstance(call.args[0], (ast.List, ast.Tuple)) else set()
            if f.attr in ACTION_SETTERS:
                outs.setdefault(d.name, set()).update(names)
            elif f.attr in WATCHERS:
                watches.setdefault(d.name, set()).update(names)
    produced = set().union(*outs.values()) if outs else set()
    for b, watched in watches.items():
        senders = [a for a, made in outs.items() if a != b and made & watched]
        for a in senders:
            c.edges.append({"from": c.agents[by_class[a]]["name"], "to": c.agents[by_class[b]]["name"],
                            "label": ", ".join(sorted(outs[a] & watched)), "conditional": False})
        if watched and not watched & produced:
            c.entry.append(c.agents[by_class[b]]["name"])


def _template_builder(action: ast.AST | None, tree: ast.AST, rel: str) -> str | None:
    """`_analyst_graph(spec, agent)` where the module's function builds a graph and returns `graph.compile()`:
    the builder, as "file:variable"."""
    if not (isinstance(action, ast.Call) and isinstance(action.func, ast.Name)):
        return None
    fn = next((st for st in getattr(tree, "body", []) if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))
               and st.name == action.func.id), None)
    for r in (n for n in ast.walk(fn) if isinstance(n, ast.Return)) if fn is not None else []:
        v = r.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "compile" and \
                isinstance(v.func.value, ast.Name):
            return f"{rel}:{v.func.value.id}"
    return None


def _fold_template_graphs(c: _Collector) -> None:
    """A graph a function builds afresh for each of several nodes (each analyst's own model + tools loop) is those
    nodes' private machinery: its inner nodes leave the map. Built for one node only, it is a subgraph like any other."""
    for builder, outer in c.templates.items():
        if len(outer) < 2:
            for o in outer:
                c.subgraphs.append((o, builder))
            continue
        inner = c.builder_nodes.get(builder, set()) - outer
        for n in inner:
            ag = c.graph_nodes.pop(n, None)
            key = bpmod.norm_key(n)
            if ag is not None and c.agents.get(key) is ag:
                del c.agents[key]
        c.edges = [e for e in c.edges if e["from"] not in inner and e["to"] not in inner]
        c.entry = [e for e in c.entry if e not in inner]
        c.exits = [x for x in c.exits if x not in inner]
        c.builder_ends.pop(builder, None)


def _link_subgraphs(c: _Collector) -> None:
    """A node that runs a whole compiled graph hands work to that graph's entry nodes and gets it back from its exit
    nodes; the inner graph's own start and end are not where a run starts or ends."""
    for node, builder in c.subgraphs:
        entries, exits = c.builder_ends.get(builder, ([], []))
        for e in dict.fromkeys(entries):
            c.edges.append({"from": node, "to": e, "label": "runs", "conditional": False})
        for x in dict.fromkeys(exits):
            c.edges.append({"from": x, "to": node, "label": "returns", "conditional": False})
        c.entry = [e for e in c.entry if e not in entries]
        c.exits = [x for x in c.exits if x not in exits]


def _declared_names(d: _ClassDef) -> list[str]:
    """Names a class gives its agent: super().__init__(agent_name="coder_agent"), name = "coder"."""
    out = []
    for node in d.node.body:
        if isinstance(node, ast.Assign) and _lit(node.value) and any(
                isinstance(t, ast.Name) and t.id in ("name", "agent_name", "role_name") for t in node.targets):
            out.append(_lit(node.value))
    for call in d.calls():
        f = call.func
        if isinstance(f, ast.Attribute) and f.attr == "__init__":
            v = _lit(_kw(call, "agent_name", "name", "role_name"))
            if v:
                out.append(v)
    return [n for n in out if n]


def _fold_into_nodes(c: _Collector, files: list[SourceFile]) -> None:
    """A graph node whose action wraps an agent by name - add_node("Coder", wrap(agents["coder_agent"])) with
    CoderAgent declaring agent_name="coder_agent" - is that agent: one block on the map, not two."""
    if not c.graph_nodes:
        return
    declared: dict[str, list[str]] = {}
    for d in _cached_defs(files, c)[0]:
        declared.setdefault(d.name, []).extend(_declared_names(d))
    owner: dict[str, str] = {}
    for node_name, keys in c.node_keys.items():
        for k in keys:
            owner.setdefault(bpmod.norm_key(k), node_name)
    # an agent handed to one node's factory - make_fan_out_node("web_scrapers", web_scraper, policy) - is that node
    # (names handed to several nodes' factories are their shared plumbing: policy engines, audit writers)
    handed: dict[str, set[str]] = {}
    for node_name, args in c.node_args.items():
        for a in args:
            handed.setdefault(bpmod.norm_key(a), set()).add(node_name)
    for k, nodes in handed.items():
        if len(nodes) == 1:
            owner.setdefault(k, next(iter(nodes)))
    # ...and so is an agent named after a node: refiner_agent / RefinerAgent next to the node "refiner"
    for node_name in c.graph_nodes:
        for suffix in ("agent", "node"):
            owner.setdefault(bpmod.norm_key(node_name) + suffix, node_name)
    graph = {id(a) for a in c.graph_nodes.values()}
    for key, agent in list(c.agents.items()):
        if id(agent) in graph:
            continue
        names = [agent["name"], *agent["aliases"]]
        names += [n for a in names for n in declared.get(a, [])]
        node_name = next((owner[bpmod.norm_key(n)] for n in names if bpmod.norm_key(n) in owner), None)
        if node_name is None:
            continue
        node_agent = c.graph_nodes[node_name]
        node_agent["role"] = node_agent["role"] or agent["role"]
        node_agent["description"] = node_agent["description"] or agent["description"]
        node_agent["tools"] += [t for t in agent["tools"] if t not in node_agent["tools"]]
        node_agent["aliases"] += [a for a in agent["aliases"] if a not in node_agent["aliases"]]
        del c.agents[key]


_SHELL_PY = re.compile(r"\bpython[\d.]*\s+(?:-[A-Za-z]+\s+)*(-m\s+[\w.]+|[^\s;|&]+?\.py)\b")


def _shell_scripts(root: Path, limit: int = 40) -> list[tuple[str, str]]:
    """(rel path, text) of the project's shell scripts: run.sh, scripts/train.sh ..."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.endswith((".sh", ".bash")):
                p = Path(dirpath) / name
                try:
                    if p.stat().st_size < 200_000:
                        out.append((p.relative_to(root).as_posix(), p.read_text(errors="replace")))
                except OSError:
                    continue
                if len(out) >= limit:
                    return out
    return out


def _script_pipelines(c: _Collector, files: list[SourceFile]) -> list[dict[str, Any]]:
    """Shell scripts that run the project's Python scripts one after another (python plan.py; python write.py) are
    a pipeline: each script's agents hand their output to the next script's. A step that calls no model is still a
    step of the run. Returns the pipelines as facts."""
    if not c.root.is_dir():
        return []
    stems: dict[str, list[str]] = {}
    for f in files:
        if f.path.suffix == ".py" and f.rel not in c.test_files:
            stems.setdefault(PurePosixPath(_posix(f.rel)).stem, []).append(f.rel)

    def resolve(token: str) -> tuple[str, list[str]]:
        if token.startswith("-m"):
            stem = token.split()[-1].split(".")[-1]
        else:  # programmer_[humaneval/mbpp].py, solver_{a,b}.py: a pattern over variants
            stem = PurePosixPath(re.sub(r"\[[^\]]*\]|\{[^}]*\}", "*", token)).stem
        if stem in stems:
            return stem, stems[stem]
        prefix = re.split(r"[\[{*?]", stem, maxsplit=1)[0]
        if prefix != stem and len(prefix) >= 3:  # python solver_[math/code].py: each variant
            hits = [r for st, rels in stems.items() if st.startswith(prefix) for r in rels]
            return prefix.strip("_-"), hits
        return stem, []

    def agents_in(rels: list[str], step: str) -> list[str]:
        out = []
        for key, a in c.agents.items():
            if a.get("file") in rels or step in a["aliases"]:
                out.append(key)
        return out

    pipelines, seen = [], set()
    for rel, text in _shell_scripts(c.root):
        text = re.sub(r"\\\n", " ", text)
        steps: list[tuple[str, list[str]]] = []
        for line in text.splitlines():
            line = line.split("#", 1)[0]
            for m in _SHELL_PY.finditer(line):
                name, hits = resolve(m.group(1))
                if hits and (not steps or steps[-1][0] != name):
                    steps.append((name, hits))
        with_agents = [s for s in steps if agents_in(s[1], s[0])]
        if len(steps) < 2 or len(with_agents) < 2:
            continue
        order = tuple(n for n, _ in steps)
        if order in seen:
            continue
        seen.add(order)
        pipelines.append({"script": rel, "runs": list(order)})
        keys: list[list[str]] = []
        for name, hits in steps:
            ks = agents_in(hits, name)
            if not ks:  # a step that runs no model (pre-processing, a test runner) is still part of the run
                ag = c.add_agent(name, file=hits[0], line=1, role="pipeline step (calls no model itself)")
                ks = [bpmod.norm_key(name)]
                _calls_into(c, hits, ag)
            keys.append(ks)
        for a, b in zip(keys, keys[1:]):
            c.edges.append({"from": c.agents[a[-1]]["name"], "to": c.agents[b[0]]["name"], "label": "then",
                            "conditional": False})
        c.entry.append(c.agents[keys[0][0]]["name"])
        c.exits.append(c.agents[keys[-1][-1]]["name"])
    return pipelines


def _calls_into(c: _Collector, rels: list[str], step: dict[str, Any]) -> None:
    """A step script that imports code from an agent's script and calls it hands work back to that agent (a test
    runner sending failures to the programmer's fix-up)."""
    by_stem: dict[str, str] = {}
    for key, ag in c.agents.items():
        if ag is not step:
            for n in [PurePosixPath(_posix(ag.get("file") or "x")).stem, *ag["aliases"]]:
                by_stem.setdefault(n, key)
    for rel in rels:
        tree = c.trees.get(rel)
        if tree is None:
            continue
        source = {a.asname or a.name: (n.module or "").split(".")[-1]
                  for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        for name in dict.fromkeys(_call_name(n) for n in ast.walk(tree) if isinstance(n, ast.Call)):
            key = by_stem.get(source.get(name, ""))
            if key:
                c.edges.append({"from": step["name"], "to": c.agents[key]["name"], "label": "calls",
                                "conditional": False})


NAME_KWARGS = ("name", "agent_name", "role_name", "role")


def _str_list(node: ast.AST | None) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts and all(_lit(e) for e in node.elts):
        return [_lit(e) for e in node.elts]  # type: ignore[misc]
    return []


def _named_instances(files: list[SourceFile], c: _Collector) -> None:
    """One agent class played by several named instances - Player(name="Affirmative"), Player(name="Negative"), or
    [Player(name=n) for n in ROLES] - is several agents: each instance is one, and the order the code asks them in
    (a.ask(); b.ask(); judge.ask() inside a loop) is how work passes between them."""
    defs, _ = _cached_defs(files, c)
    bases = {d.name: [b.split(".")[-1] for b in d.bases] for d in defs}
    asking = c.llm_methods or call_sites.class_llm_methods([f for f in files if f.rel not in c.test_files])

    def methods(cls: str, seen: frozenset = frozenset()) -> set[str]:
        out = set(asking.get(cls, ()))
        for b in bases.get(cls, ()):
            if b not in seen:
                out |= methods(b, seen | {cls})
        return out

    def framework_agent(cls: str, seen: frozenset = frozenset()) -> bool:  # MyProxy(UserProxyAgent)
        return any(b in AGENT_CLASSES or (b not in seen and framework_agent(b, seen | {cls}))
                   for b in bases.get(cls, ()))

    # the project's classes that talk to a model, or that extend a framework's agent class: each named instance of
    # one (MyProxy(name="user"), MyProxy(name="executor")) is an agent of its own
    classes = {name for name in bases if methods(name) or framework_agent(name)}
    if not classes:
        return
    consts: dict[str, list[str]] = {}  # NAMES = ["Affirmative side", "Negative side", "Moderator"]
    for tree in c.trees.values():
        for node in tree.body:
            if isinstance(node, ast.Assign) and _str_list(node.value):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = _str_list(node.value)

    found: dict[str, dict[str, Any]] = {}  # instance name -> {"cls", "file", "line", "aliases"}
    lists: dict[str, list[str]] = {}  # self.players -> its instance names, in order
    for rel, tree in c.trees.items():
        if rel in c.test_files:
            continue
        parents = {ch: n for n in ast.walk(tree) for ch in ast.iter_child_nodes(n)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _call_name(node) in classes):
                continue
            cls, kw = _call_name(node), _kw(node, *NAME_KWARGS)
            names, holder = [], parents.get(node)
            if _lit(kw):
                names = [_lit(kw)]
            elif isinstance(kw, ast.Name) and isinstance(holder, (ast.ListComp, ast.GeneratorExp)):
                gen = next((g for g in holder.generators if isinstance(g.target, ast.Name) and g.target.id == kw.id), None)
                if gen is not None:
                    names = _str_list(gen.iter) or consts.get(getattr(gen.iter, "id", ""), [])
                holder = parents.get(holder)
            if not names:
                continue
            targets = _assigned_names(holder) if holder is not None else []
            for n in names:
                rec = found.setdefault(n, {"cls": cls, "file": rel, "line": node.lineno, "aliases": []})
                if len(names) == 1:
                    rec["aliases"] += [t for t in targets if t not in rec["aliases"]]
            if len(names) > 1:
                for t in targets:
                    lists[t] = names
    if len(found) < 2:
        return
    for tree in c.trees.values():  # self.affirmative = self.players[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript):
                src, idx = _ref(node.value.value), node.value.slice
                if src in lists and isinstance(idx, ast.Constant) and isinstance(idx.value, int) \
                        and -len(lists[src]) <= idx.value < len(lists[src]):
                    rec = found[lists[src][idx.value]]
                    rec["aliases"] += [a for a in _assigned_names(node) if a not in rec["aliases"]]

    owners: dict[str, list[str]] = {}
    for n, rec in found.items():
        for a in dict.fromkeys(rec["aliases"]):
            owners.setdefault(a, []).append(n)
    alias = {a: ns[0] for a, ns in owners.items() if len(ns) == 1}
    for cls in {rec["cls"] for rec in found.values()}:  # the class is its instances now
        key = bpmod.norm_key(cls)
        if key in c.agents and cls in c.agents[key]["aliases"] + [c.agents[key]["name"]]:
            del c.agents[key]
    for n, rec in found.items():
        c.add_agent(n, file=rec["file"], line=rec["line"], role=f"a {rec['cls']}",
                    aliases=[a for a in rec["aliases"] if alias.get(a) == n])

    def who(call: ast.Call) -> str | None:
        f = call.func
        if isinstance(f, ast.Attribute) and f.attr in methods(found[next(iter(found))]["cls"]) | {
                m for rec in found.values() for m in methods(rec["cls"])}:
            return alias.get(_ref(f.value) or "")
        return None

    for rel, tree in c.trees.items():
        if rel in c.test_files:
            continue
        for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            parents = {ch: n for n in ast.walk(fn) for ch in ast.iter_child_nodes(n)}
            seq: list[tuple[str, ast.AST | None]] = []
            for call in sorted((n for n in ast.walk(fn) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)):
                speaker = who(call)
                if not speaker:
                    continue
                loop, cur = None, parents.get(call)
                while cur is not None and cur is not fn:
                    if isinstance(cur, (ast.For, ast.AsyncFor, ast.While)):
                        loop = cur
                        break
                    cur = parents.get(cur)
                if not seq or seq[-1][0] != speaker:
                    seq.append((speaker, loop))
            for (a, _), (b, _) in zip(seq, seq[1:]):
                c.edges.append({"from": a, "to": b, "label": "then", "conditional": False})
            loops: dict[int, list[str]] = {}
            for sp, loop in seq:
                if loop is not None:
                    loops.setdefault(id(loop), []).append(sp)
            for turn in loops.values():
                if len(set(turn)) > 1:
                    c.edges.append({"from": turn[-1], "to": turn[0], "label": "next round", "conditional": True})


# -- framework and structural facts ---------------------------------------------


def _strongest(votes: dict[str, int]) -> str:
    """The framework with the most evidence; LangChain only when nothing else has any."""
    return min(votes, key=lambda fw: (fw in WEAK_FRAMEWORKS, -votes[fw], fw)) if votes else "custom"


def _framework(c: _Collector) -> str:
    """What the agents are built with: every file that defines an agent counts for its framework, and one on
    LangChain chat models or no framework counts half - so agents built on a framework outvote framework-free
    helpers, and a LangGraph team outvotes forty files that import LangChain for model calls."""
    weight: dict[str, float] = {}
    for key, a in c.agents.items():
        for file in dict.fromkeys(f for f, _ in c.sites.get(key) or [(a["file"], 0)]):
            fw = _strongest(c.file_votes.get(file, {}))
            weight[fw] = weight.get(fw, 0) + (0.5 if fw in WEAK_FRAMEWORKS else 1)
    if weight:
        return max(sorted(weight), key=lambda fw: (weight[fw], fw not in WEAK_FRAMEWORKS, c.frameworks.get(fw, 0)))
    return _strongest(c.frameworks)


def _posix(rel: str) -> PurePosixPath:
    return PurePosixPath(rel.replace(os.sep, "/"))


def _own_dir(files: set[str], others: set[str]) -> str:
    """The highest folder holding `files` and none of `others`: app/autogen_port next to app/team."""
    parts = PurePosixPath(posixpath.commonpath([str(_posix(f).parent) for f in files])).parts
    for i in range(1, len(parts) + 1):
        d = "/".join(parts[:i])
        if not any(o.startswith(d + "/") for o in others):
            return d
    return "/".join(parts)


def _ports(c: _Collector) -> list[dict[str, Any]]:
    """Fact: the same agent names defined on two frameworks - most of the smaller group's names are also in the
    bigger one, each in its own folder. Usually a port of one team; the model decides."""
    groups: dict[str, set[str]] = {}
    files: dict[str, set[str]] = {}
    for key in c.agents:
        for file, _ in c.sites.get(key, []):
            fw = _strongest(c.file_votes.get(file, {}))
            if fw not in WEAK_FRAMEWORKS:
                groups.setdefault(fw, set()).add(key)
                files.setdefault(fw, set()).add(file)
    order = list(c.agents)
    ranked = sorted(groups, key=lambda fw: (-len(groups[fw]), fw))
    out = []
    for i, big in enumerate(ranked):
        for small in ranked[i + 1:]:
            shared = groups[big] & groups[small]
            if len(shared) < 2 or 2 * len(shared) < len(groups[small]):
                continue
            out.append({
                "shared_agents": [c.agents[k]["name"] for k in sorted(shared, key=order.index)],
                "implementations": [{"framework": fw, "folder": _own_dir(files[fw], files[other]),
                                     "agents": [c.agents[k]["name"] for k in sorted(groups[fw], key=order.index)]}
                                    for fw, other in ((big, small), (small, big))],
            })
    return out


def _factories(c: _Collector) -> list[dict[str, Any]]:
    """Fact: code that creates one of several agents depending on a setting - an if/elif or match on one value
    (`if config.type == "retry": return RetryAgent.from_config(config)`), or a registry dict ({"retry": RetryAgent}).
    The setting picks which one runs."""
    agents = c.agent_classes | {k for a in c.agents.values() for k in a["aliases"] if k[:1].isupper()}

    def created(body: list[ast.stmt]) -> str | None:
        for n in (x for stmt in body for x in ast.walk(stmt)):
            if isinstance(n, ast.Call):
                cls = _dotted(n.func).split(".")[0]  # Agent(...), Agent.from_config(...), Agent[Ctx](...)
                if cls in agents:
                    return cls
        return None

    def case(test: ast.AST) -> tuple[str, str] | None:
        """(what is compared, the value it selects) for x == "v", "v" == x and isinstance(x, VConfig)."""
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left, right = test.left, test.comparators[0]
            for subject, value in ((left, right), (right, left)):
                if _lit(value) is not None and _dotted(subject):
                    return _dotted(subject), _lit(value)  # type: ignore[return-value]
        if isinstance(test, ast.Call) and _call_name(test) == "isinstance" and len(test.args) == 2:
            return f"type({_dotted(test.args[0])})", _dotted(test.args[1])
        return None

    out = []
    for rel, tree in c.trees.items():
        if rel in c.test_files:
            continue
        for node in ast.walk(tree):
            switch, options = None, {}
            if isinstance(node, ast.If) and not getattr(node, "_elif", False):
                branch: ast.AST | None = node
                while isinstance(branch, ast.If):
                    picked = case(branch.test)
                    if picked is None or (switch is not None and picked[0] != switch):
                        break
                    switch = picked[0]
                    cls = created(branch.body)
                    if cls:
                        options[picked[1]] = cls
                    nxt = branch.orelse[0] if len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.If) else None
                    if nxt is not None:
                        nxt._elif = True  # type: ignore[attr-defined]
                    branch = nxt
            elif isinstance(node, ast.Match):
                switch = _dotted(node.subject)
                for m in node.cases:
                    if isinstance(m.pattern, ast.MatchValue) and _lit(m.pattern.value) is not None:
                        cls = created(m.body)
                        if cls:
                            options[_lit(m.pattern.value)] = cls
            elif isinstance(node, ast.Dict) and len(node.keys) >= 2 and all(_lit(k) for k in node.keys):
                refs = {_lit(k): _dotted(v) for k, v in zip(node.keys, node.values)}
                if all(v in agents for v in refs.values()):
                    switch, options = "a registry key", refs
            if switch and len(set(options.values())) >= 2:
                out.append({"file": rel, "line": node.lineno, "chooses_by": switch, "options": options})
    return out


def _config_lists(c: _Collector, files: list[SourceFile]) -> list[dict[str, Any]]:
    """Fact: default lists of roles / agents / workers whose names are folders or modules of the project
    (roles = ["planner", "coder"]), so the model can tell what a default run switches on. A list of chat roles
    (["user", "assistant"]) names nothing in the project and is not reported."""
    here = {part.lower() for f in files for part in (*_posix(f.rel).parent.parts, _posix(f.rel).stem)}
    out, seen = [], set()
    for rel, line, names in c.role_lists:
        if (rel, line) not in seen and sum(1 for n in names if n.lower() in here) >= 2:
            seen.add((rel, line))
            out.append({"file": rel, "line": line, "names": names})
    return out


# -- assembly -------------------------------------------------------------


def scan(root: Path, files: list[SourceFile] | None = None, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pass a dict as `facts` to get structural facts for the LLM: `ports` (the same agent names defined on two
    frameworks), `config_lists` (default roles / agents lists naming parts of the project) and `factories` (code that
    creates one of several agents depending on a setting)."""
    root = Path(root).resolve()
    files = files if files is not None else collect(root, max_files=600)
    c = _Collector()
    c.root = root if root.is_dir() else root.parent
    c.test_files = {f.rel for f in files if is_test_file(f.rel, f.text)}
    for f in files:
        if f.path.suffix.lower() in (".py", ".ipynb"):
            tree = _parse(f)
            if tree is not None:
                c.trees[f.rel] = tree
                c.own_defs.update({n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})
                c.project_frameworks.update(_frameworks_imported(_imported_modules(tree), f, c.root))
    for f in files:
        ext = f.path.suffix.lower()
        if f.rel in c.test_files and any(g.rel not in c.test_files for g in files):
            continue
        c.file = f.rel
        if ext in {".py", ".ipynb"}:
            _scan_python(f, c)
        elif ext in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}:
            _scan_js(f, c)
        elif ext in {".yaml", ".yml"}:
            _scan_yaml(f, c)
        elif ext == ".json":
            _scan_json(f, c)
    c.file = None
    _edge_tables(c)
    _group_chat_turns(c)
    _fold_template_graphs(c)

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

    c.llm_methods = call_sites.class_llm_methods([f for f in files if f.rel not in c.test_files], c.llm_functions)
    _model_callbacks(c)
    _scan_classes(files, c)
    _split_pipeline_classes(c, files)
    _fold_into_nodes(c, files)
    run_order.subgraphs_run_in_nodes(c)
    _link_subgraphs(c)
    _named_instances(files, c)
    _message_routes(files, c)
    _companion_agents(c)
    _record_hub(c)
    run_order.link(c)
    _tool_delegation(c)

    # no framework: every place that calls an LLM is an agent, even when there is only one
    if not c.agents:
        found = call_sites.find(files)
        for a in found["agents"]:
            c.add_agent(a["name"], file=a["file"], line=a["line"], role=a["role"], description=a["description"],
                        tools=a["tools"], aliases=a["aliases"] + sorted(c.instances.get(a["name"], ())))
        in_order = [e for e in found["edges"] if e.pop("_pipeline", False)]  # call order, branches ignored
        c.edges += [e for e in found["edges"] if e not in in_order]
        if not found["edges"]:  # independent features (chat, export, titles): each starts and ends on its own
            c.entry += [a["name"] for a in found["agents"]]
            c.exits += [a["name"] for a in found["agents"]]
            c.on_their_own = {a["name"] for a in found["agents"]}
        if found["agents"]:
            c.vote("custom", 1)
            before = len(c.edges)
            run_order.link(c)  # the order an orchestrator runs them in, through its helpers and branches
            if len(c.edges) == before:  # nothing followed: keep the plain call order
                c.edges += in_order
            if len(c.edges) > before and c.on_their_own:  # linked now: not each its own start and end
                to = {bpmod.norm_key(e["to"]) for e in c.edges[before:]}
                frm = {bpmod.norm_key(e["from"]) for e in c.edges[before:]}
                c.entry = [e for e in c.entry if not (e in c.on_their_own and bpmod.norm_key(e) in to)]
                c.exits = [x for x in c.exits if not (x in c.on_their_own and bpmod.norm_key(x) in frm)]

    # generic fallback: classes named *Agent with a run-ish method
    if not c.agents:
        for f in files:
            if f.path.suffix != ".py" or f.rel in c.test_files:
                continue
            for m in re.finditer(r"^class\s+(\w*Agent)\b[^\n]*:", f.text, re.M):
                if m.group(1) not in {"Agent", "BaseAgent", "AbstractAgent"}:
                    c.add_agent(m.group(1), file=f.rel, line=f.text.count("\n", 0, m.start()) + 1)
            for m in re.finditer(r"^(?:async\s+)?def\s+(?!test_)(\w+_agent)\s*\(", f.text, re.M):
                c.add_agent(m.group(1), file=f.rel, line=f.text.count("\n", 0, m.start()) + 1)

    pipelines = _script_pipelines(c, files)
    if pipelines and c.on_their_own:  # ...unless a script pipeline runs them in order
        linked_to = {bpmod.norm_key(e["to"]) for e in c.edges}
        linked_from = {bpmod.norm_key(e["from"]) for e in c.edges}
        c.entry = [e for e in c.entry if not (e in c.on_their_own and bpmod.norm_key(e) in linked_to)]
        c.exits = [x for x in c.exits if not (x in c.on_their_own and bpmod.norm_key(x) in linked_from)]

    for e in c.edges:
        e["from"] = c.node_funcs.get(e["from"], e["from"])

    _drop_example_agents(c)
    _drop_separate_programs(c)
    _merge_mode_variants(c)
    _lone_hub(c)

    if facts is not None:
        found = (("ports", _ports(c)), ("config_lists", _config_lists(c, files)), ("factories", _factories(c)),
                 ("script_pipelines", pipelines), ("agent_classes_never_used", c.unused_classes))
        facts.update({k: v for k, v in found if v})
    framework = _framework(c)
    names = [a["name"] for a in c.agents.values()]
    raw = {
        "project": root.name if root.is_dir() else root.stem,
        "framework": framework,
        "summary": (f"Static analysis found {len(names)} agent(s): {', '.join(names[:8])}"
                    + ("…" if len(names) > 8 else "") + ". Run `visualagents scan --llm` for a richer description.")
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


_RUNS = {"run", "run_sync", "run_stream", "iter", "arun", "ainvoke", "invoke", "kickoff", "kickoff_async"}
_TOOL_KWARGS = ("tools", "extra_tools", "functions", "toolsets")
_ATTACH_DECORATORS = {"tool", "tool_plain", "output_validator", "result_validator"}


_MODE_WORDS = {"cli", "only", "clionly", "plugin", "plugins", "pluginonly", "lite", "mini", "async", "sync", "stream",
               "streaming", "local", "remote", "legacy", "old", "new", "v1", "v2", "v3", "simple", "basic", "fast",
               "mock", "dummy", "offline", "online", "web", "api", "gui", "headless", "no", "with", "without", "tools",
               "tool", "code"}


def _merge_mode_variants(c: _Collector) -> None:
    """CodeInterpreterCLIOnly / CodeInterpreterPluginOnly beside CodeInterpreter are one agent built for different
    modes: the variants join the plain one (their names add only mode words to it)."""
    graph = {id(a) for a in c.graph_nodes.values()}
    slug = {k: bpmod.slugify(a["name"]).replace("-", "_") for k, a in c.agents.items()}
    for base in sorted(slug, key=lambda k: len(slug[k])):
        if base not in c.agents or id(c.agents[base]) in graph:
            continue
        for k in [k for k in list(c.agents) if k != base and slug[k].startswith(slug[base] + "_")]:
            rest = [w for w in slug[k][len(slug[base]) + 1:].split("_") if w]
            if not rest or not all(w in _MODE_WORDS for w in rest) or id(c.agents[k]) in graph:
                continue
            a, b = c.agents[base], c.agents.pop(k)
            for n in [b["name"], *b["aliases"]]:
                if n not in a["aliases"]:
                    a["aliases"].append(n)
            a["tools"] += [t for t in b["tools"] if t not in a["tools"]]
            for e in c.edges:  # its arrows are the plain one's now
                e["from"] = a["name"] if bpmod.norm_key(e["from"]) in {k, *map(bpmod.norm_key, b["aliases"])} else e["from"]
                e["to"] = a["name"] if bpmod.norm_key(e["to"]) in {k, *map(bpmod.norm_key, b["aliases"])} else e["to"]


_CALLBACK_KW = re.compile(r"callbacks?$|guardrails?$|validators?$|^(before|after)_", re.I)


def _model_callbacks(c: _Collector) -> None:
    """A callback or guardrail given to an agent that asks a model itself (an LLM-as-judge gate run after the agent,
    an input check run before it) is an agent of its own, checking that one's work."""
    for agent, fn, kw in c.callbacks:
        if fn not in c.llm_functions or bpmod.norm_key(agent) not in c.agents:
            continue
        if bpmod.norm_key(fn) not in c.agents:
            c.add_agent(fn, file="", line=0, role="Checks the agent's work with a model")
        before = kw.lower().startswith(("before", "input"))
        c.edges.append({"from": fn if before else agent, "to": agent if before else fn,
                        "label": "checks", "conditional": False})
        if not before:
            c.edges.append({"from": fn, "to": agent, "label": "verdict", "conditional": True})


def _companion_agents(c: _Collector) -> None:
    """A class that runs two or more of the agents also runs the other model-calling parts it builds beside them
    (self.skill_manager = SkillManager(...) next to self.action_agent, or solver = MLESolver(...) in one of its
    methods): those are agents too, named after the attribute, or after the class for a local variable."""
    if len(c.agents) < 2 or not c.llm_methods:
        return
    known = {n for k, a in c.agents.items() for n in (k, a["name"], *a["aliases"])}
    for rel, tree in c.trees.items():
        if rel in c.test_files:
            continue
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            held, extra = 0, []
            for st in ast.walk(cls):
                if not (isinstance(st, ast.Assign) and isinstance(st.value, ast.Call)):
                    continue
                ctor = _call_name(st.value)
                for t in st.targets:
                    attr = t.attr if isinstance(t, ast.Attribute) and _ref(t.value) == "self" else None
                    var = attr or (t.id if isinstance(t, ast.Name) else None)
                    if not var:
                        continue
                    if (attr and attr in known) or ctor in known:
                        held += 1
                    elif ctor in c.llm_methods and ctor != cls.name:
                        extra.append((attr or ctor, ctor, var, st.lineno))
            if held < 2:
                continue
            for name, ctor, var, line in extra:
                if bpmod.norm_key(name) not in c.agents:
                    c.add_agent(name, file=rel, line=line, aliases=list(dict.fromkeys([ctor, var])))
                    known |= {name, ctor, var}


def _tool_delegation(c: _Collector) -> None:
    """Hand-offs through functions given to an agent - tools=[delegate_to_math], @supervisor.tool, a helper registering
    them (register_tools(supervisor, [delegate_to_math])), or supervisor.check = validate - that run another agent
    (math_agent.run(...), or a project helper that runs the agent it is passed)."""
    owners: dict[str, str] = {}
    for key, a in c.agents.items():
        for n in (key, a["name"], *a["aliases"]):
            k = bpmod.norm_key(n)
            if k and k != "agent":
                owners.setdefault(k, key)
    if len(set(owners.values())) < 2:
        return
    defs: dict[str, list[ast.AST]] = {}  # function name -> definitions (module level and nested)
    for rel, tree in c.trees.items():
        if rel in c.test_files:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(n.name, []).append(n)

    def agent_of(e: ast.AST | None) -> str | None:
        r = _ref(e) if isinstance(e, (ast.Name, ast.Attribute)) else None
        return owners.get(bpmod.norm_key(r)) if r else None

    def runs_params(fn: ast.AST) -> set[str]:
        """Parameters this function runs: def run_agent(agent, ...): await agent.run(...)."""
        params = {a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]}
        return {n.func.value.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in _RUNS and isinstance(n.func.value, ast.Name) and n.func.value.id in params}

    def run_order(fn: ast.AST, depth: int = 0, seen: frozenset = frozenset()) -> list[str]:
        out: list[str] = []
        for call in _calls_in_order(fn):
            ran: list[str] = []
            f = call.func
            if isinstance(f, ast.Attribute) and f.attr in _RUNS and agent_of(f.value):
                ran = [agent_of(f.value)]
            elif isinstance(f, ast.Attribute) and f.attr in _RUNS and _ref(f.value) == "Runner" and call.args:
                ran = [a for a in [agent_of(call.args[0])] if a]
            elif isinstance(f, ast.Name):  # a function of the project (asyncio.run(...) is not)
                for d in defs.get(f.id, [])[:3]:
                    if id(d) in seen or depth >= 3:
                        continue
                    params = [a.arg for a in [*d.args.posonlyargs, *d.args.args]]
                    run_p = runs_params(d)
                    given = {params[i]: a for i, a in enumerate(call.args) if i < len(params)}
                    given.update({k.arg: k.value for k in call.keywords if k.arg})
                    ran += [a for p, e in given.items() if p in run_p for a in [agent_of(e)] if a]
                    ran += run_order(d, depth + 1, seen | {id(d)})
            out += [a for a in ran if a not in out]
        return out

    handed: list[tuple[str, ast.AST]] = []  # (agent key, function handed to it)
    for rel, tree in c.trees.items():
        if rel in c.test_files:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in n.decorator_list:  # @supervisor.tool / @supervisor.tool_plain(...)
                    d = d.func if isinstance(d, ast.Call) else d
                    if isinstance(d, ast.Attribute) and d.attr in _ATTACH_DECORATORS and agent_of(d.value):
                        handed.append((agent_of(d.value), n))
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Name):  # supervisor.check = validate
                for t in n.targets:
                    if isinstance(t, ast.Attribute) and agent_of(t.value):
                        handed += [(agent_of(t.value), d) for d in defs.get(n.value.id, [])]
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):  # x = make_agent(tools=[delegate])
                owner = next((agent_of(t) for t in n.targets if agent_of(t)), None)
                for kw in n.value.keywords:
                    if owner and kw.arg in _TOOL_KWARGS:
                        handed += [(owner, d) for r in _list_refs(kw.value) for d in defs.get(r, [])]
            elif isinstance(n, ast.Call):  # register_tools(supervisor, [delegate_to_math, ...])
                givers = {agent_of(a) for a in n.args} - {None}
                if len(givers) == 1:
                    for a in [*n.args, *(k.value for k in n.keywords)]:
                        if isinstance(a, (ast.List, ast.Tuple)):
                            handed += [(next(iter(givers)), d) for r in _list_refs(a) for d in defs.get(r, [])]
    seen_edges = {(bpmod.norm_key(e["from"]), bpmod.norm_key(e["to"])) for e in c.edges}

    def add(a: str, b: str, label: str) -> None:
        pair = (bpmod.norm_key(c.agents[a]["name"]), bpmod.norm_key(c.agents[b]["name"]))
        if a != b and pair not in seen_edges:
            seen_edges.add(pair)
            c.edges.append({"from": c.agents[a]["name"], "to": c.agents[b]["name"], "label": label, "conditional": True})

    for owner, fn in handed:  # what runs after what inside them is branchy (retry this or that): left to run order
        for b in run_order(fn):
            add(owner, b, "delegates")


def _calls_in_order(node: ast.AST):
    """Calls in the order they run: statement by statement, a call's arguments before the call itself (nested
    definitions are not run here)."""
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from _calls_in_order(child)
    if isinstance(node, ast.Call):
        yield node


def _split_pipeline_classes(c: _Collector, files: list[SourceFile]) -> None:
    """An agent class that runs three or more of its own differently prompted methods in sequence (sub-queries, search,
    reflection, summary) is drawn as those steps - the rule the model-call pass applies when no framework is found
    (call_sites._class_pipelines)."""
    if not c.agent_classes:
        return
    found = call_sites.find(files)
    for cls, step_names in found.get("pipelines", {}).items():
        steps = [a for a in found["agents"] if a["name"] in step_names]
        key = next((k for k, a in c.agents.items() if k == bpmod.norm_key(cls) or cls in a["aliases"]
                    or bpmod.norm_key(a["name"]) == bpmod.norm_key(cls)), None)
        if key is None or len(steps) < 3:
            continue
        old = c.agents.pop(key)
        for s in steps:
            c.add_agent(s["name"], file=s["file"], line=s["line"], role=s["role"], description=s["description"],
                        tools=s["tools"], aliases=s["aliases"])
        names = {s["name"] for s in steps}
        c.edges += [dict(e) for e in found["edges"] if e["from"] in names and e["to"] in names]
        first = steps[0]["name"]
        olds = {key, bpmod.norm_key(old["name"]), *(bpmod.norm_key(a) for a in old["aliases"])}
        for e in c.edges:  # hand-offs to or from the class now start / end at its first step
            for end in ("from", "to"):
                if bpmod.norm_key(e[end]) in olds:
                    e[end] = first


def _edge_tables(c: _Collector) -> None:
    """add_edge(e["from"], e["to"]) over edge dicts built elsewhere - edges.append({"from": "refiner", "to": "critic"})
    in the graph's patterns: each dict literal of the project with those two keys and string values is an edge
    (pairs that do not name two agents are dropped when the blueprint is normalised)."""
    for k1, k2 in sorted(c.edge_tables):
        seen: set[tuple[str, str]] = set()
        for rel, tree in c.trees.items():
            if rel in c.test_files:
                continue
            for d in ast.walk(tree):
                if not isinstance(d, ast.Dict):
                    continue
                vals = {_lit(k): v for k, v in zip(d.keys, d.values) if k is not None}
                a, b = _lit(vals.get(k1)), _lit(vals.get(k2))
                if a and b and (a, b) not in seen:
                    seen.add((a, b))
                    c.edge(a, b)


def _group_chat_turns(c: _Collector) -> None:
    """Turns in a group chat. Round-robin chats (and Swarm / AgentWorkflow hand-offs) go round in the listed order.
    When a manager or a model picks each speaker, the listed order is the likely sequence but not a ring, and the
    person in the chat (the admin, a UserProxyAgent) is not one of the scheduled speakers."""
    for g in c.group_chats:
        members = g["members"]
        if g["cls"] in ("RoundRobinGroupChat", "Swarm", "AgentWorkflow") or g["method"] == "round_robin":
            order = members + members[:1] if len(members) > 2 else members
        else:
            humans = c.humans | ({g["admin"]} if g["admin"] else set())
            order = [m for m in members if m not in humans]
        for x, y in zip(order, order[1:]):
            c.edges.append({"from": x, "to": y, "label": "turn", "conditional": True})


_NOT_PROGRAM_DIRS = re.compile(r"^(src|lib|docs?|tests?)$", re.I)


def _drop_separate_programs(c: _Collector) -> None:
    """A folder with its own README that the rest of the project never imports is another program kept in the same
    repository (babycoder/ beside babyagi.py). When the code outside it wires agents of its own together, the
    folder's agents stay off the map."""
    def program_dir(rel: str) -> str | None:
        parts = Path(rel).parts[:-1]
        for i in range(len(parts), 0, -1):
            d = Path(*parts[:i])
            if not _NOT_PROGRAM_DIRS.match(d.name) and any((c.root / d / n).is_file() for n in ("README.md", "README.rst",
                                                                                                    "README")):
                return d.as_posix()
        return None

    def files_of(key: str, agent: dict[str, Any]) -> list[str]:
        return [f for f, _ in c.sites.get(key, [])] or [agent["file"]]

    homes = {k: {program_dir(f) for f in files_of(k, a) if f} for k, a in c.agents.items()}
    dirs = {d for hs in homes.values() for d in hs if d}
    if not dirs:
        return
    imported: set[str] = set()  # module path parts imported by code outside each folder
    separate = set()
    for d in dirs:
        name = Path(d).name
        for rel, tree in c.trees.items():
            if rel.startswith(d + "/"):
                continue
            mods = _imported_modules(tree)
            if any(name in m.split(".") for m in mods):
                imported.add(d)
                break
        if d not in imported:
            separate.add(d)
    inside = {k for k, hs in homes.items() if hs and hs <= separate}
    own: set[str] = set()
    for k, a in c.agents.items():
        if k not in inside:
            own |= {bpmod.norm_key(n) for n in (k, a["name"], *a["aliases"])}
    if inside and any(bpmod.norm_key(e["from"]) in own and bpmod.norm_key(e["to"]) in own for e in c.edges):
        for k in inside:
            del c.agents[k]


EXAMPLE_DIRS = re.compile(r"(^|/)(examples?|samples?|demos?|cookbooks?|tutorials?|[\w\-]*tutorial[\w\-]*|walkthroughs?)(/|$)",
                          re.I)


def _drop_example_agents(c: _Collector) -> None:
    """Agents defined only under examples/ (samples/, demos/) show how to use the project; when the project wires
    agents of its own together outside them (a graph, hand-offs), that is the system and the examples stay off the
    map. Loose agents outside examples/ are not enough: many projects keep their real systems in examples/."""
    def only_examples(key: str, agent: dict[str, Any]) -> bool:
        files = [f for f, _ in c.sites.get(key, [])] or [agent["file"]]
        return all(f and EXAMPLE_DIRS.search(f.replace(os.sep, "/")) for f in files)

    example = {k for k, a in c.agents.items() if only_examples(k, a)}
    own: set[str] = set()
    for k, a in c.agents.items():
        if k not in example:
            own |= {bpmod.norm_key(n) for n in (k, a["name"], *a["aliases"])}
    wired = any(bpmod.norm_key(e["from"]) in own and bpmod.norm_key(e["to"]) in own for e in c.edges)
    if example and wired:
        for k in example:
            del c.agents[k]


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
