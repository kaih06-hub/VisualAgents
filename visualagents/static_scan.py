"""Offline agent detection: reads Python (AST) and JS/TS/YAML (regex) without calling an LLM.

Understands the common shapes of LangGraph, LangChain, CrewAI (code + YAML), AutoGen
(0.2 and 0.4), the OpenAI Agents SDK, Swarm, Google ADK, smolagents and LlamaIndex.
The result is a raw blueprint dict; `blueprint.normalize` resolves names to ids.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from . import blueprint as bpmod
from . import call_sites
from .collector import SourceFile, collect

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

    # -- helpers ---------------------------------------------------------
    def vote(self, fw: str, n: int = 1) -> None:
        self.frameworks[fw] = self.frameworks.get(fw, 0) + n

    def add_agent(self, name: str, *, file: str, line: int, role: str = "", description: str = "",
                  tools: list[str] | None = None, aliases: list[str] | None = None) -> dict[str, Any]:
        key = bpmod.norm_key(name)
        agent = self.agents.get(key)
        if agent is None:
            agent = {"name": _pretty(name), "role": "", "description": "", "tools": [], "file": file,
                     "line": line, "aliases": [name], "keywords": []}
            self.agents[key] = agent
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


def _scan_python(src: SourceFile, c: _Collector) -> None:
    try:
        tree = ast.parse(src.text)
    except (SyntaxError, ValueError):
        return
    text = src.text
    for fw, pat in (("langgraph", "langgraph"), ("crewai", "crewai"), ("autogen", "autogen"),
                    ("openai-agents", "from agents import"), ("swarm", "from swarm"), ("google-adk", "google.adk"),
                    ("smolagents", "smolagents"), ("llamaindex", "llama_index"), ("langchain", "langchain")):
        if pat in text:
            c.vote(fw, 2)

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


def _line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    return text.count("\n", 0, idx) + 1 if idx >= 0 else 0


# -- assembly -------------------------------------------------------------


def scan(root: Path, files: list[SourceFile] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    files = files if files is not None else collect(root, max_files=600)
    c = _Collector()
    for f in files:
        ext = f.path.suffix.lower()
        if ext == ".py":
            _scan_python(f, c)
        elif ext in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}:
            _scan_js(f, c)
        elif ext in {".yaml", ".yml"}:
            _scan_yaml(f, c)
        elif ext == ".ipynb":
            _scan_python(f, c)

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

    framework = max(c.frameworks, key=c.frameworks.get) if c.frameworks else "custom"
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
