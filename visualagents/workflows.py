"""Agent workflows wired with objects rather than names, read from the code without running it.

- Strands: `GraphBuilder` (`add_node(agent, "id")`, `add_edge(a, b)`) and `Swarm([a, b, c])`;
- Microsoft Agent Framework: `WorkflowBuilder` (`add_edge`, fan-out / fan-in edges, chains, switch-case and
  multi-selection groups) and its orchestration builders (sequential, concurrent, handoff, group chat, Magentic);
- Google ADK 2: `Workflow(edges=[("START", a, b), (b, {"route": c}), Edge(from_node=b, to_node=d)])`;
- pydantic_graph: `BaseNode` classes whose `run()` return annotation names the nodes that may come next;
- CrewAI Flows: the `@start()`, `@listen(step)` and `@router(step)` methods of a `Flow`.

Every step these wire up is a block, as a LangGraph node is: a step that calls no model (a loader, a merge, a PDF
export) is still part of the run. Agents are the ones `static_scan` already found; a step that is an agent is that
agent's block, not a second one.
"""

from __future__ import annotations

import ast
import re
from typing import Any

# what a builder or a step wrapper calls the name of the step it makes
_NAME_KW = ("name", "id", "node_id", "executor_id")
# wrappers whose first argument is the agent or function that does the work
_WRAPPERS = {"AgentExecutor", "FunctionNode", "FunctionExecutor", "ToolNode", "AgentNode", "LlmAgentWrapper"}
_START = {"START", "__start__"}
_END = {"END", "__end__"}
# MAF orchestration builders, and the keyword / method that names their manager
_ORCHESTRATIONS = {"SequentialBuilder", "ConcurrentBuilder", "HandoffBuilder", "GroupChatBuilder", "MagenticBuilder"}
_MANAGER_KW = ("orchestrator_agent", "manager_agent", "manager")
_FLOW_DECOS = {"start", "listen", "router"}


class _File:
    """One file's tree with the scanner's helpers, for resolving what a builder call refers to."""

    def __init__(self, tree: ast.Module, rel: str, c: Any, parents: dict[ast.AST, ast.AST]) -> None:
        from . import static_scan as ss
        self.ss, self.tree, self.rel, self.c, self.parents = ss, tree, rel, c, parents
        self.depth = 0

    # -- blocks ----------------------------------------------------------------
    def step(self, name: str, at: ast.AST, aliases: list[str | None]) -> str:
        """A step of the workflow: a block of its own, kept on the map like a graph node."""
        ag = self.c.add_agent(name, file=self.rel, line=getattr(at, "lineno", 0), aliases=[a for a in aliases if a])
        self.c.graph_nodes[name] = ag
        return name

    def agent_node(self, ref: str, node_id: str | None) -> str:
        """An agent placed in a graph under an id of its own: the id is another name of that agent's block."""
        key = self.c.var_to_agent.get(ref)
        ag = self.c.agents.get(key) if key else None
        if ag is None:
            return node_id or ref
        for a in (node_id, ref):
            if a and a not in ag["aliases"]:
                ag["aliases"].append(a)
            if a:
                self.c.var_to_agent[a] = key
        self.c.graph_nodes[node_id or ref] = ag
        return ref

    # -- what an expression stands for ----------------------------------------
    def node(self, expr: ast.AST | None, at: ast.AST) -> str | None:
        ss = self.ss
        if expr is None:
            return None
        lit = ss._lit(expr)
        if lit is not None:
            return lit
        if isinstance(expr, ast.Call):
            return self.from_call(expr, at, None)
        if isinstance(expr, ast.Subscript):  # specialists["auditor"]: the key names the agent
            key = ss._lit(expr.slice)
            return self.step(key, at, []) if key and key.isidentifier() else None
        if isinstance(expr, ast.Lambda):
            return None
        if isinstance(expr, (ast.Name, ast.Attribute)):
            var = expr.id if isinstance(expr, ast.Name) else expr.attr
            if var in self.c.var_to_agent:
                return var
            built = ss._assigned_value(expr, at, self.parents, self.tree)
            if isinstance(built, ast.Call):
                return self.from_call(built, at, var)
            if isinstance(built, (ast.Name, ast.Attribute)) and ss._ref(built) != var and self.depth < 4:
                self.depth += 1  # first = self._agents.strategy: what it was set to
                try:
                    return self.node(built, at)
                finally:
                    self.depth -= 1
            fn = self.c.functions.get(var) or self.c.own_defs.get(var)
            if fn is not None:
                return self.step(_decorated_id(fn, ss) or var, fn, [var])
            cls = self.c.project_classes.get(var)
            if cls:
                return self.step(var, cls[0], [var])
            # built in another file (eval_node = _build_eval_node(...)): the agent of that name once every file is
            # read, else a step of its own - whatever a workflow wires up is part of it
            self.c.pending_steps.append((var, self.rel, getattr(at, "lineno", 0)))
            return var
        return None

    def from_call(self, call: ast.Call, at: ast.AST, alias: str | None) -> str | None:
        ss = self.ss
        name = ss._call_name(call)
        if name in ss.AGENT_CLASSES or name in ss.AGENT_FACTORIES or name.endswith("Agent"):
            return ss._agent_name_kw(call, self.tree, self.c) or alias or name
        if name in _WRAPPERS:  # AgentExecutor(writer, id="writer"), FunctionNode(func=intake, name="Intake")
            inner = ss._kw(call, "agent", "func", "function") or (call.args[0] if call.args else None)
            label = ss._lit(ss._kw(call, *_NAME_KW))
            ref = ss._ref(inner) if isinstance(inner, (ast.Name, ast.Attribute)) else None
            if ref and ref in self.c.var_to_agent:
                return self.agent_node(ref, label)
            return self.step(label or ref or alias or name, call, [alias, ref])
        # a project class or helper that makes a step: UpperCase(id="upper"), _function_node(intake, "IntakeNode")
        label = ss._lit(ss._kw(call, *_NAME_KW)) or next(
            (s for s in (ss._lit(a) for a in call.args) if s and ss._name_like(s) and " " not in s.strip()), None)
        func = next((ss._ref(a) for a in [*call.args, *(k.value for k in call.keywords)]
                     if isinstance(a, ast.Name) and a.id in self.c.functions), None)
        if func and func in self.c.var_to_agent:
            return func
        if label or func or name in self.c.project_classes or name in self.c.own_defs:
            return self.step(label or alias or func or name, call, [alias, func])
        return alias

    def nodes(self, expr: ast.AST | None, at: ast.AST) -> list[str]:
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            return [n for n in (self.node(e, at) for e in expr.elts) if n]
        if isinstance(expr, ast.Name):
            built = self.ss._assigned_value(expr, at, self.parents, self.tree)
            if isinstance(built, (ast.List, ast.Tuple)):
                return self.nodes(built, at)
        n = self.node(expr, at)
        return [n] if n else []

    def edge(self, a: str | None, b: str | None, label: str = "", conditional: bool = False) -> None:
        if a and b and a != b:
            self.c.edge(a, b, label=label, conditional=conditional)


def _decorated_id(fn: ast.AST, ss: Any) -> str | None:
    """@executor(id="upper_case") names the step its function makes."""
    for d in getattr(fn, "decorator_list", []):
        if isinstance(d, ast.Call):
            v = ss._lit(ss._kw(d, *_NAME_KW))
            if v:
                return v
    return None


def _root(call: ast.Call) -> ast.AST:
    """The object a chain of builder calls starts from: WorkflowBuilder(...) in WorkflowBuilder(...).add_edge(a, b)
    .build(), or the variable the builder was kept in."""
    cur: ast.AST = call
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
        cur = cur.func.value
    return cur


def scan_file(tree: ast.Module, rel: str, c: Any, parents: dict[ast.AST, ast.AST], imported: list[str]) -> None:
    f = _File(tree, rel, c, parents)
    if "strands" in imported:
        _strands(f)
    if "microsoft-agent-framework" in imported:
        _agent_framework(f)
    if "google-adk" in imported:
        _adk_workflows(f)
    if "pydantic-ai" in imported:
        _pydantic_graph(f)
    if "crewai" in imported:
        _crewai_flows(f)


# -- Strands -------------------------------------------------------------------------------------------------------
def _strands(f: _File) -> None:
    ss, c = f.ss, f.c
    held: dict[str, str] = {}  # variable holding what add_node returned -> the node
    calls = [n for n in ast.walk(f.tree) if isinstance(n, ast.Call)]
    for call in calls:
        name = ss._call_name(call)
        if name == "add_node" and isinstance(call.func, ast.Attribute) and call.args and ss._lit(call.args[0]) is None:
            node_id = ss._lit(call.args[1]) if len(call.args) > 1 else ss._lit(ss._kw(call, "node_id"))
            ref = f.node(call.args[0], call)
            if ref is None:
                continue
            node = f.agent_node(ref, node_id) if ref in c.var_to_agent else f.step(node_id or ref, call, [ref])
            parent = f.parents.get(call)
            for t in ss._assigned_names(parent) if parent is not None else []:
                held[t] = node
                c.var_to_agent.setdefault(t, c.var_to_agent.get(node, ss.bpmod.norm_key(node)))
    for call in calls:
        name = ss._call_name(call)
        if name == "add_edge" and isinstance(call.func, ast.Attribute) and len(call.args) >= 2 and not all(
                ss._lit(a) for a in call.args[:2]):
            a, b = (held.get(ss._ref(x) or "") or f.node(x, call) for x in call.args[:2])
            f.edge(a, b, conditional=len(call.args) > 2 or ss._kw(call, "condition") is not None)
        elif name == "set_entry_point" and call.args and ss._lit(call.args[0]) is None:
            r = held.get(ss._ref(call.args[0]) or "") or f.node(call.args[0], call)
            if r:
                c.entry.append(r)
        elif name == "Swarm":
            _swarm(f, call)


def _swarm(f: _File, call: ast.Call) -> None:
    """A swarm's agents hand off to each other as they see fit: to the members their own prompt names, or, when no
    prompt names another member, from the one that starts to each of the others."""
    ss, c = f.ss, f.c
    members = f.nodes(ss._kw(call, "nodes") or (call.args[0] if call.args else None), call)
    if not members:
        return
    first = f.node(ss._kw(call, "entry_point"), call) or members[0]
    c.entry.append(first)
    named = False
    for m in members:
        prompt = _prompt_of(f, m, call)
        for other in members:
            if other != m and prompt and re.search(rf"\b{re.escape(_agent_name(f, other))}\b", prompt, re.I):
                f.edge(m, other, label="handoff", conditional=True)
                named = True
    if not named:
        for other in members:
            f.edge(first, other, label="handoff", conditional=True)
    fn = ss._enclosing_function(call, f.parents)
    if fn:  # a tool that runs the swarm hands work to its first agent
        c.group_runners[fn] = first


def _agent_name(f: _File, ref: str) -> str:
    key = f.c.var_to_agent.get(ref)
    ag = f.c.agents.get(key) if key else None
    return next((a for a in (ag or {}).get("aliases", []) if a != ref), ref) if ag else ref


def _prompt_of(f: _File, ref: str, at: ast.AST) -> str:
    built = f.ss._assigned_value(ast.Name(id=ref), at, f.parents, f.tree)
    if not isinstance(built, ast.Call):
        return ""
    return " ".join(v for v in (f.ss._lit(f.ss._kw(built, *f.ss._PROMPT_KWARGS)),) if v)


# -- Microsoft Agent Framework -------------------------------------------------------------------------------------
def _agent_framework(f: _File) -> None:
    ss, c = f.ss, f.c
    builders: dict[int, dict[str, Any]] = {}  # chain root -> what its calls say about one orchestration

    def orchestration(call: ast.Call) -> dict[str, Any] | None:
        root = _root(call)
        if isinstance(root, ast.Name):
            root = ss._assigned_value(root, call, f.parents, f.tree) or root
            root = _root(root) if isinstance(root, ast.Call) else root
        if isinstance(root, ast.Call) and ss._call_name(root) in _ORCHESTRATIONS:
            return builders.setdefault(id(root), {"cls": ss._call_name(root), "at": root, "members": [], "start": None,
                                                  "manager": None, "handoffs": [], "aggregator": None})
        return None

    for call in (n for n in ast.walk(f.tree) if isinstance(n, ast.Call)):
        name = ss._call_name(call)
        method = isinstance(call.func, ast.Attribute)
        if name == "WorkflowBuilder" or (method and name == "set_start_executor"):
            start = ss._kw(call, "start_executor") or (call.args[0] if method and call.args else None)
            r = f.node(start, call)
            if r:
                c.entry.append(r)
        elif method and name == "add_edge" and len(call.args) >= 2 and not all(ss._lit(a) for a in call.args[:2]):
            cond = len(call.args) > 2 or ss._kw(call, "condition") is not None
            f.edge(f.node(call.args[0], call), f.node(call.args[1], call), conditional=cond)
        elif method and name == "add_fan_out_edges" and len(call.args) >= 2:
            src = f.node(call.args[0], call)
            for t in f.nodes(call.args[1], call):
                f.edge(src, t, label="fan-out")
        elif method and name == "add_fan_in_edges" and len(call.args) >= 2:
            dst = f.node(call.args[1], call)
            for s in f.nodes(call.args[0], call):
                f.edge(s, dst, label="fan-in")
        elif method and name == "add_chain" and call.args:
            chain = f.nodes(call.args[0], call)
            for a, b in zip(chain, chain[1:]):
                f.edge(a, b)
        elif method and name in ("add_switch_case_edge_group", "add_multi_selection_edge_group") and call.args:
            src = f.node(call.args[0], call)
            targets = call.args[1] if len(call.args) > 1 else ss._kw(call, "cases", "targets")
            for el in getattr(targets, "elts", []):
                t = ss._kw(el, "target") if isinstance(el, ast.Call) and ss._call_name(el) in ("Case", "Default") else el
                f.edge(src, f.node(t, call), label=ss._call_name(el) if isinstance(el, ast.Call) else "", conditional=True)
        o = orchestration(call)
        if o is None:
            continue
        if name in _ORCHESTRATIONS or name == "participants":
            members = ss._kw(call, "participants") or (call.args[0] if name == "participants" and call.args else None)
            if members is not None:
                o["members"] = f.nodes(members, call)
            elif name == "participants" and call.keywords:  # MagenticBuilder().participants(researcher=a, coder=b)
                o["members"] = [m for m in (f.node(k.value, call) for k in call.keywords) if m]
            mgr = f.node(ss._kw(call, *_MANAGER_KW), call)
            o["manager"] = mgr or o["manager"]
        elif name in ("with_start_agent", "set_coordinator") and call.args:
            o["start"] = f.node(call.args[0], call)
        elif name == "add_handoff" and call.args:
            o["handoffs"].append((f.node(call.args[0], call), f.nodes(call.args[1], call) if len(call.args) > 1 else []))
        elif name in ("with_aggregator", "with_custom_aggregator") and call.args:
            o["aggregator"] = f.node(call.args[0], call)
        elif name in ("with_standard_manager", "with_manager") and (call.args or call.keywords):
            o["manager"] = f.node(ss._kw(call, "agent", "manager") or (call.args[0] if call.args else None), call)
    for o in builders.values():
        _orchestrate(f, o)


def _orchestrate(f: _File, o: dict[str, Any]) -> None:
    c, members, cls = f.c, o["members"], o["cls"]
    if not members:
        return
    if cls == "SequentialBuilder":
        for a, b in zip(members, members[1:]):
            f.edge(a, b)
        c.entry.append(members[0])
        c.exits.append(members[-1])
    elif cls == "ConcurrentBuilder":
        c.entry.extend(members)
        for m in members:
            f.edge(m, o["aggregator"], label="results")
    elif cls == "HandoffBuilder":
        start = o["start"] or members[0]
        c.entry.append(start)
        for src, dsts in o["handoffs"]:
            for d in dsts:
                f.edge(src, d, label="handoff", conditional=True)
        if not o["handoffs"]:  # by default every agent may hand off to the others: drawn from the one that starts
            for m in members:
                f.edge(start, m, label="handoff", conditional=True)
    else:  # group chat / Magentic: a manager picks who goes next
        hub = o["manager"]
        if hub is None and cls == "MagenticBuilder":
            hub = f.step("magentic_manager", o["at"], [])
        if hub is None:  # a selection function picks the speaker: the listed order is the likely one
            for a, b in zip(members, members[1:]):
                f.edge(a, b, label="turn", conditional=True)
            c.entry.append(members[0])
            return
        c.entry.append(hub)
        for m in members:
            f.edge(hub, m, label="delegates", conditional=True)
            f.edge(m, hub, label="reports", conditional=True)


# -- Google ADK 2 workflows ----------------------------------------------------------------------------------------
def _adk_workflows(f: _File) -> None:
    ss = f.ss
    for call in (n for n in ast.walk(f.tree) if isinstance(n, ast.Call)):
        if ss._call_name(call) != "Workflow":
            continue
        edges = ss._kw(call, "edges")
        if isinstance(edges, ast.Name):
            edges = ss._assigned_value(edges, call, f.parents, f.tree)
        for item in getattr(edges, "elts", []):
            if isinstance(item, ast.Tuple):
                _adk_chain(f, item.elts, call)
            elif isinstance(item, ast.Call) and ss._call_name(item) == "Edge":
                a = f.node(ss._kw(item, "from_node") or (item.args[0] if item.args else None), call)
                b = f.node(ss._kw(item, "to_node") or (item.args[1] if len(item.args) > 1 else None), call)
                route = ss._lit(ss._kw(item, "route"))
                f.edge(a, b, label=route or "", conditional=route is not None)


def _adk_chain(f: _File, elements: list[ast.AST], at: ast.AST) -> None:
    """("START", a, b, {"retry": a, "done": c}): each element follows the one before it; a tuple inside fans out, a
    dict routes by the value the step before it emits."""
    prev: list[str] = []
    for el in elements:
        if isinstance(el, ast.Dict):
            nxt = []
            for k, v in zip(el.keys, el.values):
                label = f.ss._lit(k) if k is not None else ""
                for t in f.nodes(v, at) if isinstance(v, ast.Tuple) else [f.node(v, at)]:
                    for p in prev:
                        f.edge(p, t, label=label or "", conditional=True)
                    if t:
                        nxt.append(t)
        else:
            nxt = f.nodes(el, at) if isinstance(el, ast.Tuple) else [n for n in [f.node(el, at)] if n]
            for p in prev:
                for t in nxt:
                    f.edge(p, t)
        prev = nxt


# -- pydantic_graph ------------------------------------------------------------------------------------------------
def _pydantic_graph(f: _File) -> None:
    ss, c = f.ss, f.c
    nodes: dict[str, ast.ClassDef] = {}
    for cls in (n for n in ast.walk(f.tree) if isinstance(n, ast.ClassDef)):
        if any(_base_name(b) == "BaseNode" for b in cls.bases):
            nodes[cls.name] = cls
    if not nodes:
        return
    targeted: set[str] = set()
    for name, cls in nodes.items():
        f.step(name, cls, [name])
        run = next((m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == "run"),
                   None)
        for t in _annotation_names(run.returns) if run is not None and run.returns is not None else []:
            if t == "End":
                c.exits.append(name)
            elif t in nodes:
                f.edge(name, t, conditional=True)
                targeted.add(t)
    started = [ss._call_name(a) for n in ast.walk(f.tree) if isinstance(n, ast.Call) and ss._call_name(n) in
               ("run", "run_sync", "iter", "run_stream") for a in n.args[:1] if isinstance(a, ast.Call)]
    c.entry.extend([s for s in started if s in nodes] or [n for n in nodes if n not in targeted])


def _base_name(b: ast.AST) -> str:
    b = b.value if isinstance(b, ast.Subscript) else b
    return b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else ""


def _annotation_names(ann: ast.AST) -> list[str]:
    """The classes a return annotation names: B | C | End[str], Union[B, C], Annotated[B, Edge(label=...)], "B"."""
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            return _annotation_names(ast.parse(ann.value, mode="eval").body)
        except SyntaxError:
            return []
    if isinstance(ann, ast.BinOp):
        return _annotation_names(ann.left) + _annotation_names(ann.right)
    if isinstance(ann, ast.Subscript):
        head = _base_name(ann)
        if head in ("Union", "Optional"):
            inner = ann.slice
            return [n for e in (inner.elts if isinstance(inner, ast.Tuple) else [inner]) for n in _annotation_names(e)]
        if head == "Annotated":
            inner = ann.slice
            return _annotation_names(inner.elts[0] if isinstance(inner, ast.Tuple) else inner)
        return [head]
    name = _base_name(ann)
    return [name] if name else []


# -- CrewAI Flows --------------------------------------------------------------------------------------------------
def _crewai_flows(f: _File) -> None:
    ss, c = f.ss, f.c
    for cls in (n for n in ast.walk(f.tree) if isinstance(n, ast.ClassDef)):
        if not any(_base_name(b) == "Flow" for b in cls.bases):
            continue
        methods = {m.name: m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
        wired: dict[str, tuple[str, ast.Call]] = {}  # method -> (start / listen / router, its decorator)
        for m in methods.values():
            for d in m.decorator_list:
                kind = ss._call_name(d) if isinstance(d, ast.Call) else ss._ref(d)
                if kind in _FLOW_DECOS:
                    wired[m.name] = (kind, d if isinstance(d, ast.Call) else ast.Call(func=d, args=[], keywords=[]))
        if not wired:
            continue
        c.vote("crewai", 2)
        labels: dict[str, list[str]] = {}  # a label a router returns -> the routers that return it
        for m, (kind, _) in wired.items():
            f.step(m, methods[m], [m])
            if kind == "router":
                for s in ss._returned_strings(methods[m]):
                    labels.setdefault(s, []).append(m)
        for m, (kind, deco) in wired.items():
            triggers = _flow_triggers(deco.args[0] if deco.args else None, ss)
            if kind == "start" and not triggers:
                c.entry.append(m)
            for t in triggers:
                if t in wired and t not in labels:
                    f.edge(t, m)
                for r in labels.get(t, []):
                    f.edge(r, m, label=t, conditional=True)
            for crew in _crews_kicked_off(methods[m], c, ss):
                c.flow_crews.append((m, crew))


def _flow_triggers(arg: ast.AST | None, ss: Any) -> list[str]:
    """@listen(fetch) / @listen("fetch") / @listen(or_(a, "retry")) / @listen(and_(a, b)): what a step waits for."""
    if arg is None:
        return []
    if isinstance(arg, ast.Call) and ss._call_name(arg) in ("or_", "and_"):
        return [t for a in arg.args for t in _flow_triggers(a, ss)]
    r = ss._ref(arg)
    return [r] if r else []


def _crews_kicked_off(fn: ast.AST, c: Any, ss: Any) -> list[str]:
    """The project crew classes a flow step runs: EmailFilterCrew().crew().kickoff(...)."""
    return list(dict.fromkeys(ss._call_name(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                              and ss._call_name(n) in c.project_classes))


def finish(c: Any) -> None:
    """After every file is read: a flow step that runs a crew hands work to the crew's first agent, and an agent
    whose tool runs a swarm or graph hands work to the swarm's first agent."""
    from .blueprint import norm_key
    for var, rel, line in c.pending_steps:
        if var not in c.var_to_agent and norm_key(var) not in c.agents:
            c.graph_nodes[var] = c.add_agent(var, file=rel, line=line, aliases=[var])
    for step, crew in c.flow_crews:
        agents = c.crew_agents.get(crew) or []
        if agents:
            c.edge(step, agents[0], label="runs crew")
    for ag in list(c.agents.values()):
        for t in ag["tools"]:
            first = c.group_runners.get(t)
            if first:
                c.edge(ag["name"], first, label="runs", conditional=True)
