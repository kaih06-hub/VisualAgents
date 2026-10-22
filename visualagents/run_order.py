"""Run order of agent objects: who runs after whom, read from the code that runs them.

Frameworks that build agents as objects (an SDK `Agent(...)`, a DSPy module, an AG2 `ConversableAgent`, a project's own
agent class) leave the order to plain Python: a manager method awaits `Runner.run(planner)`, then a helper that runs
`search` for each item, then `writer`. This pass follows that code - through the project's own functions and methods,
`self.helper()` and `self.stage.method()` included - and links each agent to the ones that can run right after it.
Branches of an `if` / `match` / `try` are alternatives, not a sequence; a loop's body is taken to run.

It also reads two hand-offs that are not a run order: an agent given to another as a tool (`x.as_tool()` in its
`tools=[...]`), and an AG2 chat (`a.initiate_chat(b)`, nested chats registered on an agent).
"""
from __future__ import annotations

import ast
from typing import Any

# methods that run an agent object: planner.run(...), Runner.run(planner, ...), expert(question) for a DSPy module
RUN_METHODS = {"run", "run_sync", "run_streamed", "arun", "invoke", "ainvoke", "stream", "astream", "kickoff",
               "kickoff_async", "initiate_chat", "a_initiate_chat", "generate_reply", "a_generate_reply", "chat",
               "achat", "step", "astep", "forward", "aforward", "acall", "reply", "areply", "query", "aquery",
               "batch", "abatch"}
PASSED_TO = {"run", "run_sync", "run_streamed", "arun", "invoke", "ainvoke", "stream", "astream"}  # Runner.run(agent)
SAME_AGENT = {"clone", "copy", "with_config", "with_options", "bind", "bind_tools", "with_structured_output",
              "with_retry", "with_fallbacks"}
CHATS = {"initiate_chat", "a_initiate_chat"}
# an agent's own model, called from outside it: self.action_agent.llm(messages)
MODEL_ATTR = __import__("re").compile(r"^_?(llm|lm|model|chat_model|chat_llm|client)$")
MAX_DEPTH = 6
BUDGET = 40_000  # calls looked at, per project
MAX_EDGES = 40


class _Fn:
    __slots__ = ("cls", "node", "rel")

    def __init__(self, rel: str, cls: str | None, node: ast.AST) -> None:
        self.rel, self.cls, self.node = rel, cls, node


def _lit(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def link(c: Any) -> None:
    """Add run-order, agent-as-tool and chat edges between the agents already found (`c` is the static collector)."""
    graph = {id(a) for a in c.graph_nodes.values()}
    owners: dict[str, set[str]] = {}
    for key, a in c.agents.items():
        if id(a) in graph:
            continue  # a graph node's order is the graph's edges
        for n in {a["name"], key, *a["aliases"]}:
            owners.setdefault(n, set()).add(key)
    names = {n: next(iter(ks)) for n, ks in owners.items() if len(ks) == 1 and n}  # one agent per name only
    if len(set(names.values())) < 2:
        return

    funcs: dict[str, list[_Fn]] = {}
    methods: dict[tuple[str, str], _Fn] = {}  # (class, method) -> function
    attr_cls: dict[str, set[str]] = {}  # self.stage = StageModule(...): attribute -> project classes it holds
    called: set[int] = set()  # functions some other project code calls: not where a run starts
    roots: list[_Fn] = []
    for rel, tree in c.trees.items():
        if rel in c.test_files:
            continue
        module_code = [s for s in tree.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                     ast.ClassDef, ast.Import, ast.ImportFrom))]
        if module_code:
            roots.append(_Fn(rel, None, ast.Module(body=module_code, type_ignores=[])))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(node.name, []).append(_Fn(rel, None, node))
            elif isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fn = _Fn(rel, node.name, m)
                        methods[(node.name, m.name)] = fn
                        funcs.setdefault(m.name, []).append(fn)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and isinstance(node.value.func, ast.Name):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        attr_cls.setdefault(t.attr, set()).add(node.value.func.id)
    bases: dict[str, list[str]] = {}  # project class -> the names of its bases
    for tree in c.trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases.setdefault(node.name, []).extend(
                    b.id if isinstance(b, ast.Name) else b.attr for b in node.bases if isinstance(b, (ast.Name, ast.Attribute)))
    asks = getattr(c, "llm_methods", {}) or {}

    def asks_model(cls: str | None, method: str, seen: frozenset = frozenset()) -> bool:
        """Does `cls.method` (its own or inherited) reach a model?"""
        if not cls or cls in seen or (cls, method) in seen:
            return False
        if method in asks.get(cls, ()):
            return True
        own = methods.get((cls, method))
        if own is not None:  # its own method: asks through another one it calls, inherited or not (self.inference())
            return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                       and n.func.value.id == "self" and n.func.attr != method
                       and asks_model(cls, n.func.attr, seen | {(cls, method)})
                       for n in ast.walk(own.node))
        return any(asks_model(b, method, seen | {cls}) for b in bases.get(cls, ()))

    project_classes = {cls for cls, _ in methods}
    attr_any = {a: next(iter(ks)) for a, ks in attr_cls.items() if len(ks) == 1}
    attr_cls = {a: k for a, k in attr_any.items() if k in project_classes}
    roots_later: list[_Fn] = [fn for fns in funcs.values() for fn in fns]

    edges: list[dict[str, Any]] = []
    have = {(e["from"], e["to"]) for e in c.edges}
    budget = [BUDGET]

    def add(a: str, b: str, label: str, conditional: bool = False) -> None:
        ea, eb = c.agents[a]["name"], c.agents[b]["name"]
        if a == b or len(edges) >= MAX_EDGES:
            return
        known = {(x, y) for x, y in have} | {(e["from"], e["to"]) for e in edges}
        aliases_a = {a, ea, *c.agents[a]["aliases"]}
        aliases_b = {b, eb, *c.agents[b]["aliases"]}
        if any(x in aliases_a and y in aliases_b for x, y in known):
            return
        edges.append({"from": ea, "to": eb, "label": label, "conditional": conditional})

    # the class that builds an agent (class PlannerAgent: self.agent = AssistantAgent(name="planner")) stands for it
    class_agent: dict[str, set[str]] = {}
    for key in set(names.values()):
        for rel, line in c.sites.get(key, []):
            tree = c.trees.get(rel)
            around = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                      and n.lineno <= line <= (n.end_lineno or n.lineno)] if tree else []
            if around:  # the innermost class around the definition
                inner = max(around, key=lambda n: n.lineno)
                class_agent.setdefault(inner.name, set()).add(key)
    class_agent = {k: next(iter(v)) for k, v in class_agent.items() if len(v) == 1}
    keyed: dict[str, str] = {}  # agents["browser"] = browser_agent: the key names that agent from now on
    returns: dict[int, str | None] = {}  # function -> the agent it returns (a factory: return planner.agent)

    def agent_of(e: ast.AST | None, env: dict[str, str], fn: _Fn | None = None) -> str | None:
        if e is None:
            return None
        if isinstance(e, ast.Await):
            return agent_of(e.value, env, fn)
        if isinstance(e, ast.Name):
            return env.get(e.id) or names.get(e.id)
        if isinstance(e, ast.Attribute):
            if e.attr in names:
                return names[e.attr]
            held = attr_any.get(e.attr)  # self.writer = WikiWriter(...): the attribute holds that agent
            if held and (held in class_agent or held in names):
                return class_agent.get(held) or names.get(held)
            if e.attr in ("agent", "_agent", "llm_agent"):  # BrowserNavAgent(...).agent: the agent it wraps
                return agent_of(e.value, env, fn)
            return None
        if isinstance(e, ast.Subscript):  # agents["planner"], self.agents_map["browser_nav_agent"]
            k = _lit(e.slice) or ""
            return names.get(k) or keyed.get(k)
        if isinstance(e, ast.Call):
            f = e.func
            if isinstance(f, ast.Attribute) and f.attr in SAME_AGENT:
                return agent_of(f.value, env, fn)
            ctor = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if ctor[:1].isupper() and (ctor in class_agent or ctor in names):  # PlannerAgent(...) builds it
                return class_agent.get(ctor) or names.get(ctor)
            target = callee(e, fn) if fn is not None else None
            if target is not None:
                return returned(target)
        return None

    def local_env(fn: _Fn) -> dict[str, str]:
        env: dict[str, str] = {}
        for st in sorted((n for n in ast.walk(fn.node) if isinstance(n, (ast.Assign, ast.AnnAssign))),
                         key=lambda n: n.lineno):
            if st.value is None:
                continue
            a = agent_of(st.value, env, fn)
            for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                if a and isinstance(t, ast.Name):
                    env[t.id] = a
                elif a and isinstance(t, ast.Subscript) and _lit(t.slice):
                    keyed.setdefault(_lit(t.slice), a)
        return env

    def returned(fn: _Fn) -> str | None:
        k = id(fn.node)
        if k not in returns:
            returns[k] = None  # recursion guard
            env = local_env(fn)
            found = {agent_of(r.value, env, fn) for r in ast.walk(fn.node) if isinstance(r, ast.Return)} - {None}
            returns[k] = next(iter(found)) if len(found) == 1 else None
        return returns[k]

    def tool_agents(call: ast.Call, env: dict[str, str], tools_env: dict[str, str], fn: _Fn) -> list[str]:
        out = []
        kw = next((k.value for k in call.keywords if k.arg == "tools"), None)
        for elt in kw.elts if isinstance(kw, (ast.List, ast.Tuple)) else []:
            if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) and elt.func.attr == "as_tool":
                a = agent_of(elt.func.value, env, fn)
            else:
                a = tools_env.get(elt.id) if isinstance(elt, ast.Name) else None
            if a:
                out.append(a)
        return out

    def callee(call: ast.Call, fn: _Fn) -> _Fn | None:
        f = call.func
        if _is_partial(call) and call.args:
            f = call.args[0]
        if isinstance(f, ast.Name):
            cands = [x for x in funcs.get(f.id, []) if x.cls is None]  # a bare name is not some class's method
            same = [x for x in cands if x.rel == fn.rel]
            return same[0] if same else (cands[0] if len(cands) == 1 else None)
        if isinstance(f, ast.Attribute):
            base = f.value
            if isinstance(base, ast.Name) and base.id in ("self", "cls") and fn.cls:
                return methods.get((fn.cls, f.attr))
            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "self":
                k = attr_cls.get(base.attr)  # self.curation.research(...) -> CurationModule.research
                return methods.get((k, f.attr)) if k else None
        return None

    def uses(call: ast.Call, env: dict[str, str], fn: _Fn) -> list[str]:
        f = call.func
        if isinstance(f, ast.Attribute) and f.attr in RUN_METHODS:
            me = agent_of(f.value, env, fn)
            if me and f.attr in CHATS:
                other = agent_of(call.args[0] if call.args else next(
                    (k.value for k in call.keywords if k.arg == "recipient"), None), env, fn)
                if other:
                    add(me, other, "chat")
                    add(other, me, "replies")
                    return [me, other]
            if me:
                return [me]
            if f.attr in PASSED_TO:  # Runner.run(planner, ...), Runner.run(starting_agent=planner, ...)
                arg = call.args[0] if call.args else next(
                    (k.value for k in call.keywords if k.arg in ("starting_agent", "agent")), None)
                a = agent_of(arg, env, fn)
                return [a] if a else []
            return []
        # a project agent class asked through one of its own model-calling methods: self.critic.check_task_success(),
        # solver.solve() with solver = MLESolver(...) - or through its model: self.action_agent.llm(messages)
        if isinstance(f, ast.Attribute) and isinstance(f.value, (ast.Attribute, ast.Name)):
            owner_cls = attr_any.get(f.value.attr) if isinstance(f.value, ast.Attribute) and \
                isinstance(f.value.value, ast.Name) and f.value.value.id == "self" else \
                local_cls(fn, f.value.id) if isinstance(f.value, ast.Name) else None
            if owner_cls and (asks_model(owner_cls, f.attr) or MODEL_ATTR.match(f.attr)):
                me = agent_of(f.value, env, fn)
                if me:
                    return [me]
        if isinstance(f, ast.Attribute) and f.attr == "register_nested_chats":
            me = agent_of(f.value, env, fn)
            chats = call.args[0] if call.args else None
            for d in chats.elts if isinstance(chats, (ast.List, ast.Tuple)) else []:
                if isinstance(d, ast.Dict):
                    fields = {_lit(k): v for k, v in zip(d.keys, d.values)}
                    other = agent_of(fields.get("recipient"), env, fn)
                    sender = agent_of(fields.get("sender"), env, fn)
                    if me and other:
                        add(me, other, "nested chat", True)
                        add(other, me, "replies", True)
                    if sender and other:  # the nested chat itself: sender <-> recipient
                        add(sender, other, "chat", True)
                        add(other, sender, "replies", True)
            return []
        # calling the agent object itself: self.write_outline(topic=...), expert(question) - not a constructor
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self") or \
                (isinstance(f, ast.Name) and f.id[:1].islower()):
            a = agent_of(f, env, fn)
            return [a] if a else []
        return []

    ctor_of: dict[int, dict[str, str]] = {}

    def local_cls(fn: _Fn, var: str) -> str | None:
        """The project class a local variable was built from: solver = MLESolver(...)."""
        k = id(fn.node)
        if k not in ctor_of:
            ctor_of[k] = {}
            for st in ast.walk(fn.node):
                if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name):
                    for t in st.targets:
                        if isinstance(t, ast.Name):
                            ctor_of[k][t.id] = st.value.func.id
        return ctor_of[k].get(var)

    memo: dict[tuple[int, frozenset], set[str]] = {}
    reach: set[int] = set()  # functions that run an agent, themselves or through what they call

    def walk_fn(fn: _Fn, frontier: set[str], depth: int, stack: frozenset) -> set[str]:
        key = (id(fn.node), frozenset(frontier))
        if key in memo:
            return memo[key]
        ctl: dict[str, Any] = {"ret": None, "brk": None}
        out = walk(getattr(fn.node, "body", []), set(frontier), fn, depth, stack | {id(fn.node)}, {}, {}, ctl)
        result = _join(out, ctl["ret"]) or set()
        memo[key] = result
        return result

    def walk_expr(node: ast.AST, frontier: set[str], fn: _Fn, depth: int, stack: frozenset,
                  env: dict[str, str], tools_env: dict[str, str]) -> set[str]:
        for call in _calls_in_order(node):  # arguments run before the call they are passed to
            budget[0] -= 1
            if budget[0] <= 0:
                return frontier
            f = call.func
            if isinstance(f, ast.Attribute) and f.attr in SAME_AGENT:  # writer.clone(tools=[risk_tool, ...])
                owner = agent_of(f.value, env, fn)
                for t in tool_agents(call, env, tools_env, fn) if owner else []:
                    add(owner, t, "as tool", True)
            found = uses(call, env, fn)
            if found:
                for u in found:
                    for x in frontier:
                        add(x, u, "then")
                    frontier = {u}
                continue
            target = callee(call, fn) if depth < MAX_DEPTH else None
            if target is not None and id(target.node) in reach and id(target.node) not in stack:
                frontier = walk_fn(target, frontier, depth + 1, stack)
        return frontier

    def walk(stmts: list[ast.stmt], frontier: set[str] | None, fn: _Fn, depth: int, stack: frozenset,
             env: dict[str, str], tools_env: dict[str, str], ctl: dict[str, Any]) -> set[str] | None:
        """What can have run last when control leaves these statements normally (None: it never does - every path
        returned, raised, broke out or continued)."""
        for st in stmts:
            if frontier is None or budget[0] <= 0:
                break
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(st, ast.If):
                frontier = walk_expr(st.test, frontier, fn, depth, stack, env, tools_env)
                a = walk(st.body, set(frontier), fn, depth, stack, env, tools_env, ctl)
                b = walk(st.orelse, set(frontier), fn, depth, stack, env, tools_env, ctl) if st.orelse else frontier
                frontier = _join(a, b)
            elif isinstance(st, (ast.For, ast.AsyncFor, ast.While)):
                head = st.iter if isinstance(st, (ast.For, ast.AsyncFor)) else st.test
                frontier = walk_expr(head, frontier, fn, depth, stack, env, tools_env)
                loop = {"ret": ctl["ret"], "brk": None}
                # a loop over planned items runs its body: what ran last is what the body ran last
                body = walk(st.body, set(frontier), fn, depth, stack, env, tools_env, loop)
                ran = (body or set()) - set(frontier)  # agents this round ran last (not ones it skipped past)
                if _repeats(st) and ran:  # the next round starts after this one's last agent
                    again = {"ret": loop["ret"], "brk": loop["brk"], "cont": loop.get("cont")}
                    body = _join(body, walk(st.body, ran, fn, depth, stack, env, tools_env, again))
                    loop.update(again)
                ctl["ret"] = loop["ret"]
                frontier = _join(_join(body, loop["brk"]), loop.get("cont"))  # `continue` ends a pass too
                if st.orelse and frontier is not None:
                    frontier = walk(st.orelse, frontier, fn, depth, stack, env, tools_env, ctl)
            elif isinstance(st, ast.Try) or type(st).__name__ == "TryStar":
                inner = {"ret": None, "brk": ctl.get("brk"), "cont": ctl.get("cont")}
                done = walk(st.body, set(frontier), fn, depth, stack, env, tools_env, inner)
                ran = _join(done, inner["ret"])  # what the body got to, returning or not
                ctl["ret"] = _join(ctl["ret"], inner["ret"])
                ctl["brk"], ctl["cont"] = inner["brk"], inner["cont"]
                failed: set[str] | None = None
                for h in st.handlers:
                    # a handler is the failure path of what the body ran, not a way around it
                    failed = _join(failed, walk(h.body, set(ran if ran is not None else frontier), fn, depth,
                                                stack, env, tools_env, ctl))
                if st.orelse and done is not None:
                    done = walk(st.orelse, done, fn, depth, stack, env, tools_env, ctl)
                frontier = _join(done, failed)
                if st.finalbody and frontier is not None:
                    frontier = walk(st.finalbody, frontier, fn, depth, stack, env, tools_env, ctl)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                for item in st.items:
                    frontier = walk_expr(item.context_expr, frontier, fn, depth, stack, env, tools_env)
                frontier = walk(st.body, frontier, fn, depth, stack, env, tools_env, ctl)
            elif isinstance(st, ast.Match):
                frontier = walk_expr(st.subject, frontier, fn, depth, stack, env, tools_env)
                out: set[str] | None = None
                for case in st.cases:
                    out = _join(out, walk(case.body, set(frontier), fn, depth, stack, env, tools_env, ctl))
                frontier = out if st.cases else frontier
            elif isinstance(st, ast.Return):
                if st.value is not None:
                    frontier = walk_expr(st.value, frontier, fn, depth, stack, env, tools_env)
                ctl["ret"] = _join(ctl["ret"], frontier)
                frontier = None
            elif isinstance(st, ast.Raise):
                frontier = None  # a failure: nothing after it runs on this path
            elif isinstance(st, ast.Break):
                ctl["brk"] = _join(ctl["brk"], frontier)
                frontier = None
            elif isinstance(st, ast.Continue):
                ctl["cont"] = _join(ctl.get("cont"), frontier)
                frontier = None
            else:
                frontier = walk_expr(st, frontier, fn, depth, stack, env, tools_env)
                if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None:
                    value = st.value.value if isinstance(st.value, ast.Await) else st.value
                    targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                    a = agent_of(value, env, fn)
                    tool = agent_of(value.func.value, env, fn) if isinstance(value, ast.Call) and isinstance(
                        value.func, ast.Attribute) and value.func.attr == "as_tool" else None
                    for t in targets:
                        if isinstance(t, ast.Name):
                            if a:
                                env[t.id] = a
                            if tool:
                                tools_env[t.id] = tool
        return frontier

    for fn in roots_later:
        for call in (n for n in ast.walk(fn.node) if isinstance(n, ast.Call)):
            t = callee(call, fn)
            if t is not None and t.node is not fn.node:
                called.add(id(t.node))
    for root in roots:  # module code calls functions too
        for call in (n for n in ast.walk(root.node) if isinstance(n, ast.Call)):
            t = callee(call, root)
            if t is not None:
                called.add(id(t.node))
    roots += [fn for fn in roots_later if id(fn.node) not in called]
    for fn in roots_later:  # agents stored under a key: agents_map["planner"] = make_planner()
        local_env(fn)
    # which functions can run an agent at all: only those are followed (a deep helper tree with none is skipped)
    calls_of: dict[int, list[int]] = {}
    for fn in roots_later:
        env = local_env(fn)
        outs = []
        for call in (n for n in ast.walk(fn.node) if isinstance(n, ast.Call)):
            if uses(call, env, fn):
                reach.add(id(fn.node))
            t = callee(call, fn)
            if t is not None:
                outs.append(id(t.node))
        calls_of[id(fn.node)] = outs
    grew = True
    while grew:
        grew = False
        for k, outs in calls_of.items():
            if k not in reach and any(o in reach for o in outs):
                reach.add(k)
                grew = True
    for fn in roots:
        if budget[0] <= 0:
            break
        walk_fn(fn, set(), 0, frozenset())
    c.edges += edges


def _repeats(st: ast.AST) -> bool:
    """A loop that runs the same work again (`while ...`, `for _ in range(n)`), not one pass over planned items."""
    if isinstance(st, ast.While):
        return True
    it = getattr(st, "iter", None)
    return isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"


def _join(a: set[str] | None, b: set[str] | None) -> set[str] | None:
    """Paths meeting: None is a path that never gets here."""
    if a is None:
        return None if b is None else set(b)
    return set(a) | (b or set())


def _calls_in_order(node: ast.AST):
    """Calls in the order they run: a call's arguments (and the call its method hangs on) before the call itself."""
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield from _calls_in_order(child)
    if isinstance(node, ast.Call):
        yield node


def _is_partial(call: ast.Call) -> bool:
    f = call.func
    return (isinstance(f, ast.Name) and f.id == "partial") or (isinstance(f, ast.Attribute) and f.attr == "partial")


def subgraphs_run_in_nodes(c: Any) -> None:
    """A node whose function runs a compiled graph itself (`await researcher_graph.ainvoke(...)` inside it) hands work
    to that graph like a node that *is* the graph. The compiled graph has to come from the function's own file (`graph`
    and `app` are everyone's variable names), and a node re-running its own graph is a loop, not a subgraph."""
    if not c.compiled:
        return
    have = set(c.subgraphs)
    file_of: dict[int, str] = {}
    own_builder: dict[str, set[str]] = {}  # node -> builders that add it
    for rel, tree in c.trees.items():
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                file_of[id(n)] = rel
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_node" \
                    and n.args and _lit(n.args[0]) and isinstance(n.func.value, (ast.Name, ast.Attribute)):
                b = n.func.value.id if isinstance(n.func.value, ast.Name) else n.func.value.attr
                own_builder.setdefault(_lit(n.args[0]), set()).add(f"{rel}:{b}")
    for fn_name, node_name in c.node_funcs.items():
        fn = c.functions.get(fn_name)
        if fn is None:
            continue
        rel = file_of.get(id(fn))
        for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
            f = call.func
            if isinstance(f, ast.Attribute) and f.attr in ("invoke", "ainvoke", "stream", "astream", "batch",
                                                           "abatch") and isinstance(f.value, ast.Name):
                builder = c.compiled.get(f.value.id)
                if not builder or builder.split(":", 1)[0] != rel or builder in own_builder.get(node_name, ()):
                    continue
                if (node_name, builder) not in have:
                    have.add((node_name, builder))
                    c.subgraphs.append((node_name, builder))
