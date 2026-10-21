"""Agents for code that calls an LLM without an agent framework.

Any function or class that calls a model is an agent, even when there is only one. A shared helper that
merely forwards a prompt (``generate_reply(system_prompt, user_content)``) is not: the functions that call it
with prompts of their own are. Calls are recognised through the libraries they come from - OpenAI-style
SDKs, Anthropic, Gemini, LiteLLM, Ollama, mlx-lm, llama.cpp, transformers, vLLM, LangChain chat models -
so the same rules work for any project.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from .collector import SourceFile, agent_class_names, is_test_file

LLM_MODULES = (
    "openai", "anthropic", "google.genai", "google.generativeai", "litellm", "ollama", "mlx_lm", "mlx_vlm",
    "llama_cpp", "transformers", "vllm", "huggingface_hub", "groq", "mistralai", "cohere", "together",
    "langchain_openai", "langchain_anthropic", "langchain_google_genai", "langchain_google_vertexai",
    "langchain_ollama", "langchain_groq", "langchain_mistralai", "langchain_aws", "langchain_community.chat_models",
    "langchain_community.llms", "langchain_huggingface",
)
# method chains that are LLM calls whatever the client variable is called
LLM_CHAINS = (
    "completions.create", "responses.create", "models.generate_content", "models.generate_content_stream",
    "generate_content", "chat_completion", "text_generation", "create_chat_completion",
)
# ...and ones other APIs share (Twilio's messages.create sends an SMS): only in files importing an LLM library
LLM_CHAINS_IF_IMPORTED = ("messages.create", "messages.stream", "responses.stream", "chat.complete", "chat.stream",
                          # the pre-1.0 OpenAI SDK, still in many research repos: openai.ChatCompletion.create(...)
                          "ChatCompletion.create", "ChatCompletion.acreate", "Completion.create", "Completion.acreate")
# module-level functions of LLM libraries that run the model: litellm.completion, ollama.chat, mlx_lm.generate
LLM_ENTRY = {"completion", "acompletion", "chat", "achat", "generate", "agenerate", "stream_generate",
             "create_chat_completion", "create_completion", "text_generation", "chat_completion"}
# methods called on a model/client object: llm.invoke(...), model.generate(...), llm(...)
LLM_USE = {"invoke", "ainvoke", "stream", "astream", "batch", "abatch", "generate", "agenerate", "chat", "complete",
           "predict", "create_chat_completion", "create_completion", "generate_content"}
CLIENT_CLASSES = {
    "OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI", "Anthropic", "AsyncAnthropic", "AnthropicBedrock",
    "AnthropicVertex", "Groq", "AsyncGroq", "Mistral", "Together", "Cohere", "ClientV2", "GenerativeModel",
    "ChatOpenAI", "AzureChatOpenAI", "ChatAnthropic", "ChatGoogleGenerativeAI", "ChatVertexAI", "ChatOllama",
    "ChatGroq", "ChatMistralAI", "ChatBedrock", "ChatBedrockConverse", "ChatHuggingFace", "Llama", "LLM",
    "InferenceClient", "AsyncInferenceClient",
}
# what a model/client object is built from (the last part of the call that created it)
_MODEL_ORIGIN = re.compile(r"(llm|llama|chat|client|pipeline|model|anthropic|openai|genai|gemini|groq|mistral|"
                           r"cohere|ollama|inference|load|from_pretrained)", re.I)
_NOT_MODEL = re.compile(r"(tokenizer|processor|embedding|embedder|encoder|config|sampler)", re.I)
_PROMPT_NAME = re.compile(r"(prompt|instruction|template|system|persona|preamble)", re.I)
# parameters a wrapper receives its prompt through: when the prompt comes in, the caller wrote it
_PROMPT_PARAMS = re.compile(r"^(\w*_)?(messages|msgs?|prompts?|system(_prompt|_message|_role)?|contents|"
                            r"(chat_)?history|conversation)$", re.I)
# keyword arguments that hand a wrapper the role it plays this time: helper(prompt, system_role=analyst)
_ROLE_KWARGS = ("system_role", "system", "system_prompt", "system_message", "sys_prompt", "role", "persona",
                "instructions", "agent_role")
# "You are LAMBDA, a data analysis agent ..." / "You are a professional data scientist writing reports."
_PERSONA = re.compile(r"\bYou are\s+(?:(?P<name>[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*){0,3})\s*[,(:\-—–]\s*)?"
                      r"(?:an?|the)\s+(?P<role>[^\n.;!?]{3,120})")
_MODEL_NAMES = {"chatgpt", "gpt", "claude", "gemini", "llama", "qwen", "deepseek", "mistral", "grok", "ai"}
_SKIP_PATH = re.compile(r"(^|/)(examples?|docs?|scripts?/dev)(/|$)")
MAX_AGENTS = 15
# a function named as an agent (research_agent, run_agent, AgentStep) - not one that merely contains the letters
_AGENT_WORD = re.compile(r"(?i:(^|_)agents?($|_))|Agents?($|[A-Z_0-9])")
# function names that say only that they call a model, not which agent they are
_GENERIC_NAME = re.compile(r"^_*(main|run|run_llm|call|call_llm|call_model|api_call|call_api|query_llm|get_completion|"
                           r"get_response|llm|infer|inference|request|execute|handler|process)$", re.I)
# decorators that register a function for a framework or server to call by itself
_REGISTERS = re.compile(r"(route|get|post|put|delete|websocket|command|tool|task|agent|node|app|on_\w+|listener|"
                        r"handler|callback|register|entrypoint|function_tool|hook|event|cli|main)", re.I)


@dataclass
class _Fn:
    rel: str
    cls: str | None
    name: str  # "<module>" for top-level code
    line: int
    doc: str
    node: ast.AST
    calls: list[tuple[str, int]] = field(default_factory=list)  # (simple name, line) of every call
    cls_doc: str = ""
    via: str = ""  # how it reaches the model: "mlx_lm.generate", "chat.completions.create", "generate_reply"
    own_prompt: bool = False
    tools: list[str] = field(default_factory=list)  # tool schemas it defines and hands to the model
    wrapped: list[str] = field(default_factory=list)  # wrapper methods it reaches the model through
    cls_bases: int = 0  # how many base classes its class names (an override may be called by a framework)
    picks: dict[str, set[str]] = field(default_factory=dict)  # callee -> prompt names it passes (prompt="reflect")
    self_calls: set[str] = field(default_factory=set)  # names it calls only as self.x() / cls.x()

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.rel, self.cls or "", self.name)


def find(files: list[SourceFile]) -> dict[str, Any]:
    """Agents and edges for LLM call sites: {"agents": [...], "edges": [...]} (empty when there are none)."""
    fns: list[_Fn] = []
    direct: list[_Fn] = []
    imports: dict[str, _Imports] = {}  # file -> where its names come from
    referenced: set[str] = set()  # every name and attribute the project's code uses
    for f in files:
        if f.path.suffix.lower() not in (".py", ".ipynb"):  # entry points named in pyproject, shell scripts, configs
            referenced.update(re.findall(r"[A-Za-z_]\w+", f.text))
        if f.path.suffix.lower() not in (".py", ".ipynb") or _SKIP_PATH.search(f.rel.replace("\\", "/")) \
                or is_test_file(f.rel, f.text):
            continue
        try:
            tree = ast.parse(f.text)
        except (SyntaxError, ValueError):
            continue
        imports[f.rel] = _import_sources(tree)
        referenced.update(n.id if isinstance(n, ast.Name) else n.attr
                          for n in ast.walk(tree) if isinstance(n, (ast.Name, ast.Attribute)))
        mods, names = _imports(tree)
        clients = _client_vars(tree, mods, names)
        chains = LLM_CHAINS + (LLM_CHAINS_IF_IMPORTED if _imports_llm(mods, names) else ())
        for fn in _functions(tree, f.rel):
            fns.append(fn)
            fn.via = _llm_call(fn.node, mods, names, clients, chains, in_class=fn.cls is not None)
            fn.own_prompt = _has_own_prompt(fn.node)
            fn.tools = _tool_names(fn.node)
            if fn.via:
                direct.append(fn)
    # a class that builds its client in __init__ and calls it in other methods: those calls are what count
    calling = {(fn.rel, fn.cls) for fn in direct if fn.cls and not fn.name.startswith("__")}
    direct = [fn for fn in direct if not (fn.name.startswith("__") and (fn.rel, fn.cls) in calling)]
    # code nothing calls (an old completion helper left next to the chat one) is not part of any run
    # (a library's or an app's functions may be called from outside: only one beside a used model call is dropped)
    dead = {fn.key for fn in direct if _unused(fn, referenced)}
    used_files = {fn.rel for fn in direct if fn.key not in dead}
    direct = [fn for fn in direct if fn.key not in dead or fn.rel not in used_files]
    if not direct:
        return {"agents": [], "edges": []}

    callers: dict[str, list[_Fn]] = {}
    picked_names = {name for fn in fns for name in fn.picks}
    for fn in fns:
        for name, _ in fn.calls:
            callers.setdefault(name, []).append(fn)

    def reaches(c: _Fn, fn: _Fn) -> bool:
        """Can code in c's file call fn? A function is reached from its own file or from a file importing it; two
        scripts that each define their own api_call() are not one helper. Methods are called on objects passed
        around, so any file may reach them."""
        if c.rel == fn.rel or fn.cls:
            return True
        return imports.get(c.rel, _Imports()).reaches(fn.name, fn.rel)

    def callers_of(fn: _Fn) -> list[_Fn]:
        if fn.name == "<module>" or fn.name.startswith("__"):  # self.__init__() in reset() is not a model call
            return []
        seen, out = set(), []
        for c in callers.get(fn.name, []):
            if c.key == fn.key or c.key in seen or not reaches(c, fn):
                continue
            # self._run() in a method of a class with no base is that class's own _run, not another class's
            if c.cls and fn.cls and c.cls != fn.cls and not c.cls_bases and fn.name in c.self_calls:
                continue
            # `if __name__ == "__main__": main()` does not make main plumbing - but a script whose top-level code
            # writes the prompts and sends each through its own api_call() helper is the agent
            if c.name == "<module>" and not (c.rel == fn.rel and c.own_prompt):
                continue
            seen.add(c.key)
            out.append(c)
        return out

    # callers that each pick the prompt a shared model call renders are steps of their own: climbing stops there
    choosers: set[tuple[str, str, str]] = set()
    for fn in fns:
        if fn.name in picked_names:
            ups = callers_of(fn)
            if _picks_prompt(fn, ups):
                choosers |= {c.key for c in ups if c.picks.get(fn.name)}

    sites: dict[tuple[str, str, str], _Fn] = {}
    services: set[tuple[str, str]] = set()  # LLM wrapper classes used from elsewhere (LLMService, ModelClient)

    def climb(fn: _Fn, via: str, depth: int, trail: frozenset, tools: tuple[str, ...], wrapped: tuple[str, ...],
              chose: bool = False):
        chose = chose or fn.key in choosers
        ups = [c for c in callers_of(fn) if c.key not in trail] if not chose else []  # it chose the prompt: a step
        tools = tools + tuple(t for t in fn.tools if t not in tools)
        picked = bool(ups) and _picks_prompt(fn, ups)
        # a helper that only forwards someone else's prompt is plumbing: its callers are the agents
        if ups and depth < 3 and ((not fn.own_prompt and not _AGENT_WORD.search(fn.name)) or picked):
            if fn.cls and any((c.rel, c.cls) != (fn.rel, fn.cls) for c in ups):
                services.add((fn.rel, fn.cls))
            # the runtime sees the wrapper (LLMService.stream_chat) when its caller runs it in another task
            wrapped = wrapped + ((f"{fn.cls}.{fn.name}",) if fn.cls else ())
            for c in ups:
                climb(c, fn.name, depth + 1, trail | {fn.key}, tools, wrapped, picked and bool(c.picks.get(fn.name)))
            return
        if fn.key not in sites:
            fn.via = fn.via if depth == 0 else via
            sites[fn.key] = fn
        site = sites[fn.key]
        site.tools += [t for t in tools if t not in site.tools]
        site.wrapped += [w for w in wrapped if w not in site.wrapped]

    for fn in direct:
        climb(fn, fn.via, 0, frozenset(), (), ())
    # ...and so is a caller nothing calls, beside one that is used (a leftover get_output() next to the chat helper)
    dead = {k for k, fn in sites.items() if _unused(fn, referenced)}
    if len(dead) < len(sites):
        sites = {k: fn for k, fn in sites.items() if k not in dead or fn.rel not in used_files}

    # a class named or built like an agent (SolverAgent(BaseAgent)) is one agent, whichever of its methods reach the
    # model and however much other code calls them; only a shared client class (LLMService) splits into its methods
    agent_classes = agent_class_names(files)
    agents: dict[str, dict[str, Any]] = {}  # agent id -> agent
    meta: dict[str, dict[str, Any]] = {}  # agent id -> how it was found: its file, its functions, named after
    owner: dict[str, str] = {}  # function name -> agent id
    role_edges: list[tuple[str, str]] = []
    for fn in sorted(sites.values(), key=lambda s: (s.rel, s.line)):
        stem = _stem(fn.rel)
        if fn.cls and ((fn.rel, fn.cls) not in services or fn.cls in agent_classes):
            aid, name, aliases, named = f"{fn.rel}:{fn.cls}", fn.cls, [fn.cls, fn.name], "class"
        elif fn.name == "<module>":
            aid, name, aliases, named = f"{fn.rel}:<module>", stem, [stem], "file"
        else:  # a function, or one prompt-writing method of a service class (LLMService.generate_title)
            aid, name, aliases, named = f"{fn.rel}:{fn.cls or ''}.{fn.name}", fn.name, [fn.name], "function"
            if fn.cls:
                aliases.append(f"{fn.cls}.{fn.name}")
        aliases += [w for w in fn.wrapped if w not in aliases]
        persona, role = _persona(fn.node)
        if persona:  # "You are LAMBDA, a data analysis agent": the agent's own name beats its function's
            aid, name, aliases, named = f"{fn.rel}:{persona}", persona, [persona, *aliases], "persona"
        roles = _roles_played(fn) if fn.via else []
        if len(roles) >= 2:  # one function sending a helper several roles in turn: each role is an agent
            ids = []
            for r in roles:
                rid = f"{fn.rel}:{fn.name}:{r}"
                if rid not in agents and len(agents) < MAX_AGENTS:
                    agents[rid] = {"name": r, "role": f"a role {fn.name}() gives {fn.via}()", "description": "",
                                   "tools": list(fn.tools), "file": fn.rel, "line": fn.line,
                                   "aliases": [r, f"{fn.name}.{r}"]}
                    meta[rid] = {"rel": fn.rel, "fns": {fn.name}, "named": "role"}
                if rid in agents:
                    ids.append(rid)
            order = _ordered_roles(fn, roles, ids)
            role_edges += list(zip(order, order[1:]))
            if ids:
                owner.setdefault(fn.name, ids[0])
            continue
        agent = agents.get(aid)
        if agent is None:
            if len(agents) >= MAX_AGENTS:
                continue
            agent = agents[aid] = {"name": name, "role": role or f"calls the LLM ({fn.via})",
                                   "description": fn.cls_doc or fn.doc,
                                   "tools": [], "file": fn.rel, "line": fn.line, "aliases": []}
            meta[aid] = {"rel": fn.rel, "fns": set(), "named": named}
        meta[aid]["fns"].add(fn.name)
        agent["aliases"] += [a for a in aliases if a not in agent["aliases"]]
        agent["tools"] += [t for t in fn.tools if t not in agent["tools"]]
        if fn.name != "<module>":
            owner.setdefault(fn.name, aid)
        else:
            owner.setdefault(f"<module>:{fn.rel}", aid)

    merged = _merge_variants(agents, meta)  # programmer_humaneval.py + programmer_mbpp.py: one programmer
    merged.update(_merge_step_variants(agents, meta, fns))  # one_chunk_reflect() + multichunk_reflect(): one step
    _name_after_files(agents, meta)  # api_call() in three scripts: three agents named after their scripts
    for k, v in list(owner.items()):
        owner[k] = merged.get(v, v)

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()

    def link(a: str, b: str, label: str, pipeline: bool = False) -> None:
        a, b = merged.get(a, a), merged.get(b, b)
        if a != b and a in agents and b in agents and (a, b) not in seen_edges and len(edges) < 20:
            seen_edges.add((a, b))
            edges.append({"from": agents[a]["name"], "to": agents[b]["name"], "label": label, "conditional": False,
                          **({"_pipeline": True} if pipeline else {})})

    for a, b in role_edges:
        link(a, b, "then")
    for fn in fns:
        mine = owner.get(fn.name) if fn.key in sites else None
        if fn.name == "<module>" and fn.key in sites:
            mine = owner.get(f"<module>:{fn.rel}")
        called = []
        for n, _ in sorted(fn.calls, key=lambda c: c[1]):
            target = owner.get(n)
            if target in agents or merged.get(target, "") in agents:
                fdef = next((s for s in sites.values() if s.name == n), None)
                if fdef is None or reaches(fn, fdef):
                    called.append(target)
        if mine:  # an agent that calls another agent hands work to it
            for other in dict.fromkeys(called):
                link(mine, other, "calls")
        elif len(set(called)) >= 2:  # a pipeline / orchestrator running agents one after another
            order = list(dict.fromkeys(called))
            for a, b in zip(order, order[1:]):
                link(a, b, "then", pipeline=True)  # in call order: a static pass that follows branches may redo it
    return {"agents": list(agents.values()), "edges": edges}


def _unused(fn: _Fn, referenced: set[str]) -> bool:
    """Neither called nor referenced anywhere in the project, nor registered by a decorator. A method only counts
    when the project uses its class but never that method, and the class has no base it could override."""
    if fn.name == "<module>" or fn.name.startswith("__") or fn.name in referenced or fn.cls_bases:
        return False
    if fn.cls and fn.cls not in referenced:  # a class nobody creates here may be the project's API: keep it
        return False
    decos = getattr(fn.node, "decorator_list", [])
    return not any(_REGISTERS.search(_dotted(d)) for d in decos)


@dataclass
class _Imports:
    """Where a file's names come from: `from pkg.helpers import api_call`, `from helpers import *`, `import helpers`."""
    names: dict[str, str] = field(default_factory=dict)  # local name -> the module it is imported from (last part)
    star: set[str] = field(default_factory=set)  # modules imported with *
    modules: set[str] = field(default_factory=set)  # modules imported whole (last part)

    def reaches(self, name: str, rel: str) -> bool:
        parts = rel.replace("\\", "/").split("/")
        mod = _stem(rel)
        here = {mod} if mod != "__init__" else {parts[-2] if len(parts) > 1 else mod}
        return (name in self.names and (self.names[name] in here or self.names[name] == "")) \
            or bool(here & self.star) or bool(here & self.modules)


def _import_sources(tree: ast.AST) -> _Imports:
    out = _Imports()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.modules.update(a.name.split(".")[-1] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[-1]
            for a in node.names:
                if a.name == "*":
                    out.star.add(mod)
                else:
                    out.names[a.asname or a.name] = mod
                    out.modules.add(a.name)  # from pkg import helpers; helpers.api_call()
    return out


def _stem(rel: str) -> str:
    return re.sub(r"\.(py|ipynb)$", "", rel.replace("\\", "/").rsplit("/", 1)[-1])


def _role_value(call: ast.Call) -> str | None:
    """The role a call hands its helper: system_role=analyst -> "analyst"; an empty or computed one -> None."""
    for k in call.keywords:
        if k.arg in _ROLE_KWARGS:
            v = k.value
            if isinstance(v, ast.Name):
                return v.id
            if isinstance(v, ast.Attribute):
                return v.attr
            return None
    return None


def _roles_played(fn: _Fn) -> list[str]:
    """Distinct named roles fn hands to the helper it reaches the model through, in source order."""
    out: list[str] = []
    for call in sorted(_own_calls(fn.node), key=lambda c: (c.lineno, c.col_offset)):
        if _dotted(call.func).replace("()", "").split(".")[-1] != fn.via:
            continue
        r = _role_value(call)
        if r and r not in out and not _PROMPT_PARAMS.match(r):
            out.append(r)
    return out


def _ordered_roles(fn: _Fn, roles: list[str], ids: list[str]) -> list[str]:
    """Agent ids in the order fn runs their roles, a role repeated only when another ran in between."""
    rid = dict(zip(roles, ids))
    seq: list[str] = []
    for call in sorted(_own_calls(fn.node), key=lambda c: (c.lineno, c.col_offset)):
        if _dotted(call.func).replace("()", "").split(".")[-1] == fn.via:
            r = _role_value(call)
            if r in rid and (not seq or seq[-1] != rid[r]):
                seq.append(rid[r])
    return seq


def _merge_variants(agents: dict[str, dict[str, Any]], meta: dict[str, dict[str, Any]]) -> dict[str, str]:
    """One script per dataset or backend (solver_math.py, solver_code.py) defining the same functions: the same agent
    run on different inputs. Merged into the first, named after what the files share. Returns merged id -> kept id."""
    groups: dict[tuple[str, str, frozenset], list[str]] = {}
    for aid, m in meta.items():
        if m["named"] not in ("function", "file") or "_" not in _stem(m["rel"]):
            continue
        folder = m["rel"].replace("\\", "/").rsplit("/", 1)[0] if "/" in m["rel"] else ""
        prefix = _stem(m["rel"]).rsplit("_", 1)[0]
        fns = frozenset(m["fns"]) if m["named"] == "function" else frozenset({"<module>"})
        if prefix and not prefix.isdigit():
            groups.setdefault((folder, prefix, fns), []).append(aid)
    merged: dict[str, str] = {}
    for (_, prefix, _), ids in groups.items():
        if len({meta[i]["rel"] for i in ids}) < 2:
            continue
        keep = agents[ids[0]]
        keep["name"] = prefix
        meta[ids[0]]["named"] = "merged"
        keep["aliases"] = [prefix, *keep["aliases"]]
        for other in ids[1:]:
            keep["aliases"] += [a for a in agents[other]["aliases"] if a not in keep["aliases"]]
            keep["tools"] += [t for t in agents[other]["tools"] if t not in keep["tools"]]
            keep["aliases"] += [_stem(meta[other]["rel"])] if _stem(meta[other]["rel"]) not in keep["aliases"] else []
            del agents[other]
            merged[other] = ids[0]
        if _stem(meta[ids[0]]["rel"]) not in keep["aliases"]:
            keep["aliases"].append(_stem(meta[ids[0]]["rel"]))
    return merged


def _feeds(fns: list[_Fn]) -> set[tuple[str, str]]:
    """(producer, consumer) pairs of functions where the result of one call is passed to another: b(a(x)), or
    y = a(x) and later b(y)."""
    out: set[tuple[str, str]] = set()

    def name(call: ast.Call) -> str:
        return _dotted(call.func).replace("()", "").split(".")[-1]

    for f in fns:
        made: dict[str, set[str]] = {}  # variable -> the calls whose result it holds
        for node in sorted((n for n in ast.walk(f.node) if isinstance(n, (ast.Assign, ast.Call))),
                           key=lambda n: (n.lineno, n.col_offset)):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        made[t.id] = {name(c) for c in ast.walk(node.value) if isinstance(c, ast.Call)}
            elif isinstance(node, ast.Call):
                me = name(node)
                for arg in [*node.args, *(k.value for k in node.keywords)]:
                    for sub in ast.walk(arg):
                        if isinstance(sub, ast.Call):
                            out.add((name(sub), me))
                        elif isinstance(sub, ast.Name):
                            out |= {(p, me) for p in made.get(sub.id, ())}
    return out


def _merge_step_variants(agents: dict[str, dict[str, Any]], meta: dict[str, dict[str, Any]],
                         fns: list[_Fn]) -> dict[str, str]:
    """Steps written twice for two modes in one file (one_chunk_translate / one_chunk_reflect next to
    multichunk_translate / multichunk_reflect): the same step, in a variant. Two name prefixes that pair up on two or
    more steps are variants - unless one's output is fed to the other (review_code(generate_code(x)): two roles on one
    thing, not one step twice). Each pair is merged into one agent named after the step. Returns merged id -> kept id."""
    feeds = _feeds(fns)
    by_step: dict[tuple[str, str], dict[str, str]] = {}  # (file, step) -> prefix -> agent id
    for aid, m in meta.items():
        if m["named"] != "function" or aid not in agents:
            continue
        tokens = agents[aid]["name"].lstrip("_").split("_")
        for k in (1, 2):
            step = "_".join(tokens[k:])
            if len(tokens) > k + 1 or len(step) >= 8:
                by_step.setdefault((m["rel"], step), {})["_".join(tokens[:k])] = aid
    pairs: dict[tuple[str, str, str], list[tuple[str, str]]] = {}  # (file, prefix a, prefix b) -> steps
    for (rel, step), pre in by_step.items():
        names = sorted(pre)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if pre[a] != pre[b]:
                    pairs.setdefault((rel, a, b), []).append((step, pre[a]))
    merged: dict[str, str] = {}
    for (rel, a, b), steps in pairs.items():
        if len(steps) < 2:
            continue
        both = [(agents[keep_id]["name"], agents[by_step[(rel, step)][b]]["name"]) for step, keep_id in steps
                if keep_id in agents and by_step[(rel, step)][b] in agents]
        if any((x, y) in feeds or (y, x) in feeds for x, y in both):
            continue
        for step, keep_id in steps:
            other_id = by_step[(rel, step)][b]
            keep_id = merged.get(keep_id, keep_id)
            if other_id in merged or other_id not in agents or keep_id not in agents or other_id == keep_id:
                continue
            keep, other = agents[keep_id], agents.pop(other_id)
            keep["aliases"] += [x for x in other["aliases"] if x not in keep["aliases"]]
            keep["tools"] += [t for t in other["tools"] if t not in keep["tools"]]
            keep["name"] = step
            keep["aliases"] = [step, *[x for x in keep["aliases"] if x != step]]
            merged[other_id] = keep_id
    return merged


def _name_after_files(agents: dict[str, dict[str, Any]], meta: dict[str, dict[str, Any]]) -> None:
    """Functions of the same name in different files (each script's own api_call()) are different agents: name each
    after its file so they stay apart on the map. So is one whose name only says it calls a model (main, run_llm)."""
    by_name: dict[str, list[str]] = {}
    for aid, a in agents.items():
        if meta.get(aid, {}).get("named") == "function":
            by_name.setdefault(a["name"], []).append(aid)
    for name, ids in by_name.items():
        # a name that says nothing about the agent (main, api_call, run_llm) is named after its file too
        if len({meta[i]["rel"] for i in ids}) < 2 and not _GENERIC_NAME.match(name):
            continue
        for i in ids:
            stem = _stem(meta[i]["rel"])
            agents[i]["name"] = stem
            if stem not in agents[i]["aliases"]:
                agents[i]["aliases"].insert(0, stem)


# -- helpers -------------------------------------------------------------------------------------------------

def _is_llm_module(mod: str) -> bool:
    return any(mod == m or mod.startswith(m + ".") for m in LLM_MODULES)


def _imports_llm(mods: dict[str, str], names: dict[str, tuple[str, str]]) -> bool:
    return any(map(_is_llm_module, mods.values())) or any(_is_llm_module(m) for m, _ in names.values())


def _imports(tree: ast.AST) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """alias -> module for `import x as y`; name -> (module, original) for `from x import y` (anywhere)."""
    mods: dict[str, str] = {}
    names: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                full = f"{node.module}.{a.name}"
                if _is_llm_module(full) and a.name[:1].islower() and a.name not in LLM_ENTRY:
                    mods[a.asname or a.name] = full  # from google import genai
                names[a.asname or a.name] = (node.module, a.name)
    return mods, names


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _dotted(node.func) + "()"
    if isinstance(node, ast.Subscript):
        return _dotted(node.value) + "[]"
    return "?"


def _from_llm(root: str, mods: dict[str, str], names: dict[str, tuple[str, str]]) -> bool:
    return (root in mods and _is_llm_module(mods[root])) or (root in names and _is_llm_module(names[root][0]))


def _client_vars(tree: ast.AST, mods, names) -> dict[str, str]:
    """Variables / attributes holding a model or client: name -> what built it ("Llama", "pipeline", ...)."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(value, ast.Call):
            continue
        parts = _dotted(value.func).replace("()", "").split(".")
        origin = parts[-1]
        if not (origin in CLIENT_CLASSES or (_from_llm(parts[0], mods, names) and _MODEL_ORIGIN.search(origin))):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            for el in (t.elts if isinstance(t, ast.Tuple) else [t]):
                var = el.id if isinstance(el, ast.Name) else el.attr if isinstance(el, ast.Attribute) else None
                if var and not _NOT_MODEL.search(var):
                    out[var] = origin
    return out


def _llm_call(node: ast.AST, mods, names, clients, chains=LLM_CHAINS, in_class=False) -> str:
    """How this function calls a model ("" when it doesn't): the call as written, e.g. "mlx_lm.generate"."""
    for call in _own_calls(node):
        d = _dotted(call.func).replace("()", "").replace("[]", "")
        parts = d.split(".")
        root, last = parts[0], parts[-1]
        for chain in chains:
            if d == chain or d.endswith("." + chain):
                return chain
        if len(parts) > 1 and root in mods and _is_llm_module(mods[root]) and last in LLM_ENTRY:
            return f"{mods[root]}.{last}"  # litellm.completion, ollama.chat
        if len(parts) == 1 and root in names and _is_llm_module(names[root][0]) and names[root][1] in LLM_ENTRY:
            return f"{names[root][0]}.{names[root][1]}"  # from mlx_lm import generate; generate(...)
        if in_class and last in CLIENT_CLASSES and len(parts) <= 2:
            return last  # a class that builds its own client / chat model is an LLM-backed component
        var = parts[1] if root == "self" and len(parts) > 1 else root
        origin = clients.get(var)
        if origin and ((len(parts) == (2 if root == "self" else 1) and origin in ("pipeline", "Llama", "LLM"))
                       or (len(parts) > 1 and last in LLM_USE)):
            return f"{origin}.{last}" if last != var else origin
    return ""


def _own_calls(node: ast.AST):
    """Calls in this function's body, not in nested classes (nested functions count: they run as part of it)."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, ast.ClassDef):
            continue
        if isinstance(n, ast.Call):
            yield n
        stack.extend(ast.iter_child_nodes(n))


def _not_prompt_text(node: ast.AST) -> set[int]:
    """ids of nodes inside the docstring, raise statements, log/print calls and command-line options: their text is
    not a prompt."""
    body = getattr(node, "body", [])
    doc = body[0] if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None),
                                                                            ast.Constant) else None
    skip = {id(doc.value)} if doc is not None else set()
    for n in ast.walk(node):
        if isinstance(n, ast.Raise) or (isinstance(n, ast.Call) and re.match(
                r"(print|warn|(.*\.)?ArgumentParser|.*\.(debug|info|warning|error|exception|critical|warn|log|"
                r"add_argument|add_option|add_parser))$", _dotted(n.func))):
            skip |= {id(x) for x in ast.walk(n)}
    return skip


def _has_own_prompt(node: ast.AST) -> bool:
    """Does the function write its own prompt (text, f-strings, a *_PROMPT constant) or only pass one through?"""
    skip = _not_prompt_text(node)
    if _receives_prompt(node):
        # the prompt comes in as an argument: tool schemas and error text around it are not a prompt of its own,
        # only a system message it adds itself is (LLMService.stream_chat(messages) is a wrapper)
        return _writes_system_message(node, skip)
    chars = 0
    for n in ast.walk(node):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and " " in n.value:
            chars += len(n.value)
        elif isinstance(n, ast.Name) and _PROMPT_NAME.search(n.id) and n.id.isupper():
            return True
        elif isinstance(n, ast.Attribute) and _PROMPT_NAME.search(n.attr) and n.attr.isupper():
            return True
        elif isinstance(n, ast.keyword) and n.arg in ("system", "system_prompt", "instructions") and \
                isinstance(n.value, (ast.Constant, ast.JoinedStr)):
            return True
    return chars >= 24


def _receives_prompt(node: ast.AST) -> bool:
    """Does the function take its messages / prompt as a parameter and use it?"""
    args = getattr(node, "args", None)
    if args is None:
        return False
    params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs] if _PROMPT_PARAMS.match(a.arg)}
    return bool(params) and any(isinstance(n, ast.Name) and n.id in params for n in ast.walk(node))


def _writes_system_message(node: ast.AST, skip: set[int]) -> bool:
    """{"role": "system", "content": "..."}, system="...", or a *_PROMPT constant written by this function."""
    for n in ast.walk(node):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Dict):
            fields = {k.value: v for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant)}
            role, content = fields.get("role"), fields.get("content")
            if isinstance(role, ast.Constant) and role.value in ("system", "developer") and \
                    isinstance(content, (ast.Constant, ast.JoinedStr, ast.BinOp)):
                return True
        elif isinstance(n, ast.keyword) and n.arg in ("system", "system_prompt", "instructions") and \
                isinstance(n.value, (ast.Constant, ast.JoinedStr)):
            return True
        elif isinstance(n, (ast.Name, ast.Attribute)):
            ident = n.id if isinstance(n, ast.Name) else n.attr
            if ident.isupper() and _PROMPT_NAME.search(ident):
                return True
    return False


def _persona(node: ast.AST) -> tuple[str, str]:
    """(name, role) from the first "You are X, a ..." in the function's prompt text; "" for what it doesn't say."""
    skip = _not_prompt_text(node)
    texts = []
    for n in ast.walk(node):
        if id(n) not in skip and isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) >= 40:
            texts.append((n.lineno, n.col_offset, n.value))
    for _, _, text in sorted(texts):
        m = _PERSONA.search(text)
        if not m:
            continue
        name = (m.group("name") or "").strip()
        if name.lower() in _MODEL_NAMES:
            name = ""
        role = re.split(r"\s+(?:that|who|which|whose|responsible)\s+|,", m.group("role"), maxsplit=1)[0].strip()
        return name, role
    return "", ""


def _tool_names(node: ast.AST) -> list[str]:
    """Tools described to the model in this function: OpenAI {"type": "function", "function": {"name": ...}}
    and Anthropic / Gemini {"name": ..., "input_schema" | "parameters": ...} schemas."""
    out: list[str] = []

    def fields(d: ast.Dict) -> dict[str, ast.AST]:
        return {k.value: v for k, v in zip(d.keys, d.values) if isinstance(k, ast.Constant)}

    for n in ast.walk(node):
        if not isinstance(n, ast.Dict):
            continue
        f = fields(n)
        name = None
        if isinstance(f.get("function"), ast.Dict) and isinstance(f.get("type"), ast.Constant) and \
                f["type"].value == "function":
            inner = fields(f["function"])
            if "parameters" in inner or "description" in inner:
                name = inner.get("name")
        elif "input_schema" in f or ("parameters" in f and "description" in f):
            name = f.get("name")
        if isinstance(name, ast.Constant) and isinstance(name.value, str) and name.value and name.value not in out:
            out.append(name.value)
    return out


def _functions(tree: ast.Module, rel: str) -> list[_Fn]:
    """Top-level functions, methods (with their class) and the module's own top-level code."""
    out: list[_Fn] = []

    def first_line(node) -> str:
        return (ast.get_docstring(node) or "").strip().split("\n")[0][:200]

    def add(node, cls, cls_node=None):
        fn = _Fn(rel, cls, node.name, node.lineno, first_line(node), node)
        fn.cls_doc = first_line(cls_node) if cls_node is not None else ""
        fn.cls_bases = len(cls_node.bases) if cls_node is not None else 0
        fn.calls, fn.picks = _calls_and_picks(node)
        fn.self_calls = _self_calls(node)
        out.append(fn)

    top = ast.Module(body=[], type_ignores=[])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, None)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(item, node.name, node)
        else:
            top.body.append(node)
    if top.body:
        fn = _Fn(rel, None, "<module>", top.body[0].lineno, "", top)
        fn.calls, fn.picks = _calls_and_picks(top)
        out.append(fn)
    return out


# a parameter that says which prompt / template the call renders: _run(problem, prompt="code_prompt_reflect")
_PICK_PARAMS = re.compile(r"^(prompt|prompt_name|prompt_key|prompt_id|prompt_type|template|template_name|"
                          r"template_key|prompt_template)$")


def _self_calls(node: ast.AST) -> set[str]:
    """Names this function calls only on self / cls (partial(self._run) included)."""
    on_self, other = set(), set()
    args = getattr(node, "args", None)
    local = {a.arg for a in (args.args + args.kwonlyargs if args else [])}
    local |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for c in _own_calls(node):
        f = c.args[0] if _dotted(c.func).split(".")[-1] == "partial" and c.args else c.func
        if isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name) and f.value.id in ("self", "cls"):
                on_self.add(f.attr)
            elif not isinstance(f.value, ast.Name) or f.value.id in local:  # asyncio.run() is not an object's run
                other.add(f.attr)
        elif isinstance(f, ast.Name):
            other.add(f.id)
    return on_self - other


def _calls_and_picks(node: ast.AST) -> tuple[list[tuple[str, int]], dict[str, set[str]]]:
    """Every call's simple name - a callable handed to functools.partial counts as called - and, per callee, the
    prompt names passed to it as string literals."""
    calls: list[tuple[str, int]] = []
    picks: dict[str, set[str]] = {}
    for c in _own_calls(node):
        name = _dotted(c.func).replace("()", "").split(".")[-1]
        target, args = name, c
        if name == "partial" and c.args:  # functools.partial(self._run, prompt="...") is a call to come
            target = _dotted(c.args[0]).replace("()", "").split(".")[-1]
            calls.append((target, c.lineno))
        calls.append((name, c.lineno))
        for k in args.keywords:
            if k.arg and _PICK_PARAMS.match(k.arg):  # prompt="reflect", or prompt=choose_prompt()
                lit = k.value.value if isinstance(k.value, ast.Constant) and isinstance(k.value.value, str) else None
                picks.setdefault(target, set()).add(lit or f"<{k.arg}@{c.lineno}>")
    return calls, picks


def _picks_prompt(fn: _Fn, ups: list[_Fn]) -> bool:
    """Is fn a shared model call whose callers each choose the prompt it renders? Then each caller is a step of its
    own (the reflection step, the test-generation step), not one agent called from many places."""
    params = {a.arg for a in getattr(getattr(fn.node, "args", None), "args", []) + getattr(
        getattr(fn.node, "args", None), "kwonlyargs", [])}
    if not any(_PICK_PARAMS.match(p) for p in params):
        return False
    # named prompts decide it (prompt="reflect" here, prompt="review" there); a caller passing a variable only joins
    chosen = [frozenset(p for p in c.picks.get(fn.name, ()) if not p.startswith("<")) for c in ups]
    return sum(1 for x in chosen if x) >= 2 and len({x for x in chosen if x}) >= 2


def class_llm_methods(files: list[SourceFile]) -> dict[str, set[str]]:
    """class name -> its methods that reach a model, directly or through another method of the class
    (ask() -> self.query() -> openai...). Inherited methods are the caller's to resolve."""
    out: dict[str, set[str]] = {}
    for f in files:
        if f.path.suffix.lower() not in (".py", ".ipynb") or is_test_file(f.rel, f.text):
            continue
        try:
            tree = ast.parse(f.text)
        except (SyntaxError, ValueError):
            continue
        mods, names = _imports(tree)
        clients = _client_vars(tree, mods, names)
        chains = LLM_CHAINS + (LLM_CHAINS_IF_IMPORTED if _imports_llm(mods, names) else ())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            methods = {m.name: m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
            hits = {n for n, m in methods.items() if _llm_call(m, mods, names, clients, chains, in_class=True)
                    and n != "__init__"}
            grew = True
            while grew:
                grew = False
                for n, m in methods.items():
                    if n in hits or n == "__init__":
                        continue
                    if any(isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name)
                           and c.func.value.id == "self" and c.func.attr in hits for c in _own_calls(m)):
                        hits.add(n)
                        grew = True
            if hits:
                out.setdefault(cls.name, set()).update(hits)
    return out
