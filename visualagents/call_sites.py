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

from .collector import SourceFile, agent_class_names

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
LLM_CHAINS_IF_IMPORTED = ("messages.create", "messages.stream", "responses.stream", "chat.complete", "chat.stream")
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
_PROMPT_PARAMS = re.compile(r"^(\w*_)?(messages|msgs|prompts?|system(_prompt|_message)?|contents|"
                            r"(chat_)?history|conversation)$", re.I)
# "You are LAMBDA, a data analysis agent ..." / "You are a professional data scientist writing reports."
_PERSONA = re.compile(r"\bYou are\s+(?:(?P<name>[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*){0,3})\s*[,(:\-—–]\s*)?"
                      r"(?:an?|the)\s+(?P<role>[^\n.;!?]{3,120})")
_MODEL_NAMES = {"chatgpt", "gpt", "claude", "gemini", "llama", "qwen", "deepseek", "mistral", "grok", "ai"}
_SKIP_PATH = re.compile(r"(^|/)(tests?|testing|examples?|docs?|scripts?/dev)(/|$)|(^|/)test_[^/]*$|_test\.py$")
MAX_AGENTS = 15


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

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.rel, self.cls or "", self.name)


def find(files: list[SourceFile]) -> dict[str, Any]:
    """Agents and edges for LLM call sites: {"agents": [...], "edges": [...]} (empty when there are none)."""
    fns: list[_Fn] = []
    direct: list[_Fn] = []
    for f in files:
        if f.path.suffix.lower() not in (".py", ".ipynb") or _SKIP_PATH.search(f.rel.replace("\\", "/")):
            continue
        try:
            tree = ast.parse(f.text)
        except (SyntaxError, ValueError):
            continue
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
    if not direct:
        return {"agents": [], "edges": []}

    callers: dict[str, list[_Fn]] = {}
    for fn in fns:
        for name, _ in fn.calls:
            callers.setdefault(name, []).append(fn)

    def callers_of(fn: _Fn) -> list[_Fn]:
        if fn.name == "<module>" or fn.name.startswith("__"):  # self.__init__() in reset() is not a model call
            return []
        seen, out = set(), []
        for c in callers.get(fn.name, []):  # `if __name__ == "__main__": main()` does not make main plumbing
            if c.key != fn.key and c.key not in seen and c.name != "<module>":
                seen.add(c.key)
                out.append(c)
        return out

    sites: dict[tuple[str, str, str], _Fn] = {}
    services: set[tuple[str, str]] = set()  # LLM wrapper classes used from elsewhere (LLMService, ModelClient)

    def climb(fn: _Fn, via: str, depth: int, trail: frozenset, tools: tuple[str, ...], wrapped: tuple[str, ...]):
        ups = [c for c in callers_of(fn) if c.key not in trail]
        tools = tools + tuple(t for t in fn.tools if t not in tools)
        # a helper that only forwards someone else's prompt is plumbing: its callers are the agents
        if ups and not fn.own_prompt and depth < 3 and "agent" not in fn.name.lower():
            if fn.cls and any((c.rel, c.cls) != (fn.rel, fn.cls) for c in ups):
                services.add((fn.rel, fn.cls))
            # the runtime sees the wrapper (LLMService.stream_chat) when its caller runs it in another task
            wrapped = wrapped + ((f"{fn.cls}.{fn.name}",) if fn.cls else ())
            for c in ups:
                climb(c, fn.name, depth + 1, trail | {fn.key}, tools, wrapped)
            return
        if fn.key not in sites:
            fn.via = fn.via if depth == 0 else via
            sites[fn.key] = fn
        site = sites[fn.key]
        site.tools += [t for t in tools if t not in site.tools]
        site.wrapped += [w for w in wrapped if w not in site.wrapped]

    for fn in direct:
        climb(fn, fn.via, 0, frozenset(), (), ())

    # a class named or built like an agent (SolverAgent(BaseAgent)) is one agent, whichever of its methods reach the
    # model and however much other code calls them; only a shared client class (LLMService) splits into its methods
    agent_classes = agent_class_names(files)
    agents: dict[str, dict[str, Any]] = {}  # agent id -> agent
    owner: dict[str, str] = {}  # function name -> agent id
    for fn in sorted(sites.values(), key=lambda s: (s.rel, s.line)):
        if fn.cls and ((fn.rel, fn.cls) not in services or fn.cls in agent_classes):
            aid, name, aliases = f"{fn.rel}:{fn.cls}", fn.cls, [fn.cls, fn.name]
        elif fn.name == "<module>":
            stem = re.sub(r"\.(py|ipynb)$", "", fn.rel.rsplit("/", 1)[-1])
            aid, name, aliases = f"{fn.rel}:<module>", stem, [stem]
        else:  # a function, or one prompt-writing method of a service class (LLMService.generate_title)
            aid, name, aliases = f"{fn.rel}:{fn.cls or ''}.{fn.name}", fn.name, [fn.name]
            if fn.cls:
                aliases.append(f"{fn.cls}.{fn.name}")
        aliases += [w for w in fn.wrapped if w not in aliases]
        persona, role = _persona(fn.node)
        if persona:  # "You are LAMBDA, a data analysis agent": the agent's own name beats its function's
            aid, name, aliases = f"{fn.rel}:{persona}", persona, [persona, *aliases]
        agent = agents.get(aid)
        if agent is None:
            if len(agents) >= MAX_AGENTS:
                continue
            agent = agents[aid] = {"name": name, "role": role or f"calls the LLM ({fn.via})",
                                   "description": fn.cls_doc or fn.doc,
                                   "tools": [], "file": fn.rel, "line": fn.line, "aliases": []}
        agent["aliases"] += [a for a in aliases if a not in agent["aliases"]]
        agent["tools"] += [t for t in fn.tools if t not in agent["tools"]]
        if fn.name != "<module>":
            owner.setdefault(fn.name, aid)

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()

    def link(a: str, b: str, label: str) -> None:
        if a != b and (a, b) not in seen_edges and len(edges) < 20:
            seen_edges.add((a, b))
            edges.append({"from": agents[a]["name"], "to": agents[b]["name"], "label": label, "conditional": False})

    for fn in fns:
        mine = owner.get(fn.name) if fn.key in sites else None
        called = [owner[n] for n, _ in sorted(fn.calls, key=lambda c: c[1]) if n in owner and owner[n] in agents]
        if mine:  # an agent that calls another agent hands work to it
            for other in dict.fromkeys(called):
                link(mine, other, "calls")
        elif len(set(called)) >= 2:  # a pipeline / orchestrator running agents one after another
            order = list(dict.fromkeys(called))
            for a, b in zip(order, order[1:]):
                link(a, b, "then")
    return {"agents": list(agents.values()), "edges": edges}


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
    """ids of nodes inside the docstring, raise statements and log/print calls: their text is not a prompt."""
    body = getattr(node, "body", [])
    doc = body[0] if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None),
                                                                            ast.Constant) else None
    skip = {id(doc.value)} if doc is not None else set()
    for n in ast.walk(node):
        if isinstance(n, ast.Raise) or (isinstance(n, ast.Call) and re.match(
                r"(print|warn|.*\.(debug|info|warning|error|exception|critical|warn|log))$", _dotted(n.func))):
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
        fn.calls = [(_dotted(c.func).replace("()", "").split(".")[-1], c.lineno) for c in _own_calls(node)]
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
        fn.calls = [(_dotted(c.func).replace("()", "").split(".")[-1], c.lineno) for c in _own_calls(top)]
        out.append(fn)
    return out
