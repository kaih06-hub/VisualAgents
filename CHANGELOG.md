# Changelog

## Unreleased

### Added
- Anything that calls an LLM is an agent system, with or without a framework, one agent or many:
  - the map: every function or class that calls a model is an agent; a shared helper that only forwards a prompt is
    not - the functions that call it with their own prompts are. Recognised through OpenAI, Anthropic, Gemini,
    LiteLLM, Ollama, mlx-lm, llama.cpp, transformers, vLLM, Groq, Hugging Face and LangChain chat models.
  - live: those libraries are hooked like the OpenAI SDK already was - each call shows the prompt, the reply,
    `<think>` reasoning and tool calls, credited to the code that made it (looking past shared helpers).
  - an app with no mapped agents shows each caller as its own agent instead of nothing.
  - hook mode now wakes for CrewAI, AutoGen and other frameworks that print their own agents.

- Static analysis follows the run order of agent objects, found missing in a second held-out check of ten more
  projects against their README graphs (no rule names a project):
  - a manager that runs SDK agents through its own helpers (`await self._plan()`, then a search per item, then the
    writer) links them in the order they run; `if` / `match` branches are alternatives, a `try` handler is the
    failure path, and `return` / `raise` / `break` / `continue` end a path; arguments run before the call they feed;
  - an agent given to another as a tool through a variable (`t = risk.as_tool(...)`, `writer.clone(tools=[t])`);
  - AG2 chats with agents kept in a dict or built by factories (`agents["user"].initiate_chat(agents["planner"])`,
    `return Planner(cfg).agent`) and nested chats registered on an agent;
  - named instances of a project subclass of a framework agent (`MyProxy(name="user")`, `MyProxy(name="executor")`);
  - a LangGraph node whose function invokes a compiled graph of its own file (`await researcher.ainvoke(...)`);
  - steps that each pick the prompt of one shared model call (`partial(self._ask, prompt="reflect")`) are agents,
    and `self._ask()` in a class with no base is that class's own method;
  - a step written once per mode (`one_chunk_reflect` / `multichunk_reflect`) is one step, unless one feeds the other.
- Six gaps found by a third check of ten more projects against their README graphs (no rule names a project):
  - a model called over raw HTTP (`requests.post` to a model endpoint, or reading `["choices"]` from the reply) is a
    model call; code that only builds or calls an embedding model is not an agent;
  - an agent class that runs three or more of its own differently prompted methods in sequence is drawn as those
    steps, with the order its own methods run them in;
  - graphs built from a node table (`add_node(key, self.node_functions[key])`) or from edge dicts
    (`add_edge(e["from"], e["to"])`); a node is never named after the variable holding its name, and
    `refiner_agent` / `RefinerAgent` beside the node `refiner` is that node;
  - a group chat whose manager picks the speaker is drawn in the listed order, not as a ring, and without the person
    in it (round-robin chats still go round);
  - routers that return computed node names (`f"speech_{p}"`, `[Send("search", s) ...]`) and path maps written as
    dict comprehensions; an agent handed to one node's factory (`make_node("search", web_scraper, ...)`) is that node;
  - a folder with its own README that the rest of the project never imports is a separate program and stays off the
    map; a loop that runs several agents (`while`, `for _ in range(n)`) comes back to its first one; front ends
    (a CLI, a server, a bot) calling one core make that core the agent; module-level prompt templates count.
- Live tracking works when VisualAgents is installed under a folder named `agents/` (its own frames looked like the
  OpenAI Agents SDK, so every model call was skipped).
- Left off the map: an older implementation kept in `legacy/` / `deprecated/` / `old/` beside the current one; usage
  examples under `examples/` when the project wires agents of its own; the project's own `create_agent()` that
  compiles its graph; pyvis graphs; and a LangChain text splitter no longer makes a project "langchain".
- `benchmarks/readme_graphs`: a second set of ten projects (`ground_truth_set2.py`, `run_set2.sh`, results in
  `results-set2-2026-10-21.json`) and a `node_recall_code` score that leaves out components the README drawing has
  and today's code no longer does.

- Static analysis maps more shapes of agent system, found missing in a held-out check of ten projects against the
  graphs in their READMEs (no rule names a project):
  - several named instances of one model-calling class (`Player(name="Proposer")`, `[Player(name=n) for n in SIDES]`)
    are the agents, and the order the code asks them in - with a loop's next round - is how work passes;
  - one function handing a model helper a different role each call (`ask(system_role=triager)`, then `reviewer`,
    then `chair`) is one agent per role, in call order;
  - roles that publish actions others watch (`self.set_actions([WriteSpec])` / `self._watch([WriteSpec])`) hand work
    to each other; a role watching what no role produces is where a run starts;
  - a LangGraph node that runs a compiled subgraph (`add_node("write", inner.compile())`) leads into the subgraph's
    entry and back from its exits, and the inner graph's ends are not the run's;
  - a LangGraph node whose action wraps an agent by name (`wrap(agents["coder_agent"])`, with the agent class
    declaring `agent_name="coder_agent"`) is that agent: one block, not two;
  - shell scripts that run the project's scripts in order (`python plan.py`, `python solver_[a/b].py`) are a
    pipeline: steps link in order, a step that calls no model is still a step, and a step calling into another
    script's code hands work back to it. Reported to the LLM as the fact `script_pipelines`.
- New fact for the LLM: `agent_classes_never_used` - agent classes nothing in the project creates or names.

### Changed
- The guided start offers `gemini-flash-latest` instead of the pinned `gemini-2.5-flash`, and says that Gemini Pro
  needs billing on the key (a free key has no Pro quota).

### Fixed
- Model calls through the pre-1.0 OpenAI SDK (`openai.ChatCompletion.create`, `openai.Completion.create`) are found.
- Files merely named like tests (`test_designer.py`, `test_executor.py`: agents that write or run tests) are part of
  the project; files in a tests folder, or holding test-runner code, are left out of the map entirely - stub agents
  and graphs in a test suite no longer show up as agents.
- A helper is credited to callers in other files only when they import it: two scripts that each define their own
  `api_call()` are two agents, named after their scripts, not one. A script whose top-level code writes the prompts
  is the agent, and an agent named only `main` / `run_llm` / `api_call` is named after its file.
- One script per dataset or backend defining the same model-calling functions (`coder_math.py`, `coder_code.py`) is
  one agent; a model helper nothing calls, next to one that is used, is not an agent; a function whose name merely
  contains "agent" (`generate_response_multiagent`) is not treated as one.
- Cloning from the guided start: answering `y` / `yes` at "Clone it into [~/agents/…]" clones into the
  suggested folder instead of one called `y` (`n` / `no` cancels), and an existing folder is reused only when it
  is a clone of the same repository - not a clone of something else that happens to have that name.
- Start-command detection reads a project's console scripts on Python 3.10 too (no `tomllib` there).
- Start-command detection: a project's own console scripts in its README count as start commands, an interactive
  chat command is preferred over a server or a web front end in a subfolder, and they run from the project's
  virtualenv. A missing command now says so instead of exiting silently.
- Apps built around a shared LLM client class (e.g. LAMBDA's `LLMService`) map to the agents that use it: the
  agent is the code that writes the system prompt and runs the loop, named after its prompt ("You are LAMBDA, ..."),
  with the tools it hands the model. The client wrapper and code kernels are no longer shown as agents, and calls
  made from another asyncio task still light up the right agent.
- The LLM scan is taught the common shapes of agent systems rather than particular projects: pipelines, routers
  and supervisors, turn-taking pairs and group chats, critic loops, planners with executors, fan-out, nested teams
  and several independent systems in one repository - and what to leave off the map and name in the summary
  instead: ports to another framework, legacy copies, role variants a setting picks between, opt-in roles, and
  apps built with the project rather than part of it.
- Static analysis hands the LLM facts to weigh instead of deciding for it: the same agent names defined on two
  frameworks (with each one's folder), default roles / agents lists in configuration, and which class creates each
  DSPy program. Subclasses of an agent base class and DSPy programs (`dspy.Module`s with their own predictors) are
  found as agents; a program an agent holds is its tool, and one that calls programs in turn gives their handoffs.
- Static analysis finds more of what a project really defines: subclasses of the project's own agent base
  (`BaseAgent`, `LLMAgent`, `Role`), OpenAI Agents SDK agents built as `Agent[Context](...)` with handoffs wired
  after the fact (`triage.handoffs = [...]`, `.append(...)`), and code that picks one agent by a setting
  (`if config.type == "retry": ...`, a registry dict), reported to the LLM as a fact. Model calls inside an agent
  class count for that class instead of each method. `add_node` / `add_edge` are LangGraph only where LangGraph is
  imported - not networkx or a project's own graph of actions.
- The framework badge comes from what the files defining the agents import, not from a name mentioned anywhere: a
  LangGraph team is LangGraph even when many other files import LangChain for model calls, another framework's
  `Task` class is not CrewAI, and a project's own `agents.py` is not the OpenAI Agents SDK (one inside a package
  does not hide it).

## 0.1.1 — 2026-10-19

### Fixed
- The LLM scan no longer reads files git ignores (local config with credentials, data, your own documents), skips
  build-tool configs, and redacts more secrets: JWTs, `KEY = "..."`, `"api_key": "..."` and unquoted
  `password: ...` lines in config files.

## 0.1.0 — 2026-10-19

First public release (alpha).

### Map your agents
- `visualagents scan`: static analysis of LangGraph, LangChain, CrewAI (code + YAML), AutoGen 0.2/0.4, the OpenAI
  Agents SDK, Swarm, Google ADK, smolagents, LlamaIndex, raw OpenAI-SDK apps and JS/TS LangGraph, plus an optional
  LLM pass (Claude, OpenAI / OpenAI-compatible, Gemini) that writes `.visualagents/blueprint.json`.
- `visualagents models` to see providers, keys and models; per-command or saved provider/model choice.

### Watch them live
- Guided start: bare `visualagents` asks for the project (path or GitHub URL), the LLM and model, and an API key,
  then opens the map and offers to start or hook the project.
- `visualagents run <folder>` finds how a project starts, remembers it, and uses the project's virtualenv.
- `visualagents hook <folder>` + a background multi-project dashboard (`up`, `down`, `status`, `open`, `unhook`).
- Auto-instrumentation for LangChain/LangGraph, the OpenAI Agents SDK and direct OpenAI-SDK calls; console
  patterns for CrewAI, AutoGen, ReAct and more; optional LLM narration (`--smart`); a stdlib-only SDK (`emit`).
- Works when start scripts background the app or redirect its logs; `pipe`, `watch` and `replay` inputs.

### Dashboard
- Live graph with thought bubbles, animated handoffs, step tracker and a colour-coded chain-of-thought feed.
- Time ruler: rewind to any second and replay at 1–16×; every run recorded under `.visualagents/runs/`.
- Draggable blocks, hover-to-highlight, agent drawer, blueprint popup, light/dark, phone layout,
  ▶ Start / Run again buttons, shareable `?t=` links.
