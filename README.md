# VisualAgents

**A live map of which agent in a multi-agent AI system is doing what — built from its chain of thought.**

Multi-agent apps (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, Swarm, Google ADK, smolagents, LlamaIndex, or your own
framework) usually show you one long stream of thoughts and tool calls. You can't tell which specialist is
working, where the request is in the workflow, or why it looped back.

VisualAgents fixes that from the terminal:

1. **Reads the codebase** (static analysis + an LLM pass) and works out how many agents there are, what each
   one does, which tools it owns and how work is handed between them — the **blueprint**.
2. **Runs your agent** (or reads its log / piped output) and maps every line of chain-of-thought onto the
   blueprint as it happens.
3. **Opens a local dashboard** — a live graph that lights up the agent that is working, animates each
   handoff, shows what it's thinking right now, which workflow step you're on, and the exact path taken.

```
 ┌ VisualAgents · research_crew ─────────────────────────────── ● Running 01:15 ┐
 │ ✓ Plan ─ ✓ Research ─ ③ Draft ─ ④ Review                                      │
 ├──────────────────────────────────────────────────────┬────────────────────────┤
 │                                                      │ WORKING NOW            │
 │  START ─▶ [Planner ✓] ─▶ [Researcher ✓] ─▶ [Writer ●]─▶ [Critic] ─▶ END       │
 │                                        ┌──────────┐ ◀─ revise ─┘  │ ● Writer  │
 │                                        │ Thought: │                │ Drafting  │
 │                                        │ drafting │                │ section 2 │
 │                                        └──────────┘                ├───────────┤
 │                                                      │ Live chain of thought  │
 ├──────────────────────────────────────────────────────┴────────────────────────┤
 │ PATH  Start › Planner 18s › Researcher 51s › Writer 9s                         │
 └───────────────────────────────────────────────────────────────────────────────┘
```

## Install

Requires Python 3.10+.

```bash
pipx install git+https://github.com/kaih06-hub/VisualAgents.git
```

or `uv tool install git+https://github.com/kaih06-hub/VisualAgents.git`, or `pip install` it into any environment.
Claude support is built in. For Gemini or OpenAI (incl. Ollama / LM Studio), install the extra:
`pipx install "visualagents[google] @ git+https://github.com/kaih06-hub/VisualAgents.git"` (`[openai]`, or `[all]`).

## Start here

```bash
visualagents
```

Three questions, then your agents' map pops up:

1. **Which project?** — a recent one, the current folder, any path (Tab completes), or a GitHub URL (cloned for you).
2. **Which LLM draws the map?** — Gemini, Claude, OpenAI, a local / OpenAI-compatible server (Ollama, LM Studio,
   vLLM), or offline. Then pick a model (best / cheaper / cheapest, or any id).
3. **API key** — pasted hidden, checked with a free model-list call, and (if you say so) saved to
   `~/.visualagents/keys.json`, readable only by you. Next time you just press Enter through the defaults.

VisualAgents then maps the agents, opens the dashboard with the blueprint popup, and offers to **start the project
now**, **hook it** so it appears whenever you run it yourself, or just show the map.

**Making it live.** The map moves only while the agent is actually working - VisualAgents watches, it doesn't do
the work. If nothing is running, the map shows a **▶ Start** button: it starts the project in your terminal, and
you then give the agent its task the way it expects (answer its questions in the terminal, or use its own web
page). If the start command has a `{task}` slot - e.g. `visualagents run . -- python main.py "{task}"` - the
dashboard shows a box to type the task in instead. After a run, **▶ Run again** starts the next one.

No API key? `visualagents demo` runs a simulated four-agent crew in 10 seconds.

## Use it on your own agents

Two ways, depending on how much you want to change your routine.

### 1. `visualagents run <folder>` — start it through VisualAgents

```bash
visualagents run ~/projects/some-agent
```

The first time, VisualAgents works out how the project starts — commands in the README, `start.sh`-style
scripts, FastAPI / Gradio / Streamlit / Chainlit apps, `__main__` scripts, `pyproject` scripts, `package.json`,
the Makefile — asks you to pick, and remembers the answer in `.visualagents/config.json`. It runs `python`,
`uvicorn`, `streamlit`… from the project's own virtualenv (or `uv run` / `poetry run`), maps the agents if there is
no blueprint yet, and opens the dashboard. Start scripts that push the app into the background or redirect its
output to a log file still work: events reach the dashboard over HTTP, and log files the app writes are followed.

```bash
visualagents run ~/projects/some-agent --pick                 # choose the start command again
visualagents run ~/projects/some-agent -- python main.py "q"  # or give an exact command
```

### 2. `visualagents hook <folder>` — hook once, then run it however you like

```bash
visualagents hook ~/projects/some-agent     # once
cd ~/projects/some-agent && ./start.sh      # ...or your IDE, a notebook, `uv run`, cron - anything
```

`hook` installs a one-line startup file (`visualagents_hook.pth`) into the folder's virtualenv(s) and starts a
background dashboard at http://127.0.0.1:7400. From then on every Python process from that environment shows
up live as soon as it calls an LLM — no wrapper command. Processes that never do (pip, pytest, your editor's
tools) stay invisible. One dashboard follows all your hooked projects, with a switcher at the top.

```bash
visualagents status                 # dashboard + hooked projects
visualagents open ~/projects/x      # open a project's dashboard
visualagents unhook ~/projects/x    # remove the hook
visualagents up / down              # start / stop the background dashboard
visualagents hook . --python ~/miniconda3/envs/agents/bin/python   # conda or any other interpreter
```

The hook needs the project's virtualenv to exist, so run the project once (or create the venv) first.

### Other inputs

```bash
python main.py 2>&1 | visualagents pipe ./my-agent-app      # output you already produce
visualagents watch ./my-agent-app --log logs/agent.log        # follow a log file (tail -F)
visualagents replay ./my-agent-app/.visualagents/runs/20261019-091542.jsonl --speed 4
visualagents show ./my-agent-app                              # just the blueprint
visualagents blueprint ./my-agent-app                         # print the blueprint in the terminal
```

Every run is recorded under `.visualagents/runs/` and can be replayed later — even against an improved
blueprint (`replay … --project ./my-agent-app`).

## How it attributes work to agents

VisualAgents combines three sources, from most to least precise:

| Source | When | What you get |
|---|---|---|
| **Auto-instrumentation** | `visualagents run` or `hook` with a Python app using LangChain / LangGraph, the OpenAI Agents SDK, or the **OpenAI SDK directly** (`chat.completions.create` — sync, async, streaming, any OpenAI-compatible endpoint) | Exact agent boundaries, LLM replies as thoughts, every tool call and tool result — no code changes. Raw SDK calls are credited to the class or function that made them (e.g. `Programmer`, `Inspector`). Injected via `PYTHONPATH`; disable with `--no-instrument`. |
| **Console patterns** | Always | Recognises CrewAI (`# Agent: …`, rich panels), AutoGen (`coder (to critic):`, `---------- coder ----------`), LangGraph stream output (`Update from node …`, `{'node': …}`), ReAct (`Thought:` / `Action:` / `Observation:` / `Final Answer:`), handoff phrases (`Handing off to Writer`, `transfer_to_billing`), `[Agent] …` prefixes, and tool names owned by a single agent. |
| **LLM narration** (`--smart`) | Optional | Every few seconds an LLM reads the latest output and says which agent is working and summarises what it's doing in one line. |

Agents the blueprint missed but that show up at runtime are added to the map (dashed), and handoffs that
weren't in the blueprint are drawn as dotted "observed" edges.

### Instrument anything yourself

From any language, print a line to stdout/stderr:

```
@@visualagents {"type": "agent_start", "agent": "researcher", "text": "find sources"}
```

From Python:

```python
from visualagents import agent, emit, handoff, track

with agent("researcher", task="find sources"):
    emit("thought", "researcher", "Start with peer-reviewed numbers")
    emit("tool_start", "researcher", "solid-state battery Wh/kg", tool="web_search")
handoff("researcher", "writer", "findings ready")

@track("critic")          # the whole function is the critic's turn
def review(draft): ...
```

Event types: `agent_start`, `agent_end`, `thought`, `tool_start`, `tool_end`, `handoff`, `output`, `log`, `error`.
Under `visualagents run` the stdlib-only runtime is importable even if VisualAgents isn't installed in your
app's environment: `from visualagents_runtime import agent, emit`. Marker lines are hidden from your terminal.
For LangChain without `run`, pass `visualagents.langchain_handler()` in `callbacks=[…]`; for the OpenAI Agents
SDK, `agents.add_trace_processor(visualagents.openai_agents_processor())`.

### Example: LAMBDA (a data-science agent)

[LAMBDA](https://github.com/AMA-CMFAI/LAMBDA) is a FastAPI backend plus a React UI that calls the OpenAI SDK
directly, and its `start.sh` sends both to the background:

```bash
git clone https://github.com/AMA-CMFAI/LAMBDA.git ~/LAMBDA
cp ~/LAMBDA/backend/.env.example ~/LAMBDA/backend/.env    # add OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_LIST
visualagents run ~/LAMBDA                                  # pick ./start.sh the first time
```

Ask LAMBDA something at http://localhost:3000 and watch it at http://127.0.0.1:7420. Or, after LAMBDA has run
once (so `backend/venv` exists), `visualagents hook ~/LAMBDA` and keep using `./start.sh` as usual. The current
LAMBDA is one agent with tools (Python, shell, files, a to-do list); the paper's Programmer ⇄ Inspector design is
on the `legacy-open-source` branch, where the map shows the two agents handing work back and forth.

## The blueprint

`visualagents scan` writes `.visualagents/blueprint.json` in the project (commit it or hand-edit it):

- **Static analysis** (offline, instant) understands LangGraph `add_node` / `add_edge` /
  `add_conditional_edges` / `Command(goto=…)`, CrewAI `Agent`/`Task`/`Crew` and `config/agents.yaml` +
  `tasks.yaml`, AutoGen agents and group chats, OpenAI Agents SDK `handoffs` and `as_tool`, Swarm
  `transfer_to_*` functions, Google ADK `sub_agents`, smolagents `managed_agents`, LlamaIndex `AgentWorkflow`,
  LangGraph.js `addNode`/`addEdge`, and `new Agent({ name })` in JS/TS.
- **LLM analysis** (when a key is available) reads the most agent-relevant files and returns each agent's role,
  description, tools, source location, log aliases and keywords, the handoff graph, entry/exit agents, and the
  general step-by-step workflow, as schema-checked JSON.

### Choosing the model that draws the map

| Provider | Key | Default scan model | Default `--smart` model |
|---|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` or `ant auth login` | `claude-opus-5-5` | `claude-opus-5-5` |
| `openai` | `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for Ollama etc.) | `gpt-5` | `gpt-5` |
| `google` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` (or Vertex AI) | `gemini-pro-latest` | `gemini-flash-lite-latest` |

```bash
visualagents models                               # which providers have keys, and their defaults
visualagents models --provider google             # every model your account can use
visualagents scan . --provider google --model gemini-flash-latest
visualagents run . --smart --narrate-model gemini-flash-lite-latest -- python app.py
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama visualagents scan . --provider openai --model qwen2.5-coder
```

To avoid passing flags every time, set `VISUALAGENTS_PROVIDER`, `VISUALAGENTS_MODEL` (scan) and
`VISUALAGENTS_NARRATE_MODEL` (`--smart`) in your shell or the project's `.env`, next to the key:

```bash
# .env (keep it out of git)
GEMINI_API_KEY=...
VISUALAGENTS_PROVIDER=google
VISUALAGENTS_MODEL=gemini-flash-latest
```

Without `--provider`, VisualAgents uses `VISUALAGENTS_PROVIDER`, else the first of Anthropic → OpenAI → Google
that has a key, else offline static analysis. The dashboard's blueprint popup shows which model drew the map.
Before anything is sent to an LLM, `.env`/key/credential files are skipped and strings that look like API keys
are redacted.

## The dashboard

- **Graph** — agents laid out by workflow order (left→right, or top→bottom on narrow screens); the active agent
  glows with a thought bubble showing what it's doing; handoffs animate along the edges with traversal counts.
  **Drag any block** to rearrange the map — the arrows follow, and positions are remembered per project
  (the layout button in the corner restores the automatic layout).
- **Steps** — the blueprint's phases, checked off as the run moves through them.
- **Live chain of thought** — every thought, tool call, result and answer, colour-coded by agent and filterable.
  Hover any line and the agent that produced it lights up and bounces on the map (handoffs light up the arrow).
- **Time ruler** — every agent's turn laid out on a timeline with the workflow steps above it. Drag the playhead
  back to any moment and the whole dashboard rewinds: who was working, which step, what it was thinking, and the
  chain of thought up to that second. Press ▶ (or Space) to replay at 1–16×, and **Live** (or `L`) to return.
- **Blueprint popup** (`B`) — summary, steps, agents with roles/tools/source, and every handoff.
- Click an agent for its details and its own chain of thought. `F` fits the graph; light/dark/auto theme.

The server binds to `127.0.0.1` only. Ingest requests need a per-run token.

## CLI reference

```
visualagents          guided start (same as `visualagents setup`)
visualagents demo     [--speed N]
visualagents run      [FOLDER] [--pick] [--smart] [--cwd DIR] [--no-instrument] [--no-follow-logs] [-- COMMAND…]
visualagents hook     [FOLDER] [--python PY]…  /  unhook [FOLDER]  /  open [FOLDER]
visualagents up       [--port 7400] [--foreground]  /  down  /  status
visualagents models   [PROJECT] [--provider anthropic|openai|google]
visualagents scan     [PROJECT] [--static | --llm] [--provider auto|anthropic|openai|google] [--model M] [--max-chars N] [--json] [-o FILE]
visualagents pipe     [PROJECT] [--quiet]
visualagents watch    [PROJECT] --log FILE [--from-start] [--echo]
visualagents replay   RECORDING [--project DIR] [--speed N] [--instant]
visualagents show     [PROJECT]
visualagents blueprint [PROJECT] [--json]

dashboard options: --port 7420 --host 127.0.0.1 --no-browser --blueprint FILE --exit --rescan
```

`vagents` is a short alias for `visualagents`.

## Examples

- [`examples/langgraph_supervisor`](examples/langgraph_supervisor) — a LangGraph supervisor team that runs
  offline (`pip install langgraph`): `visualagents run examples/langgraph_supervisor -- python app.py`
- [`examples/crewai_project`](examples/crewai_project) — the standard CrewAI template (YAML agents & tasks).
- [`examples/manual_sdk`](examples/manual_sdk) — a hand-rolled multi-agent loop instrumented with `emit`.

## Development

```bash
git clone https://github.com/kaih06-hub/VisualAgents.git && cd VisualAgents
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,all]"
pytest
```

The dashboard is plain HTML/CSS/JS in `visualagents/web/` (no build step). New console formats go in
`visualagents/patterns.py`; new static detectors in `visualagents/static_scan.py`.

## License

MIT
