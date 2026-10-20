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

### Fixed
- Start-command detection: a project's own console scripts in its README count as start commands, an interactive
  chat command is preferred over a server or a web front end in a subfolder, and they run from the project's
  virtualenv. A missing command now says so instead of exiting silently.
- Apps built around a shared LLM client class (e.g. LAMBDA's `LLMService`) map to the agents that use it: the
  agent is the code that writes the system prompt and runs the loop, named after its prompt ("You are LAMBDA, ..."),
  with the tools it hands the model. The client wrapper and code kernels are no longer shown as agents, and calls
  made from another asyncio task still light up the right agent.

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
