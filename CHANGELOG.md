# Changelog

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
