"""LLM access for VisualAgents: codebase -> blueprint, and (optionally) live narration.

Providers (pick with --provider / --model, or VISUALAGENTS_PROVIDER / VISUALAGENTS_MODEL):
- anthropic: Claude via the official `anthropic` SDK (installed by default)
- openai:    OpenAI or any OpenAI-compatible server (Ollama, LM Studio, vLLM via OPENAI_BASE_URL);
             `pip install "visualagents[openai]"`
- google:    Gemini via the official `google-genai` SDK (GEMINI_API_KEY, or Vertex AI);
             `pip install "visualagents[google]"`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROVIDERS = ("anthropic", "openai", "google")
# Default model for reading the codebase (quality matters) and for --smart narration (runs every few seconds).
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-5", "google": "gemini-pro-latest"}
NARRATE_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-5", "google": "gemini-flash-lite-latest"}
PROVIDER_LABELS = {"anthropic": "Anthropic Claude", "openai": "OpenAI / compatible", "google": "Google Gemini"}
KEY_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"),
}
_DOTENV_VARS = {
    *(v for vs in KEY_VARS.values() for v in vs), "ANTHROPIC_BASE_URL", "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION", "VISUALAGENTS_PROVIDER", "VISUALAGENTS_MODEL", "VISUALAGENTS_NARRATE_MODEL",
}
# Models that accept the server-side refusal fallback (beta server-side-fallback-2026-07-01).
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")
_NO_EFFORT = ("claude-3", "claude-haiku-4", "claude-sonnet-4-5", "claude-sonnet-4-0", "claude-opus-4-0", "claude-opus-4-1")


class LLMError(RuntimeError):
    pass


def load_dotenv_keys(project: Path) -> None:
    """Pick up provider keys and model choices from the project's .env if they aren't already set."""
    for candidate in (Path(project) / ".env", Path.cwd() / ".env"):
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k in _DOTENV_VARS and v and not os.environ.get(k):
                os.environ[k] = v
    from . import home

    home.apply_saved()  # keys and choices saved by the setup wizard come last


def has_credentials(provider: str) -> bool:
    if any(os.environ.get(v) for v in KEY_VARS.get(provider, ())):
        return True
    # `ant auth login` stores a profile the Anthropic SDK picks up without any env var
    return provider == "anthropic" and bool(
        os.environ.get("ANTHROPIC_PROFILE") or (Path.home() / ".config" / "anthropic").is_dir()
    )


def detect_provider(preferred: str = "auto") -> str | None:
    """--provider wins, then VISUALAGENTS_PROVIDER, then the first provider with credentials."""
    for choice in (preferred, os.environ.get("VISUALAGENTS_PROVIDER", "")):
        if choice and choice != "auto":
            if choice not in PROVIDERS:
                raise LLMError(f"unknown provider {choice!r} - use one of: {', '.join(PROVIDERS)}")
            return choice
    for provider in ("anthropic", "openai", "google"):
        if any(os.environ.get(v) for v in KEY_VARS[provider]):
            return provider
    return "anthropic" if has_credentials("anthropic") else None


def default_model(provider: str, purpose: str = "scan") -> str:
    """--model is handled by callers; this is the env override or the provider's default."""
    if purpose == "narrate":
        return os.environ.get("VISUALAGENTS_NARRATE_MODEL") or NARRATE_MODELS.get(provider, "")
    return os.environ.get("VISUALAGENTS_MODEL") or DEFAULT_MODELS.get(provider, "")


def complete_json(
    *,
    provider: str,
    model: str | None,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    effort: str = "high",
    max_tokens: int = 64000,
) -> dict[str, Any]:
    model = model or default_model(provider)
    if provider == "anthropic":
        return _anthropic_json(model, system, prompt, schema, effort, max_tokens)
    if provider == "openai":
        return _openai_json(model, system, prompt, schema)
    if provider == "google":
        return _google_json(model, system, prompt, schema, max_tokens)
    raise LLMError(f"unknown provider {provider!r} - use one of: {', '.join(PROVIDERS)}")


def _anthropic_json(model: str, system: str, prompt: str, schema: dict[str, Any], effort: str, max_tokens: int) -> dict[str, Any]:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - anthropic is a hard dependency
        raise LLMError("the 'anthropic' package is missing: pip install anthropic") from exc

    client = anthropic.Anthropic()
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
    if not model.startswith(_NO_EFFORT):
        output_config["effort"] = effort
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": output_config,
    }
    try:
        if model.startswith(_FALLBACK_MODELS):
            with client.beta.messages.stream(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            ) as stream:
                message = stream.get_final_message()
        else:
            with client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise LLMError("Anthropic rejected the credentials - set ANTHROPIC_API_KEY") from exc
    except anthropic.NotFoundError as exc:
        raise LLMError(f"model {model!r} not found - pass --model") from exc
    except anthropic.RateLimitError as exc:
        raise LLMError("Anthropic rate limit hit - try again shortly") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("could not reach the Anthropic API - check your network") from exc

    if message.stop_reason == "refusal":
        raise LLMError("the model declined this request")
    if message.stop_reason == "max_tokens":
        raise LLMError("the model ran out of output tokens - try a smaller --max-chars")
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except ValueError as exc:
        raise LLMError("the model returned invalid JSON") from exc


def _openai_json(model: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        import openai
    except ImportError as exc:
        raise LLMError('OpenAI support needs: pip install "visualagents[openai]"') from exc

    client = openai.OpenAI()
    sys_prompt = (
        f"{system}\n\nRespond with a single JSON object that matches this JSON schema exactly:\n"
        f"{json.dumps(schema)}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    except openai.AuthenticationError as exc:
        raise LLMError("OpenAI rejected the credentials - set OPENAI_API_KEY") from exc
    except openai.NotFoundError as exc:
        raise LLMError(f"model {model!r} not found - pass --model") from exc
    except openai.RateLimitError as exc:
        raise LLMError("OpenAI rate limit hit - try again shortly") from exc
    except openai.APIStatusError as exc:
        raise LLMError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
    except openai.APIConnectionError as exc:
        raise LLMError("could not reach the OpenAI-compatible endpoint") from exc

    choice = resp.choices[0]
    if getattr(choice.message, "refusal", None):
        raise LLMError("the model declined this request")
    try:
        return json.loads(choice.message.content or "")
    except ValueError as exc:
        raise LLMError("the model returned invalid JSON") from exc


def _google_json(model: str, system: str, prompt: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError as exc:
        raise LLMError('Gemini support needs: pip install "visualagents[google]"') from exc

    # Reads GEMINI_API_KEY / GOOGLE_API_KEY, or Vertex AI via GOOGLE_GENAI_USE_VERTEXAI + project/location.
    # Retry rate limits and "high demand" 5xx responses with backoff, like the Anthropic/OpenAI SDKs do.
    client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(
        attempts=4, initial_delay=2.0, max_delay=20.0, http_status_codes=[408, 429, 500, 502, 503, 504],
    )))
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=schema,
        max_output_tokens=min(max_tokens, 65536),
    )
    try:
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
    except errors.ClientError as exc:
        if exc.code in (401, 403):
            raise LLMError("Google rejected the credentials - set GEMINI_API_KEY") from exc
        if exc.code == 404:
            raise LLMError(f"model {model!r} not found - see `visualagents models --provider google`") from exc
        if exc.code == 429:
            raise LLMError("Gemini rate limit or quota hit - try again shortly") from exc
        raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
    except errors.ServerError as exc:
        raise LLMError(f"Gemini is unavailable right now ({exc.code}: {exc.message}) - retry, or pick another "
                       f"model with --model (see `visualagents models --provider google`)") from exc
    except errors.APIError as exc:
        raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
    except OSError as exc:
        raise LLMError("could not reach the Gemini API - check your network") from exc

    feedback = getattr(resp, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise LLMError("the model declined this request")
    candidate = resp.candidates[0] if resp.candidates else None
    finish = str(getattr(candidate, "finish_reason", "") or "")
    if "MAX_TOKENS" in finish:
        raise LLMError("the model ran out of output tokens - try a smaller --max-chars")
    if any(r in finish for r in ("SAFETY", "PROHIBITED", "BLOCKLIST", "SPII", "RECITATION")):
        raise LLMError("the model declined this request")
    try:
        return json.loads(resp.text or "")
    except ValueError as exc:
        raise LLMError("the model returned invalid JSON") from exc


def list_models(provider: str) -> list[str]:
    """Model ids the configured account can use (best effort; listing models is free)."""
    try:
        if provider == "anthropic":
            import anthropic

            return [m.id for m in anthropic.Anthropic().models.list()]
        if provider == "openai":
            import openai

            return sorted(m.id for m in openai.OpenAI().models.list())
        if provider == "google":
            from google import genai

            client = genai.Client()  # keep a reference: the pager needs the client open
            out = []
            for m in client.models.list():
                actions = getattr(m, "supported_actions", None) or []
                if "generateContent" in actions and "gemini" in (m.name or ""):
                    out.append(m.name.split("/")[-1])
            return out
    except ImportError as exc:
        raise LLMError(f'the {provider} SDK is not installed: pip install "visualagents[{provider}]"') from exc
    except Exception as exc:  # each SDK has its own error types; listing is informational only
        raise LLMError(f"could not list {provider} models: {exc}") from exc
    raise LLMError(f"unknown provider {provider!r}")


# -- prompts ----------------------------------------------------------------

SCAN_SYSTEM = """You are the code analyst inside VisualAgents, a tool that draws a live map of multi-agent AI \
applications. You read a project's source and return its blueprint: which agents exist, what each one does, \
and how work flows between them. The blueprint is rendered as a node graph, and later used to recognise each \
agent in the program's printed chain-of-thought logs.

Guidelines:
- An agent is a distinct LLM-driven worker (own prompt / role / tools) or a graph node that performs a \
distinct reasoning step. Tools, LLM clients, routers that make no LLM call, and plain helpers are not agents; \
list a tool under the agent that calls it.
- A single-agent program yields one agent. Do not invent agents the code does not define.
- Without an agent framework, every distinct place that prompts a model for its own purpose is an agent \
(an answerer, a planner, a summariser, an extractor), even if they share one model or one client. A shared \
helper that only forwards a prompt it was given (e.g. generate_reply(system_prompt, user_content)) is plumbing, \
not an agent: the functions that call it with their own prompts are the agents. A program with one such place \
is a one-agent system.
- The agent is where its system prompt is written and its loop runs (often a request handler or a run() \
method), not the client class that sends the request: an LLMService / ModelClient whose methods take messages \
and pass them on is plumbing, as are code executors, kernels, sandboxes, databases and file exporters with no \
prompt of their own. Name the agent after its prompt ("You are LAMBDA, a data analysis agent" -> LAMBDA).
- One agent that loops - call the model, run the code or tools it asks for, read the output or error, fix and \
retry - is still one agent. Show that loop through steps (e.g. plan, write code, run it, inspect errors and fix, \
report) instead of inventing a separate programmer or inspector the code does not define.
- ids are stable snake_case names taken from the code: LangGraph node names, CrewAI agents.yaml keys, \
Agent(name=...) values, AutoGen agent names.
- aliases: every string that may identify the agent in logs - node name, variable names, class name, the \
name= value, CrewAI role text with template variables removed. Always include the name of the class (or \
function) that builds this agent's prompt and calls the model, e.g. a Programmer class holding its own OpenAI \
client - never a class or helper shared by several agents, such as a common LLMService.
- keywords: 3-8 distinctive words or short phrases this agent's own output is likely to contain.
- edges: handoffs / control flow (who passes work to whom). conditional=true when a router, supervisor or \
LLM decision picks the next agent. Include loops such as critic -> writer.
- entry: the agents that receive the user's query first. exits: the agents that produce the final answer.
- steps: the general blueprint a typical query moves through, as 2-8 ordered phases, each naming its agents.
- file and line: where the agent is defined (use the line numbers shown in the source). line=0 if unsure.
- The source starts with <repo_outline>: every source file with its classes (and their base classes), \
functions and prompt personas, each with its line number - including files not shown in full. The files after it \
are the most relevant ones; those marked excerpt show only selected lines. An agent you can only see in the \
outline (another subclass of the same agent base class, another module of the same pipeline) is still an agent: \
include it, with the outline's file and line.
- Static-analysis hints are heuristic: reuse their names, aliases and line numbers, but drop hint agents the \
source shows are plumbing and add agents the hints missed."""

NARRATE_SYSTEM = """You watch the live console output of a multi-agent AI program for VisualAgents. Given the \
program's agents and its most recent log lines, decide which agent is working right now and describe what it \
is doing. Use an agent id from the list, or an empty string when the log does not say. activity is a \
present-tense phrase of at most 12 words, e.g. "Searching the web for 2024 battery benchmarks"."""

NARRATE_SCHEMA = {
    "type": "object",
    "properties": {
        "agent": {"type": "string"},
        "activity": {"type": "string"},
    },
    "required": ["agent", "activity"],
    "additionalProperties": False,
}
