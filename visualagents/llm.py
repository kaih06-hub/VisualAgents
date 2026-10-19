"""LLM access for VisualAgents: codebase -> blueprint, and (optionally) live narration.

Claude is the default provider (official `anthropic` SDK). OpenAI or any OpenAI-compatible
server (Ollama, LM Studio, vLLM via OPENAI_BASE_URL) works with `pip install "visualagents[openai]"`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-5"}
# Models that accept the server-side refusal fallback (beta server-side-fallback-2026-07-01).
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")
_NO_EFFORT = ("claude-3", "claude-haiku-4", "claude-sonnet-4-5", "claude-sonnet-4-0", "claude-opus-4-0", "claude-opus-4-1")


class LLMError(RuntimeError):
    pass


def load_dotenv_keys(project: Path) -> None:
    """Pick up provider keys from the project's .env if they aren't already in the environment."""
    wanted = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL"}
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
            if k in wanted and v and not os.environ.get(k):
                os.environ[k] = v


def detect_provider(preferred: str = "auto") -> str | None:
    if preferred and preferred != "auto":
        return preferred
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return "openai"
    # `ant auth login` stores a profile the Anthropic SDK picks up without any env var
    if os.environ.get("ANTHROPIC_PROFILE") or (Path.home() / ".config" / "anthropic").is_dir():
        return "anthropic"
    return None


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
    model = model or DEFAULT_MODELS.get(provider)
    if provider == "anthropic":
        return _anthropic_json(model, system, prompt, schema, effort, max_tokens)
    if provider == "openai":
        return _openai_json(model, system, prompt, schema)
    raise LLMError(f"unknown provider {provider!r} (use 'anthropic' or 'openai')")


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
- ids are stable snake_case names taken from the code: LangGraph node names, CrewAI agents.yaml keys, \
Agent(name=...) values, AutoGen agent names.
- aliases: every string that may identify the agent in logs - node name, variable names, class name, the \
name= value, CrewAI role text with template variables removed.
- keywords: 3-8 distinctive words or short phrases this agent's own output is likely to contain.
- edges: handoffs / control flow (who passes work to whom). conditional=true when a router, supervisor or \
LLM decision picks the next agent. Include loops such as critic -> writer.
- entry: the agents that receive the user's query first. exits: the agents that produce the final answer.
- steps: the general blueprint a typical query moves through, as 2-8 ordered phases, each naming its agents.
- file and line: where the agent is defined (use the line numbers shown in the source). line=0 if unsure.
- Static-analysis hints are provided; trust them for names, correct them when the code says otherwise."""

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
