"""`visualagents` with no arguments: pick a project, pick an LLM, paste a key - then the map pops up."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from . import blueprint as bpmod
from . import detect, home, llm, termui, tui
from .hub import Hub
from .server import Server

DAEMON_PORT = 7400
_GIT_URL = re.compile(r"^(https?://|git@|ssh://)\S+$|\.git$")
# Google's -latest aliases follow its current models, so a choice here cannot be retired from under the user
# (pinned versions are: gemini-2.5-pro already is for new keys). Pro has no free-tier quota.
MODEL_CHOICES = {
    "google": [("gemini-pro-latest", "best, needs billing on the key"), ("gemini-flash-latest", "fast, works on a free key"),
               ("gemini-flash-lite-latest", "cheapest")],
    "anthropic": [("claude-opus-5-5", "best"), ("claude-sonnet-5-5", "faster, cheaper"),
                  ("claude-haiku-5-5", "cheapest")],
    "openai": [("gpt-5", "best"), ("gpt-5-mini", "cheaper")],
}
KEY_VAR = {"google": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
KEY_HELP = {
    "google": "get one at https://aistudio.google.com/apikey",
    "anthropic": "get one at https://console.anthropic.com/settings/keys",
    "openai": "get one at https://platform.openai.com/api-keys",
}


def run(args: argparse.Namespace) -> int:
    termui_banner()
    try:
        project = ask_project()
        provider, model = ask_provider(project)
        bp = map_agents(project, provider, model)
        return show_and_watch(project, bp, args)
    except (KeyboardInterrupt, EOFError):
        tui.note("\n  Cancelled.")
        return 130


def termui_banner() -> None:
    tui.note("")
    tui.note(f"{termui.bold('VisualAgents')}  {termui.dim('· a live map of your agents')}")
    tui.note(termui.dim("Three questions, then your agents' map pops up. Ctrl+C to quit."))


# -- 1. project -----------------------------------------------------------------

def ask_project() -> Path:
    tui.heading("1/3", "Which project should VisualAgents map?")
    recent = [p for p in home.settings().get("recent", []) if Path(p).is_dir()][:5]
    cwd = Path.cwd().resolve()
    options: list[tuple[str, str]] = []
    paths: list[Path | None] = []
    for p in recent:
        options.append((tui.short(p), "last used" if not paths else "recent"))
        paths.append(Path(p))
    if _looks_like_project(cwd) and str(cwd) not in recent:
        options.append((f"This folder  {tui.short(str(cwd))}", ""))
        paths.append(cwd)
    options.append(("Another folder or a GitHub URL…", "Tab completes paths"))
    paths.append(None)
    choice = tui.select("", options) if len(options) > 1 else 0
    project = paths[choice]
    while project is None:
        raw = tui.ask("Folder path or GitHub URL", paths=True)
        project = _resolve_project(raw)
    project = project.resolve()
    home.remember_project(project)
    return project


def _resolve_project(raw: str) -> Path | None:
    raw = raw.strip()
    if not raw:
        return None
    if _GIT_URL.search(raw) and not os.path.exists(tui.clean_path(raw)):
        return _clone(raw)
    path = Path(tui.clean_path(raw))
    if not path.is_dir():
        tui.warn(f"{raw} is not a folder")
        return None
    return path


def _clone(url: str) -> Path | None:
    name = re.sub(r"\.git$", "", url.rstrip("/").split("/")[-1].split(":")[-1]) or "project"
    default = os.path.join("~", "agents", name)
    answer = tui.ask("Clone it into", default, paths=True).strip()
    if answer.lower() in ("y", "yes"):  # "yes, clone it" - not a folder called y
        answer = default
    elif answer.lower() in ("n", "no"):
        return None
    dest = Path(tui.clean_path(answer or default))
    if dest.exists() and any(dest.iterdir()):
        if (dest / ".git").exists():
            origin = _origin(dest)
            if origin and _same_repo(origin, url):
                tui.ok(f"Using the existing clone in {tui.short(str(dest))}")
                return dest
            tui.warn(f"{tui.short(str(dest))} is a clone of {origin or 'another repository'}, not {url} - "
                     "pick another folder")
            return None
        tui.warn(f"{tui.short(str(dest))} already exists and isn't empty")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tui.note(termui.dim(f"git clone --depth 1 {url} {tui.short(str(dest))}"))
    result = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)])
    if result.returncode != 0:
        tui.warn("git clone failed - check the URL (and your access to the repo)")
        return None
    tui.ok(f"Cloned into {tui.short(str(dest))}")
    return dest


def _origin(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"], capture_output=True, text=True)
    except OSError:
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def _same_repo(a: str, b: str) -> bool:
    """https://github.com/o/r.git, git@github.com:o/r and ssh://git@github.com/o/r are the same repository."""
    def key(url: str) -> str:
        url = re.sub(r"^[a-z+]+://", "", url.strip().lower())  # https://, ssh://, git://
        url = re.sub(r"^[^@/]+@", "", url)  # git@ or user:token@
        url = re.sub(r"^([^/:]+):\d+/", r"\1/", url)  # host:443/owner/repo
        url = url.replace(":", "/", 1) if "/" not in url.split(":", 1)[0] else url  # host:owner/repo
        return re.sub(r"(\.git)?/*$", "", url)
    return key(a) == key(b)


def _looks_like_project(path: Path) -> bool:
    markers = ("README.md", "pyproject.toml", "requirements.txt", "package.json", "setup.py", "main.py", "app.py")
    return any((path / m).exists() for m in markers) and path != Path.home()


# -- 2. provider, model, key ----------------------------------------------------------

def ask_provider(project: Path) -> tuple[str | None, str | None]:
    llm.load_dotenv_keys(project)
    tui.heading("2/3", "Which LLM should read the code and draw the map?")
    saved = home.settings()
    order = ["google", "anthropic", "openai"]
    options = []
    for p in order:
        hint = _key_hint(p)
        options.append((llm.PROVIDER_LABELS[p] if p != "openai" else "OpenAI", hint))
    options.append(("Local or OpenAI-compatible server", "Ollama, LM Studio, vLLM… no key needed"))
    options.append(("Skip - map the code offline", "no LLM, less detail"))
    default = order.index(saved["provider"]) if saved.get("provider") in order else next(
        (i for i, p in enumerate(order) if llm.has_credentials(p)), 0)
    if saved.get("provider") == "openai" and saved.get("base_url"):
        default = 3
    choice = tui.select("", options, default)
    if choice == len(options) - 1:
        home.save_settings({"provider": None})
        return None, None
    if choice == 3:
        return _local_server(saved)
    provider = order[choice]
    model = _ask_model(provider, saved.get("models", {}).get(provider))
    if not _ensure_key(provider):
        return None, None
    models = dict(saved.get("models", {}))
    models[provider] = model
    home.save_settings({"provider": provider, "models": models, "base_url": None})
    os.environ["VISUALAGENTS_PROVIDER"] = provider
    os.environ["VISUALAGENTS_MODEL"] = model
    return provider, model


def _key_hint(provider: str) -> str:
    var = next((v for v in llm.KEY_VARS[provider] if os.environ.get(v) and not v.endswith("BASE_URL")), None)
    if var and not var.startswith("GOOGLE_GENAI"):
        return f"key ready ••••{os.environ[var][-4:]}"
    if provider == "anthropic" and llm.has_credentials("anthropic"):
        return "signed in with `ant`"
    return ""


def _ensure_key(provider: str) -> bool:
    var = KEY_VAR[provider]
    tui.heading("3/3", f"{llm.PROVIDER_LABELS[provider]} API key")
    existing = next((os.environ[v] for v in llm.KEY_VARS[provider]
                     if os.environ.get(v) and not v.endswith(("BASE_URL", "VERTEXAI"))), None)
    if existing:
        idx = tui.select("", [(f"Use the key I already have  ••••{existing[-4:]}", ""), ("Paste a different key", "")])
        if idx == 0 and _check_key(provider):
            return True
    elif provider == "anthropic" and llm.has_credentials("anthropic"):
        if _check_key(provider):
            return True
    for attempt in range(3):
        tui.note(termui.dim(f"Paste it below - it stays hidden · {KEY_HELP[provider]} · empty = skip"))
        key = tui.secret("API key")
        if not key:
            tui.note(termui.dim("No key - the map will be drawn offline."))
            return False
        os.environ[var] = key
        if _check_key(provider):
            if tui.confirm(f"Save it for next time? {termui.dim('(~/.visualagents/keys.json, readable only by you)')}"):
                path = home.save_key(var, key)
                tui.ok(f"Saved to {tui.short(str(path))}")
            return True
        os.environ.pop(var, None)
    tui.note(termui.dim("Continuing without an LLM - the map will be drawn offline."))
    return False


def _check_key(provider: str) -> bool:
    with tui.Spinner("Checking the key (free - just lists your models)"):
        try:
            models = llm.list_models(provider)
        except llm.LLMError as exc:
            problem = str(exc)
            models = None
    if models is None:
        tui.warn(problem.replace("could not list", "That key didn't work for") if "401" in problem or "403" in problem
                 or "auth" in problem.lower() or "key" in problem.lower() else problem)
        return False
    tui.ok(f"Key works · {len(models)} models available")
    return True


def _ask_model(provider: str, saved: str | None) -> str:
    choices = list(MODEL_CHOICES[provider])
    if saved and saved not in [m for m, _ in choices]:
        choices.insert(0, (saved, "last used"))
    options = [(m, hint) for m, hint in choices] + [("Another model…", f"see: visualagents models --provider {provider}")]
    default = next((i for i, (m, _) in enumerate(choices) if m == saved), 0)
    tui.note(termui.dim("Model"))
    idx = tui.select("", options, default)
    if idx < len(choices):
        return choices[idx][0]
    return tui.ask("Model id", choices[0][0])


def _local_server(saved: dict[str, Any]) -> tuple[str | None, str | None]:
    base = tui.ask("Server URL", saved.get("base_url") or "http://localhost:11434/v1")
    os.environ["OPENAI_BASE_URL"] = base
    os.environ.setdefault("OPENAI_API_KEY", "local")
    with tui.Spinner("Asking the server which models it has"):
        try:
            models = llm.list_models("openai")
        except llm.LLMError as exc:
            models, problem = [], str(exc)
    if not models:
        tui.warn(f"Couldn't reach {base} - is the server running? Continuing offline.")
        return None, None
    tui.ok(f"Server is up · {len(models)} models")
    previous = saved.get("models", {}).get("openai")
    default = models.index(previous) if previous in models else 0
    model = models[tui.select("Model", [(m, "") for m in models[:12]], min(default, 11))]
    home.save_key("OPENAI_BASE_URL", base)
    if os.environ.get("OPENAI_API_KEY") == "local":
        home.save_key("OPENAI_API_KEY", "local")
    models_saved = dict(saved.get("models", {}))
    models_saved["openai"] = model
    home.save_settings({"provider": "openai", "models": models_saved, "base_url": base})
    os.environ["VISUALAGENTS_PROVIDER"], os.environ["VISUALAGENTS_MODEL"] = "openai", model
    return "openai", model


# -- map + popup ------------------------------------------------------------------------

def map_agents(project: Path, provider: str | None, model: str | None) -> dict[str, Any]:
    from .scanner import scan_project

    tui.note("")
    label = f"Reading {project.name} with {model}" if provider else f"Reading {project.name} (offline)"
    with tui.Spinner(label) as sp:
        try:
            bp = scan_project(project, mode="llm" if provider else "static", provider=provider or "auto",
                              model=model, log=sp.update)
        except llm.LLMError as exc:
            sp.update(str(exc))
            bp = None
    if bp is None:
        tui.warn("The LLM couldn't map it - using the offline map instead.")
        bp = scan_project(project, mode="static", log=lambda m: None)
    bpmod.save(bp, bpmod.blueprint_path(project))
    names = ", ".join(a["name"] for a in bp["agents"][:6]) + ("…" if len(bp["agents"]) > 6 else "")
    tui.ok(f"Found {len(bp['agents'])} agent(s){': ' + names if names else ''}  {termui.dim('(' + bp['framework'] + ')')}")
    return bp


def show_and_watch(project: Path, bp: dict[str, Any], args: argparse.Namespace) -> int:
    from .cli import _hold, _choose_start_command
    from .runner import follow_new_logs, install_sigterm_handler, run_command

    record = bpmod.state_dir(project) / "runs"
    record.mkdir(parents=True, exist_ok=True)
    from .launcher import Launcher

    hub = Hub(bp, record_dir=record, meta={"mode": "setup", "project": str(project)})
    server = Server(hub, port=getattr(args, "port", 7420) or 7420).start()
    Launcher(project, hub, server.url)
    if not getattr(args, "no_browser", False) and not os.environ.get("VISUALAGENTS_NO_BROWSER"):
        webbrowser.open(server.url)
    tui.ok(f"Map is open at {termui.accent(server.url)}")

    saved = detect.load_saved(project)
    cands = detect.detect(project) if not saved else []
    first = saved or (cands[0] if cands else None)
    tui.heading("▶", "Watch it live?")
    options = []
    if first:
        options.append(("Start it now", first.label()))
    options += [("Start it with a different command…", ""),
                ("Hook it - I'll start it myself", "IDE, scripts, notebooks: it shows up whenever it runs"),
                ("Just show the map", "")]
    idx = tui.select("", options)
    action = options[idx][0]
    if action.startswith("Hook"):
        server.stop()
        from . import cli

        ns = argparse.Namespace(project=project, python=[], no_scan=True, no_daemon=False, port=DAEMON_PORT,
                                no_browser=getattr(args, "no_browser", False), static=True, llm=False,
                                provider="auto", model=None)
        return cli.cmd_hook(ns, [])
    if action.startswith("Just"):
        if not getattr(args, "exit", False):
            _hold(server, "Showing the map. Press ▶ Start in the dashboard whenever you want to run it.")
        server.stop()
        return 0
    cand = first if action == "Start it now" else _choose_start_command(project, force_menu=True)
    if cand is None:
        server.stop()
        return 2
    detect.save_choice(project, cand)
    command, notes = detect.resolve(project, cand)
    for n in notes:
        tui.note(termui.dim(n))
    tui.note(termui.dim(f"$ {' '.join(command)}"))
    tui.note(f"{termui.bold(project.name + ' is starting below.')} Give it a task the way it expects - answer its "
             f"questions here, or open its own web page - and watch the map light up.\n")
    install_sigterm_handler()
    stop_logs = threading.Event()
    threading.Thread(target=follow_new_logs, args=(project, hub), daemon=True,
                     kwargs={"since": time.time() - 1, "stop": stop_logs}).start()
    code = run_command(command, hub, cwd=(project / cand.cwd).resolve(), server_url=server.url, instrument=True)
    hub.close()
    if not getattr(args, "exit", False):
        _hold(server, f"Run {'finished' if code == 0 else f'exited with code {code}'}. "
                      "Anything it left running keeps reporting here.")
    stop_logs.set()
    server.stop()
    return code
