"""`visualagents` command line."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import webbrowser
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from . import blueprint as bpmod
from . import termui
from .hub import Hub
from .llm import PROVIDERS, LLMError, detect_provider, load_dotenv_keys
from .server import Server

DEFAULT_PORT = 7420
_VALUE_OPTS = {"--port", "--host", "--provider", "--model", "--narrate-model", "--blueprint", "--cwd", "--interval"}

EPILOG = """examples:
  visualagents demo                                  see it work in 10 seconds, no API key needed
  visualagents scan ./my-crew                        map the agents in a project (LLM if a key is set)
  visualagents run ./my-crew -- python main.py "q"   run it and watch the live map
  python main.py 2>&1 | visualagents pipe ./my-crew  monitor output you already have
  visualagents watch ./my-crew --log run.log         follow a log file
"""


def _server_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--port", type=int, default=int(os.environ.get("VISUALAGENTS_PORT", DEFAULT_PORT)),
                   help=f"dashboard port (default {DEFAULT_PORT}; next free port is used if taken)")
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    p.add_argument("--no-browser", action="store_true", help="don't open the dashboard in a browser")
    p.add_argument("--blueprint", type=Path, help="use this blueprint.json instead of the project's")
    p.add_argument("--exit", action="store_true", help="exit when the input ends instead of keeping the dashboard open")


def _scan_opts(p: argparse.ArgumentParser, *, standalone: bool = False) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--static", action="store_true", help="offline analysis only (no LLM)")
    g.add_argument("--llm", action="store_true", help="require an LLM pass (fail if no key is configured)")
    p.add_argument("--provider", default="auto", choices=["auto", *PROVIDERS],
                   help="LLM provider (default: $VISUALAGENTS_PROVIDER, else whichever has a key)")
    p.add_argument("--model", help="model that reads the code and draws the map (default: $VISUALAGENTS_MODEL, "
                                   "else claude-opus-5-5 / gpt-5 / gemini-pro-latest) - see `visualagents models`")
    if standalone:
        p.add_argument("--max-chars", type=int, default=180_000, help="source budget sent to the LLM")
    else:
        p.add_argument("--rescan", action="store_true", help="rebuild the blueprint before starting")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualagents",
        description="A live map of which agent in a multi-agent AI system is doing what.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"visualagents {__version__}")
    sub = parser.add_subparsers(dest="cmd", metavar="command")

    p = sub.add_parser("scan", help="read a codebase and write .visualagents/blueprint.json")
    p.add_argument("project", nargs="?", default=".", type=Path)
    _scan_opts(p, standalone=True)
    p.add_argument("--json", action="store_true", help="print the blueprint as JSON")
    p.add_argument("-o", "--output", type=Path, help="write the blueprint here instead")

    p = sub.add_parser("blueprint", help="print a project's blueprint in the terminal")
    p.add_argument("project", nargs="?", default=".", type=Path)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("show", help="open the dashboard with just the blueprint")
    p.add_argument("project", nargs="?", default=".", type=Path)
    _server_opts(p)
    _scan_opts(p)

    p = sub.add_parser("run", help="run an agent command and map it live:  run [project] -- <command>",
                       description="Run a command (your agent) with live mapping. Put the command after `--`.")
    p.add_argument("project", nargs="?", default=None, type=Path, help="project folder (default: current folder)")
    _server_opts(p)
    _scan_opts(p)
    p.add_argument("--cwd", type=Path, help="working directory for the command (default: the project folder)")
    p.add_argument("--smart", action="store_true", help="also use an LLM to narrate what each agent is doing")
    p.add_argument("--narrate-model", help="model for --smart narration (default: $VISUALAGENTS_NARRATE_MODEL, "
                                           "else the provider's fast default - see `visualagents models`)")
    p.add_argument("--interval", type=float, default=4.0, help="seconds between --smart narrations")
    p.add_argument("--no-instrument", action="store_true", help="don't auto-hook LangChain/LangGraph/OpenAI Agents")
    p.add_argument("--no-pty", action="store_true", help="use plain pipes instead of a pseudo-terminal")
    p.add_argument("--no-record", action="store_true", help="don't save the run under .visualagents/runs/")

    p = sub.add_parser("pipe", help="map output piped on stdin:  python app.py 2>&1 | visualagents pipe")
    p.add_argument("project", nargs="?", default=".", type=Path)
    _server_opts(p)
    _scan_opts(p)
    p.add_argument("--quiet", action="store_true", help="don't echo the piped output")
    p.add_argument("--no-record", action="store_true")

    p = sub.add_parser("watch", help="follow a log file:  watch [project] --log agent.log")
    p.add_argument("project", nargs="?", default=".", type=Path)
    p.add_argument("--log", type=Path, required=True, help="log file to follow")
    p.add_argument("--from-start", action="store_true", help="read the whole file, not only new lines")
    p.add_argument("--echo", action="store_true", help="print followed lines in this terminal too")
    _server_opts(p)
    _scan_opts(p)
    p.add_argument("--no-record", action="store_true")

    p = sub.add_parser("replay", help="replay a recorded run from .visualagents/runs/")
    p.add_argument("recording", type=Path)
    p.add_argument("--project", type=Path, help="use this project's current blueprint instead of the recorded one")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--instant", action="store_true", help="jump straight to the end state")
    _server_opts(p)

    p = sub.add_parser("models", help="show LLM providers, which have keys, and the models you can pick")
    p.add_argument("project", nargs="?", default=".", type=Path, help="also read keys from this project's .env")
    p.add_argument("--provider", choices=list(PROVIDERS), help="list the models this provider's account can use")

    p = sub.add_parser("demo", help="a simulated 4-agent research crew - no API key needed")
    p.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    _server_opts(p)
    return parser


def _split_run_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """`run [opts] [project] -- cmd...` (or without `--`: everything after the project is the command)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    if not argv or argv[0] != "run":
        return argv, []
    ours, i = ["run"], 1
    while i < len(argv) and argv[i].startswith("-"):
        ours.append(argv[i])
        if argv[i] in _VALUE_OPTS and i + 1 < len(argv):
            ours.append(argv[i + 1])
            i += 1
        i += 1
    rest = argv[i:]
    if rest and Path(rest[0]).is_dir():
        ours.append(rest[0])
        rest = rest[1:]
    return ours, rest


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    ours, command = _split_run_argv(raw)
    parser = build_parser()
    args = parser.parse_args(ours)
    if not args.cmd:
        parser.print_help()
        return 0
    try:
        return {
            "scan": cmd_scan, "blueprint": cmd_blueprint, "show": cmd_show, "run": cmd_run, "pipe": cmd_pipe,
            "watch": cmd_watch, "replay": cmd_replay, "demo": cmd_demo, "models": cmd_models,
        }[args.cmd](args, command)
    except LLMError as exc:
        termui.error(str(exc))
        return 2
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


# -- blueprint helpers --------------------------------------------------------

def _scan(project: Path, args: argparse.Namespace, **kw: Any) -> dict[str, Any]:
    from .scanner import scan_project

    mode = "static" if getattr(args, "static", False) else "llm" if getattr(args, "llm", False) else "auto"
    return scan_project(project, mode=mode, provider=getattr(args, "provider", "auto"),
                        model=getattr(args, "model", None), log=termui.info, **kw)


def _project_blueprint(project: Path, args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "blueprint", None):
        return bpmod.load(args.blueprint)
    path = bpmod.blueprint_path(project)
    if path.exists() and not getattr(args, "rescan", False):
        return bpmod.load(path)
    if not path.exists():
        termui.info(f"No blueprint yet for {project} - scanning the code first.")
    bp = _scan(project, args)
    try:
        bpmod.save(bp, path)
        termui.info(f"Saved {path}")
    except OSError as exc:
        termui.info(f"Could not save the blueprint ({exc}); continuing.")
    return bp


def _record_dir(project: Path, args: argparse.Namespace) -> Path | None:
    if getattr(args, "no_record", False):
        return None
    d = bpmod.state_dir(project) / "runs"
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        return None


def _serve(hub: Hub, args: argparse.Namespace, extra: str = "") -> Server:
    server = Server(hub, host=args.host, port=args.port).start()
    termui.banner(server.url, hub.blueprint, extra)
    if not args.no_browser and not os.environ.get("VISUALAGENTS_NO_BROWSER"):
        threading.Timer(0.4, lambda: webbrowser.open(server.url)).start()
    return server


def _hold(server: Server, message: str) -> None:
    termui.info(message + f" Dashboard stays live at {server.url} - press Ctrl+C to close.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


# -- commands ----------------------------------------------------------------

def cmd_scan(args, _command) -> int:
    project = args.project.resolve()
    if not project.exists():
        termui.error(f"{project} does not exist")
        return 2
    bp = _scan(project, args, max_chars=args.max_chars)
    out = args.output or bpmod.blueprint_path(project if project.is_dir() else project.parent)
    bpmod.save(bp, out)
    if args.json:
        print(json.dumps(bp, indent=2, ensure_ascii=False))
    else:
        termui.print_blueprint(bp)
    termui.info(f"Saved {out}")
    return 0


def cmd_blueprint(args, _command) -> int:
    path = bpmod.blueprint_path(args.project.resolve())
    if not path.exists():
        termui.error(f"no blueprint at {path} - run `visualagents scan {args.project}` first")
        return 1
    bp = bpmod.load(path)
    print(json.dumps(bp, indent=2, ensure_ascii=False)) if args.json else termui.print_blueprint(bp)
    return 0


def cmd_show(args, _command) -> int:
    project = args.project.resolve()
    bp = _project_blueprint(project, args)
    hub = Hub(bp, meta={"mode": "blueprint", "project": str(project)})
    server = _serve(hub, args, "blueprint only - use `visualagents run` to watch a live run")
    _hold(server, "Showing the blueprint.")
    return 0


def cmd_run(args, command: list[str]) -> int:
    from .runner import install_sigterm_handler, run_command

    if not command:
        termui.error("give the command to run after `--`, e.g.  visualagents run ./my-crew -- python main.py")
        return 2
    project = (args.project or Path.cwd()).resolve()
    if not project.is_dir():
        termui.error(f"{project} is not a folder")
        return 2
    if command[0].endswith(".py") and not shutil.which(command[0]) and (project / command[0]).exists():
        command = [shutil.which("python3") or shutil.which("python") or sys.executable, *command]
    bp = _project_blueprint(project, args)
    hub = Hub(bp, record_dir=_record_dir(project, args),
              meta={"mode": "run", "project": str(project), "command": " ".join(command)})
    server = _serve(hub, args, f"running: {' '.join(command)}")

    narrator = None
    if args.smart:
        load_dotenv_keys(project)
        provider = detect_provider(args.provider)
        if provider is None:
            termui.info("--smart needs ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY - continuing without narration.")
        else:
            from .narrator import Narrator

            from .llm import default_model

            model = args.narrate_model or default_model(provider, "narrate")
            termui.info(f"Smart narration: {provider}:{model} every {args.interval:g}s while output changes.")
            narrator = Narrator(hub, provider=provider, model=model, interval=args.interval).start()

    install_sigterm_handler()
    code = run_command(command, hub, cwd=(args.cwd or project).resolve(), server_url=server.url,
                       instrument=not args.no_instrument, use_pty=False if args.no_pty else None)
    if narrator:
        narrator.stop.set()
    hub.close()
    if hub.record_path:
        termui.info(f"Recorded to {hub.record_path}  (replay: visualagents replay {hub.record_path})")
    if not args.exit:
        status = "finished" if code == 0 else f"exited with code {code}"
        _hold(server, f"Run {status}.")
    server.stop()
    return code


def cmd_pipe(args, _command) -> int:
    from .runner import read_stream

    project = args.project.resolve()
    bp = _project_blueprint(project, args)
    hub = Hub(bp, record_dir=_record_dir(project, args), meta={"mode": "pipe", "project": str(project)})
    server = _serve(hub, args, "reading stdin")
    read_stream(None, hub, echo=not args.quiet)
    hub.close()
    if not args.exit:
        _hold(server, "Input ended.")
    server.stop()
    return 0


def cmd_watch(args, _command) -> int:
    from .runner import follow_file

    project = args.project.resolve()
    bp = _project_blueprint(project, args)
    hub = Hub(bp, record_dir=_record_dir(project, args),
              meta={"mode": "watch", "project": str(project), "log": str(args.log)})
    server = _serve(hub, args, f"following {args.log}")
    stop = threading.Event()
    try:
        follow_file(args.log.resolve(), hub, from_start=args.from_start, echo=args.echo, stop=stop)
    except KeyboardInterrupt:
        stop.set()
    hub.close()
    server.stop()
    return 0


def cmd_replay(args, _command) -> int:
    from .runner import read_recording_meta, replay

    meta = read_recording_meta(args.recording)
    if args.blueprint:
        bp = bpmod.load(args.blueprint)
    elif args.project:
        bp = _project_blueprint(args.project.resolve(), args)
    elif meta.get("blueprint"):
        bp = bpmod.normalize(meta["blueprint"])
    else:
        termui.error("this recording has no blueprint - pass --project or --blueprint")
        return 2
    hub = Hub(bp, meta={"mode": "replay", "recording": str(args.recording), **(meta.get("meta") or {})})
    server = _serve(hub, args, f"replaying {args.recording.name} at {args.speed}x")
    stop = threading.Event()
    try:
        replay(args.recording, hub, speed=args.speed, instant=args.instant, stop=stop)
        if not args.exit:
            _hold(server, "Replay finished.")
    except KeyboardInterrupt:
        stop.set()
    server.stop()
    return 0


def cmd_models(args, _command) -> int:
    from . import llm

    load_dotenv_keys(args.project.resolve())
    if args.provider:
        models = llm.list_models(args.provider)
        print("\n".join(models) if models else f"(no models returned for {args.provider})")
        return 0
    active = detect_provider("auto")
    print(f"\n{termui.bold('LLM providers')}  (the provider + model that read your code and draw the map)\n")
    for p in PROVIDERS:
        mark = termui.good("● key found") if llm.has_credentials(p) else termui.dim("○ no key")
        chosen = termui.accent("  ← used by default") if p == active else ""
        print(f"  {termui.bold(f'{p:<10}')} {llm.PROVIDER_LABELS[p]:<22} {mark}{chosen}")
        print(f"             scan: {llm.default_model(p)}   --smart: {llm.default_model(p, 'narrate')}")
        print(f"             {termui.dim('keys: ' + ' / '.join(llm.KEY_VARS[p]))}")
    print(f"""
{termui.bold('Choose per command')}   visualagents scan . --provider google --model gemini-flash-latest
                     visualagents run . --smart --provider google --narrate-model gemini-flash-lite-latest -- python app.py
{termui.bold('Or set defaults')}      VISUALAGENTS_PROVIDER, VISUALAGENTS_MODEL, VISUALAGENTS_NARRATE_MODEL
                     (in your shell or the project's .env)
{termui.bold('See what you can use')} visualagents models --provider google
""")
    return 0


def cmd_demo(args, _command) -> int:
    from .runner import run_command

    demo_dir = Path(str(resources.files("visualagents") / "demo"))
    bp = bpmod.load(demo_dir / "blueprint.json")
    hub = Hub(bp, meta={"mode": "demo", "project": "demo"})
    command = [sys.executable, "-m", "visualagents.demo.research_crew", "--speed", str(args.speed)]
    server = _serve(hub, args, "demo: a simulated research crew (planner → researcher → writer ⇄ critic)")
    if not args.no_browser:
        import time

        time.sleep(1.5)  # give the browser a moment so the run starts on screen
    code = run_command(command, hub, cwd=demo_dir, server_url=server.url, instrument=False)
    if not args.exit:
        _hold(server, "Demo finished.")
    server.stop()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
