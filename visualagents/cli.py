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
from .server import Server, exposure_warning

DEFAULT_PORT = 7420
DAEMON_PORT = 7400
_VALUE_OPTS = {"--port", "--host", "--provider", "--model", "--narrate-model", "--blueprint", "--cwd", "--interval"}

EPILOG = """examples:
  visualagents demo                                  see it work in 10 seconds, no API key needed
  visualagents run ./my-agent                        start a project and watch it (asks how, once)
  visualagents hook ./my-agent                       once: then run it however you like (IDE, scripts...)
  visualagents scan ./my-agent                       (re)map the agents in a project
  visualagents run ./my-agent -- python main.py "q"  run an exact command
  python main.py 2>&1 | visualagents pipe ./my-agent monitor output you already have
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

    p = sub.add_parser("run", help="start a project and map it live:  run [folder] [-- command]",
                       description="Start a project with live mapping. Without a command, VisualAgents finds how "
                                   "the project starts (README, start scripts, app files), asks once and remembers.")
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
    p.add_argument("--pick", action="store_true", help="choose the start command again")
    p.add_argument("--no-follow-logs", action="store_true", help="don't follow *.log files the app writes")

    p = sub.add_parser("hook", help="hook a folder once - then run its agent however you normally do",
                       description="Install a startup hook into the folder's virtualenv(s) so every Python process "
                                   "from them reports to the background dashboard (`visualagents up`).")
    p.add_argument("project", nargs="?", default=".", type=Path)
    p.add_argument("--python", type=Path, action="append", default=[],
                   help="also hook this interpreter (e.g. a conda env's python); repeatable")
    p.add_argument("--no-scan", action="store_true", help="don't build a blueprint now")
    p.add_argument("--no-daemon", action="store_true", help="don't start the background dashboard")
    p.add_argument("--port", type=int, default=DAEMON_PORT)
    p.add_argument("--no-browser", action="store_true")
    _scan_opts(p)

    p = sub.add_parser("unhook", help="remove the startup hook from a folder's virtualenv(s)")
    p.add_argument("project", nargs="?", default=".", type=Path)

    p = sub.add_parser("up", help="start the background dashboard for hooked projects")
    p.add_argument("--port", type=int, default=DAEMON_PORT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--foreground", action="store_true", help="stay in this terminal instead of the background")

    sub.add_parser("down", help="stop the background dashboard")
    sub.add_parser("status", help="show the background dashboard and hooked projects")

    p = sub.add_parser("open", help="open the dashboard for a hooked project")
    p.add_argument("project", nargs="?", default=".", type=Path)

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

    p = sub.add_parser("setup", help="guided start: pick a project, an LLM and a key - then the map pops up "
                                     "(also what bare `visualagents` does in a terminal)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--exit", action="store_true", help=argparse.SUPPRESS)

    p = sub.add_parser("models", help="show LLM providers, which have keys, and the models you can pick")
    p.add_argument("project", nargs="?", default=".", type=Path, help="also read keys from this project's .env")
    p.add_argument("--provider", choices=list(PROVIDERS), help="list the models this provider's account can use")

    p = sub.add_parser("demo", help="a simulated 4-agent research crew - no API key needed")
    p.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    _server_opts(p)
    return parser


def _split_run_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """`run [opts] [folder] [opts] [--] cmd...`; without `--`, the command starts at the first token that is
    neither an option nor the project folder. A `--` after that is the command's own (`npm run dev -- --port 3000`)."""
    if not argv or argv[0] != "run":
        if "--" in argv:
            i = argv.index("--")
            return argv[:i], argv[i + 1:]
        return argv, []
    ours, rest, i, have_project = ["run"], argv[1:], 0, False
    while i < len(rest):
        tok = rest[i]
        if tok == "--":
            return ours, rest[i + 1:]
        if tok.startswith("-"):
            ours.append(tok)
            if tok in _VALUE_OPTS and i + 1 < len(rest) and rest[i + 1] != "--":
                ours.append(rest[i + 1])
                i += 1
        elif not have_project and Path(tok).is_dir():
            ours.append(tok)
            have_project = True
        else:
            break
        i += 1
    return ours, rest[i:]


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    ours, command = _split_run_argv(raw)
    parser = build_parser()
    if not ours and sys.stdin.isatty() and sys.stderr.isatty():
        ours = ["setup"]  # bare `visualagents` in a terminal: the guided start
    args = parser.parse_args(ours)
    if not args.cmd:
        parser.print_help()
        return 0
    try:
        return {
            "scan": cmd_scan, "blueprint": cmd_blueprint, "show": cmd_show, "run": cmd_run, "pipe": cmd_pipe,
            "watch": cmd_watch, "replay": cmd_replay, "demo": cmd_demo, "models": cmd_models,
            "hook": cmd_hook, "unhook": cmd_unhook, "up": cmd_up, "down": cmd_down, "status": cmd_status,
            "open": cmd_open, "setup": cmd_setup,
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
    if exposure_warning(args.host):
        termui.info(termui.warn(exposure_warning(args.host)))
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
    from .launcher import Launcher

    hub = Hub(bp, record_dir=_record_dir(project, args), meta={"mode": "blueprint", "project": str(project)})
    server = _serve(hub, args, "press ▶ Start in the dashboard (or Ctrl+C) - the agent runs in this terminal")
    Launcher(project, hub, server.url)
    _hold(server, "Showing the map.")
    return 0


def cmd_run(args, command: list[str]) -> int:
    from .runner import install_sigterm_handler, run_command

    from . import detect

    project = (args.project or Path.cwd()).resolve()
    if not project.is_dir():
        termui.error(f"{project} is not a folder")
        return 2
    cwd = args.cwd.resolve() if args.cwd else project
    explicit = bool(command)
    if command:
        if command[0].endswith(".py") and not shutil.which(command[0]) and (cwd / command[0]).exists():
            command = [shutil.which("python3") or shutil.which("python") or sys.executable, *command]
        elif os.sep not in command[0] and not shutil.which(command[0]):
            venv = detect.best_venv(project, os.path.relpath(cwd, project))
            if venv is not None and detect.venv_bin(venv, command[0]).exists():  # e.g. -- redrock-chat
                command = [str(detect.venv_bin(venv, command[0])), *command[1:]]
                termui.info(f"using the project's virtualenv: {os.path.relpath(venv, project)}")
    else:
        cand = None if args.pick else detect.load_saved(project)
        if cand is None:
            cand = _choose_start_command(project)
            if cand is None:
                return 2
            saved = detect.save_choice(project, cand)
            termui.info(f"Saved to {saved.relative_to(project)} - next time just run: visualagents run {args.project or '.'}")
        command, notes = detect.resolve(project, cand)
        cwd = (project / cand.cwd).resolve() if not args.cwd else cwd
        for note in notes:
            termui.info(note)
    bp = _project_blueprint(project, args)
    hub = Hub(bp, record_dir=_record_dir(project, args),
              meta={"mode": "run", "project": str(project), "command": " ".join(command)})
    server = _serve(hub, args, f"running: {' '.join(command)}")
    from .launcher import Launcher

    exact = detect.Candidate(list(command), os.path.relpath(cwd, project)) if explicit else None
    Launcher(project, hub, server.url, instrument=not args.no_instrument, fixed=exact)  # "Run again" from the dashboard

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
    stop_logs = threading.Event()
    if not args.no_follow_logs:
        import time

        from .runner import follow_new_logs

        threading.Thread(target=follow_new_logs, args=(project, hub), daemon=True,
                         kwargs={"since": time.time() - 1, "stop": stop_logs}).start()
    code = run_command(command, hub, cwd=cwd, server_url=server.url,
                       instrument=not args.no_instrument, use_pty=False if args.no_pty else None)
    if narrator:
        narrator.stop.set()
    hub.close()
    if hub.record_path:
        termui.info(f"Recorded to {hub.record_path}  (replay: visualagents replay {hub.record_path})")
    if not args.exit:
        status = "finished" if code == 0 else f"exited with code {code}"
        _hold(server, f"Run {status}. Anything it left running in the background keeps reporting here.")
    stop_logs.set()
    server.stop()
    return code


def _choose_start_command(project: Path, force_menu: bool = False):
    from . import detect, tui

    cands = detect.detect(project)
    if not cands and not force_menu:
        termui.error(f"Couldn't work out how to start {project.name}. Give the command once and it is remembered:\n"
                     f"    visualagents run {project} -- python main.py")
        return None
    if not tui.interactive() and not force_menu:
        termui.info(f"Starting with: {cands[0].label()}  ({cands[0].source})  - use --pick to choose another")
        return cands[0]
    tui.heading("▶", f"How should {project.name} start?")
    shown = cands[:9]
    options = [(c.label(), c.source) for c in shown] + [("Type a command…", "")]
    idx = tui.select("", options)
    if idx < len(shown):
        return shown[idx]
    import shlex

    text = tui.ask("Command")
    where = tui.ask("Run it in which subfolder", ".")
    return detect.Candidate(shlex.split(text), where, "typed") if text else None


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


def _daemon_url(info: dict, project_id: str | None = None) -> str:
    return info["url"] + (f"/?project={project_id}" if project_id else "")


def _ensure_daemon(port: int) -> dict:
    """Background dashboard info, starting it if needed."""
    import subprocess
    import time

    from . import home

    info = home.read_daemon()
    if info:
        return info
    log = home.home() / "daemon.log"
    log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        subprocess.Popen([sys.executable, "-m", "visualagents", "up", "--foreground", "--no-browser", "--port", str(port)],
                         stdout=fh, stderr=fh, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        time.sleep(0.1)
        info = home.read_daemon()
        if info:
            return info
    raise SystemExit(termui.error(f"the background dashboard didn't start - see {log}") or 1)


def cmd_up(args, _command) -> int:
    from . import home
    from .projects import Projects

    info = home.read_daemon()
    if info and not args.foreground:
        termui.info(f"Dashboard already running at {info['url']}")
    elif not args.foreground:
        info = _ensure_daemon(args.port)
        termui.info(f"Dashboard running in the background at {info['url']}  (stop: visualagents down)")
    else:
        if info:
            termui.error(f"a dashboard is already running at {info['url']} (pid {info['pid']})")
            return 1
        from .runner import install_sigterm_handler

        projects = Projects(daemon=True)
        for pid, meta in home.registry().items():
            if Path(meta.get("path", "")).is_dir():
                try:
                    projects.open(Path(meta["path"]))
                except Exception as exc:  # a broken project must not take the daemon down
                    termui.error(f"skipping {meta['path']}: {exc}")
        server = Server(projects, host=args.host, port=args.port).start()
        if exposure_warning(args.host):
            termui.info(termui.warn(exposure_warning(args.host)))
        home.write_daemon(server.url, projects.token, os.getpid())
        termui.info(f"Dashboard for hooked projects at {server.url}  (Ctrl+C to stop)")
        install_sigterm_handler()
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            home.clear_daemon(os.getpid())
            server.stop()
        return 0
    if not args.no_browser and not os.environ.get("VISUALAGENTS_NO_BROWSER"):
        webbrowser.open(info["url"])
    return 0


def cmd_down(args, _command) -> int:
    import signal
    import time

    from . import home

    info = home.read_daemon()
    if not info:
        termui.info("No background dashboard is running.")
        return 0
    try:
        os.kill(int(info["pid"]), signal.SIGTERM)
    except ProcessLookupError:  # it stopped on its own a moment ago
        pass
    for _ in range(30):
        time.sleep(0.1)
        if not home.read_daemon():
            break
    home.clear_daemon(int(info["pid"]))
    termui.info("Stopped the background dashboard.")
    return 0


def cmd_status(args, _command) -> int:
    from . import home, hooks

    info = home.read_daemon()
    print(f"\n{termui.bold('Dashboard')}  " + (termui.good(f"running at {info['url']}") if info
                                                else termui.dim("not running  (visualagents up)")))
    projects = home.registry()
    print(f"\n{termui.bold('Hooked projects')} ({len(projects)})")
    for pid, meta in projects.items():
        print(f"  {termui.accent('●')} {termui.bold(pid)}  {meta.get('path')}")
        for site in meta.get("envs", []):
            ok = hooks.hooked_project(Path(site)) is not None
            print(f"      {termui.good('hooked') if ok else termui.bad('missing')}  {site}")
    if not projects:
        print(termui.dim("  none yet - visualagents hook <folder>"))
    print()
    return 0


def cmd_hook(args, _command) -> int:
    from . import detect, home, hooks

    project = args.project.resolve()
    if not project.is_dir():
        termui.error(f"{project} is not a folder")
        return 2
    pythons = [detect.venv_bin(v, "python") for v in detect.find_venvs(project)] + [p.expanduser() for p in args.python]
    pythons = [p for p in pythons if p.exists()]
    if not pythons:
        termui.error(f"No virtualenv found in {project}.\n"
                     f"  Create one:      python3 -m venv {project}/.venv && {project}/.venv/bin/pip install -r requirements.txt\n"
                     f"  or name another:  visualagents hook {args.project} --python /path/to/env/bin/python\n"
                     f"  or skip hooking:  visualagents run {args.project}")
        return 2
    projects = home.registry()
    pid = home.project_id(project, projects)
    envs = []
    for py in pythons:
        try:
            target = hooks.install(py, project)
        except Exception as exc:
            termui.error(f"could not hook {py}: {exc}")
            continue
        envs.append(str(target.parent))
        termui.info(f"Hooked {py}  ->  {target}")
    if not envs:
        return 1
    projects[pid] = {"path": str(project), "envs": sorted(set(envs) | set(projects.get(pid, {}).get("envs", [])))}
    home.save_registry(projects)
    if not args.no_scan and not bpmod.blueprint_path(project).exists():
        try:
            bpmod.save(_scan(project, args), bpmod.blueprint_path(project))
        except LLMError as exc:
            termui.error(f"scan failed ({exc}); the dashboard will use a quick static map")
    url = None
    if not args.no_daemon:
        info = _ensure_daemon(args.port)
        url = _daemon_url(info, pid)
        _notify_daemon(info, project)
    example = detect.load_saved(project) or next(iter(detect.detect(project)), None)
    print(f"""
{termui.good('✔')} {termui.bold(project.name)} is hooked. Start it however you normally do{' - for example:' if example else '.'}
{('    ' + example.label()) if example else ''}
  Every Python process from {'its virtualenv' if len(envs) == 1 else 'its virtualenvs'} appears on the dashboard once it calls an LLM.
  Dashboard: {url or 'visualagents up'}    Undo: visualagents unhook {args.project}
""", file=sys.stderr)
    if url and not args.no_browser and not os.environ.get("VISUALAGENTS_NO_BROWSER"):
        webbrowser.open(url)
    return 0


def _notify_daemon(info: dict, project: Path) -> None:
    """Let a running dashboard know about a newly hooked project right away."""
    import json
    import urllib.request

    req = urllib.request.Request(info["url"] + "/api/ingest", method="POST",
                                 data=json.dumps([{"type": "hello", "project": str(project)}]).encode(),
                                 headers={"Content-Type": "application/json", "X-VisualAgents-Token": info["token"]})
    try:
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=5).read()
    except Exception:
        pass


def cmd_unhook(args, _command) -> int:
    from . import detect, home, hooks

    project = args.project.resolve()
    projects = home.registry()
    pid = home.project_id(project, projects)
    sites = set(projects.get(pid, {}).get("envs", []))
    for v in detect.find_venvs(project):
        try:
            sites.add(str(hooks.site_packages(detect.venv_bin(v, "python"))))
        except Exception:
            pass
    removed = [s for s in sites if hooks.uninstall(Path(s))]
    projects.pop(pid, None)
    home.save_registry(projects)
    for s in removed:
        termui.info(f"Removed hook from {s}")
    if not removed:
        termui.info(f"{project.name} had no hooks installed.")
    return 0


def cmd_open(args, _command) -> int:
    from . import home

    project = args.project.resolve()
    info = _ensure_daemon(DAEMON_PORT)
    url = _daemon_url(info, home.project_id(project))
    termui.info(f"Dashboard: {url}")
    webbrowser.open(url)
    return 0


def cmd_setup(args, _command) -> int:
    from . import wizard

    return wizard.run(args)


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
    code = run_command(command, hub, cwd=demo_dir, server_url=server.url, instrument=False, label="visualagents demo")
    if not args.exit:
        _hold(server, "Demo finished.")
    server.stop()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
