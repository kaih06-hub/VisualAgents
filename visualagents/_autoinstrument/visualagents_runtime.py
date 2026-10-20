"""VisualAgents runtime - stdlib only, safe to import in any Python 3.8+ process.

`visualagents run` puts this directory on PYTHONPATH so frameworks are instrumented with no
code changes. You can also emit events yourself:

    from visualagents import emit, agent

    with agent("researcher", task="find sources"):
        emit("thought", "researcher", "Looking for 2024 benchmarks")
        emit("tool_start", "researcher", "solid-state batteries", tool="web_search")

Events travel in-band as `@@visualagents {json}` lines on stderr when stderr is the pipe VisualAgents is
reading (ordered exactly with printed output, hidden from the terminal). When it isn't - a start script sent
the app to the background or redirected its logs, or the app runs on its own under `visualagents hook` -
they are POSTed to the dashboard (VISUALAGENTS_URL / VISUALAGENTS_TOKEN, or the hook's daemon settings).
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import functools
import importlib.abc
import importlib.util
import inspect
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.request

MARKER = "@@visualagents "
_lock = threading.Lock()
_queue = None
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy


_config = {}  # set by configure(): url, token, project (hook mode)
_channel = {}  # pid -> True when events go in-band on stderr
_down_until = 0.0


def configure(url=None, token=None, project=None, daemon_file=None, lazy=False):
    """Send events to a dashboard over HTTP (used by `visualagents hook`).

    lazy: stay silent until the process does something agent-like (an LLM call or a framework event),
    so pip / pytest / language servers running in the same virtualenv never show up on the dashboard.
    """
    _config.update({k: v for k, v in (("url", url), ("token", token), ("project", project),
                                      ("daemon_file", daemon_file)) if v})
    _config["http"] = True
    _config["lazy"] = bool(lazy)
    _config.setdefault("armed", not lazy)
    _channel.clear()


def _url():
    url = _config.get("url") or os.environ.get("VISUALAGENTS_URL")
    if not url and _config.get("daemon_file"):
        _reload_daemon()
        url = _config.get("url")
    return url


def _reload_daemon():
    """The dashboard may start after this process: look for it again every few seconds."""
    now = time.time()
    if now - _config.get("daemon_checked", 0) < 5:
        return
    _config["daemon_checked"] = now
    try:
        with open(_config["daemon_file"], encoding="utf-8") as fh:
            info = json.load(fh)
        _config["url"], _config["token"] = info.get("url"), info.get("token")
    except Exception:
        pass


def _token():
    return _config.get("token") or os.environ.get("VISUALAGENTS_TOKEN", "")


def _in_band():
    """True when fd 2 is the very pipe/pty VisualAgents reads, so markers stay ordered with output."""
    pid = os.getpid()
    if pid not in _channel:
        if _config.get("http"):
            ok = False
        elif os.environ.get("VISUALAGENTS_STREAM_ID"):
            try:
                st = os.fstat(2)
                ok = "%d:%d" % (st.st_dev, st.st_ino) == os.environ["VISUALAGENTS_STREAM_ID"]
            except OSError:
                ok = False
        elif os.environ.get("VISUALAGENTS_CHANNEL") == "stderr":
            ok = True
        else:
            ok = not _url()
        _channel[pid] = ok
    return _channel[pid]


def _text(value, limit=4000):
    if value is None:
        return None
    if not isinstance(value, str):
        content = getattr(value, "content", None)
        if isinstance(content, str):
            value = content
        else:
            try:
                value = json.dumps(value, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                value = str(value)
    return value if len(value) <= limit else value[:limit] + "…"


def emit(type, agent=None, text=None, **data):
    """Send one event. type: agent_start, agent_end, thought, tool_start, tool_end, handoff, output, log, error."""
    if os.environ.get("VISUALAGENTS_DISABLE") == "1":
        return
    payload = {"type": type, "ts": time.time()}
    if agent is not None:
        payload["agent"] = str(agent)
    if text is not None:
        payload["text"] = _text(text)
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float)):
            payload[k] = v
        elif isinstance(v, (list, tuple)):
            payload[k] = [_text(x, 200) for x in v][:20]
        else:
            payload[k] = _text(v, 1000)
    if not _in_band():
        if _config.get("project"):
            payload["project"] = _config["project"]
        if not _config.get("armed", True):
            if type == "line":
                _pending.append(payload)  # remember recent output in case this process turns out to matter
                return
            _arm()
        if _url():
            _enqueue(payload)
        return
    line = MARKER + json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    try:
        # Straight to fd 2: keeps exact ordering with printed output and survives sys.stderr redirection.
        os.write(2, line.encode("utf-8", "replace"))
    except Exception:
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass


_pending = __import__("collections").deque(maxlen=300)


def _arm():
    """First agent-like event in a lazily hooked process: announce it, then replay its recent output."""
    _config["armed"] = True
    hello = {"type": "process_start", "ts": time.time(), "pid": os.getpid(),
             "text": " ".join(sys.argv) or sys.executable, "cwd": os.getcwd()}
    if _config.get("project"):
        hello["project"] = _config["project"]
    if _url():
        _enqueue(hello)
        for item in list(_pending):
            _enqueue(item)
    _pending.clear()
    atexit.register(_goodbye)


def _goodbye():
    emit("process_end", None, None, pid=os.getpid(), code=0)
    flush()


def _enqueue(payload):
    global _queue
    with _lock:
        if _queue is None:
            _queue = queue.Queue()
            threading.Thread(target=_sender, args=(_queue,), name="visualagents-emit", daemon=True).start()
    _queue.put(payload)


def _sender(q):
    while True:
        batch = [q.get()]
        try:
            while len(batch) < 200:
                batch.append(q.get_nowait())
        except queue.Empty:
            pass
        _post(batch)
        for _ in batch:
            q.task_done()


def _post(batch):
    global _down_until
    url = _url()
    if not url or time.time() < _down_until:
        return  # no dashboard listening right now: drop rather than slow the app down
    req = urllib.request.Request(
        url.rstrip("/") + "/api/ingest",
        data=json.dumps(batch, ensure_ascii=False, default=str).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-VisualAgents-Token": _token()},
        method="POST",
    )
    try:
        _opener.open(req, timeout=3).read()
    except Exception:
        _down_until = time.time() + 3.0


def flush(timeout=3.0):
    q = _queue
    if q is None:
        return
    deadline = time.time() + timeout
    while q.unfinished_tasks and time.time() < deadline:
        time.sleep(0.02)


atexit.register(flush)
if hasattr(os, "register_at_fork"):
    def _after_fork():
        global _queue
        _queue = None
        _channel.clear()
        if _config.get("lazy"):
            _config["armed"] = False  # a forked worker announces itself when it does agent work
            _pending.clear()

    os.register_at_fork(after_in_child=_after_fork)


# -- output capture (hook mode: the dashboard can't read this process's terminal) ----------

class _Tee:
    """Writes through to the real stream and forwards complete lines to the dashboard."""

    def __init__(self, stream, name):
        self._stream = stream
        self._name = name
        self._buf = ""
        self._busy = False

    def write(self, s):
        n = self._stream.write(s)
        if isinstance(s, str) and s and not self._busy:
            self._busy = True
            try:
                self._buf += s
                if "\n" in self._buf:
                    *lines, self._buf = self._buf.split("\n")
                    for line in lines:
                        if line.strip() and MARKER.strip() not in line:
                            emit("line", None, line, stream=self._name)
                if len(self._buf) > 8000:
                    self._buf = ""
            finally:
                self._busy = False
        return n

    def __getattr__(self, name):
        return getattr(self._stream, name)


def capture_output():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and not isinstance(stream, _Tee):
            setattr(sys, name, _Tee(stream, name))


# -- convenience API --------------------------------------------------------

def agent_start(name, task=None):
    emit("agent_start", name, task)


def agent_end(name, output=None):
    emit("agent_end", name, output)


def thought(name, text):
    emit("thought", name, text)


def tool_call(name, tool, input=None):
    emit("tool_start", name, input, tool=tool)


def tool_result(name, tool, output=None):
    emit("tool_end", name, output, tool=tool)


def handoff(src, dst, reason=None):
    emit("handoff", None, reason, **{"from": src, "to": dst})


def log(name, text):
    emit("log", name, text)


@contextlib.contextmanager
def agent(name, task=None):
    """Mark a block of code as one agent's turn."""
    agent_start(name, task)
    try:
        yield
    except BaseException as exc:
        emit("error", name, "%s: %s" % (type(exc).__name__, exc))
        raise
    finally:
        agent_end(name)


def track(name=None):
    """Decorator: the wrapped function (sync or async) is an agent's turn."""
    def deco(fn):
        label = name or fn.__name__
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a, **kw):
                with agent(label):
                    return await fn(*a, **kw)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with agent(label):
                return fn(*a, **kw)
        return wrapper

    return deco(name) if callable(name) else deco


# -- LangChain / LangGraph ------------------------------------------------------

def _message_text(msg):
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text") or c.get("thinking") or "")
            else:
                parts.append(str(c))
        content = "\n".join(p for p in parts if p)
    return content if isinstance(content, str) else ""


def langchain_handler():
    """A LangChain callback handler that reports LangGraph nodes as agents."""
    from langchain_core.callbacks.base import BaseCallbackHandler

    class VisualAgentsCallbackHandler(BaseCallbackHandler):
        raise_error = False
        run_inline = True

        def __init__(self):
            self._node_of = {}
            self._node_runs = {}
            self._soft_runs = {}
            self._tools = {}

        def _node(self, run_id, parent_run_id, metadata):
            md = metadata or {}
            ns = md.get("langgraph_checkpoint_ns") or md.get("checkpoint_ns") or ""
            node = ns.split("|")[0].split(":")[0] if ns else None
            node = node or md.get("langgraph_node")
            if node in (None, "", "__start__", "__end__"):
                node = self._node_of.get(parent_run_id)
            if node:
                self._node_of[run_id] = node
            return node

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kw):
            try:
                md = metadata or {}
                name = kw.get("name") or (serialized or {}).get("name") or ""
                node = self._node(run_id, parent_run_id, md)
                ns = md.get("langgraph_checkpoint_ns") or ""
                if node and name == node and "|" not in ns:
                    self._node_runs[run_id] = node
                    emit("agent_start", node)
                elif not md.get("langgraph_node") and parent_run_id is None and name:
                    self._soft_runs[run_id] = name
                    emit("agent_start", name, soft=True)
            except Exception:
                pass

        def on_chain_end(self, outputs, *, run_id, **kw):
            try:
                if run_id in self._node_runs:
                    emit("agent_end", self._node_runs.pop(run_id))
                elif run_id in self._soft_runs:
                    emit("agent_end", self._soft_runs.pop(run_id), soft=True)
            except Exception:
                pass

        def on_chain_error(self, error, *, run_id, **kw):
            node = self._node_runs.pop(run_id, None) or self._node_of.get(run_id)
            emit("error", node, "%s: %s" % (type(error).__name__, error))
            if node and run_id not in self._soft_runs:
                emit("agent_end", node)

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kw):
            self._node(run_id, parent_run_id, metadata)

        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, metadata=None, **kw):
            self._node(run_id, parent_run_id, metadata)

        def on_llm_end(self, response, *, run_id, **kw):
            try:
                node = self._node_of.get(run_id)
                gen = response.generations[0][0] if response.generations and response.generations[0] else None
                if gen is None:
                    return
                msg = getattr(gen, "message", None)
                text = _message_text(msg) if msg is not None else (gen.text or "")
                calls = [c.get("name") for c in (getattr(msg, "tool_calls", None) or []) if isinstance(c, dict)]
                if text.strip():
                    emit("thought", node, text.strip())
                elif calls:
                    emit("thought", node, "Decided to call " + ", ".join(c for c in calls if c))
            except Exception:
                pass

        def on_llm_error(self, error, *, run_id, **kw):
            emit("error", self._node_of.get(run_id), "%s: %s" % (type(error).__name__, error))

        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, metadata=None, **kw):
            try:
                node = self._node(run_id, parent_run_id, metadata)
                name = (serialized or {}).get("name") or kw.get("name") or "tool"
                self._tools[run_id] = name
                emit("tool_start", node, input_str, tool=name)
            except Exception:
                pass

        def on_tool_end(self, output, *, run_id, **kw):
            emit("tool_end", self._node_of.get(run_id), _message_text(output) or _text(output), tool=self._tools.pop(run_id, None))

        def on_tool_error(self, error, *, run_id, **kw):
            emit("error", self._node_of.get(run_id), "%s failed: %s" % (self._tools.pop(run_id, "tool"), error))

        def on_agent_action(self, action, *, run_id, **kw):
            node = self._node_of.get(run_id)
            log_text = (getattr(action, "log", "") or "").strip()
            emit("thought", node, log_text or "Action: %s" % getattr(action, "tool", "?"))

        def on_agent_finish(self, finish, *, run_id, **kw):
            node = self._node_of.get(run_id)
            values = getattr(finish, "return_values", {}) or {}
            emit("output", node, values.get("output") or getattr(finish, "log", ""))

    return VisualAgentsCallbackHandler()


def _hook_langchain(_module=None):
    from langchain_core.tracers import context as ctx

    if getattr(ctx, "_visualagents_hooked", False):
        return
    from contextvars import ContextVar

    ctx.register_configure_hook(ContextVar("visualagents_handler", default=langchain_handler()), True)
    ctx._visualagents_hooked = True


# -- OpenAI Agents SDK ------------------------------------------------------------

def openai_agents_processor():
    from agents.tracing import TracingProcessor

    class VisualAgentsProcessor(TracingProcessor):
        def __init__(self):
            self._agent_of = {}

        def on_trace_start(self, trace):
            pass

        def on_trace_end(self, trace):
            pass

        def on_span_start(self, span):
            try:
                sd = span.span_data
                typ = getattr(sd, "type", "")
                if typ == "agent":
                    self._agent_of[span.span_id] = sd.name
                    emit("agent_start", sd.name)
                    return
                agent_name = self._agent_of.get(span.parent_id)
                self._agent_of[span.span_id] = agent_name
                if typ == "function":
                    emit("tool_start", agent_name, getattr(sd, "input", None), tool=getattr(sd, "name", "tool"))
            except Exception:
                pass

        def on_span_end(self, span):
            try:
                sd = span.span_data
                typ = getattr(sd, "type", "")
                agent_name = self._agent_of.pop(span.span_id, None)
                if typ == "agent":
                    emit("agent_end", sd.name)
                elif typ == "function":
                    emit("tool_end", agent_name, getattr(sd, "output", None), tool=getattr(sd, "name", "tool"))
                elif typ == "handoff":
                    emit("handoff", None, None, **{"from": getattr(sd, "from_agent", None), "to": getattr(sd, "to_agent", None)})
                elif typ == "response":
                    text = getattr(getattr(sd, "response", None), "output_text", None)
                    if text:
                        emit("thought", agent_name, text)
                elif typ == "generation":
                    out = getattr(sd, "output", None) or []
                    texts = [m.get("content") for m in out if isinstance(m, dict) and isinstance(m.get("content"), str)]
                    if texts:
                        emit("thought", agent_name, "\n".join(texts))
                err = getattr(span, "error", None)
                if err:
                    emit("error", agent_name, err.get("message") if isinstance(err, dict) else str(err))
            except Exception:
                pass

        def shutdown(self):
            flush()

        def force_flush(self):
            flush()

    return VisualAgentsProcessor()


def _hook_openai_agents(module):
    # "agents" is also a common name for a project's own package - only hook the real SDK.
    if not hasattr(module, "add_trace_processor") or getattr(module, "_visualagents_hooked", False):
        return
    module.add_trace_processor(openai_agents_processor())
    module._visualagents_hooked = True


# -- raw OpenAI SDK (apps that call chat.completions.create themselves, e.g. LAMBDA) ------

# Frameworks that already report their own agents (instrumented above, or printing their own logs).
_FRAMEWORK_MARKERS = ("langchain", "langgraph", "crewai", "litellm", "autogen", "llama_index", "smolagents",
                      os.sep + "agents" + os.sep, "dspy", "haystack", "semantic_kernel")
_LIB_MARKERS = ("site-packages", "dist-packages")
_HERE = os.path.dirname(os.path.abspath(__file__))
_last_task = {}


def _stdlib_dirs():
    import sysconfig

    paths = sysconfig.get_paths()
    # os.__file__ is the real standard library even inside a virtualenv
    return tuple(p for p in {os.path.dirname(os.__file__), paths.get("stdlib"), paths.get("platstdlib")} if p)


_STDLIB = _stdlib_dirs()


def _is_library(filename):
    if not filename or filename.startswith("<"):
        return True
    return (any(m in filename for m in _LIB_MARKERS) or filename.startswith(_STDLIB)
            or os.path.dirname(os.path.abspath(filename)) == _HERE)


def _caller_stack():
    """Names of the user code on the stack, innermost first: a method's class (Programmer, Inspector, ...)
    followed by Class.method (a shared LLMService serves several agents), a function's name, or the script's
    file name for top-level code. False when a framework that reports its own agents is on the stack."""
    frame = sys._getframe(1)
    names = []
    while frame is not None:
        filename = frame.f_code.co_filename or ""
        if any(m in filename for m in _FRAMEWORK_MARKERS) and _is_library(filename):
            return False
        if not _is_library(filename):
            code = frame.f_code.co_name
            self_obj = frame.f_locals.get("self")
            found = []
            if self_obj is not None:
                found = [type(self_obj).__name__]
                if not code.startswith("<"):
                    found.append(f"{found[0]}.{code}")
            elif code == "<module>":
                found = [os.path.splitext(os.path.basename(filename))[0]]
            elif not code.startswith("<"):
                found = [code]
            names += [n for n in found if n not in names]
        frame = frame.f_back
    return names[:16]


def _caller_agent():
    """Name of the code that made the LLM call (see _caller_stack); False under a self-reporting framework."""
    stack = _caller_stack()
    if stack is False:
        return False
    return stack[0] if stack else None


def _ensure_armed():
    """A framework that prints its own agents just called a model: make sure its output reaches the dashboard."""
    if not _config.get("armed", True) and not _in_band():
        _arm()


_busy = contextvars.ContextVar("visualagents_llm_call", default=False)  # inside a model call we already report


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class _Call:
    """One model call, reported as the code that made it taking a turn: what it asked, what came back."""

    def __init__(self):
        self.finished = False
        self.content, self.reasoning, self.tools = [], [], {}
        busy = _busy.get()
        stack = False if busy else _caller_stack()
        self.off = stack is False
        if self.off:
            if not busy:
                _ensure_armed()
            return
        self.agent = stack[0] if stack else "LLM"
        self.hints = stack[1:] or None  # outer callers, so the dashboard can match whichever is on the map

    def emit(self, typ, text=None, **data):
        emit(typ, self.agent, text, hint=True, hints=self.hints, **data)

    def start(self, task=None, model=None):
        if self.off:
            return self
        if task and _last_task.get(self.agent) == task:
            task = None  # the same question again (a tool loop): don't repeat it
        elif task:
            _last_task[self.agent] = task
        self.emit("agent_start", task, model=model)
        return self

    def add(self, content=None, reasoning=None):
        if isinstance(content, str) and content:
            self.content.append(content)
        if isinstance(reasoning, str) and reasoning:
            self.reasoning.append(reasoning)

    def tool(self, key, name=None, args=None):
        slot = self.tools.setdefault(key, ["", ""])
        slot[0] = slot[0] or name or ""
        if args:
            slot[1] += args if isinstance(args, str) else json.dumps(args, default=str, ensure_ascii=False)

    def finish(self, error=None):
        if self.off or self.finished:
            return
        self.finished = True
        text = "".join(self.content).strip()
        reasoning = "".join(self.reasoning).strip()
        if not reasoning and "</think>" in text:  # Qwen, DeepSeek-R1...: reasoning inline in <think> tags
            head, _, text = text.partition("</think>")
            reasoning, text = head.replace("<think>", "").strip(), text.strip()
        if reasoning:
            self.emit("thought", reasoning)
        if text:
            self.emit("thought" if self.tools else "output", text)
        for name, args in self.tools.values():
            self.emit("tool_start", args, tool=name or "tool")
        if error is not None:
            self.emit("error", "%s: %s" % (type(error).__name__, error))
        self.emit("agent_end")


class _LLMCall(_Call):
    """An OpenAI-style chat completion (openai, groq, litellm, llama.cpp, huggingface_hub)."""

    def __init__(self, kwargs):
        super().__init__()
        if self.off:
            return
        messages = list(kwargs.get("messages") or [])
        # results of the tools requested last turn arrive as trailing tool/function messages
        results = []
        for m in reversed(messages):
            if _get(m, "role") in ("tool", "function"):
                results.append(m)
            else:
                break
        for m in reversed(results):
            self.emit("tool_end", _content_text(_get(m, "content")), tool=_get(m, "name"))
        self.start(_last_user(messages), kwargs.get("model"))

    def absorb_message(self, msg):
        if msg is None:
            return
        self.add(_get(msg, "content"), _get(msg, "reasoning_content") or _get(msg, "reasoning"))
        for i, tc in enumerate(_get(msg, "tool_calls") or []):
            fn = _get(tc, "function")
            self.tool(i, _get(fn, "name") or "tool", _get(fn, "arguments") or "")
        fc = _get(msg, "function_call")
        if fc is not None:
            self.tool("legacy", _get(fc, "name") or "function", _get(fc, "arguments") or "")

    def absorb_chunk(self, chunk):
        try:
            choices = _get(chunk, "choices") or []
            if not choices:
                return
            delta = _get(choices[0], "delta")
            if delta is None:
                return
            self.add(_get(delta, "content"), _get(delta, "reasoning_content") or _get(delta, "reasoning"))
            for tc in _get(delta, "tool_calls") or []:
                fn = _get(tc, "function")
                self.tool(_get(tc, "index", 0), _get(fn, "name") if fn is not None else None,
                          _get(fn, "arguments") if fn is not None else None)
            fc = _get(delta, "function_call")
            if fc is not None:
                self.tool("legacy", _get(fc, "name"), _get(fc, "arguments"))
        except Exception:
            pass


def _last_user(messages):
    if not isinstance(messages, (list, tuple)):
        return None
    last_user = next((m for m in reversed(list(messages)) if _get(m, "role") == "user"), None)
    return _content_text(_get(last_user, "content")) if last_user is not None else None


def _content_text(content):
    if isinstance(content, list):
        parts = [_get(p, "text") for p in content]
        content = "\n".join(p for p in parts if isinstance(p, str))
    return content if isinstance(content, str) else (None if content is None else str(content))


class _StreamProxy:
    """Passes a streamed completion through untouched while collecting what the model said."""

    def __init__(self, stream, call, feed=None):
        self._stream = stream
        self._call = call
        self._feed = feed
        self._it = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = iter(self._stream)
        token = _busy.set(True)
        try:
            chunk = next(self._it)
        except StopIteration:
            self._call.finish()
            raise
        except BaseException as exc:
            self._call.finish(exc)
            raise
        finally:
            _busy.reset(token)
        self._absorb(chunk)
        return chunk

    def _absorb(self, chunk):
        try:
            if self._feed is not None:
                self._feed(self._call, chunk)
            else:
                self._call.absorb_chunk(chunk)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._call.finish()
        close = getattr(self._stream, "close", None)
        if close:
            close()
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __del__(self):
        try:
            self._call.finish()
        except Exception:
            pass


class _AsyncStreamProxy(_StreamProxy):
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._it is None:
            self._it = self._stream.__aiter__()
        token = _busy.set(True)
        try:
            chunk = await self._it.__anext__()
        except StopAsyncIteration:
            self._call.finish()
            raise
        except BaseException as exc:
            self._call.finish(exc)
            raise
        finally:
            _busy.reset(token)
        self._absorb(chunk)
        return chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._call.finish()
        close = getattr(self._stream, "close", None)
        if close:
            result = close()
            if inspect.isawaitable(result):
                await result
        return False


def _finish_response(call, result, streaming, is_async):
    if call.off:
        return result
    if streaming:
        return _AsyncStreamProxy(result, call) if is_async else _StreamProxy(result, call)
    try:
        choices = _get(result, "choices") or []
        call.absorb_message(_get(choices[0], "message") if choices else None)
    finally:
        call.finish()
    return result


def _hook_openai(_module=None):
    from openai.resources.chat import completions as cc

    for name, is_async in (("Completions", False), ("AsyncCompletions", True)):
        cls = getattr(cc, name, None)
        if cls is None or getattr(cls.create, "_visualagents", False):
            continue
        original = cls.create
        if is_async:
            async def create(self, *args, _original=original, **kwargs):
                try:
                    call = _LLMCall(kwargs)
                except Exception:
                    return await _original(self, *args, **kwargs)
                try:
                    result = await _original(self, *args, **kwargs)
                except BaseException as exc:
                    call.finish(exc)
                    raise
                return _finish_response(call, result, bool(kwargs.get("stream")), True)
        else:
            def create(self, *args, _original=original, **kwargs):
                try:
                    call = _LLMCall(kwargs)
                except Exception:
                    return _original(self, *args, **kwargs)
                try:
                    result = _original(self, *args, **kwargs)
                except BaseException as exc:
                    call.finish(exc)
                    raise
                return _finish_response(call, result, bool(kwargs.get("stream")), False)
        functools.update_wrapper(create, original)
        create._visualagents = True
        cls.create = create


# -- any other LLM library: each call is the calling code taking a turn ---------------------

def _patch(owner, attr, begin, on_result=None, on_chunk=None, streams=None, manager=False):
    """Wrap a library function or method so every call is reported. begin(args, kwargs) -> _Call; on_result
    reads a full response, on_chunk one streamed piece; streams(args, kwargs, result) says it is a stream."""
    original = getattr(owner, attr, None)
    if original is None or getattr(original, "_visualagents", False):
        return

    def _begin(args, kwargs):
        if _busy.get():
            return None  # made by a call we already report (retries, wrappers, generate -> stream_generate)
        try:
            call = begin(args, kwargs)
        except Exception:
            return None
        return None if call is None or call.off else call

    def _after(call, result, args, kwargs):
        try:
            if manager:
                return _ManagerProxy(result, call, on_result)
            streaming = streams(args, kwargs, result) if streams else (
                inspect.isgenerator(result) or inspect.isasyncgen(result))
            if streaming and on_chunk is not None:
                proxy = _AsyncStreamProxy if hasattr(result, "__aiter__") else _StreamProxy
                return proxy(result, call, on_chunk)
            if on_result is not None:
                on_result(call, result)
        except Exception:
            pass
        call.finish()
        return result

    async def _awaited(call, awaitable, args, kwargs):
        token = _busy.set(True)
        try:
            result = await awaitable
        except BaseException as exc:
            call.finish(exc)
            raise
        finally:
            _busy.reset(token)
        return _after(call, result, args, kwargs)

    if inspect.iscoroutinefunction(original):
        async def wrapper(*args, **kwargs):
            call = _begin(args, kwargs)
            if call is None:
                return await original(*args, **kwargs)
            return await _awaited(call, original(*args, **kwargs), args, kwargs)
    else:
        def wrapper(*args, **kwargs):
            call = _begin(args, kwargs)
            if call is None:
                return original(*args, **kwargs)
            token = _busy.set(True)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                call.finish(exc)
                raise
            finally:
                _busy.reset(token)
            if inspect.isawaitable(result):  # an async method behind a sync decorator
                return _awaited(call, result, args, kwargs)
            return _after(call, result, args, kwargs)
    try:
        functools.update_wrapper(wrapper, original)
    except Exception:
        pass
    wrapper._visualagents = True
    setattr(owner, attr, wrapper)


class _ManagerProxy:
    """`with client.messages.stream(...) as stream:` - the reply is reported when the block ends."""

    def __init__(self, manager, call, on_result):
        self._m, self._call, self._on_result, self._s = manager, call, on_result, None

    def __enter__(self):
        self._s = self._m.__enter__()
        return self._s

    def __exit__(self, *exc):
        try:
            return self._m.__exit__(*exc)
        finally:
            self._done(exc[1])

    async def __aenter__(self):
        self._s = await self._m.__aenter__()
        return self._s

    async def __aexit__(self, *exc):
        try:
            return await self._m.__aexit__(*exc)
        finally:
            self._done(exc[1])

    def _done(self, error):
        try:
            snapshot = getattr(self._s, "current_message_snapshot", None)
            if snapshot is not None and self._on_result:
                self._on_result(self._call, snapshot)
        except Exception:
            pass
        self._call.finish(error)

    def __getattr__(self, name):
        return getattr(self._m, name)


def _stream_kwarg(args, kwargs, result):
    return bool(kwargs.get("stream"))


def _always(args, kwargs, result):
    return True


def _arg(args, kwargs, name, index):
    """A parameter passed by name or position (index counts `self` for methods)."""
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


def _openai_shaped(args, kwargs, messages_index=1):
    """litellm.completion(model, messages), Llama.create_chat_completion(self, messages), ..."""
    kw = dict(kwargs)
    if "messages" not in kw and len(args) > messages_index:
        kw["messages"] = args[messages_index]
    return _LLMCall(kw)


def _openai_result(call, result):
    choices = _get(result, "choices") or []
    call.absorb_message(_get(choices[0], "message") if choices else None)


def _openai_chunk(call, chunk):
    call.absorb_chunk(chunk)


_TURN = re.compile(r"<\|im_start\|>user|<\|start_header_id\|>user<\|end_header_id\|>|\[INST\]|<start_of_turn>user|"
                   r"<\|user\|>|^### (?:User|Instruction|Human):|^(?:User|Human):", re.I | re.M)


def _prompt_text(prompt):
    """The last user turn of a prompt already rendered with a chat template (local models)."""
    if not isinstance(prompt, str):
        return None
    turns = list(_TURN.finditer(prompt))
    text = prompt[turns[-1].end():] if turns else prompt
    text = re.sub(r"<\|[^|>]{1,40}\|>|</?s>|\[/INST\]|<end_of_turn>|<start_of_turn>model|<think>\s*</think>", " ", text)
    text = re.sub(r"\s*\b(assistant|model)\s*$", "", text.strip(), flags=re.I)
    return text.strip()[:4000] or None


def _simple_call(task=None, model=None):
    call = _Call()
    return call if call.off else call.start(task, model)


# Anthropic: messages.create (plain or stream=True) and messages.stream(...)
def _anthropic_begin(args, kwargs):
    call = _Call()
    if call.off:
        return call
    messages = list(kwargs.get("messages") or [])
    last = messages[-1] if messages else None
    if last is not None and _get(last, "role") == "user" and isinstance(_get(last, "content"), list):
        for block in _get(last, "content"):
            if _get(block, "type") == "tool_result":
                call.emit("tool_end", _content_text(_get(block, "content")))
    return call.start(_last_user(messages), kwargs.get("model"))


def _anthropic_result(call, message):
    for i, block in enumerate(_get(message, "content") or []):
        kind = _get(block, "type")
        if kind == "text":
            call.add(_get(block, "text"))
        elif kind == "thinking":
            call.add(reasoning=_get(block, "thinking"))
        elif kind in ("tool_use", "server_tool_use", "mcp_tool_use"):
            call.tool(i, _get(block, "name"), _get(block, "input"))


def _anthropic_chunk(call, event):
    kind = _get(event, "type")
    if kind == "content_block_start":
        block = _get(event, "content_block")
        if _get(block, "type") in ("tool_use", "server_tool_use", "mcp_tool_use"):
            call.tool(_get(event, "index", 0), _get(block, "name"))
    elif kind == "content_block_delta":
        delta = _get(event, "delta")
        dkind = _get(delta, "type")
        if dkind == "text_delta":
            call.add(_get(delta, "text"))
        elif dkind == "thinking_delta":
            call.add(reasoning=_get(delta, "thinking"))
        elif dkind == "input_json_delta":
            call.tool(_get(event, "index", 0), None, _get(delta, "partial_json"))


def _hook_anthropic(_module=None):
    for modname in ("anthropic.resources.messages", "anthropic.resources.beta.messages"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for cls_name in ("Messages", "AsyncMessages"):
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                _patch(cls, "create", _anthropic_begin, _anthropic_result, _anthropic_chunk, streams=_stream_kwarg)
                _patch(cls, "stream", _anthropic_begin, _anthropic_result, manager=True)


# Google Gemini: google-genai (client.models.generate_content[_stream]) and the older google-generativeai
def _genai_text(contents):
    if isinstance(contents, str):
        return contents
    if isinstance(contents, (list, tuple)):
        if not contents:
            return None
        if all(isinstance(c, str) for c in contents):
            return "\n".join(contents)
        contents = contents[-1]  # the latest turn
    if isinstance(contents, str):
        return contents
    texts = [_get(p, "text") for p in (_get(contents, "parts") or [])]
    return "\n".join(t for t in texts if isinstance(t, str)) or None


def _genai_begin(args, kwargs):
    return _simple_call(_genai_text(kwargs.get("contents")), kwargs.get("model"))


def _genai_legacy_begin(args, kwargs):
    return _simple_call(_genai_text(_arg(args, kwargs, "contents", 1)), _get(args[0], "model_name") if args else None)


def _genai_result(call, response):
    for candidate in (_get(response, "candidates") or [])[:1]:
        for part in _get(_get(candidate, "content"), "parts") or []:
            text = _get(part, "text")
            if isinstance(text, str) and text:
                call.add(None, text) if _get(part, "thought") else call.add(text)
            fc = _get(part, "function_call")
            if fc is not None:
                call.tool(("fc", len(call.tools)), _get(fc, "name"), _get(fc, "args"))


def _hook_genai(_module=None):
    mod = importlib.import_module("google.genai.models")
    for cls_name in ("Models", "AsyncModels"):
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            _patch(cls, "generate_content", _genai_begin, _genai_result)
            _patch(cls, "generate_content_stream", _genai_begin, None, _genai_result, streams=_always)


def _hook_genai_legacy(module):
    cls = getattr(module, "GenerativeModel", None)
    if cls is not None:
        _patch(cls, "generate_content", _genai_legacy_begin, _genai_result, _genai_result, streams=_stream_kwarg)
        _patch(cls, "generate_content_async", _genai_legacy_begin, _genai_result, _genai_result, streams=_stream_kwarg)


# LiteLLM: litellm.completion / acompletion (OpenAI-shaped responses)
def _hook_litellm(module):
    for name in ("completion", "acompletion"):
        _patch(module, name, _openai_shaped, _openai_result, _openai_chunk, streams=_stream_kwarg)


# Ollama: ollama.chat / ollama.generate and Client / AsyncClient
def _ollama_result(call, response):
    message = _get(response, "message")
    if message is not None:
        call.add(_get(message, "content"), _get(message, "thinking"))
        for tc in _get(message, "tool_calls") or []:
            fn = _get(tc, "function")
            call.tool(("t", len(call.tools)), _get(fn, "name"), _get(fn, "arguments"))
    else:
        call.add(_get(response, "response"), _get(response, "thinking"))


def _hook_ollama(module):
    client_mod = sys.modules.get("ollama._client") or importlib.import_module("ollama._client")
    chat = lambda a, k: _simple_call(_last_user(_arg(a, k, "messages", 2)), _arg(a, k, "model", 1))  # noqa: E731
    generate = lambda a, k: _simple_call(_arg(a, k, "prompt", 2), _arg(a, k, "model", 1))  # noqa: E731
    for cls_name in ("Client", "AsyncClient"):
        cls = getattr(client_mod, cls_name, None)
        if cls is not None:
            _patch(cls, "chat", chat, _ollama_result, _ollama_result, streams=_stream_kwarg)
            _patch(cls, "generate", generate, _ollama_result, _ollama_result, streams=_stream_kwarg)
    default = getattr(module, "_client", None)
    if default is not None:  # ollama.chat / ollama.generate are bound methods of a default client
        for name in ("chat", "generate"):
            if hasattr(module, name):
                setattr(module, name, getattr(default, name))


# mlx-lm (Apple silicon): generate(model, tokenizer, prompt) and stream_generate(...)
def _hook_mlx_lm(module):
    begin = lambda a, k: _simple_call(_prompt_text(_arg(a, k, "prompt", 2)))  # noqa: E731
    for owner in (module, sys.modules.get("mlx_lm.generate")):
        if inspect.ismodule(owner):
            _patch(owner, "generate", begin, lambda call, text: call.add(text if isinstance(text, str) else None))
            _patch(owner, "stream_generate", begin, None, lambda call, r: call.add(_get(r, "text")), streams=_always)


# llama.cpp: Llama.create_chat_completion / create_completion (and llm(prompt))
def _completion_text(call, response):
    choices = _get(response, "choices") or []
    call.add(_get(choices[0], "text") if choices else None)


def _hook_llama_cpp(module):
    cls = getattr(module, "Llama", None)
    if cls is None:
        return
    _patch(cls, "create_chat_completion", _openai_shaped, _openai_result, _openai_chunk, streams=_stream_kwarg)
    _patch(cls, "create_completion", lambda a, k: _simple_call(_arg(a, k, "prompt", 1)),
           _completion_text, _completion_text, streams=_stream_kwarg)


# transformers: text-generation pipelines, and model.generate for everything else
def _tf_pipe_begin(args, kwargs):
    inputs = _arg(args, kwargs, "text_inputs", 1)
    if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], dict):
        return _simple_call(_last_user(inputs))
    return _simple_call(inputs if isinstance(inputs, str) else None)


def _tf_pipe_result(call, out):
    item = out[0] if isinstance(out, list) and out else out
    if isinstance(item, list) and item:
        item = item[0]
    text = _get(item, "generated_text")
    if isinstance(text, list) and text:
        text = _get(text[-1], "content")
    call.add(text if isinstance(text, str) else None)


def _hook_tf_pipeline(module):
    cls = getattr(module, "TextGenerationPipeline", None)
    if cls is not None:
        _patch(cls, "__call__", _tf_pipe_begin, _tf_pipe_result)


def _hook_tf_generate(module):
    cls = getattr(module, "GenerationMixin", None)
    if cls is not None:
        _patch(cls, "generate", lambda a, k: _simple_call())


# vLLM: LLM.generate / LLM.chat
def _vllm_begin(args, kwargs):
    p = _arg(args, kwargs, "prompts", 1)
    if p is None:
        p = kwargs.get("messages")
    if isinstance(p, (list, tuple)) and p and isinstance(p[0], dict):
        return _simple_call(_last_user(p))
    if isinstance(p, (list, tuple)) and p and isinstance(p[0], str):
        return _simple_call(p[0] if len(p) == 1 else "%d prompts" % len(p))
    return _simple_call(p if isinstance(p, str) else None)


def _vllm_result(call, outputs):
    for out in list(outputs or [])[:1]:
        for completion in list(_get(out, "outputs") or [])[:1]:
            call.add(_get(completion, "text"))


def _hook_vllm(module):
    cls = getattr(module, "LLM", None)
    if cls is not None:
        for name in ("generate", "chat"):
            _patch(cls, name, _vllm_begin, _vllm_result)


# OpenAI-compatible SDKs that are not the openai package
def _hook_groq(_module=None):
    mod = importlib.import_module("groq.resources.chat.completions")
    for cls_name in ("Completions", "AsyncCompletions"):
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            _patch(cls, "create", lambda a, k: _LLMCall(k), _openai_result, _openai_chunk, streams=_stream_kwarg)


def _hook_hf_inference(module):
    for cls_name in ("InferenceClient", "AsyncInferenceClient"):
        cls = getattr(module, cls_name, None)
        if cls is not None:
            _patch(cls, "chat_completion", _openai_shaped, _openai_result, _openai_chunk, streams=_stream_kwarg)


# -- import hooks -----------------------------------------------------------------

_HOOKS = {
    "langchain_core.callbacks.manager": _hook_langchain, "agents": _hook_openai_agents, "openai": _hook_openai,
    "anthropic": _hook_anthropic, "google.genai": _hook_genai, "google.generativeai": _hook_genai_legacy,
    "litellm": _hook_litellm, "ollama": _hook_ollama, "mlx_lm": _hook_mlx_lm, "llama_cpp": _hook_llama_cpp,
    "transformers.pipelines.text_generation": _hook_tf_pipeline, "transformers.generation.utils": _hook_tf_generate,
    "vllm": _hook_vllm, "groq": _hook_groq, "huggingface_hub.inference._client": _hook_hf_inference,
    "huggingface_hub.inference._generated._async_client": _hook_hf_inference,
}


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._busy = set()

    def find_spec(self, name, path=None, target=None):
        if name not in _HOOKS or name in self._busy:
            return None
        self._busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        finally:
            self._busy.discard(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module):
            original(module)
            try:
                _HOOKS[name](module)
            except Exception:
                pass

        loader.exec_module = exec_module
        return spec


def instrument():
    """Hook supported frameworks: immediately if already imported, otherwise when they are imported."""
    for name, hook in _HOOKS.items():
        mod = sys.modules.get(name)
        if mod is not None:
            try:
                hook(mod)
            except Exception:
                pass
    if not any(isinstance(f, _PostImportFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportFinder())
