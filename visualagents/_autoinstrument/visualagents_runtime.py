"""VisualAgents runtime - stdlib only, safe to import in any Python 3.8+ process.

`visualagents run` puts this directory on PYTHONPATH so frameworks are instrumented with no
code changes. You can also emit events yourself:

    from visualagents import emit, agent

    with agent("researcher", task="find sources"):
        emit("thought", "researcher", "Looking for 2024 benchmarks")
        emit("tool_start", "researcher", "solid-state batteries", tool="web_search")

Events are written to stderr as `@@visualagents {json}` lines, which `visualagents run/pipe/watch` parse
(and hide from the terminal). Set VISUALAGENTS_URL (and VISUALAGENTS_TOKEN) to POST them to a running
dashboard instead, e.g. from a process whose output VisualAgents does not see.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import importlib.abc
import importlib.util
import inspect
import json
import os
import queue
import sys
import threading
import time
import urllib.request

MARKER = "@@visualagents "
_lock = threading.Lock()
_queue = None
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy


def _url():
    return os.environ.get("VISUALAGENTS_URL")


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
        if v is not None:
            payload[k] = v if isinstance(v, (bool, int, float)) else _text(v, 1000)
    if _url() and os.environ.get("VISUALAGENTS_CHANNEL") != "stderr":
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
    url = _url()
    if not url:
        return
    req = urllib.request.Request(
        url.rstrip("/") + "/api/ingest",
        data=json.dumps(batch, ensure_ascii=False, default=str).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-VisualAgents-Token": os.environ.get("VISUALAGENTS_TOKEN", "")},
        method="POST",
    )
    try:
        _opener.open(req, timeout=3).read()
    except Exception:
        pass


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

    os.register_at_fork(after_in_child=_after_fork)


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


# -- import hooks -----------------------------------------------------------------

_HOOKS = {"langchain_core.callbacks.manager": _hook_langchain, "agents": _hook_openai_agents}


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
