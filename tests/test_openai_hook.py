"""The raw OpenAI SDK hook: apps that call chat.completions.create themselves (e.g. LAMBDA)."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from visualagents import blueprint as bpm
from visualagents.hub import Hub
from visualagents.runner import run_command

pytest.importorskip("openai")


class _FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req["messages"]
        system = msgs[0]["content"].lower()
        n_tool = sum(1 for m in msgs if m["role"] == "tool")
        base = {"id": "c1", "created": 0, "model": "m"}
        if not req.get("stream"):
            text = "Define df first." if "inspector" in system else "ok"
            body = dict(base, object="chat.completion",
                        choices=[{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}])
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(delta, finish=None):
            chunk = dict(base, object="chat.completion.chunk", choices=[{"index": 0, "delta": delta, "finish_reason": finish}])
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")

        if req.get("tools") and n_tool == 0:
            send({"role": "assistant", "content": "Checking the data."})
            send({"tool_calls": [{"index": 0, "id": "t1", "type": "function", "function": {"name": "execute_python", "arguments": '{"code": '}}]})
            send({"tool_calls": [{"index": 0, "function": {"arguments": '"df.head()"}'}}]})
        else:
            for word in ("print(", "df)"):
                send({"content": word})
        send({}, "stop")
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture(scope="module")
def fake_openai():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


SYNC_APP = """
import sys, openai
class Programmer:
    def __init__(self):
        self.client = openai.OpenAI(api_key="x", base_url=sys.argv[1])
    def write(self, task):
        msgs = [{"role": "system", "content": "programmer"}, {"role": "user", "content": task}]
        return "".join(c.choices[0].delta.content or "" for c in self.client.chat.completions.create(model="m", messages=msgs, stream=True) if c.choices)
class Inspector:
    def __init__(self):
        self.client = openai.OpenAI(api_key="x", base_url=sys.argv[1])
    def review(self, err):
        msgs = [{"role": "system", "content": "inspector"}, {"role": "user", "content": err}]
        return self.client.chat.completions.create(model="m", messages=msgs).choices[0].message.content
p, i = Programmer(), Inspector()
p.write("plot it"); print("Executing result: NameError", flush=True); p.write("fix: " + i.review("NameError"))
"""

ASYNC_APP = """
import asyncio, sys, openai
class LLMService:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key="x", base_url=sys.argv[1])
    async def turn(self, msgs):
        calls, text = {}, ""
        stream = await self.client.chat.completions.create(model="m", messages=msgs, stream=True,
            tools=[{"type": "function", "function": {"name": "execute_python", "parameters": {"type": "object"}}}])
        async for c in stream:
            d = c.choices[0].delta if c.choices else None
            text += (d.content or "") if d else ""
            for tc in (d.tool_calls or []) if d else []:
                calls.setdefault(tc.index, tc.id)
        return text, calls
async def main():
    llm = LLMService()
    msgs = [{"role": "system", "content": "agent"}, {"role": "user", "content": "explore"}]
    text, calls = await llm.turn(msgs)
    msgs += [{"role": "assistant", "content": text, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "execute_python", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "t1", "name": "execute_python", "content": "5 rows"}]
    await llm.turn(msgs)
asyncio.run(main())
"""


def _run(tmp_path, source, agents, url):
    app = tmp_path / "app.py"
    app.write_text(source)
    hub = Hub(bpm.normalize({"agents": [{"name": a} for a in agents]}))
    code = run_command([sys.executable, str(app), url], hub, cwd=tmp_path, server_url="http://127.0.0.1:9",
                       instrument=True, use_pty=False, echo=False)
    assert code == 0
    return hub.tracker


def test_sync_classes_become_agents(tmp_path, fake_openai):
    t = _run(tmp_path, SYNC_APP, ["Programmer", "Inspector"], fake_openai)
    s = t.snapshot()
    assert s["precise"]
    assert [p["agent"] for p in s["path"]] == ["programmer", "inspector", "programmer"]
    outputs = [(e["agent"], e["text"]) for e in t.events if e["source"] == "sdk" and e["kind"] == "output"]
    assert ("programmer", "print(df)") in outputs
    assert ("inspector", "Define df first.") in outputs
    line = next(e for e in t.events if e["text"] == "Executing result: NameError")
    assert line["agent"] == "programmer"


def test_async_streamed_tool_calls_and_results(tmp_path, fake_openai):
    t = _run(tmp_path, ASYNC_APP, ["LLMService"], fake_openai)
    sdk = [(e["kind"], (e.get("data") or {}).get("tool"), e["text"]) for e in t.events if e["source"] == "sdk"]
    assert ("tool", "execute_python", 'execute_python({"code": "df.head()"})') in sdk
    assert ("observation", "execute_python", "5 rows") in sdk
    assert t.snapshot()["agents"]["llmservice"]["tools"] == 1
