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


# LAMBDA's real shape: a shared LLMService (client wrapper + title helper), the agent loop in a route handler
# that names itself in its system prompt and runs the model call in a keepalive task, and a report exporter.
LAMBDA_APP = """
import asyncio, sys, openai


class LLMService:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key="x", base_url=sys.argv[1])

    async def stream_chat(self, messages, model=None):
        \"\"\"Stream chat completion with tool support.\"\"\"
        tools = [{"type": "function", "function": {"name": "execute_python", "description": "Run Python code",
                  "parameters": {"type": "object"}}}]
        stream = await self.client.chat.completions.create(model="m", messages=messages, tools=tools, stream=True)
        async for c in stream:
            d = c.choices[0].delta if c.choices else None
            if d and d.tool_calls:
                yield {"type": "tool_call", "id": d.tool_calls[0].id}
            elif d and d.content:
                yield {"type": "content", "content": d.content}

    async def chat(self, messages, model=None):
        r = await self.client.chat.completions.create(model="m", messages=messages)
        return r.choices[0].message.content

    async def generate_title(self, first_message):
        msgs = [{"role": "system", "content": "Generate a short, concise title for this conversation."},
                {"role": "user", "content": first_message}]
        r = await self.client.chat.completions.create(model="m", messages=msgs)
        return r.choices[0].message.content


llm_service = LLMService()


class NotebookKernel:
    def __init__(self, session_id):
        self.session_id = session_id

    def reset(self):
        self.__init__(self.session_id)


async def stream_with_keepalive(source):
    queue = asyncio.Queue()

    async def produce():
        async for item in source:
            await queue.put(item)
        await queue.put(None)

    asyncio.create_task(produce())
    while (item := await queue.get()) is not None:
        yield item


async def stream_chat_response(question):
    system_prompt = f\"\"\"You are LAMBDA, a data analysis agent that helps users inspect datasets and run analysis.
Today is {1}.\"\"\"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    for step in range(3):
        calls = [c async for c in stream_with_keepalive(llm_service.stream_chat(messages)) if c["type"] == "tool_call"]
        if not calls:
            return
        NotebookKernel("s").reset()
        messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
                      "function": {"name": "execute_python", "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": "t1", "content": "5 rows"}]


async def export_report():
    return await llm_service.chat([
        {"role": "system", "content": "You are a professional data scientist writing comprehensive analysis reports."},
        {"role": "user", "content": "the session"}])


async def generate_conversation_title():
    return await llm_service.generate_title("explore the housing data")


async def main():
    await generate_conversation_title()
    await stream_chat_response("explore the housing data")
    await export_report()

asyncio.run(main())
"""


def test_a_shared_llm_service_maps_to_the_agents_that_use_it(tmp_path, fake_openai):
    """LAMBDA: the agent is the code that writes the prompt and runs the loop, not the client wrapper it calls,
    and its calls still light it up when the wrapper runs in another asyncio task."""
    from visualagents import static_scan

    (tmp_path / "app.py").write_text(LAMBDA_APP)
    bp = static_scan.scan(tmp_path)
    agents = {a["id"]: a for a in bp["agents"]}
    assert sorted(agents) == ["export_report", "generate_title", "lambda"]  # no LLMService, no NotebookKernel
    assert agents["lambda"]["role"] == "data analysis agent"
    assert agents["lambda"]["tools"] == ["execute_python"]
    assert agents["export_report"]["role"] == "professional data scientist writing comprehensive analysis reports"

    hub = Hub(bp)
    code = run_command([sys.executable, str(tmp_path / "app.py"), fake_openai], hub, cwd=tmp_path,
                       server_url="http://127.0.0.1:9", instrument=True, use_pty=False, echo=False)
    assert code == 0
    snap = hub.tracker.snapshot()
    assert [p["agent"] for p in snap["path"]] == ["generate_title", "lambda", "export_report"]
    assert not snap["extra_agents"]
    tools = [(e["agent"], e["text"]) for e in hub.tracker.events if e["source"] == "sdk" and e["kind"] == "tool"]
    assert ("lambda", 'execute_python({"code": "df.head()"})') in tools
