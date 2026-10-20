"""Any program that calls an LLM is an agent system - one agent or many, with or without a framework."""

import importlib.util
import sys
from pathlib import Path

from visualagents import blueprint as bpm
from visualagents import static_scan
from visualagents.hub import Hub
from visualagents.runner import run_command

RUNTIME = Path(__file__).parent.parent / "visualagents" / "_autoinstrument" / "visualagents_runtime.py"


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


# -- the map: static analysis of LLM call sites ------------------------------------------------------------

def test_a_shared_helper_is_plumbing_and_its_callers_are_the_agents(tmp_path):
    """RedRockDB-like: one generate_reply() runs the local model for three features with their own prompts."""
    _write(tmp_path, {
        "app/local_model.py": (
            "def generate_reply(system_prompt, user_content):\n"
            "    \"\"\"Run the chat model over one prompt.\"\"\"\n"
            "    from mlx_lm import generate\n"
            "    if not user_content:\n"
            "        raise ValueError('user_content must not be empty for the local chat model')\n"
            "    return generate(None, None, prompt=system_prompt + user_content)\n"),
        "app/rag.py": (
            "from app.local_model import generate_reply\n"
            "SYSTEM_PROMPT = 'Answer only from the excerpts.'\n"
            "def answer(question):\n"
            "    \"\"\"Answer a question with citations.\"\"\"\n"
            "    return generate_reply(SYSTEM_PROMPT, question)\n"),
        "app/interpret.py": (
            "from app.local_model import generate_reply\n"
            "def interpret(instruction):\n"
            "    return generate_reply('Turn the instruction into exactly one structured action, as JSON.', instruction)\n"),
        "app/enrich.py": (
            "from app.local_model import generate_reply\n"
            "def add_context(chunks):\n"
            "    return [generate_reply(f'Write one sentence placing this passage in its document: {c}', c) for c in chunks]\n"),
        "app/pipeline.py": (
            "from app.enrich import add_context\nfrom app.rag import answer\n"
            "def load_then_ask(chunks, q):\n    add_context(chunks)\n    return answer(q)\n"),
    })
    bp = static_scan.scan(tmp_path)
    by_id = {a["id"]: a for a in bp["agents"]}
    assert sorted(by_id) == ["add_context", "answer", "interpret"]  # generate_reply is not an agent
    assert by_id["answer"]["role"] == "calls the LLM (generate_reply)"
    assert by_id["answer"]["description"] == "Answer a question with citations."
    assert {"from": "add_context", "to": "answer", "label": "then", "conditional": False} in bp["edges"]


def test_one_llm_call_is_a_one_agent_system(tmp_path):
    _write(tmp_path, {"summarize.py": (
        "import sys\nfrom openai import OpenAI\nclient = OpenAI()\n"
        "text = open(sys.argv[1]).read()\n"
        "r = client.chat.completions.create(model='gpt-5', messages=[{'role': 'user', 'content': text}])\n"
        "print(r.choices[0].message.content)\n")})
    bp = static_scan.scan(tmp_path)
    assert [(a["id"], a["role"]) for a in bp["agents"]] == [("summarize", "calls the LLM (completions.create)")]


def test_llm_libraries_are_recognised_by_their_imports(tmp_path):
    _write(tmp_path, {
        "chat_bot.py": "import ollama\ndef reply(msgs):\n    return ollama.chat(model='llama3', messages=msgs)\n",
        "router.py": "from litellm import completion\ndef route(q):\n    return completion('gpt-5', [q])\n",
        "local.py": ("from llama_cpp import Llama\nllm = Llama(model_path='m.gguf')\n"
                     "def complete(p):\n    return llm(p)\n"),
        "lc.py": ("from langchain_openai import ChatOpenAI\nclass Critic:\n    def __init__(self):\n"
                  "        self.llm = ChatOpenAI()\n    def review(self, d):\n        return self.llm.invoke(d)\n"),
        "sms.py": "def notify(client):\n    return client.messages.create(to='+1', body='done')\n",  # Twilio, not an LLM
        "embed.py": ("from transformers import AutoTokenizer\ntok = AutoTokenizer.from_pretrained('x')\n"
                     "def count(t):\n    return len(tok(t))\n"),
    })
    ids = sorted(a["id"] for a in static_scan.scan(tmp_path)["agents"])
    assert ids == ["complete", "critic", "reply", "route"]  # building Llama() at import time is not a call


# -- live: library hooks report each call as the code that made it ------------------------------------------

FAKE_MLX = {
    "site-packages/mlx_lm/__init__.py": "from .generate import generate, stream_generate\n",
    "site-packages/mlx_lm/generate.py": (
        "class R:\n    def __init__(self, text):\n        self.text = text\n"
        "def stream_generate(model, tokenizer, prompt, **kw):\n"
        "    for w in ['<think>check the passages</think>', 'Paris ', 'is the capital.']:\n        yield R(w)\n"
        "def generate(model, tokenizer, prompt, verbose=False, **kw):\n"
        "    return ''.join(r.text for r in stream_generate(model, tokenizer, prompt, **kw))\n"),
}

MLX_APP = """
import sys
sys.path.insert(0, sys.argv[1])
def generate_reply(system_prompt, user_content):
    from mlx_lm import generate
    rendered = f"<|im_start|>system\\n{system_prompt}<|im_end|>\\n<|im_start|>user\\n{user_content}<|im_end|>\\n<|im_start|>assistant\\n"
    return generate(None, None, prompt=rendered, verbose=False)
def answer(question):
    return generate_reply("Answer only from the excerpts.", question)
def interpret(instruction):
    return generate_reply("Turn the instruction into one action.", instruction)
print(answer("What is the capital of France?"), flush=True)
from mlx_lm import stream_generate
for r in stream_generate(None, None, "<|im_start|>user\\nsummarise the corpus<|im_end|>"):
    pass
print(interpret("remove commas from chunk 9"), flush=True)
"""


def _run(tmp_path, source, bp):
    app = tmp_path / "app.py"
    app.write_text(source)
    hub = Hub(bp)
    code = run_command([sys.executable, str(app), str(tmp_path / "site-packages")], hub, cwd=tmp_path,
                       server_url="http://127.0.0.1:9", instrument=True, use_pty=False, echo=False)
    assert code == 0
    return hub.tracker


def test_a_local_model_app_lights_up_the_code_that_called_it(tmp_path):
    _write(tmp_path, FAKE_MLX)
    (tmp_path / "app.py").write_text(MLX_APP)
    bp = static_scan.scan(tmp_path)  # the map comes from the same rules
    assert sorted(a["id"] for a in bp["agents"]) == ["answer", "app", "interpret"]
    t = _run(tmp_path, MLX_APP, bp)
    assert [p["agent"] for p in t.snapshot()["path"]] == ["answer", "app", "interpret"]
    sdk = [(e["agent"], e["kind"], e["text"]) for e in t.events if e["source"] == "sdk"]
    assert ("answer", "task", "What is the capital of France?") in sdk
    assert ("answer", "thought", "check the passages") in sdk  # <think> ... </think> is reasoning
    assert ("answer", "output", "Paris is the capital.") in sdk
    assert ("app", "task", "summarise the corpus") in sdk
    assert sum(1 for a, k, _ in sdk if k == "output") == 3  # generate -> stream_generate is one call, not two


FAKE_OLLAMA = {
    "site-packages/ollama/__init__.py": (
        "from ._client import Client, AsyncClient\n_client = Client()\nchat = _client.chat\ngenerate = _client.generate\n"),
    "site-packages/ollama/_client.py": (
        "class Client:\n"
        "    def chat(self, model='', messages=None, *, stream=False, **kw):\n"
        "        if stream:\n"
        "            return iter([{'message': {'content': 'Hel'}}, {'message': {'content': 'lo'}}])\n"
        "        return {'message': {'role': 'assistant', 'content': 'Looking it up.', 'thinking': 'the user greets me',\n"
        "                            'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'q': 'x'}}}]}}\n"
        "    def generate(self, model='', prompt='', **kw):\n"
        "        return {'response': '42'}\n"
        "class AsyncClient:\n    pass\n"),
}

OLLAMA_APP = """
import sys
sys.path.insert(0, sys.argv[1])
import ollama
class Greeter:
    def run(self):
        return ollama.chat(model="llama3", messages=[{"role": "user", "content": "hi there"}])
def ask():
    return "".join(c["message"]["content"] for c in ollama.chat(model="llama3", messages=[{"role": "user", "content": "stream please"}], stream=True))
Greeter().run()
print(ask(), flush=True)
"""


def test_with_no_map_every_caller_appears_as_its_own_agent(tmp_path):
    _write(tmp_path, FAKE_OLLAMA)
    t = _run(tmp_path, OLLAMA_APP, bpm.normalize({"project": "p", "agents": []}))
    snap = t.snapshot()
    assert [p["agent"] for p in snap["path"]] == ["greeter", "ask"]
    assert {a["id"] for a in snap["extra_agents"]} >= {"greeter", "ask"}
    sdk = [(e["agent"], e["kind"], e["text"]) for e in t.events if e["source"] == "sdk"]
    assert ("greeter", "thought", "the user greets me") in sdk
    assert ("greeter", "tool", 'lookup({"q": "x"})') in sdk
    assert ("ask", "output", "Hello") in sdk


def test_a_one_agent_map_takes_every_call(tmp_path):
    _write(tmp_path, FAKE_OLLAMA)
    t = _run(tmp_path, OLLAMA_APP, bpm.normalize({"project": "p", "agents": [{"name": "Assistant"}]}))
    assert {p["agent"] for p in t.snapshot()["path"]} == {"assistant"}


def test_a_self_reporting_framework_still_wakes_a_hooked_process(tmp_path):
    """CrewAI / AutoGen call models through litellm and print their own agents: the hook must not stay asleep."""
    spec = importlib.util.spec_from_file_location("va_runtime_under_test", RUNTIME)
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    rt.configure(url="http://127.0.0.1:9", token="t", lazy=True)
    assert rt._config["armed"] is False
    framework = {"rt": rt}
    exec(compile("def kickoff():\n    return rt._Call()\n", "/x/site-packages/crewai/agent.py", "exec"), framework)
    call = framework["kickoff"]()
    assert call.off  # CrewAI's own logs name its agents; no duplicate events
    assert rt._config["armed"] is True


# -- the real Anthropic and Gemini SDKs against a fake server ---------------------------------------------

def _fake_llm_server():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if ":generateContent" in self.path:
                return self._json({"candidates": [{"content": {"role": "model", "parts": [
                    {"text": "weighing formal vs casual", "thought": True}, {"text": "Bonjour"}]}}]})
            usage = {"input_tokens": 1, "output_tokens": 1}
            if not req.get("stream"):
                return self._json({"id": "m1", "type": "message", "role": "assistant", "model": "m",
                                   "stop_reason": "tool_use", "usage": usage, "content": [
                                       {"type": "thinking", "thinking": "outline first", "signature": "s"},
                                       {"type": "text", "text": "Drafting."},
                                       {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                ("message_start", {"type": "message_start", "message": {
                    "id": "m2", "type": "message", "role": "assistant", "model": "m", "content": [],
                    "stop_reason": None, "usage": usage}}),
                ("content_block_start", {"type": "content_block_start", "index": 0,
                                         "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                         "delta": {"type": "text_delta", "text": "Streamed "}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                         "delta": {"type": "text_delta", "text": "draft."}}),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage}),
                ("message_stop", {"type": "message_stop"}),
            ]
            for name, data in events:
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


SDK_APP = """
import sys, anthropic
from google import genai
url = sys.argv[1]
class Writer:
    def draft(self):
        c = anthropic.Anthropic(api_key="x", base_url=url)
        return c.messages.create(model="claude-opus-5-5", max_tokens=99, messages=[{"role": "user", "content": "draft it"}])
    def stream_draft(self):
        c = anthropic.Anthropic(api_key="x", base_url=url)
        with c.messages.stream(model="claude-opus-5-5", max_tokens=99, messages=[{"role": "user", "content": "stream it"}]) as s:
            for _ in s.text_stream:
                pass
class Translator:
    def run(self):
        g = genai.Client(api_key="x", http_options={"base_url": url})
        return g.models.generate_content(model="gemini-flash-latest", contents="translate hello").text
Writer().draft()
Writer().stream_draft()
print(Translator().run(), flush=True)
"""


def test_real_anthropic_and_gemini_sdks_are_tracked(tmp_path):
    import pytest

    pytest.importorskip("anthropic")
    pytest.importorskip("google.genai")
    server = _fake_llm_server()
    try:
        app = tmp_path / "app.py"
        app.write_text(SDK_APP)
        hub = Hub(bpm.normalize({"agents": [{"name": "Writer"}, {"name": "Translator"}]}))
        code = run_command([sys.executable, str(app), f"http://127.0.0.1:{server.server_address[1]}"], hub,
                           cwd=tmp_path, server_url="http://127.0.0.1:9", instrument=True, use_pty=False, echo=False)
    finally:
        server.shutdown()
    assert code == 0
    t = hub.tracker
    assert [p["agent"] for p in t.snapshot()["path"]] == ["writer", "translator"]
    sdk = [(e["agent"], e["kind"], e["text"]) for e in t.events if e["source"] == "sdk"]
    assert ("writer", "task", "draft it") in sdk
    assert ("writer", "thought", "outline first") in sdk
    assert ("writer", "tool", 'search({"q": "x"})') in sdk
    assert ("writer", "output", "Streamed draft.") in sdk  # messages.stream(...) as a context manager
    assert ("translator", "task", "translate hello") in sdk
    assert ("translator", "thought", "weighing formal vs casual") in sdk
    assert ("translator", "output", "Bonjour") in sdk


def test_call_sites_in_an_agent_class_are_named_by_the_class(tmp_path):
    """A class built like an agent is one agent, even when other code calls its methods (a shared LLM client class,
    called from everywhere, still splits into the methods that write their own prompts)."""
    from visualagents import call_sites
    from visualagents.collector import collect

    (tmp_path / "llm.py").write_text(
        "import openai\n"
        "class Chat:\n"
        "    def __init__(self):\n"
        "        self.client = openai.OpenAI()\n"
        "    def generate_response(self, prompt):\n"
        "        return self.client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': prompt}])\n"
    )
    (tmp_path / "agents.py").write_text(
        "class BaseAgent:\n"
        "    pass\n"
        "class CriticAgent(BaseAgent):\n"
        "    def step(self, prompt):\n"
        "        return self.llm.generate_response(prompt)\n"
        "    def add_message_to_memory(self, messages):\n"
        "        return self.llm.generate_response(messages)\n"
    )
    (tmp_path / "env.py").write_text(
        "class Env:\n"
        "    def run(self, critic):\n"
        "        return critic.step('You are a careful critic. Review the draft and list every unsupported claim.')\n"
    )
    names = {a["name"] for a in call_sites.find(collect(tmp_path))["agents"]}
    assert "CriticAgent" in names
    assert "add_message_to_memory" not in names
