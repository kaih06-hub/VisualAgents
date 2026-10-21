"""Static-analysis gaps found by the third README-graph set, each checked on a made-up project."""

import importlib.util
import shutil
from pathlib import Path

from visualagents.static_scan import scan

RUNTIME = Path(__file__).parent.parent / "visualagents" / "_autoinstrument" / "visualagents_runtime.py"


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def ids(bp):
    return sorted(a["id"] for a in bp["agents"])


def edges(bp):
    return {(e["from"], e["to"]) for e in bp["edges"]}


def test_a_model_called_over_raw_http_is_found(tmp_path):
    _write(tmp_path, {
        "app.py": (
            "import requests\n"
            "PLAN_PROMPT = 'Split the request into tasks.'\n"
            "SUMMARY_PROMPT = 'Summarise the results for the user.'\n"
            "def send(messages, endpoint):\n"
            "    r = requests.post(endpoint, json={'model': 'm', 'messages': messages})\n"
            "    return r.json()['choices'][0]['message']['content']\n"
            "def plan(request, endpoint):\n"
            "    return send([{'role': 'system', 'content': PLAN_PROMPT}, {'role': 'user', 'content': request}], endpoint)\n"
            "def summarize(results, endpoint):\n"
            "    return send([{'role': 'system', 'content': SUMMARY_PROMPT}, {'role': 'user', 'content': results}], endpoint)\n"
            "def notify(url, text):\n"
            "    return requests.post(url, json={'text': text, 'channel': 'ops'})  # a webhook, not a model\n"
            "def run(request, endpoint):\n"
            "    notify('https://hooks.example/x', 'start')\n"
            "    return summarize(plan(request, endpoint), endpoint)\n"),
    })
    bp = scan(tmp_path)
    assert ids(bp) == ["plan", "summarize"]
    assert ("plan", "summarize") in edges(bp)


def test_an_embedding_model_is_not_an_agent(tmp_path):
    _write(tmp_path, {
        "store.py": (
            "from llama_cpp import Llama\n"
            "class VectorStore:\n"
            "    def get_embedding(self, text):\n"
            "        model = Llama(model_path='e.gguf', embedding=True)\n"
            "        return model.embed(text)\n"),
        "writer.py": (
            "from openai import OpenAI\n"
            "def write(topic):\n"
            "    return OpenAI().chat.completions.create(model='m', messages=[{'role': 'user', 'content': f'Write about {topic} in two paragraphs.'}])\n"),
    })
    assert ids(scan(tmp_path)) == ["write"]


PIPELINE = """
from openai import OpenAI
SUB_QUERY_PROMPT = "Break the question into sub-questions."
RERANK_PROMPT = "Keep only passages that answer the question."
REFLECT_PROMPT = "List what is still unknown."
ANSWER_PROMPT = "Answer from the passages."
class BaseAgent:
    pass
class Researcher(BaseAgent):
    def __init__(self):
        self.client = OpenAI()
    def ask(self, prompt):
        return self.client.chat.completions.create(model="m", messages=[{"role": "user", "content": prompt}])
    def _sub_queries(self, q):
        return self.ask(SUB_QUERY_PROMPT + q)
    def _rerank(self, q, docs):
        return self.ask(RERANK_PROMPT + q + docs)
    def _reflect(self, q, notes):
        return self.ask(REFLECT_PROMPT + q + notes)
    def retrieve(self, q):
        queries = self._sub_queries(q)
        for _ in range(3):
            notes = self._rerank(q, queries)
            queries = self._reflect(q, notes)
        return notes
    def answer(self, q):
        notes = self.retrieve(q)
        return self.ask(ANSWER_PROMPT + notes)
class Greeter(BaseAgent):
    def greet(self, name):
        return OpenAI().chat.completions.create(model="m", messages=[{"role": "user", "content": "Say hello to " + name}])
"""


def test_a_class_running_its_own_prompted_steps_is_drawn_as_steps(tmp_path):
    _write(tmp_path, {"research.py": PIPELINE})
    bp = scan(tmp_path)
    assert {"sub_queries", "rerank", "reflect", "answer"} <= set(ids(bp))
    assert "greeter" in ids(bp) and "researcher" not in ids(bp)  # one prompt: still one agent
    assert {("sub_queries", "rerank"), ("rerank", "reflect"), ("reflect", "rerank"), ("reflect", "answer")} <= edges(bp)
    first = next(a for a in bp["agents"] if a["id"] == "sub_queries")
    assert "Researcher" in first["aliases"] and "SUB_QUERY_PROMPT" in first["aliases"]


def test_a_graph_built_from_a_node_table_and_edge_dicts(tmp_path):
    _write(tmp_path, {
        "graph.py": (
            "from langgraph.graph import StateGraph, END\n"
            "from nodes import refine, critique, recall, synthesize\n"
            "from patterns import standard_edges\n"
            "class Builder:\n"
            "    def __init__(self):\n"
            "        self.node_functions = {'refiner': refine, 'critic': critique, 'historian': recall,\n"
            "                               'synthesis': synthesize}\n"
            "    def build(self, agents):\n"
            "        g = StateGraph(dict)\n"
            "        for name in agents:\n"
            "            agent_key = name.lower()\n"
            "            fn = self.node_functions[agent_key]\n"
            "            g.add_node(agent_key, fn)\n"
            "        for edge in standard_edges(agents):\n"
            "            g.add_edge(edge['from'], edge['to'] if edge['to'] != 'END' else END)\n"
            "        return g.compile()\n"),
        "patterns.py": (
            "def standard_edges(agents):\n"
            "    edges = [{'from': 'refiner', 'to': 'critic'}, {'from': 'refiner', 'to': 'historian'}]\n"
            "    edges += [{'from': 'critic', 'to': 'synthesis'}, {'from': 'historian', 'to': 'synthesis'}]\n"
            "    edges.append({'from': 'synthesis', 'to': 'END'})\n"
            "    return edges\n"),
        "nodes.py": "".join(f"def {f}(state):\n    return state\n" for f in ("refine", "critique", "recall", "synthesize")),
        "prototype.py": (
            "from agents import Agent\n"
            "refiner_agent = Agent(name='refiner_agent', instructions='Refine the question')\n"),
    })
    bp = scan(tmp_path)
    assert ids(bp) == ["critic", "historian", "refiner", "synthesis"]  # no `agent_key`; refiner_agent is refiner
    assert edges(bp) == {("refiner", "critic"), ("refiner", "historian"), ("critic", "synthesis"),
                         ("historian", "synthesis")}


def test_routing_to_computed_node_names_and_factory_built_nodes(tmp_path):
    _write(tmp_path, {
        "game.py": (
            "from langgraph.graph import StateGraph, START\n"
            "from langgraph.types import Send\n"
            "from factory import make_node, web_scraper, audit\n"
            "def route(state):\n"
            "    if state['phase'] == 'speak':\n"
            "        return f\"speech_{state['next']}\"\n"
            "    return [f\"vote_{p}\" for p in state['alive']]\n"
            "def fan_out(state):\n"
            "    return [Send('scrape', {'q': q}) for q in state['queries']]\n"
            "def build(players):\n"
            "    g = StateGraph(dict)\n"
            "    g.add_node('switch', lambda s: s)\n"
            "    g.add_node('plan', make_node('plan', audit))\n"
            "    g.add_node('scrape', make_node('scrape', web_scraper, audit))\n"
            "    for p in players:\n"
            "        g.add_node(f'speech_{p}', lambda s: s)\n"
            "        g.add_node(f'vote_{p}', lambda s: s)\n"
            "    g.add_edge(START, 'plan')\n"
            "    g.add_conditional_edges('plan', fan_out)\n"
            "    g.add_edge('scrape', 'switch')\n"
            "    g.add_conditional_edges('switch', route, {**{f'speech_{p}': f'speech_{p}' for p in players},\n"
            "                                               **{f'vote_{p}': f'vote_{p}' for p in players}})\n"
            "    return g.compile()\n"),
        "factory.py": (
            "from agents import Agent\n"
            "def make_node(name, *deps):\n"
            "    return lambda s: s\n"
            "audit = object()\n"
            "web_scraper = Agent(name='web_scraper', instructions='Search the web')\n"),
    })
    bp = scan(tmp_path)
    assert "web_scraper" not in ids(bp)  # handed to the scrape node's factory: it is that node
    assert {("plan", "scrape"), ("switch", "speech"), ("switch", "vote")} <= edges(bp)


def test_a_separate_program_in_the_same_repository_stays_off_the_map(tmp_path):
    main = (
        "import openai\n"
        "def execution_agent(task):\n"
        "    return openai.chat.completions.create(model='m', messages=[{'role': 'user', 'content': 'Do this task well: ' + task}])\n"
        "def creation_agent(result):\n"
        "    return openai.chat.completions.create(model='m', messages=[{'role': 'user', 'content': 'Make new tasks from: ' + result}])\n"
        "def main():\n"
        "    while True:\n"
        "        creation_agent(execution_agent('next'))\n")
    coder = (
        "import openai\n"
        "def code_writer_agent(spec):\n"
        "    return openai.chat.completions.create(model='m', messages=[{'role': 'user', 'content': 'Write code for: ' + spec}])\n"
        "def code_review_agent(code):\n"
        "    return openai.chat.completions.create(model='m', messages=[{'role': 'user', 'content': 'Review this code: ' + code}])\n"
        "def run(spec):\n"
        "    return code_review_agent(code_writer_agent(spec))\n")
    _write(tmp_path, {"loop.py": main, "coder/coder.py": coder, "coder/README.md": "# A coding helper\n"})
    bp = scan(tmp_path)
    assert ids(bp) == ["creation_agent", "execution_agent"]
    assert {("execution_agent", "creation_agent"), ("creation_agent", "execution_agent")} <= edges(bp)  # the loop

    (tmp_path / "loop.py").write_text("from coder.coder import run\n" + main)  # imported: part of the system
    assert "code_writer_agent" in ids(scan(tmp_path))


def test_the_runtime_works_when_installed_under_an_agents_folder(tmp_path):
    """`/agents/` in a library path marks the OpenAI Agents SDK; VisualAgents' own frames never count."""
    home = tmp_path / "agents" / "VisualAgents"
    home.mkdir(parents=True)
    shutil.copy(RUNTIME, home / "visualagents_runtime.py")
    spec = importlib.util.spec_from_file_location("va_runtime_agents_dir", home / "visualagents_runtime.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    rt._HERE = str(home)
    rt.configure(url="http://127.0.0.1:9", token="t")

    def plan_trip():
        return rt._Call()

    call = plan_trip()
    assert not call.off and call.agent == "plan_trip"
