"""How the scanner is put together, rather than which code shapes it knows.

These hold the scan to a few general rules: both passes always run, a model call is recognised by what it does, an
agent is where a prompt is written, and a file is never lost to how it was saved. Each is checked on a made-up
project.
"""

import ast
from pathlib import Path

from visualagents import call_sites
from visualagents.collector import collect
from visualagents.static_scan import scan


def _write(root: Path, files: dict[str, str], encoding: str = "utf-8") -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding=encoding)
    return root


def ids(bp):
    return sorted(a["id"] for a in bp["agents"])


def edges(bp):
    return {(e["from"], e["to"]) for e in bp["edges"]}


def _via(src: str) -> str:
    return call_sites._named_llm_call(ast.parse(src).body[0])


# -- a model call is recognised by what it does ------------------------------------------------------------------

def test_a_model_call_is_known_by_its_names_whatever_library_it_goes_through():
    assert _via("def f(self, p):\n return self.llm(p)") == "llm"
    assert _via("def f(self, p):\n return self.llm_engine(p, max_tokens=64)") == "llm_engine"
    assert _via("def f(self, m):\n return self.planner_llm.bind_tools(t).ainvoke(m)")
    assert _via("def f(p):\n return call_gpt(prompt=p)") == "call_gpt"
    assert _via("def f(p):\n return get_completion(p)") == "get_completion"
    # a name that only may be a model counts when the call carries a prompt
    assert _via("def f(self, messages):\n return self.client.chat(messages=messages)") == "client.chat"
    assert _via("def f(self, prompt):\n return self.model(prompt)") == "model"


def test_what_only_looks_like_a_model_call_is_left_alone():
    assert _via("def f(X):\n return model.predict(X)") == ""  # scikit-learn
    assert _via("def f(x):\n return self.model(x)") == ""  # a torch module
    assert _via("def f(cfg):\n return get_llm(cfg)") == ""  # hands the model out, does not ask it
    assert _via("def f(cfg):\n return create_openai_client(cfg)") == ""
    assert _via("def f(t):\n return llm.get_num_tokens(t)") == ""
    assert _via("def f(m):\n return self.skillmanager.run(m)") == ""  # "llm" inside another word
    assert _via("def f(q):\n return slack_client.chat(q)") == ""


def test_agents_behind_an_engine_object_are_found_and_the_engines_are_not_agents(tmp_path):
    """Provider engines are built by a factory and called as self.llm_engine(prompt): no caller of them can be traced,
    yet the classes that write the prompts are the agents - and the engines are plumbing, not agents."""
    bp = scan(_write(tmp_path, {
        "engines.py": (
            "from openai import OpenAI\nimport anthropic\n"
            "class OpenAIEngine:\n"
            "    def __init__(self):\n        self.client = OpenAI()\n"
            "    def __call__(self, prompt, system_prompt=None):\n        return self.generate(prompt, system_prompt)\n"
            "    def generate(self, prompt, system_prompt=None):\n"
            "        return self.client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': prompt}])\n"
            "class ClaudeEngine:\n"
            "    def __init__(self):\n        self.client = anthropic.Anthropic()\n"
            "    def __call__(self, prompt, system_prompt=None):\n        return self.generate(prompt, system_prompt)\n"
            "    def generate(self, prompt, system_prompt=None):\n"
            "        return self.client.messages.create(model='m', max_tokens=9, messages=[{'role': 'user', 'content': prompt}])\n"
            "def create_engine(name):\n    return {'openai': OpenAIEngine, 'claude': ClaudeEngine}[name]()\n"
        ),
        "planner.py": (
            "from engines import create_engine\n"
            "class Planner:\n"
            "    def __init__(self, name):\n        self.llm_engine = create_engine(name)\n"
            "    def plan(self, question):\n"
            "        prompt = f'You are a careful planner. Break this question into steps: {question}'\n"
            "        return self.llm_engine(prompt)\n"
        ),
        "verifier.py": (
            "from engines import create_engine\n"
            "class Verifier:\n"
            "    def __init__(self, name):\n        self.llm_engine = create_engine(name)\n"
            "    def verify(self, answer):\n"
            "        prompt = f'You are a strict verifier. Is this answer complete and correct? {answer}'\n"
            "        return self.llm_engine(prompt)\n"
        ),
        "main.py": (
            "from planner import Planner\nfrom verifier import Verifier\n"
            "def solve(q):\n    steps = Planner('openai').plan(q)\n    return Verifier('openai').verify(steps)\n"
        ),
    }))
    assert ids(bp) == ["planner", "verifier"]


def test_a_name_mangled_request_method_is_followed_like_any_other(tmp_path):
    """self.__request() is an ordinary private method: the code that writes the prompts is found through it."""
    bp = scan(_write(tmp_path, {
        "model.py": (
            "import requests\n"
            "class APIModel:\n"
            "    def __req(self, text):\n"
            "        body = {'model': 'm', 'messages': [{'role': 'user', 'content': text}]}\n"
            "        return requests.post('https://api.openai.com/v1/chat/completions', json=body).json()\n"
            "    def chat(self, text):\n        return self.__req(text)\n"
        ),
        "writer.py": (
            "from model import APIModel\n"
            "OUTLINE_PROMPT = 'You are an expert survey writer. Draft an outline for the topic: {topic}'\n"
            "class OutlineWriter:\n"
            "    def __init__(self):\n        self.api_model = APIModel()\n"
            "    def draft(self, topic):\n        return self.api_model.chat(OUTLINE_PROMPT.format(topic=topic))\n"
        ),
    }))
    assert ids(bp) == ["outline_writer"]


# -- both passes always run ---------------------------------------------------------------------------------------

_WRITERS = {
    "writer.py": (
        "from openai import OpenAI\nclient = OpenAI()\n"
        "def write_article(notes):\n"
        "    prompt = f'You are a magazine writer. Write an article from these notes: {notes}'\n"
        "    return client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': prompt}])\n"
    ),
    "editor.py": (
        "from openai import OpenAI\nclient = OpenAI()\n"
        "def edit_article(draft):\n"
        "    prompt = f'You are a demanding editor. Tighten this draft and fix its errors: {draft}'\n"
        "    return client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': prompt}])\n"
    ),
    "main.py": (
        "from researcher import research\nfrom writer import write_article\nfrom editor import edit_article\n"
        "def run(topic):\n    notes = research(topic)\n    return edit_article(write_article(notes))\n"
    ),
}


def test_plain_model_callers_join_the_agents_a_framework_builds(tmp_path):
    """One step is an SDK agent, the others prompt the model themselves: all three are the system."""
    bp = scan(_write(tmp_path, {
        **_WRITERS,
        "researcher.py": (
            "from agents import Agent, Runner\n"
            "research_agent = Agent(name='researcher', instructions='Find sources for the topic.')\n"
            "def research(topic):\n    return Runner.run_sync(research_agent, topic).final_output\n"
        ),
    }))
    assert {"researcher", "write_article", "edit_article"} <= set(ids(bp))


def test_a_class_that_is_only_called_an_agent_does_not_hide_the_system(tmp_path):
    """QLearningAgent(lr=0.1) has no model, prompt or framework behind it: it is not an agent, and the functions
    that do prompt a model are found as they would be without it."""
    bp = scan(_write(tmp_path, {
        **_WRITERS,
        "researcher.py": (
            "from openai import OpenAI\nclient = OpenAI()\n"
            "def research(topic):\n"
            "    prompt = f'You are a thorough researcher. List the key facts about: {topic}'\n"
            "    return client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': prompt}])\n"
        ),
        "experiments/bandit.py": (
            "class QLearningAgent:\n"
            "    def __init__(self, lr, gamma):\n        self.lr, self.gamma, self.q = lr, gamma, {}\n"
            "    def act(self, state):\n        return max(self.q.get(state, {0: 0}), key=lambda a: a)\n"
            "q_agent = QLearningAgent(lr=0.1, gamma=0.9)\n"
        ),
    }))
    assert "q_agent" not in ids(bp)
    assert {"research", "write_article", "edit_article"} <= set(ids(bp))


def test_what_an_agent_runs_itself_is_part_of_that_agent(tmp_path):
    """A graph node's helper that prompts the model, and a tool class, are not more agents."""
    bp = scan(_write(tmp_path, {
        "graph.py": (
            "from langgraph.graph import StateGraph, START, END\n"
            "from helpers import summarize_sources\nfrom tools import LookupTool\n"
            "def research(state):\n    return {'notes': summarize_sources(state['sources'])}\n"
            "def answer(state):\n    return {'answer': LookupTool().run(state['notes'])}\n"
            "g = StateGraph(dict)\n"
            "g.add_node('research', research)\ng.add_node('answer', answer)\n"
            "g.add_edge(START, 'research')\ng.add_edge('research', 'answer')\ng.add_edge('answer', END)\n"
        ),
        "helpers.py": (
            "from langchain_openai import ChatOpenAI\nllm = ChatOpenAI()\n"
            "def summarize_sources(sources):\n"
            "    prompt = f'You are a research assistant. Summarize these sources faithfully: {sources}'\n"
            "    return llm.invoke(prompt)\n"
        ),
        "tools.py": (
            "from langchain_openai import ChatOpenAI\nfrom langchain_core.tools import BaseTool\n"
            "class LookupTool(BaseTool):\n"
            "    def _run(self, notes):\n"
            "        prompt = f'You are a lookup assistant. Answer from these notes only: {notes}'\n"
            "        return ChatOpenAI().invoke(prompt)\n"
        ),
    }))
    assert ids(bp) == ["answer", "research"]


# -- an agent is where a prompt is written ------------------------------------------------------------------------

def test_a_prompt_read_from_a_file_is_still_that_class_s_own_prompt(tmp_path):
    """The node classes read their system prompt from a file and send it through a shared helper: each class that
    composes its own system message is an agent - the map does not collapse into whatever builds them."""
    node = (
        "from langchain_core.messages import SystemMessage\nimport model\n"
        "class {name}:\n"
        "    def __init__(self, prompt_file):\n        self.prompt_file = prompt_file\n"
        "    def respond(self, state):\n"
        "        with open(self.prompt_file) as fh:\n            system_prompt = fh.read()\n"
        "        messages = [SystemMessage(content=system_prompt), *state['messages']]\n"
        "        return model.query_model_safe(messages)\n"
    )
    bp = scan(_write(tmp_path, {
        "model.py": (
            "from langchain_openai import ChatOpenAI\n"
            "def query_model_safe(messages):\n    return ChatOpenAI(model='m').invoke(messages)\n"
        ),
        "architect.py": node.format(name="Architect"),
        "technician.py": node.format(name="Technician"),
        "build.py": (
            "from architect import Architect\nfrom technician import Technician\n"
            "def build():\n    return [Architect('prompts/architect.txt'), Technician('prompts/technician.txt')]\n"
        ),
    }))
    assert ids(bp) == ["architect", "technician"]


def test_subclasses_stand_in_for_a_model_calling_base_that_is_never_built(tmp_path):
    bp = scan(_write(tmp_path, {
        "base.py": (
            "from openai import OpenAI\n"
            "class BaseStage:\n"
            "    role = ''\n"
            "    def __init__(self):\n        self.client = OpenAI()\n"
            "    def run(self, task):\n"
            "        system = f'You are the {self.role} of a research team. Do your part of the task.'\n"
            "        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': task}]\n"
            "        return self.client.chat.completions.create(model='m', messages=msgs)\n"
        ),
        "stages.py": (
            "from base import BaseStage\n"
            "class Surveyor(BaseStage):\n    role = 'surveyor'\n    def tidy(self, x):\n        return x.strip()\n"
            "class Analyst(BaseStage):\n    role = 'analyst'\n    def tidy(self, x):\n        return x.strip()\n"
            "class Reporter(BaseStage):\n    role = 'reporter'\n    def tidy(self, x):\n        return x.strip()\n"
        ),
        "main.py": (
            "from stages import Surveyor, Analyst, Reporter\n"
            "def run(task):\n    a = Surveyor().run(task)\n    b = Analyst().run(a)\n    return Reporter().run(b)\n"
        ),
    }))
    assert ids(bp) == ["analyst", "reporter", "surveyor"]


def test_an_agent_class_that_picks_one_step_per_call_is_one_agent(tmp_path):
    """Three prompted methods, but run() picks one by the state it is in: one agent, not a pipeline of three."""
    bp = scan(_write(tmp_path, {
        "bug_hunter.py": (
            "from openai import OpenAI\nclient = OpenAI()\n"
            "def ask(prompt):\n    return client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': prompt}])\n"
            "class BugHunter:\n"
            "    def run(self, state):\n"
            "        if state == 'hunting':\n            return self.check_logs()\n"
            "        elif state == 'reproducing':\n            return self.reproduce()\n"
            "        return self.pair_program()\n"
            "    def check_logs(self):\n        return ask('You are a debugger. Read these logs and name the failing call.')\n"
            "    def reproduce(self):\n        return ask('You are a tester. Write the steps that reproduce this bug.')\n"
            "    def pair_program(self):\n        return ask('You are a pair programmer. Propose the smallest fix for it.')\n"
        ),
        "main.py": "from bug_hunter import BugHunter\ndef go(state):\n    return BugHunter().run(state)\n",
    }))
    assert ids(bp) == ["bug_hunter"]


# -- explicit wiring ----------------------------------------------------------------------------------------------

def test_a_node_built_by_a_factory_is_named_after_the_function_it_returns(tmp_path):
    bp = scan(_write(tmp_path, {
        "nodes.py": (
            "def create_guard_node(llm):\n"
            "    async def guardrails(state):\n        return await llm.ainvoke(state['question'])\n"
            "    return guardrails\n"
            "def create_answer_node(llm):\n"
            "    async def final_answer(state):\n        return await llm.ainvoke(state['question'])\n"
            "    return final_answer\n"
        ),
        "graph.py": (
            "from langgraph.graph import StateGraph, START, END\n"
            "from nodes import create_guard_node, create_answer_node\n"
            "def build(llm):\n"
            "    g = StateGraph(dict)\n"
            "    guard = create_guard_node(llm)\n"
            "    final_answer = create_answer_node(llm)\n"
            "    g.add_node(guard)\n    g.add_node(final_answer)\n"
            "    g.add_edge(START, 'guardrails')\n    g.add_edge('guardrails', 'final_answer')\n"
            "    g.add_edge('final_answer', END)\n"
            "    return g.compile()\n"
        ),
    }))
    assert ids(bp) == ["final_answer", "guardrails"]
    assert ("guardrails", "final_answer") in edges(bp)


def test_an_agent_class_run_by_one_graph_node_is_that_node(tmp_path):
    """The node function builds SearchAgent in a helper and runs it: one block, not a node and a class."""
    bp = scan(_write(tmp_path, {
        "agents.py": (
            "from openai import OpenAI\n"
            "class BaseAgent:\n    def __init__(self):\n        self.client = OpenAI()\n"
            "class SearchAgent(BaseAgent):\n"
            "    def run(self, topic):\n"
            "        prompt = f'You are a literature search agent. Plan the queries for: {topic}'\n"
            "        return self.client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': prompt}])\n"
            "class WriteAgent(BaseAgent):\n"
            "    def run(self, notes):\n"
            "        prompt = f'You are a survey writing agent. Write the survey from: {notes}'\n"
            "        return self.client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': prompt}])\n"
        ),
        "nodes.py": (
            "from agents import SearchAgent, WriteAgent\n"
            "def do_search(state):\n    return {'notes': SearchAgent().run(state['topic'])}\n"
            "def do_write(state):\n    return {'survey': WriteAgent().run(state['notes'])}\n"
        ),
        "graph.py": (
            "from langgraph.graph import StateGraph, START, END\nfrom nodes import do_search, do_write\n"
            "def run_search(state):\n    return do_search(state)\n"
            "def run_write(state):\n    return do_write(state)\n"
            "g = StateGraph(dict)\n"
            "g.add_node('run_search', run_search)\ng.add_node('run_write', run_write)\n"
            "g.add_edge(START, 'run_search')\ng.add_edge('run_search', 'run_write')\ng.add_edge('run_write', END)\n"
        ),
    }))
    assert ids(bp) == ["run_search", "run_write"]
    by_id = {a["id"]: a for a in bp["agents"]}
    assert "SearchAgent" in by_id["run_search"]["aliases"]


def test_an_engine_a_class_keeps_as_self_agent_is_that_class(tmp_path):
    """self.agent = CodeAgent(...) in the shared base is each subclass's engine, not one more agent called 'agent';
    the pipeline that calls a method on each of them runs them in that order."""
    bp = scan(_write(tmp_path, {
        "base.py": (
            "from smolagents import CodeAgent\n"
            "class BaseAgent:\n"
            "    def __init__(self, model):\n        self.agent = CodeAgent(tools=[], model=model)\n"
        ),
        "roles.py": (
            "from base import BaseAgent\n"
            "class EntityAgent(BaseAgent):\n"
            "    def recognize(self, q):\n        return self.agent.run(f'Find the entities in: {q}')\n"
            "class SqlAgent(BaseAgent):\n"
            "    def generate(self, q, entities):\n        return self.agent.run(f'Write SQL for {q} using {entities}')\n"
        ),
        "pipeline.py": (
            "from roles import EntityAgent, SqlAgent\n"
            "class Pipeline:\n"
            "    def __init__(self, model, shared=None):\n"
            "        self.entity_agent = shared or EntityAgent(model)\n"
            "        self.sql_agent = SqlAgent(model)\n"
            "    def answer(self, q):\n"
            "        entities = self.entity_agent.recognize(q)\n"
            "        return self.sql_agent.generate(q, entities)\n"
        ),
    }))
    assert ids(bp) == ["entity_agent", "sql_agent"]
    assert ("entity_agent", "sql_agent") in edges(bp)


def test_an_agent_that_runs_other_agents_and_asks_the_model_itself_is_their_supervisor(tmp_path):
    bp = scan(_write(tmp_path, {
        "debate.py": (
            "from openai import OpenAI\n"
            "class Agent:\n    def __init__(self, llm):\n        self.llm = llm\n"
            "class BullAgent(Agent):\n"
            "    def argue(self, topic):\n        return self.llm.chat(f'You are a bullish analyst. Argue for: {topic}')\n"
            "class BearAgent(Agent):\n"
            "    def argue(self, topic):\n        return self.llm.chat(f'You are a bearish analyst. Argue against: {topic}')\n"
            "class ModeratorAgent(Agent):\n"
            "    def run(self, topic):\n"
            "        opening = self.llm.chat(f'You are the moderator of a debate. Open the debate on: {topic}')\n"
            "        bull = BullAgent(self.llm)\n        bear = BearAgent(self.llm)\n"
            "        a = bull.argue(topic)\n        b = bear.argue(topic)\n"
            "        return self.llm.chat(f'You are the moderator. Weigh {a} against {b} and decide.')\n"
        ),
        "main.py": "from debate import ModeratorAgent\ndef go(llm, topic):\n    return ModeratorAgent(llm).run(topic)\n",
    }))
    assert ids(bp) == ["bear_agent", "bull_agent", "moderator_agent"]
    assert {("moderator_agent", "bull_agent"), ("bull_agent", "moderator_agent"),
            ("moderator_agent", "bear_agent"), ("bear_agent", "moderator_agent")} <= edges(bp)
    assert ("bull_agent", "bear_agent") not in edges(bp)


# -- a file is never lost to how it was saved ---------------------------------------------------------------------

_GRAPH = (
    "from langgraph.graph import StateGraph, START, END\n"
    "def plan(state):\n    return state\n"
    "def act(state):\n    return state\n"
    "g = StateGraph(dict)\n"
    "g.add_node('plan', plan)\ng.add_node('act', act)\n"
    "g.add_edge(START, 'plan')\ng.add_edge('plan', 'act')\ng.add_edge('act', END)\n"
)


def test_a_file_saved_with_a_byte_order_mark_is_scanned_like_any_other(tmp_path):
    """Editors on Windows put U+FEFF at the start of a file; Python's parser rejects it in a string."""
    bp = scan(_write(tmp_path, {"graph.py": _GRAPH}, encoding="utf-8-sig"))
    assert ids(bp) == ["act", "plan"]
    assert ("plan", "act") in edges(bp)
    assert not collect(tmp_path)[0].text.startswith("﻿")


def test_an_instruction_is_never_taken_for_an_agent_s_name(tmp_path):
    bp = scan(_write(tmp_path, {
        "team.py": (
            "from openai import OpenAI\n"
            "class Agent:\n"
            "    def __init__(self, instruction, role):\n"
            "        self.instruction, self.role, self.client = instruction, role, OpenAI()\n"
            "    def chat(self, text):\n"
            "        msgs = [{'role': 'system', 'content': self.instruction}, {'role': 'user', 'content': text}]\n"
            "        return self.client.chat.completions.create(model='m', messages=msgs)\n"
            "moderator = Agent('You are a final decision maker who reviews all the opinions and decides.', 'Moderator')\n"
            "recruiter = Agent(instruction='You recruit the experts a question needs.', role='recruiter')\n"
        ),
    }))
    assert all(len(a["name"]) < 40 for a in bp["agents"])
    assert bp["framework"] != "crewai"  # the project's own Agent class with a role= is not CrewAI


# -- a graph is read however it is written down --------------------------------------------------------------------

def test_nodes_added_in_a_loop_over_a_table_are_all_found(tmp_path):
    """for name, fn in {...}.items(): g.add_node(name, fn) - one node per row, each with its own function."""
    bp = scan(_write(tmp_path, {
        "agent.py": (
            "from langgraph.graph import StateGraph, START, END\n"
            "class Reviewer:\n"
            "    def __init__(self, model):\n"
            "        self.model = model\n"
            "        g = StateGraph(dict)\n"
            "        for name, node in {'route': self._route, 'retrieve': self._retrieve,\n"
            "                           'generate': self._generate}.items():\n"
            "            g.add_node(name, node)\n"
            "        g.add_edge(START, 'route')\n        g.add_edge('route', 'retrieve')\n"
            "        g.add_edge('retrieve', 'generate')\n        g.add_edge('generate', END)\n"
            "        self.graph = g.compile()\n"
            "    def _route(self, s):\n        return s\n"
            "    def _retrieve(self, s):\n        return s\n"
            "    def _generate(self, s):\n        return {'answer': self.model.invoke('You are a tutor. Answer.')}\n"
        ),
    }))
    assert ids(bp) == ["generate", "retrieve", "route"]
    assert {("route", "retrieve"), ("retrieve", "generate")} <= edges(bp)


def test_nodes_named_by_an_enum_kept_in_another_module_are_all_found(tmp_path):
    """The loop runs over a list picked out of a table of (enum member, function) pairs defined elsewhere."""
    bp = scan(_write(tmp_path, {
        "names.py": (
            "from enum import StrEnum\n"
            "class AgentName(StrEnum):\n    SUPERVISOR = 'supervisor'\n    DATA = 'data'\n    REPORT = 'report'\n"
        ),
        "steps.py": (
            "from names import AgentName\n"
            "def _supervisor(state):\n    return state\n"
            "def _data(state):\n    return state\n"
            "def _report(state):\n    return state\n"
            "WORKFLOW: tuple = ((AgentName.SUPERVISOR, _supervisor), (AgentName.DATA, _data), (AgentName.REPORT, _report))\n"
        ),
        "graph.py": (
            "from langgraph.graph import StateGraph, START\n"
            "from steps import WORKFLOW\n"
            "def build(provider):\n"
            "    builder = StateGraph(dict)\n"
            "    agents = [agent for agent, _ in WORKFLOW]\n"
            "    for agent in agents:\n"
            "        builder.add_node(agent.value, make_node(agent, provider))\n"
            "    builder.add_edge(START, agents[0].value)\n"
            "    return builder.compile()\n"
        ),
    }))
    assert ids(bp) == ["data", "report", "supervisor"]


def test_arrows_added_in_a_loop_over_pairs_are_one_arrow_per_pair(tmp_path):
    bp = scan(_write(tmp_path, {
        "graph.py": (
            "from langgraph.graph import StateGraph, START, END\n"
            "def plan(s):\n    return s\n"
            "def write(s):\n    return s\n"
            "def review(s):\n    return s\n"
            "EDGES = [('plan', 'write'), ('write', 'review')]\n"
            "g = StateGraph(dict)\n"
            "for fn in (plan, write, review):\n    g.add_node(fn)\n"
            "g.add_edge(START, 'plan')\n"
            "for a, b in EDGES:\n    g.add_edge(a, b)\n"
            "g.add_edge('review', END)\n"
        ),
    }))
    assert ids(bp) == ["plan", "review", "write"]
    assert {("plan", "write"), ("write", "review")} <= edges(bp)
    assert ("plan", "review") not in edges(bp) and ("write", "write") not in edges(bp)


def test_a_record_put_in_the_project_s_own_graph_is_not_an_agent(tmp_path):
    """A parking map with its own add_node next to a LangGraph workflow: only the workflow's nodes are agents."""
    bp = scan(_write(tmp_path, {
        "workflow.py": _GRAPH,
        "navigation.py": (
            "from dataclasses import dataclass\n"
            "@dataclass\nclass NavNode:\n    id: str\n    x: float\n"
            "class NavGraph:\n"
            "    def __init__(self):\n        self._nodes = {}\n"
            "    def add_node(self, node):\n        self._nodes[node.id] = node\n"
            "def build_map(spots):\n"
            "    graph = NavGraph()\n"
            "    graph.add_node(NavNode('entrance', 0.0))\n"
            "    for spot in spots:\n"
            "        spot_node = NavNode(spot.id, spot.x)\n"
            "        graph.add_node(spot_node)\n"
            "    return graph\n"
        ),
    }))
    assert ids(bp) == ["act", "plan"]


def test_a_helper_of_a_wrapped_node_function_is_part_of_that_node(tmp_path):
    """add_node("modeler", _wrap(modeler_node, "modeler")): what modeler_node calls is the modeler, not a new agent."""
    bp = scan(_write(tmp_path, {
        "graph.py": (
            "from langgraph.graph import StateGraph, START, END\n"
            "from nodes import modeler_node, critic_node\n"
            "def _wrap(fn, name):\n"
            "    def _inner(state):\n        return fn(state)\n"
            "    return _inner\n"
            "g = StateGraph(dict)\n"
            "g.add_node('modeler', _wrap(modeler_node, 'modeler'))\n"
            "g.add_node('critic', _wrap(critic_node, 'critic'))\n"
            "g.add_edge(START, 'modeler')\ng.add_edge('modeler', 'critic')\ng.add_edge('critic', END)\n"
        ),
        "nodes.py": (
            "from langchain_openai import ChatOpenAI\nllm = ChatOpenAI()\n"
            "def _build_model_draft(state):\n"
            "    prompt = f'You are a mathematical modeler. Draft a model for: {state}'\n"
            "    return llm.invoke(prompt)\n"
            "def modeler_node(state):\n    return {'draft': _build_model_draft(state)}\n"
            "def critic_node(state):\n    return {'review': llm.invoke('You are a critic. Review the draft.')}\n"
        ),
    }))
    assert ids(bp) == ["critic", "modeler"]


def test_model_vendor_wrappers_and_one_off_scripts_do_not_join_the_agents(tmp_path):
    """Beside a graph, the classes agents reach a model through and a probe script are not more agents."""
    bp = scan(_write(tmp_path, {
        "app/workflow.py": _GRAPH,
        "app/client/base_client.py": (
            "from langchain_core.messages import SystemMessage, HumanMessage\n"
            "class BaseClient:\n"
            "    def llm_model(self):\n        raise NotImplementedError\n"
            "    def chat(self, text):\n"
            "        system = 'You are a helpful assistant. Answer briefly and truthfully.'\n"
            "        llm = self.llm_model()\n"
            "        return llm.invoke([SystemMessage(content=system), HumanMessage(content=text)])\n"
        ),
        "app/client/vendors.py": (
            "from langchain_openai import ChatOpenAI\nfrom app.client.base_client import BaseClient\n"
            "class DeepSeekClient(BaseClient):\n    def llm_model(self):\n        return ChatOpenAI(model='deepseek')\n"
            "class OllamaClient(BaseClient):\n    def llm_model(self):\n        return ChatOpenAI(model='llama')\n"
        ),
        "app/factory.py": (
            "from app.client.vendors import DeepSeekClient, OllamaClient\n"
            "def client_for(name):\n    return DeepSeekClient() if name == 'deepseek' else OllamaClient()\n"
        ),
        "scripts/probe.py": (
            "from langchain_openai import ChatOpenAI\n"
            "def probe_raw(question):\n"
            "    prompt = f'You are an analyst. Answer this probe question: {question}'\n"
            "    return ChatOpenAI().invoke(prompt)\n"
            "def main():\n    print(probe_raw('ping'))\n"
        ),
        "app/cli.py": "from scripts.probe import probe_raw\n",
    }))
    assert ids(bp) == ["act", "plan"]
    from visualagents.static_scan import MODEL_CLASS
    assert all(MODEL_CLASS.search(n) for n in ("DeepSeekClient", "OpenAIProvider", "LocalLLM", "BaseChatModel"))
    assert not any(MODEL_CLASS.search(n) for n in ("ScriptParserAgent", "Planner", "ClientSupportAgent", "ReviewNode"))
