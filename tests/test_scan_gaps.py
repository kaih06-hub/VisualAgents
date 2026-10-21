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


def test_an_agent_named_like_a_test_is_part_of_the_project(tmp_path):
    """sub_agents/test_runner.py builds the agent that runs the user's tests; tests/ or a pytest module is a test."""
    from visualagents.collector import is_test_file

    runner = (
        "from google.adk.agents import Agent\n"
        "async def test_runner_instruction_provider(context):\n"
        "    return 'Run the tests in the submitted code and report failures.'\n"
        "test_runner_agent = Agent(name='TestRunner', model='gemini-2.5-pro',\n"
        "                          instruction=test_runner_instruction_provider)\n")
    pytest_module = "from review import analyzer\ndef test_analyzer_scores_style():\n    assert analyzer.score('x') >= 0\n"
    assert not is_test_file("review/sub_agents/test_runner.py", runner)
    assert is_test_file("review/test_analyzer.py", pytest_module)
    assert is_test_file("tests/test_runner.py", runner)  # anything in a tests folder still is

    _write(tmp_path, {
        "review/agent.py": (
            "from google.adk.agents import Agent, SequentialAgent\n"
            "from review.sub_agents.test_runner import test_runner_agent\n"
            "analyzer = Agent(name='CodeAnalyzer', model='gemini-2.5-flash', instruction='Analyse the code')\n"
            "pipeline = SequentialAgent(name='ReviewPipeline', sub_agents=[analyzer, test_runner_agent])\n"),
        "review/sub_agents/test_runner.py": runner,
        "review/test_analyzer.py": pytest_module,
    })
    assert "testrunner" in {a["id"].replace("_", "") for a in scan(tmp_path)["agents"]}


# -- gaps found by the fifth README-graph set ----------------------------------------------------------------

def test_a_big_notebook_with_shell_magics_is_read(tmp_path):
    """Saved outputs make notebooks megabytes, and `!pip install` is not Python: neither hides the agents."""
    import json

    cells = [
        {"cell_type": "code", "source": ["!pip install -q crewai\n", "%matplotlib inline\n"], "outputs": []},
        {"cell_type": "code", "source": [
            "from crewai import Agent, Task, Crew, Process\n",
            "analyst = Agent(role='Research Analyst', goal='Analyse the paper', backstory='x')\n",
            "writer = Agent(role='Academic Writer', goal='Write the report', backstory='x')\n",
            "t1 = Task(description='analyse', expected_output='notes', agent=analyst)\n",
            "t2 = Task(description='write', expected_output='report', agent=writer)\n",
            "crew = Crew(agents=[analyst, writer], tasks=[t1, t2], process=Process.sequential)\n"],
         "outputs": [{"output_type": "display_data", "data": {"image/png": "A" * 600_000}}]},
        {"cell_type": "code", "source": ["%%bash\n", "echo not python\n"], "outputs": []},
    ]
    (tmp_path / "paper.ipynb").write_text(json.dumps({"cells": cells, "nbformat": 4, "nbformat_minor": 5}))
    bp = scan(tmp_path)
    assert ids(bp) == ["academic_writer", "research_analyst"]
    assert ("research_analyst", "academic_writer") in edges(bp)


def test_agents_built_by_factories_with_constant_names(tmp_path):
    _write(tmp_path, {"team.py": (
        "from smolagents import CodeAgent\n"
        "WEB_NAME = 'web_researcher'\n"
        "MEDIA_NAME = 'media_analyst'\n"
        "def build_agent(model):\n"
        "    return CodeAgent(tools=[], model=model, instructions='Solve the task')\n"
        "def build_team(model):\n"
        "    team = []\n"
        "    team.append(CodeAgent(tools=[], model=model, name=WEB_NAME, description='searches'))\n"
        "    team.append(CodeAgent(tools=[], model=model, name=MEDIA_NAME, description='reads media'))\n"
        "    return CodeAgent(tools=[], model=model, managed_agents=team, instructions='Lead the team')\n"),
        "registry.py": (
            "from agents_lib import ResearchAgent\n"
            "def create_agent(name, registry):\n"
            "    return registry.create_agent(name)  # a generic registry: no agent of its own\n"
            "def make_researcher(llm):\n"
            "    return ResearchAgent(llm=llm, cache=None)  # the project's own class, not a framework agent\n"),
    })
    bp = scan(tmp_path)
    assert {"build_agent", "build_team", "web_researcher", "media_analyst"} <= set(ids(bp))
    assert "create_agent" not in ids(bp) and "make_researcher" not in ids(bp)
    assert {("build_team", "web_researcher"), ("build_team", "media_analyst")} <= edges(bp)


def test_delegation_through_tool_functions_and_run_helpers(tmp_path):
    _write(tmp_path, {
        "helpers.py": "async def run_agent(agent, prompt):\n    return await agent.run(prompt)\n",
        "team.py": (
            "from pydantic_ai import Agent\n"
            "from helpers import run_agent\n"
            "def register(agent, tools):\n"
            "    for t in tools:\n"
            "        agent.tool_plain(t)\n"
            "def build():\n"
            "    research = Agent('openai:gpt-5', name='research_agent', instructions='Research')\n"
            "    math = Agent('openai:gpt-5', name='math_agent', instructions='Do maths')\n"
            "    critic = Agent('openai:gpt-5', name='critic_agent', instructions='Check the answer')\n"
            "    supervisor = Agent('openai:gpt-5', name='supervisor_agent', instructions='Delegate')\n"
            "    async def ask_research(q):\n"
            "        return await run_agent(agent=research, prompt=q)\n"
            "    async def ask_math(q):\n"
            "        return await math.run(q)\n"
            "    async def check(answer):\n"
            "        return await critic.run(answer)\n"
            "    register(supervisor, [ask_research, ask_math])\n"
            "    supervisor.validate = check\n"
            "    return supervisor\n"),
    })
    assert {("supervisor_agent", "research_agent"), ("supervisor_agent", "math_agent"),
            ("supervisor_agent", "critic_agent")} <= edges(scan(tmp_path))


def test_a_hierarchical_crew_with_members_built_in_a_method(tmp_path):
    _write(tmp_path, {"crew.py": (
        "from crewai import Agent, Crew, Process\n"
        "class SalesCrew:\n"
        "    def __init__(self, cfg):\n"
        "        self.cfg = cfg\n"
        "        self.manager = Agent(role='Sales Manager', goal='g', backstory='b')\n"
        "        self.agents = self._build_agents()\n"
        "    def _build_agents(self):\n"
        "        agents = {}\n"
        "        agents['lead_finder'] = Agent(role=self.cfg['lead_finder'], goal='g', backstory='b')\n"
        "        agents['email_writer'] = Agent(role=self.cfg['email_writer'], goal='g', backstory='b')\n"
        "        return agents\n"
        "    def crew(self):\n"
        "        workers = list(self.agents.values())\n"
        "        mode = Process.hierarchical\n"
        "        return Crew(agents=workers, tasks=[], process=mode, manager_agent=self.manager)\n"),
        "agents.yaml": "lead_finder:\n  role: Lead Finder\nemail_writer:\n  role: Email Writer\n",
    })
    assert {("sales_manager", "lead_finder"), ("sales_manager", "email_writer")} <= edges(scan(tmp_path))


def test_a_dspy_program_running_three_predictors_is_drawn_as_steps(tmp_path):
    _write(tmp_path, {"rag.py": (
        "import dspy\n"
        "class InputGuard(dspy.Signature):\n    'Block unsafe questions.'\n"
        "class Answer(dspy.Signature):\n    'Answer from the context.'\n"
        "class OutputGuard(dspy.Signature):\n    'Block unsupported answers.'\n"
        "class SafeRAG(dspy.Module):\n"
        "    def __init__(self):\n"
        "        self.input_guard = dspy.Predict(InputGuard)\n"
        "        self.answer = dspy.ChainOfThought(Answer)\n"
        "        self.output_guard = dspy.Predict(OutputGuard)\n"
        "    def forward(self, question):\n"
        "        if self.input_guard(question=question).blocked:\n"
        "            return None\n"
        "        reply = self.answer(question=question)\n"
        "        return self.output_guard(answer=reply.answer)\n"
        "rag = SafeRAG()\n"),
        "summary.py": (
            "import dspy\n"
            "class Summarize(dspy.Module):\n"
            "    def __init__(self):\n"
            "        self.draft = dspy.ChainOfThought('text -> summary')\n"
            "    def forward(self, text):\n"
            "        return self.draft(text=text)\n"
            "s = Summarize()\n"),
    })
    bp = scan(tmp_path)
    assert {"input_guard", "answer", "output_guard", "summarize"} <= set(ids(bp))  # one predictor: one agent
    assert {("input_guard", "answer"), ("answer", "output_guard")} <= edges(bp)
