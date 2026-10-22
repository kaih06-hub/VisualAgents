from pathlib import Path

from visualagents.static_scan import scan

ROOT = Path(__file__).resolve().parents[1]


def edges(bp):
    return {(e["from"], e["to"]) for e in bp["edges"]}


def test_demo_crew():
    bp = scan(ROOT / "visualagents" / "demo")
    assert [a["id"] for a in bp["agents"]] == ["planner", "researcher", "writer", "critic"]
    assert ("critic", "writer") in edges(bp)
    assert bp["entry"] == ["planner"]
    tools = {a["id"]: a["tools"] for a in bp["agents"]}
    assert tools["researcher"] == ["web_search", "read_paper"]


def test_langgraph_supervisor():
    bp = scan(ROOT / "examples" / "langgraph_supervisor")
    assert bp["framework"] == "langgraph"
    assert {a["id"] for a in bp["agents"]} == {"supervisor", "researcher", "coder", "reviewer"}
    assert {("supervisor", "coder"), ("coder", "supervisor"), ("reviewer", "supervisor")} <= edges(bp)
    assert bp["entry"] == ["supervisor"]
    assert {a["id"]: a["tools"] for a in bp["agents"]}["reviewer"] == ["run_tests"]


def test_crewai_yaml_project():
    bp = scan(ROOT / "examples" / "crewai_project")
    assert bp["framework"] == "crewai"
    ids = [a["id"] for a in bp["agents"]]
    assert ids == ["researcher", "reporting_analyst"]
    assert ("researcher", "reporting_analyst") in edges(bp)
    assert "Senior Data Researcher" in bp["agents"][0]["role"]


def test_openai_agents_handoffs(tmp_path):
    (tmp_path / "triage.py").write_text(
        "from agents import Agent, Runner\n"
        "billing = Agent(name='Billing agent', instructions='Handle billing')\n"
        "refunds = Agent(name='Refund agent', instructions='Handle refunds')\n"
        "triage = Agent(name='Triage agent', instructions='Route', handoffs=[billing, refunds])\n"
    )
    bp = scan(tmp_path)
    assert bp["framework"] == "openai-agents"
    assert edges(bp) == {("triage_agent", "billing_agent"), ("triage_agent", "refund_agent")}
    assert bp["entry"] == ["triage_agent"]


def test_langgraph_create_react_agent_is_folded_into_its_node(tmp_path):
    (tmp_path / "graph.py").write_text(
        "from langgraph.graph import StateGraph, START, END\n"
        "from langgraph.prebuilt import create_react_agent\n"
        "research_agent = create_react_agent(llm, tools=[web_search], prompt='You research')\n"
        "def research_node(state):\n"
        "    return research_agent.invoke(state)\n"
        "def write_node(state):\n"
        "    return state\n"
        "g = StateGraph(dict)\n"
        "g.add_node('research', research_node)\n"
        "g.add_node('write', write_node)\n"
        "g.add_edge(START, 'research')\n"
        "g.add_edge('research', 'write')\n"
        "g.add_edge('write', END)\n"
    )
    bp = scan(tmp_path)
    assert [a["id"] for a in bp["agents"]] == ["research", "write"]
    assert bp["agents"][0]["tools"] == ["web_search"]
    assert bp["exits"] == ["write"]


def test_autogen_group_chat(tmp_path):
    (tmp_path / "chat.py").write_text(
        "import autogen\n"
        "planner = autogen.AssistantAgent(name='planner', system_message='Plan')\n"
        "coder = autogen.AssistantAgent(name='coder', system_message='Code')\n"
        "critic = autogen.AssistantAgent(name='critic', system_message='Review')\n"
        "user = autogen.UserProxyAgent(name='user')\n"
        "chat = autogen.GroupChat(agents=[user, planner, coder, critic], messages=[])\n"
    )
    bp = scan(tmp_path)
    assert bp["framework"] == "autogen"
    # the manager picks each speaker: the listed order, not a ring, and the person is not a scheduled speaker
    assert edges(bp) == {("planner", "coder"), ("coder", "critic")}

    (tmp_path / "chat.py").write_text((tmp_path / "chat.py").read_text().replace(
        "messages=[])", "messages=[], speaker_selection_method='round_robin')"))
    assert ("critic", "user") in edges(scan(tmp_path))  # round robin goes round


def test_classes_that_call_an_llm_are_agents_without_a_framework(tmp_path):
    (tmp_path / "programmer.py").write_text(
        "import openai\n"
        "class Programmer:\n"
        "    \"\"\"Writes analysis code.\"\"\"\n"
        "    def __init__(self):\n"
        "        self.client = openai.OpenAI()\n"
        "class Helper:\n"
        "    def run(self):\n"
        "        return 1\n"
    )
    (tmp_path / "inspector.py").write_text(
        "class Inspector:\n"
        "    def check(self, client):\n"
        "        return client.chat.completions.create(model='m', messages=[])\n"
    )
    (tmp_path / "conversation.py").write_text(
        "from programmer import Programmer\n"
        "class Conversation:\n"
        "    def __init__(self):\n"
        "        self.programmer = Programmer()\n"
    )
    bp = scan(tmp_path)
    assert sorted(a["id"] for a in bp["agents"]) == ["inspector", "programmer"]
    prog = next(a for a in bp["agents"] if a["id"] == "programmer")
    assert prog["description"] == "Writes analysis code."
    assert "programmer" in prog["aliases"]


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_camel_agents_are_camel_not_crewai(tmp_path):
    _write(tmp_path, {
        "examples/run.py": (
            "from camel.agents import ChatAgent\n"
            "from camel.tasks.task import Task\n"
            "from camel.societies import Workforce\n"
            "web_agent = ChatAgent(system_message='You search the web', tools=[search])\n"
            "coder = ChatAgent(system_message='You write code')\n"
            "task = Task(content=question, id='0')\n"
        ),
        # a "crewai" string is not an import, and framework-free agents count for less than CAMEL ones
        "usecase/sec_tools.py": "HEADERS = {'User-Agent': 'crewai.com bisan@crewai.com'}\n",
        "usecase/main.py": "".join(f"a{i} = Analyst{i}Agent(model_name=m)\n" for i in range(3)),
    })
    bp = scan(tmp_path)
    assert bp["framework"] == "camel"
    assert {"web_agent", "coder"} <= {a["id"] for a in bp["agents"]}
    assert not any(s["title"].lower() == "task" for s in bp["steps"])  # CAMEL's Task is not a CrewAI task


def test_a_langgraph_team_outvotes_files_that_use_langchain_for_model_calls(tmp_path):
    for i in range(6):
        _write(tmp_path, {f"lib/provider_{i}.py": "from langchain_openai import ChatOpenAI\nllm = ChatOpenAI()\n"})
    _write(tmp_path, {"team/graph.py": (
        "from langgraph.graph import StateGraph, START, END\n"
        "from langchain_openai import ChatOpenAI\n"
        "g = StateGraph(dict)\n"
        "g.add_node('planner', plan)\n"
        "g.add_node('writer', write)\n"
        "g.add_edge(START, 'planner')\n"
        "g.add_edge('planner', 'writer')\n"
    )})
    assert scan(tmp_path)["framework"] == "langgraph"


def test_a_local_agents_module_is_not_the_openai_agents_sdk(tmp_path):
    _write(tmp_path, {
        "agents.py": "class PhDStudentAgent:\n    pass\nclass PostdocAgent:\n    pass\n",
        "lab.py": ("from agents import *\n"
                   "phd = PhDStudentAgent(model='gpt-5', notes=[])\n"
                   "postdoc = PostdocAgent(model='gpt-5', notes=[])\n"),
    })
    bp = scan(tmp_path)
    assert bp["framework"] == "custom"
    assert {a["id"] for a in bp["agents"]} == {"phd", "postdoc"}


_DSPY_PIPELINE = {
    "storm/curation.py": (
        "import dspy\n"
        "class AskQuestion(dspy.Signature):\n"
        "    \"\"\"You are a Wikipedia writer asking an expert.\"\"\"\n"
        "class WikiWriter(dspy.Module):\n"
        "    \"\"\"Asks questions from a persona.\"\"\"\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.ask = dspy.ChainOfThought(AskQuestion)\n"
        "    def forward(self, topic):\n"
        "        return self.ask(topic=topic)\n"
        "class TopicExpert(dspy.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.answer = dspy.Predict('question -> answer')\n"
        "    def forward(self, question):\n"
        "        return self.answer(question=question)\n"
        "class ConvSimulator(dspy.Module):\n"
        "    def __init__(self, max_turn):\n"
        "        super().__init__()\n"
        "        self.wiki_writer = WikiWriter()\n"
        "        self.topic_expert = TopicExpert()\n"
        "        self.max_turn = max_turn\n"
        "    def forward(self, topic):\n"
        "        for _ in range(self.max_turn):\n"
        "            q = self.wiki_writer(topic=topic)\n"
        "            a = self.topic_expert(question=q)\n"
        "        return a\n"
    ),
    "storm/outline.py": (
        "from dspy import Module, Predict\n"
        "class WriteOutline(Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.draft = Predict('topic -> outline')\n"
        "class NaiveOutlineGen(Module):\n"  # nothing creates it: not part of a run
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.write = Predict('topic -> outline')\n"
    ),
    "storm/runner.py": (
        "from .curation import ConvSimulator\n"
        "from .outline import WriteOutline\n"
        "class WikiRunner:\n"
        "    def __init__(self):\n"
        "        self.curation = ConvSimulator(max_turn=3)\n"
        "        self.outline = WriteOutline()\n"
    ),
    "interface.py": "from abc import ABC\nclass Agent(ABC):\n    pass\n",
    # a second system whose agents are classes; a program one of them holds is its tool
    "costorm/agents.py": (
        "import dspy\n"
        "from interface import Agent\n"
        "from .utterance import UtteranceModule\n"
        "class CoStormExpert(Agent):\n"
        "    \"\"\"A domain expert in the roundtable.\"\"\"\n"
        "    def __init__(self):\n"
        "        self.speak = UtteranceModule()\n"
        "class Moderator(Agent):\n"
        "    pass\n"
    ),
    "costorm/utterance.py": (
        "import dspy\n"
        "class UtteranceModule(dspy.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.gen = dspy.Predict('context -> utterance')\n"
    ),
    "costorm/summary.py": (
        "import dspy\n"
        "class SummaryModule(dspy.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.gen = dspy.Predict('kb -> summary')\n"
        "s = SummaryModule()\n"
    ),
}


def test_dspy_programs_and_agent_subclasses_are_agents_in_every_pipeline(tmp_path):
    _write(tmp_path, _DSPY_PIPELINE)
    bp = scan(tmp_path)
    assert bp["framework"] == "dspy"
    by_id = {a["id"]: a for a in bp["agents"]}
    # NaiveOutlineGen is never created: not part of a run. UtteranceModule belongs to the agent that holds it.
    assert set(by_id) == {"wiki_writer", "topic_expert", "write_outline", "summary_module", "co_storm_expert",
                          "moderator"}
    assert {("wiki_writer", "topic_expert"), ("topic_expert", "wiki_writer")} <= edges(bp)  # the turns loop
    assert by_id["co_storm_expert"]["tools"] == ["UtteranceModule"]
    assert by_id["co_storm_expert"]["description"] == "A domain expert in the roundtable."
    assert by_id["wiki_writer"]["file"] == "storm/curation.py" and by_id["wiki_writer"]["line"] == 4
    # who creates each program is a fact for the model: a step of a runner, or part of something bigger
    assert by_id["wiki_writer"]["role"] == "DSPy program, created by ConvSimulator"
    assert by_id["write_outline"]["role"] == "DSPy program, created by WikiRunner"
    assert by_id["summary_module"]["role"] == "DSPy program"


def test_the_same_team_on_two_frameworks_is_reported_not_dropped(tmp_path):
    _write(tmp_path, {
        "multi_agents/agents/orchestrator.py": (
            "from langgraph.graph import StateGraph, END\n"
            "g = StateGraph(dict)\n"
            + "".join(f"g.add_node('{n}', {n})\n" for n in ("planner", "researcher", "writer", "reviewer", "publisher"))
            + "g.set_entry_point('planner')\n"
            "g.add_edge('planner', 'researcher')\n"
            "g.add_edge('researcher', 'writer')\n"
            "g.add_edge('writer', 'reviewer')\n"
            "g.add_edge('reviewer', 'publisher')\n"
            "g.add_edge('publisher', END)\n"
        ),
        "multi_agents/ag2/agents/orchestrator.py": (
            "from autogen import ConversableAgent, GroupChat\n"
            "chief = ConversableAgent(name='ChiefEditor', system_message='You coordinate.')\n"
            "researcher = ConversableAgent(name='Researcher', system_message='You research.')\n"
            "reviewer = ConversableAgent(name='Reviewer', system_message='You review.')\n"
            "writer = ConversableAgent(name='Writer', system_message='You write.')\n"
            "chat = GroupChat(agents=[chief, researcher, reviewer, writer], messages=[])\n"
        ),
    })
    facts = {}
    bp = scan(tmp_path, facts=facts)
    assert bp["framework"] == "langgraph"  # every file that defines an agent counts, whatever the file order
    assert "chief_editor" in {a["id"] for a in bp["agents"]}  # static analysis reports; the model decides
    port = facts["ports"][0]
    assert port["shared_agents"] == ["Researcher", "Writer", "Reviewer"]
    assert [(i["framework"], i["folder"]) for i in port["implementations"]] == [
        ("langgraph", "multi_agents/agents"), ("autogen", "multi_agents/ag2")]


_ROLES = {
    "app/session/session.py": (
        "class SessionConfig(ModuleConfig):\n"
        "    def _configure(self):\n"
        "        self.roles = self._get_list('roles', ['planner', 'code_interpreter'])\n"
    ),
    "app/planner/planner.py": (
        "class Planner(Role):\n"
        "    def reply(self, client):\n"
        "        return client.chat.completions.create(model='m', messages=[{'role': 'system', 'content': 'Plan'}])\n"
    ),
    **{f"app/code_interpreter/{folder}/{stem}.py": (
        f"class {cls}(Role):\n"
        "    def reply(self, client):\n"
        "        return client.chat.completions.create(model='m', messages=[{'role': 'system', 'content': 'Code'}])\n"
    ) for folder, stem, cls in (("code_interpreter", "code_generator", "CodeGenerator"),
                                ("code_interpreter_cli_only", "code_generator_cli_only", "CodeGeneratorCLIOnly"),
                                ("code_interpreter_plugin_only", "code_generator_plugin_only", "CodeGeneratorPluginOnly"))},
    "app/ext_role/image_reader/image_reader.py": (
        "class ImageReader(Role):\n"
        "    def reply(self, client):\n"
        "        return client.chat.completions.create(model='m', messages=[{'role': 'system', 'content': 'Read'}])\n"
    ),
}


def test_a_default_roles_list_naming_parts_of_the_project_is_reported(tmp_path):
    _write(tmp_path, _ROLES)
    facts = {}
    bp = scan(tmp_path, facts=facts)
    assert facts["config_lists"] == [{"file": "app/session/session.py", "line": 3,
                                      "names": ["planner", "code_interpreter"]}]
    # nothing is dropped for it: which roles a default run uses is for the model to read
    assert {"Planner", "CodeGenerator", "ImageReader"} <= {a["name"] for a in bp["agents"]}
    # the CLI-only build of the code generator is the same agent in another mode: it joins CodeGenerator
    gen = next(a for a in bp["agents"] if a["name"] == "CodeGenerator")
    assert "CodeGeneratorCLIOnly" in gen["aliases"]


def test_a_list_of_chat_roles_is_not_a_config_fact(tmp_path):
    _write(tmp_path, {k: v for k, v in _ROLES.items() if "session" not in k})
    _write(tmp_path, {"app/chat.py": "ROLES = ['user', 'assistant']\n"})
    facts = {}
    scan(tmp_path, facts=facts)
    assert "config_lists" not in facts


# -- static facts that carry over to unseen projects ---------------------------------------------------------


def test_graph_calls_are_langgraph_only_where_langgraph_is_imported(tmp_path):
    _write(tmp_path, {
        "viz.py": "import networkx as nx\nG = nx.DiGraph()\nG.add_node('parent_id')\nG.add_edge('a', 'b')\n",
        # a project's own graph of actions, built from constants: not an agent graph
        "actions.py": "from .graph import ActionGraph\ng = ActionGraph()\ng.add_node(ISSUE_TYPE)\ng.add_node(PRODUCT_GOALS)\n",
        "team.py": "from agents import Agent\nwriter = Agent(name='Writer', instructions='Write')\n",
    })
    bp = scan(tmp_path)
    assert {a["id"] for a in bp["agents"]} == {"writer"}
    assert bp["framework"] == "openai-agents"


def test_a_graph_builder_split_from_the_module_that_imports_langgraph_still_counts(tmp_path):
    _write(tmp_path, {
        "graph.py": "from langgraph.graph import StateGraph\nfrom nodes import build\ng = build(StateGraph(dict))\n",
        "nodes.py": "def build(g):\n    g.add_node('plan', plan)\n    g.add_node('act', act)\n    g.add_edge('plan', 'act')\n"
                    "    return g\n",
    })
    bp = scan(tmp_path)
    assert {a["id"] for a in bp["agents"]} == {"plan", "act"} and ("plan", "act") in edges(bp)
    assert bp["framework"] == "langgraph"


def test_agent_classes_on_a_projects_own_base_are_agents(tmp_path):
    _write(tmp_path, {
        "agents/base.py": "class BaseAgent:\n    pass\n",
        "agents/solver.py": "from .base import BaseAgent\nclass SolverAgent(BaseAgent):\n    \"\"\"Proposes a solution.\"\"\"\n",
        "agents/critic.py": "from .base import BaseAgent\nclass CriticAgent(BaseAgent):\n    pass\n",
        "roles/role.py": "class Role:\n    pass\nclass Architect(Role):\n    pass\n",
        "loop.py": (  # a concrete agent other classes extend is still an agent when the project creates it
            "from abc import ABC\n"
            "class AbstractAgent(ABC):\n    pass\n"
            "class DefaultAgent(AbstractAgent):\n    pass\n"
            "class ShellAgent(DefaultAgent):\n    pass\n"
            "agent = DefaultAgent.from_config(cfg)\n"
        ),
        "state.py": "from typing import TypedDict\nclass AgentState(TypedDict):\n    x: int\n"
                    "class ResearchState(AgentState):\n    y: int\n",
    })
    bp = scan(tmp_path)
    names = {a["name"] for a in bp["agents"]}
    assert names == {"SolverAgent", "CriticAgent", "Architect", "DefaultAgent", "ShellAgent"}
    assert next(a for a in bp["agents"] if a["name"] == "SolverAgent")["description"] == "Proposes a solution."


def test_code_that_picks_one_agent_by_a_setting_is_a_fact(tmp_path):
    _write(tmp_path, {
        "agents.py": (
            "class AbstractAgent:\n    pass\n"
            "class DefaultAgent(AbstractAgent):\n    pass\n"
            "class RetryAgent(AbstractAgent):\n    pass\n"
            "class ShellAgent(AbstractAgent):\n    pass\n"
            "def get_agent_from_config(config):\n"
            "    if config.type == 'default':\n"
            "        return DefaultAgent.from_config(config)\n"
            "    elif config.type == 'retry':\n"
            "        return RetryAgent.from_config(config)\n"
            "    elif config.type == 'shell':\n"
            "        return ShellAgent.from_config(config)\n"
            "    raise ValueError(config.type)\n"
            "REGISTRY = {'default': DefaultAgent, 'retry': RetryAgent}\n"
            "LABELS = {'default': 'Default', 'retry': 'Retry'}\n"  # strings, not agents
        ),
    })
    facts = {}
    scan(tmp_path, facts=facts)
    by_line = {f["line"]: f for f in facts["factories"]}
    assert by_line[10] == {"file": "agents.py", "line": 10, "chooses_by": "config.type",
                           "options": {"default": "DefaultAgent", "retry": "RetryAgent", "shell": "ShellAgent"}}
    assert by_line[17] == {"file": "agents.py", "line": 17, "chooses_by": "a registry key",
                           "options": {"default": "DefaultAgent", "retry": "RetryAgent"}}
    assert len(facts["factories"]) == 2  # the elif branches are one chain, and LABELS maps to strings


def test_an_agents_module_inside_a_package_does_not_hide_the_sdk(tmp_path):
    _write(tmp_path, {
        "backend/__init__.py": "",
        "backend/airline/__init__.py": "",
        "backend/airline/agents.py": (  # imported as airline.agents; `from agents import` here is the SDK
            "from agents import Agent, handoff\n"
            "faq = Agent[Context](name='FAQ Agent', instructions='Answer policy questions')\n"
            "billing = Agent[Context](name='Billing Agent', instructions='Handle billing')\n"
            "triage = Agent[Context](name='Triage Agent', instructions='Route', handoffs=[])\n"
            "triage.handoffs = [faq, handoff(agent=billing, on_handoff=log)]\n"
            "faq.handoffs.append(triage)\n"
            "billing.handoffs.extend([faq, triage])\n"
        ),
    })
    bp = scan(tmp_path)
    assert bp["framework"] == "openai-agents"
    assert {a["id"] for a in bp["agents"]} == {"faq_agent", "billing_agent", "triage_agent"}
    assert edges(bp) == {("triage_agent", "faq_agent"), ("triage_agent", "billing_agent"),
                         ("faq_agent", "triage_agent"), ("billing_agent", "faq_agent"), ("billing_agent", "triage_agent")}
