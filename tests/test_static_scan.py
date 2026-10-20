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
        "chat = autogen.GroupChat(agents=[planner, coder, critic], messages=[])\n"
    )
    bp = scan(tmp_path)
    assert bp["framework"] == "autogen"
    assert ("critic", "planner") in edges(bp)


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
    # a second system whose agents are classes: the DSPy programs next to them are their parts, not agents
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
    assert set(by_id) == {"wiki_writer", "topic_expert", "write_outline", "co_storm_expert", "moderator"}
    assert {("wiki_writer", "topic_expert"), ("topic_expert", "wiki_writer")} <= edges(bp)  # the turns loop
    assert by_id["co_storm_expert"]["tools"] == ["UtteranceModule"]
    assert by_id["co_storm_expert"]["description"] == "A domain expert in the roundtable."
    assert by_id["wiki_writer"]["file"] == "storm/curation.py" and by_id["wiki_writer"]["line"] == 4


def test_the_same_team_on_another_framework_is_an_alternative_not_more_agents(tmp_path):
    _write(tmp_path, {
        "multi_agents/agents/orchestrator.py": (
            "from langgraph.graph import StateGraph, END\n"
            "g = StateGraph(dict)\n"
            "g.add_node('planner', editor.plan)\n"
            "g.add_node('researcher', research)\n"
            "g.add_node('writer', write)\n"
            "g.add_node('reviewer', review)\n"
            "g.set_entry_point('planner')\n"
            "g.add_edge('planner', 'researcher')\n"
            "g.add_edge('researcher', 'writer')\n"
            "g.add_edge('writer', 'reviewer')\n"
            "g.add_edge('reviewer', END)\n"
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
    alternatives = []
    bp = scan(tmp_path, alternatives=alternatives)
    assert bp["framework"] == "langgraph"
    assert [a["id"] for a in bp["agents"]] == ["planner", "researcher", "writer", "reviewer"]
    assert {a["file"] for a in bp["agents"]} == {"multi_agents/agents/orchestrator.py"}
    assert edges(bp) == {("planner", "researcher"), ("researcher", "writer"), ("writer", "reviewer")}
    assert alternatives[0]["kind"] == "port" and alternatives[0]["paths"] == ["multi_agents/ag2"]
    assert "ChiefEditor" in alternatives[0]["agents"]
    assert "multi_agents/ag2" in bp["summary"]


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


def test_role_variants_a_config_picks_between_are_not_parallel_workers(tmp_path):
    _write(tmp_path, _ROLES)
    alternatives = []
    bp = scan(tmp_path, alternatives=alternatives)
    names = {a["name"] for a in bp["agents"]}
    assert {"Planner", "CodeGenerator", "ImageReader"} <= names  # an optional extra role is not a variant
    assert not names & {"CodeGeneratorCLIOnly", "CodeGeneratorPluginOnly"}
    alt = alternatives[0]
    assert alt["kind"] == "config" and alt["default"] == ["planner", "code_interpreter"]
    assert alt["file"] == "app/session/session.py" and alt["line"] == 3
    assert alt["paths"] == ["app/code_interpreter/code_interpreter_cli_only",
                            "app/code_interpreter/code_interpreter_plugin_only"]


def test_a_roles_list_of_chat_roles_changes_nothing(tmp_path):
    _write(tmp_path, {k: v for k, v in _ROLES.items() if "session" not in k})
    _write(tmp_path, {"app/chat.py": "ROLES = ['user', 'assistant']\n"})
    alternatives = []
    bp = scan(tmp_path, alternatives=alternatives)
    assert alternatives == []
    assert "CodeGeneratorCLIOnly" in {a["name"] for a in bp["agents"]}
