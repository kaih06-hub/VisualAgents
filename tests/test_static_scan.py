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
