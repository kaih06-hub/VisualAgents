"""Static-analysis gaps found by the sixth README-graph set, each checked on a made-up project."""

from pathlib import Path

from visualagents.static_scan import scan


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


_SPEC_GRAPH = {
    "specs.py": (
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Spec:\n    key: str\n    node: str\n"
        "SPECS = [Spec(key='a', node='Alpha Scout'), Spec(key='b', node='Beta Scout')]\n"
    ),
    "graph.py": (
        "from langgraph.graph import StateGraph, START, END\n"
        "from specs import SPECS\n"
        "ROUTES = {'Judge': 'Judge', 'Pro': 'Pro', 'Con': 'Con'}\n"
        "def _scout_graph(spec, llm):\n"
        "    g = StateGraph(dict)\n"
        "    g.add_node('agent', llm)\n"
        "    g.add_node('tools', run_tools)\n"
        "    g.add_edge(START, 'agent')\n"
        "    g.add_conditional_edges('agent', pick, ['tools', END])\n"
        "    g.add_edge('tools', 'agent')\n"
        "    return g.compile()\n"
        "def build(llm, logic):\n"
        "    w = StateGraph(dict)\n"
        "    for spec in SPECS:\n"
        "        w.add_node(spec.node, _scout_graph(spec, llm))\n"
        "    w.add_node('Pro', pro)\n    w.add_node('Con', con)\n    w.add_node('Judge', judge)\n"
        "    first = [spec.node for spec in SPECS]\n"
        "    for n in first:\n"
        "        w.add_edge(START, n)\n"
        "    w.add_edge(first, 'Pro')\n"
        "    for side in ('Pro', 'Con'):\n"
        "        w.add_conditional_edges(side, logic.should_continue, ROUTES)\n"
        "    w.add_edge('Judge', END)\n"
        "    return w.compile()\n"
    ),
}


def test_nodes_from_a_table_of_records_and_routes_from_a_loop_with_a_path_map_constant(tmp_path):
    bp = scan(_write(tmp_path, _SPEC_GRAPH))
    assert {"alpha_scout", "beta_scout", "pro", "con", "judge"} <= set(ids(bp))
    e = edges(bp)
    assert {("alpha_scout", "pro"), ("beta_scout", "pro"), ("pro", "con"), ("con", "pro"), ("pro", "judge"),
            ("con", "judge")} <= e


def test_a_graph_built_afresh_for_each_node_is_that_nodes_own_machinery(tmp_path):
    bp = scan(_write(tmp_path, _SPEC_GRAPH))
    assert "agent" not in ids(bp) and "tools" not in ids(bp)


def test_a_graph_built_by_a_function_for_one_node_is_a_subgraph(tmp_path):
    files = dict(_SPEC_GRAPH)
    files["specs.py"] = "class Spec:\n    pass\n"
    files["graph.py"] = (
        "from langgraph.graph import StateGraph, START, END\n"
        "def _inner():\n"
        "    g = StateGraph(dict)\n    g.add_node('draft', draft)\n    g.add_node('check', check)\n"
        "    g.add_edge(START, 'draft')\n    g.add_edge('draft', 'check')\n    g.add_edge('check', END)\n"
        "    return g.compile()\n"
        "w = StateGraph(dict)\nw.add_node('plan', plan)\nw.add_node('write', _inner())\n"
        "w.add_edge(START, 'plan')\nw.add_edge('plan', 'write')\nw.add_edge('write', END)\n"
    )
    bp = scan(_write(tmp_path, files))
    assert {("write", "draft"), ("check", "write")} <= edges(bp)


def test_an_agent_wrapped_in_an_agent_tool_is_called_by_its_owner(tmp_path):
    bp = scan(_write(tmp_path, {
        "subs.py": "from google.adk.agents import LlmAgent\n"
                   "pricing_agent = LlmAgent(name='PricingAgent', instruction='Price it')\n"
                   "risk_agent = LlmAgent(name='RiskAgent', instruction='Assess it')\n",
        "agent.py": "from google.adk.agents import LlmAgent\nfrom google.adk.tools.agent_tool import AgentTool\n"
                    "from subs import pricing_agent, risk_agent\n"
                    "root_agent = LlmAgent(name='Desk', instruction='Run the desk',\n"
                    "    tools=[lookup, AgentTool(pricing_agent), AgentTool(agent=risk_agent)])\n",
    }))
    assert {("desk", "pricing_agent"), ("desk", "risk_agent")} <= edges(bp)
    desk = next(a for a in bp["agents"] if a["id"] == "desk")
    assert "AgentTool" not in desk["tools"]


def test_a_model_calling_callback_given_to_an_agent_is_a_checking_agent(tmp_path):
    bp = scan(_write(tmp_path, {
        "judge.py": "from google.genai import Client\n"
                    "async def quality_gate(ctx):\n"
                    "    client = Client()\n"
                    "    return await client.aio.models.generate_content(model='m', contents='Judge this')\n",
        "agent.py": "from google.adk.agents import LlmAgent\nfrom judge import quality_gate\n"
                    "root_agent = LlmAgent(name='Desk', instruction='Run', after_agent_callback=quality_gate)\n",
    }))
    assert "quality_gate" in ids(bp)
    assert ("desk", "quality_gate") in edges(bp)


_LAB = {
    "llm.py": "import openai\n"
              "def ask(prompt, system):\n"
              "    return openai.chat.completions.create(model='m', messages=[{'role': 'system', 'content': system}])\n",
    "agents.py": (
        "from llm import ask\n"
        "class BaseAgent:\n"
        "    def inference(self, task):\n        return ask(task, self.role)\n"
        "class ScoutAgent(BaseAgent):\n    role = 'You scout.'\n"
        "class CriticAgent(BaseAgent):\n    role = 'You critique.'\n"
        "    def check(self, work):\n        return self.inference(work)\n"
    ),
    "tools.py": (
        "from langchain_openai import ChatOpenAI\n"
        "class Library:\n"
        "    def __init__(self):\n        self.llm = ChatOpenAI()\n"
        "    def add(self, item):\n        return self.llm([item])\n"
    ),
    "run.py": (
        "from agents import ScoutAgent, CriticAgent\nfrom tools import Library\n"
        "class Lab:\n"
        "    def __init__(self):\n"
        "        self.scout = ScoutAgent()\n        self.critic = CriticAgent()\n        self.library = Library()\n"
        "    def loop(self):\n"
        "        while True:\n"
        "            work = self.scout.inference('go')\n"
        "            if self.critic.check(work):\n"
        "                self.library.add(work)\n"
        "                break\n"
        "Lab().loop()\n"
    ),
}


def test_agent_classes_asked_through_their_own_model_methods_run_in_order(tmp_path):
    bp = scan(_write(tmp_path, _LAB))
    assert {("scout_agent", "critic_agent"), ("critic_agent", "library")} <= edges(bp)


def test_a_model_caller_built_beside_the_agents_is_an_agent(tmp_path):
    bp = scan(_write(tmp_path, _LAB))
    assert "library" in ids(bp)


def test_a_retry_loop_comes_back_to_its_first_agent(tmp_path):
    bp = scan(_write(tmp_path, _LAB))
    assert ("critic_agent", "scout_agent") in edges(bp)


def test_agents_written_as_records_are_routed_by_the_orchestrator_step(tmp_path):
    bp = scan(_write(tmp_path, {
        "flow.py": (
            "from llama_index.core.workflow import Workflow, step\nfrom pydantic import BaseModel\n"
            "class AgentConfig(BaseModel):\n    name: str\n    system_prompt: str\n"
            "class Desk(Workflow):\n"
            "    @step\n    async def orchestrator(self, ctx, ev):\n        return ev\n"
            "    @step\n    async def speak_with_sub_agent(self, ctx, ev):\n        return ev\n"
        ),
        "main.py": (
            "from flow import AgentConfig, Desk\n"
            "def configs():\n"
            "    return [AgentConfig(name='Billing Agent', system_prompt='You bill.'),\n"
            "            AgentConfig(name='Refund Agent', system_prompt='You refund.')]\n"
            "workflow = Desk(timeout=None)\nworkflow.run(agent_configs=configs())\n"
        ),
        "tutorials/01_intro.py": "from flow import AgentConfig\n"
                                 "AgentConfig(name='Toy Agent', system_prompt='You are a toy.')\n",
    }))
    assert ids(bp) == ["billing_agent", "orchestrator", "refund_agent"]
    assert {("orchestrator", "billing_agent"), ("billing_agent", "orchestrator"), ("orchestrator", "refund_agent")} \
        <= edges(bp)


def test_mode_variants_of_an_agent_are_one_agent_and_a_lone_planner_leads(tmp_path):
    role = ("from app.role import Role\nclass {cls}(Role):\n    def reply(self, memory):\n"
            "        return self.llm_api.chat_completion(messages=memory)\n")
    bp = scan(_write(tmp_path, {
        "app/role.py": "class Role:\n    pass\n",
        "app/planner.py": role.format(cls="Planner"),
        "app/interp/interpreter.py": role.format(cls="CodeInterpreter"),
        "app/interp_cli/interpreter_cli.py": role.format(cls="CodeInterpreterCLIOnly"),
        "app/interp_plugin/interpreter_plugin.py": role.format(cls="CodeInterpreterPluginOnly"),
        "app/search/search.py": role.format(cls="WebSearch"),
    }))
    assert ids(bp) == ["code_interpreter", "planner", "web_search"]
    assert {("planner", "code_interpreter"), ("code_interpreter", "planner"), ("planner", "web_search")} <= edges(bp)
