"""Agent graphs wired with objects: Strands, Microsoft Agent Framework, Google ADK 2 workflows, PydanticAI and
pydantic_graph, CrewAI Flows. Small projects written the way each framework's own samples are."""

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


# -- Strands ---------------------------------------------------------------------------------------------------------
def test_strands_graph_names_nodes_after_the_agent_object(tmp_path):
    _write(tmp_path, {"app.py": (
        "from strands import Agent\n"
        "from strands.multiagent import GraphBuilder\n"
        "researcher = Agent(name='researcher', system_prompt='Research the topic')\n"
        "analyst = Agent(name='analyst', system_prompt='Analyse the findings')\n"
        "writer = Agent(name='writer', system_prompt='Write the report')\n"
        "builder = GraphBuilder()\n"
        "r = builder.add_node(researcher, 'research')\n"
        "builder.add_node(analyst, 'analysis')\n"
        "builder.add_node(writer, 'report')\n"
        "builder.add_edge(r, 'analysis')\n"
        "builder.add_edge('analysis', 'report')\n"
        "builder.set_entry_point('research')\n"
        "graph = builder.build()\n")})
    bp = scan(tmp_path)
    assert bp["framework"] == "strands"
    assert ids(bp) == ["analyst", "researcher", "writer"]  # one block per agent, not one more per node id
    assert {("researcher", "analyst"), ("analyst", "writer")} <= edges(bp)
    assert "researcher" in bp["entry"]


def test_strands_swarm_hands_off_to_the_members_a_prompt_names(tmp_path):
    _write(tmp_path, {"swarm.py": (
        "from strands import Agent, tool\n"
        "from strands.multiagent import Swarm\n"
        "@tool\n"
        "def analyse(ticker: str) -> str:\n"
        "    strategist = Agent(name='strategist', system_prompt='Study the business, then hand off to analyst.')\n"
        "    analyst = Agent(name='analyst', system_prompt='Check the numbers, then hand off to reporter.')\n"
        "    reporter = Agent(name='reporter', system_prompt='Write the recommendation.')\n"
        "    swarm = Swarm([strategist, analyst, reporter])\n"
        "    return str(swarm(ticker))\n"
        "orchestrator = Agent(system_prompt='Answer stock questions', tools=[analyse])\n")})
    bp = scan(tmp_path)
    assert bp["framework"] == "strands"
    assert {("strategist", "analyst"), ("analyst", "reporter"), ("orchestrator", "strategist")} <= edges(bp)
    assert ("reporter", "strategist") not in edges(bp)  # handed off as the prompts say, not round a ring


def test_strands_agents_as_tools(tmp_path):
    _write(tmp_path, {"sre.py": (
        "from strands import Agent, tool\n"
        "_logs = Agent(name='logs_agent', system_prompt='Read the logs')\n"
        "_fixer = Agent(name='fixer_agent', system_prompt='Fix the service')\n"
        "@tool\n"
        "def logs_agent(task: str) -> str:\n"
        "    return str(_logs(task))\n"
        "triage = Agent(name='triage', system_prompt='Triage incidents', tools=[logs_agent, _fixer])\n")})
    bp = scan(tmp_path)
    assert {("triage", "logs_agent"), ("triage", "fixer_agent")} <= edges(bp)


# -- Microsoft Agent Framework ---------------------------------------------------------------------------------------
def test_agent_framework_workflow_of_agents_and_executors(tmp_path):
    _write(tmp_path, {"flow.py": (
        "from agent_framework import Agent, Executor, WorkflowBuilder, executor, handler\n"
        "from agent_framework.openai import OpenAIChatClient\n"
        "client = OpenAIChatClient()\n"
        "writer = client.as_agent(name='writer', instructions='Write the copy')\n"
        "reviewer = Agent(client, name='reviewer', instructions='Review the copy')\n"
        "class Publish(Executor):\n"
        "    @handler\n"
        "    async def run(self, text, ctx):\n"
        "        await ctx.yield_output(text)\n"
        "@executor(id='load_brief')\n"
        "async def load(brief, ctx):\n"
        "    await ctx.send_message(brief)\n"
        "publish = Publish(id='publish')\n"
        "workflow = (WorkflowBuilder(start_executor=load)\n"
        "            .add_edge(load, writer)\n"
        "            .add_edge(writer, reviewer)\n"
        "            .add_edge(reviewer, publish, condition=lambda r: r.approved)\n"
        "            .build())\n")})
    bp = scan(tmp_path)
    assert bp["framework"] == "microsoft-agent-framework"
    assert {"writer", "reviewer", "load_brief", "publish"} <= set(ids(bp))  # steps with no model are blocks too
    assert {("load_brief", "writer"), ("writer", "reviewer"), ("reviewer", "publish")} <= edges(bp)
    assert "load_brief" in bp["entry"]


def test_agent_framework_orchestrations(tmp_path):
    _write(tmp_path, {"team.py": (
        "from agent_framework import Agent\n"
        "from agent_framework.orchestrations import SequentialBuilder, HandoffBuilder, MagenticBuilder\n"
        "def build_agent(client, name, instructions):\n"
        "    return Agent(client, name=name, instructions=instructions)\n"
        "def pipeline(client):\n"
        "    plan = build_agent(client, 'planner', 'Plan it')\n"
        "    draft = build_agent(client, 'drafter', 'Draft it')\n"
        "    edit = build_agent(client, 'editor', 'Edit it')\n"
        "    return SequentialBuilder(participants=[plan, draft, edit]).build()\n"
        "def desk(client):\n"
        "    triage = Agent(client, name='triage', instructions='Route the request')\n"
        "    billing = Agent(client, name='billing', instructions='Billing questions')\n"
        "    refunds = Agent(client, name='refunds', instructions='Refunds')\n"
        "    return (HandoffBuilder(participants=[triage, billing, refunds]).with_start_agent(triage)\n"
        "            .add_handoff(triage, [billing, refunds]).build())\n")})
    bp = scan(tmp_path)
    assert {"planner", "drafter", "editor", "triage", "billing", "refunds"} <= set(ids(bp))
    assert "build_agent" not in ids(bp)  # the factory is not an agent: the names its callers pass are
    assert {("planner", "drafter"), ("drafter", "editor"), ("triage", "billing"), ("triage", "refunds")} <= edges(bp)


# -- Google ADK 2 workflows ------------------------------------------------------------------------------------------
def test_adk_workflow_chains_routes_and_fan_out(tmp_path):
    _write(tmp_path, {"workflow.py": (
        "from google.adk.agents import LlmAgent\n"
        "from google.adk.workflow import FunctionNode, Workflow\n"
        "def _node(func, name):\n"
        "    return FunctionNode(func=func, name=name)\n"
        "def intake(ctx): ...\n"
        "def report(ctx): ...\n"
        "auditor = LlmAgent(name='auditor', model='gemini-flash-latest', instruction='Audit the page')\n"
        "fixer = LlmAgent(name='fixer', model='gemini-flash-latest', instruction='Fix the issues')\n"
        "def build():\n"
        "    start = _node(intake, 'IntakeNode')\n"
        "    done = _node(report, 'ReportNode')\n"
        "    return Workflow(name='pilot', edges=[\n"
        "        ('START', start, auditor),\n"
        "        (auditor, {'fix': fixer, 'report': done}),\n"
        "        (fixer, done),\n"
        "    ])\n")})
    bp = scan(tmp_path)
    assert {"auditor", "fixer", "intake_node", "report_node"} <= set(ids(bp))
    assert {("intake_node", "auditor"), ("auditor", "fixer"), ("auditor", "report_node"),
            ("fixer", "report_node")} <= edges(bp)
    assert "intake_node" in bp["entry"]


# -- PydanticAI ------------------------------------------------------------------------------------------------------
def test_pydantic_ai_model_id_is_not_a_name_and_a_table_of_agents_is_delegated_to(tmp_path):
    _write(tmp_path, {"clinic.py": (
        "from pydantic_ai import Agent, RunContext\n"
        "gp = Agent('openai:gpt-5.2', instructions='General practice')\n"
        "cardio = Agent('openai:gpt-5.2', instructions='Heart questions')\n"
        "triage = Agent('openai:gpt-5.2', instructions='Send the patient to a specialist')\n"
        "SPECIALISTS = {'gp': gp, 'cardio': cardio}\n"
        "@triage.tool\n"
        "async def consult(ctx: RunContext, kind: str, question: str) -> str:\n"
        "    specialist = SPECIALISTS.get(kind)\n"
        "    return (await specialist.run(question)).output\n")})
    bp = scan(tmp_path)
    assert bp["framework"] == "pydantic-ai"
    assert ids(bp) == ["cardio", "gp", "triage"]  # three agents, not one named after the model
    assert {("triage", "gp"), ("triage", "cardio")} <= edges(bp)


def test_pydantic_graph_edges_come_from_return_annotations(tmp_path):
    _write(tmp_path, {"graph.py": (
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from pydantic_graph import BaseNode, End, Graph, GraphRunContext\n"
        "@dataclass\n"
        "class Ask(BaseNode[None]):\n"
        "    async def run(self, ctx: GraphRunContext) -> Answer:\n"
        "        return Answer()\n"
        "@dataclass\n"
        "class Answer(BaseNode[None]):\n"
        "    async def run(self, ctx: GraphRunContext) -> Evaluate:\n"
        "        return Evaluate()\n"
        "@dataclass\n"
        "class Evaluate(BaseNode[None, None, str]):\n"
        "    async def run(self, ctx: GraphRunContext) -> Ask | End[str]:\n"
        "        return End('done')\n"
        "graph = Graph(nodes=(Ask, Answer, Evaluate))\n"
        "graph.run_sync(Ask())\n")})
    bp = scan(tmp_path)
    assert {"ask", "answer", "evaluate"} <= set(ids(bp))
    assert {("ask", "answer"), ("answer", "evaluate"), ("evaluate", "ask")} <= edges(bp)
    assert bp["entry"] == ["ask"] and "evaluate" in bp["exits"]


# -- CrewAI Flows ----------------------------------------------------------------------------------------------------
def test_crewai_flow_steps_routes_and_the_crews_they_run(tmp_path):
    _write(tmp_path, {
        "crews/review_crew.py": (
            "from crewai import Agent, Task, Crew\n"
            "from crewai.project import CrewBase, agent, task, crew\n"
            "@CrewBase\n"
            "class ReviewCrew:\n"
            "    @agent\n"
            "    def critic(self): return Agent(config=self.agents_config['critic'])\n"
            "    @agent\n"
            "    def editor(self): return Agent(config=self.agents_config['editor'])\n"
            "    @task\n"
            "    def critique(self): return Task(config=self.tasks_config['critique'], agent=self.critic())\n"
            "    @task\n"
            "    def edit(self): return Task(config=self.tasks_config['edit'], agent=self.editor())\n"
            "    @crew\n"
            "    def crew(self): return Crew(agents=self.agents, tasks=self.tasks)\n"),
        "flow.py": (
            "from crewai.flow.flow import Flow, listen, router, start\n"
            "from crews.review_crew import ReviewCrew\n"
            "class PostFlow(Flow):\n"
            "    @start()\n"
            "    def draft(self): return 'text'\n"
            "    @listen(draft)\n"
            "    def review(self):\n"
            "        return ReviewCrew().crew().kickoff(inputs={})\n"
            "    @router(review)\n"
            "    def decide(self):\n"
            "        if self.state.ok:\n"
            "            return 'publish'\n"
            "        return 'redo'\n"
            "    @listen('publish')\n"
            "    def save(self): ...\n"
            "    @listen('redo')\n"
            "    def retry(self): ...\n"),
    })
    bp = scan(tmp_path)
    assert bp["framework"] == "crewai"
    assert {"draft", "review", "decide", "save", "retry", "critic", "editor"} <= set(ids(bp))
    assert {("draft", "review"), ("review", "decide"), ("decide", "save"), ("decide", "retry"),
            ("review", "critic"), ("critic", "editor")} <= edges(bp)
    assert "draft" in bp["entry"]
