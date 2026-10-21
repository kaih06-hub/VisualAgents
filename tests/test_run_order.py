"""Shapes found missing or wrong in the second held-out README-graph check, each written as a made-up project: run
order through helper methods, agents handed over as tools, chats, prompt-keyed steps, step variants, and code that is
not part of today's system (older implementations, usage examples, a visualisation library)."""

from visualagents.static_scan import scan


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _ids(bp):
    return sorted(a["id"] for a in bp["agents"])


def _edges(bp):
    return {(e["from"], e["to"]) for e in bp["edges"]}


SDK_AGENTS = (
    "from agents import Agent\n"
    "scout = Agent(name='scout', instructions='Plan the lookups.')\n"
    "fetcher = Agent(name='fetcher', instructions='Look one thing up.')\n"
    "auditor = Agent(name='auditor', instructions='Check the figures.')\n"
    "critic = Agent(name='critic', instructions='Check the draft.')\n"
    "author = Agent(name='author', instructions='Write the brief.')\n"
)


def test_a_manager_running_sdk_agents_through_its_helpers_is_a_pipeline(tmp_path):
    _write(tmp_path, {
        "crew/team.py": SDK_AGENTS,
        "crew/manager.py": (
            "import asyncio\n"
            "from agents import Runner\n"
            "from .team import scout, fetcher, auditor, critic, author\n"
            "class Manager:\n"
            "    async def go(self, q):\n"
            "        plan = await self._plan(q)\n"
            "        found = await asyncio.gather(*[self._fetch(x) for x in plan])\n"
            "        brief = await self._write(found)\n"
            "        verdict = await Runner.run(critic, brief)\n"
            "        return verdict\n"
            "    async def _plan(self, q):\n"
            "        return (await Runner.run(scout, q)).final_output\n"
            "    async def _fetch(self, item):\n"
            "        try:\n"
            "            return await Runner.run(fetcher, item)\n"
            "        except Exception:\n"
            "            return None\n"
            "    async def _write(self, found):\n"
            "        numbers = auditor.as_tool(tool_name='audit', tool_description='figures')\n"
            "        writer = author.clone(tools=[numbers])\n"
            "        return await Runner.run(writer, str(found))\n"
            "if __name__ == '__main__':\n"
            "    asyncio.run(Manager().go('q'))\n"
        ),
    })
    edges = _edges(scan(tmp_path))
    assert {("scout", "fetcher"), ("fetcher", "author"), ("author", "critic"), ("author", "auditor")} <= edges
    assert ("scout", "author") not in edges  # a failed lookup is the failure path, not a way around the lookup


def test_branches_are_alternatives_not_a_sequence(tmp_path):
    _write(tmp_path, {
        "app/team.py": SDK_AGENTS,
        "app/run.py": (
            "from agents import Runner\n"
            "from .team import scout, fetcher, author\n"
            "def main(fast):\n"
            "    if fast:\n"
            "        Runner.run_sync(fetcher, 'q')\n"
            "    else:\n"
            "        Runner.run_sync(scout, 'q')\n"
            "    Runner.run_sync(author, 'q')\n"
        ),
    })
    edges = _edges(scan(tmp_path))
    assert {("fetcher", "author"), ("scout", "author")} <= edges
    assert ("fetcher", "scout") not in edges and ("scout", "fetcher") not in edges


def test_chats_between_agents_kept_in_a_dict_and_built_by_factories(tmp_path):
    _write(tmp_path, {
        "bot/roles.py": (
            "import autogen\n"
            "class Proxy(autogen.UserProxyAgent):\n"
            "    pass\n"
            "class Thinker:\n"
            "    def __init__(self, cfg):\n"
            "        self.agent = autogen.AssistantAgent(name='thinker', llm_config=cfg)\n"
            "class Clicker:\n"
            "    def __init__(self, cfg):\n"
            "        self.agent = autogen.AssistantAgent(name='clicker', llm_config=cfg)\n"
        ),
        "bot/wire.py": (
            "from .roles import Proxy, Thinker, Clicker\n"
            "class Wiring:\n"
            "    def build(self, cfg):\n"
            "        team = {}\n"
            "        team['person'] = self._person()\n"
            "        team['runner'] = Proxy(name='runner', human_input_mode='NEVER')\n"
            "        team['thinker'] = self._thinker(cfg)\n"
            "        team['clicker'] = Clicker(cfg).agent\n"
            "        self.team = team\n"
            "        self.team['person'].register_nested_chats([\n"
            "            {'sender': self.team['runner'], 'recipient': self.team['clicker'], 'max_turns': 3}],\n"
            "            trigger=lambda m: True)\n"
            "    def _person(self):\n"
            "        person = Proxy(name='person', human_input_mode='ALWAYS')\n"
            "        return person\n"
            "    def _thinker(self, cfg):\n"
            "        t = Thinker(cfg)\n"
            "        return t.agent\n"
            "    async def ask(self, text):\n"
            "        return await self.team['person'].a_initiate_chat(self.team['thinker'], message=text)\n"
        ),
    })
    bp = scan(tmp_path)
    assert {"person", "runner", "thinker", "clicker"} <= set(_ids(bp))  # each named Proxy is an agent of its own
    edges = _edges(bp)
    assert {("person", "thinker"), ("thinker", "person"), ("person", "clicker"), ("runner", "clicker")} <= edges


def test_steps_that_pick_the_prompt_of_one_shared_model_call_are_agents(tmp_path):
    _write(tmp_path, {
        "solver/core.py": (
            "import functools\n"
            "from openai import AsyncOpenAI\n"
            "from .stages import draft_stage, review_stage, polish_stage\n"
            "class Solver:\n"
            "    def __init__(self):\n"
            "        self.client = AsyncOpenAI()\n"
            "    async def _ask(self, item, prompt='draft'):\n"
            "        text = TEMPLATES[prompt].format(item=item)\n"
            "        r = await self.client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': text}])\n"
            "        return r.choices[0].message.content\n"
            "    async def solve(self, item):\n"
            "        item = await draft_stage(self, item)\n"
            "        item = await review_stage(self, item)\n"
            "        return await polish_stage(self, item)\n"
            "TEMPLATES = {}\n"
        ),
        "solver/stages.py": (
            "import functools\n"
            "async def draft_stage(self, item):\n"
            "    f = functools.partial(self._ask, item=item, prompt='draft')\n"
            "    return await f()\n"
            "async def review_stage(self, item):\n"
            "    f = functools.partial(self._ask, item=item, prompt='review')\n"
            "    return await f()\n"
            "async def polish_stage(self, item):\n"
            "    f = functools.partial(self._ask, item=item, prompt='polish')\n"
            "    return await f()\n"
        ),
    })
    bp = scan(tmp_path)
    assert {"draft_stage", "review_stage", "polish_stage"} <= set(_ids(bp))
    assert {("draft_stage", "review_stage"), ("review_stage", "polish_stage")} <= _edges(bp)


def test_a_step_written_once_per_mode_is_one_step(tmp_path):
    _write(tmp_path, {
        "tr/steps.py": (
            "import openai\n"
            "def ask(p):\n"
            "    return openai.chat.completions.create(model='m', messages=[{'role': 'user', 'content': p}])\n"
            "def short_text_first_draft(t):\n    return ask('You are a drafter. Draft ' + t)\n"
            "def short_text_critique_draft(t):\n    return ask('You are a critic. Critique ' + t)\n"
            "def long_text_first_draft(t):\n    return ask('You are a drafter. Draft each part of ' + t)\n"
            "def long_text_critique_draft(t):\n    return ask('You are a critic. Critique each part of ' + t)\n"
            "def run_long(t):\n    return long_text_critique_draft(long_text_first_draft(t))\n"
            "def run_short(t):\n    return short_text_critique_draft(short_text_first_draft(t))\n"
            "def run(t, long):\n    return run_long(t) if long else run_short(t)\n"
        ),
    })
    bp = scan(tmp_path)
    assert len(bp["agents"]) == 2, _ids(bp)


def test_two_roles_on_the_same_things_are_not_variants(tmp_path):
    _write(tmp_path, {
        "cr/steps.py": (
            "import openai\n"
            "def ask(p):\n"
            "    return openai.chat.completions.create(model='m', messages=[{'role': 'user', 'content': p}])\n"
            "def write_unit_code(t):\n    return ask('You are a senior coder. Write clean, tested code for: ' + t)\n"
            "def check_unit_code(t):\n    return ask('You are a strict reviewer. Check this code for bugs: ' + t)\n"
            "def write_test_cases(t):\n    return ask('You are a careful tester. Write unit tests for: ' + t)\n"
            "def check_test_cases(t):\n    return ask('You are a strict reviewer. Check these tests for gaps: ' + t)\n"
            "def run(t):\n"
            "    code = check_unit_code(write_unit_code(t))\n"
            "    return check_test_cases(write_test_cases(code))\n"
        ),
    })
    bp = scan(tmp_path)
    assert len(bp["agents"]) == 4, _ids(bp)
    assert ("write_unit_code", "check_unit_code") in _edges(bp)


def test_early_returns_and_nested_runs_keep_their_order(tmp_path):
    _write(tmp_path, {
        "app/team.py": SDK_AGENTS,
        "app/route.py": (
            "from agents import Runner\n"
            "from .team import scout, fetcher, author\n"
            "async def handle(kind, q):\n"
            "    if kind == 'quick':\n"
            "        return await Runner.run(fetcher, q)\n"
            "    return await Runner.run(starting_agent=author, input=(await Runner.run(scout, q)).final_output)\n"
        ),
    })
    edges = _edges(scan(tmp_path))
    assert ("scout", "author") in edges
    assert ("fetcher", "author") not in edges and ("fetcher", "scout") not in edges and ("author", "scout") not in edges


def test_an_older_implementation_in_a_legacy_folder_stays_off_the_map(tmp_path):
    graph = ("from langgraph.graph import StateGraph, START\n"
             "def {a}(s):\n    return s\ndef {b}(s):\n    return s\n"
             "g = StateGraph(dict)\ng.add_node('{a}', {a})\ng.add_node('{b}', {b})\n"
             "g.add_edge(START, '{a}')\ng.add_edge('{a}', '{b}')\n")
    _write(tmp_path, {"src/app/graph.py": graph.format(a="gather", b="report"),
                      "src/legacy/graph.py": graph.format(a="old_search", b="old_write")})
    assert _ids(scan(tmp_path)) == ["gather", "report"]
    only_old = tmp_path / "only_old"
    _write(only_old, {"legacy/graph.py": graph.format(a="old_search", b="old_write")})
    assert _ids(scan(only_old)) == ["old_search", "old_write"]  # all there is: it is the system


def test_usage_examples_stay_off_the_map_beside_the_projects_own_graph(tmp_path):
    graph = ("from langgraph.graph import StateGraph, START\n"
             "def {a}(s):\n    return s\ndef {b}(s):\n    return s\n"
             "g = StateGraph(dict)\ng.add_node('{a}', {a})\ng.add_node('{b}', {b})\n"
             "g.add_edge(START, '{a}')\ng.add_edge('{a}', '{b}')\n")
    _write(tmp_path, {"src/lib/core.py": graph.format(a="act", b="judge"),
                      "examples/demo.py": graph.format(a="demo_call", b="demo_check")})
    assert _ids(scan(tmp_path)) == ["act", "judge"]


def test_a_node_that_invokes_a_compiled_graph_hands_work_to_it(tmp_path):
    _write(tmp_path, {"g.py": (
        "from langgraph.graph import StateGraph, START, END\n"
        "def dig(s):\n    return s\ndef sum_up(s):\n    return s\n"
        "inner = StateGraph(dict)\ninner.add_node('dig', dig)\ninner.add_node('sum_up', sum_up)\n"
        "inner.add_edge(START, 'dig')\ninner.add_edge('dig', 'sum_up')\ninner.add_edge('sum_up', END)\n"
        "digger = inner.compile()\n"
        "async def lead(s):\n    return s\n"
        "async def lead_tools(s):\n    return [await digger.ainvoke(t) for t in s]\n"
        "outer = StateGraph(dict)\nouter.add_node('lead', lead)\nouter.add_node('lead_tools', lead_tools)\n"
        "outer.add_edge(START, 'lead')\nouter.add_edge('lead', 'lead_tools')\nouter.add_edge('lead_tools', 'lead')\n"
    )})
    edges = _edges(scan(tmp_path))
    assert {("lead_tools", "dig"), ("sum_up", "lead_tools")} <= edges


def test_a_create_agent_wrapper_of_a_framework_factory_still_makes_agents(tmp_path):
    _write(tmp_path, {"team.py": (
        "from langgraph.prebuilt import create_react_agent\n"
        "def create_agent(llm, tools, system_message):\n"
        "    return create_react_agent(llm, tools, prompt=system_message)\n"
        "research_agent = create_agent(llm, [search], 'You research.')\n"
        "chart_agent = create_agent(llm, [plot], 'You draw charts.')\n"
    )})
    assert {"chart_agent", "research_agent"} <= set(_ids(scan(tmp_path)))


def test_the_projects_own_create_agent_and_a_drawing_library_are_not_agents(tmp_path):
    _write(tmp_path, {
        "pipe.py": ("from langgraph.graph import StateGraph, START\n"
                    "def plan(s):\n    return s\ndef act(s):\n    return s\n"
                    "def create_agent():\n"
                    "    g = StateGraph(dict)\n    g.add_node('plan', plan)\n    g.add_node('act', act)\n"
                    "    g.add_edge(START, 'plan')\n    g.add_edge('plan', 'act')\n    return g.compile()\n"),
        "ui.py": ("from pyvis.network import Network\nfrom pipe import *\n"
                  "def show(nodes):\n    net = Network()\n"
                  "    for n in nodes:\n        net.add_node(n['id'], label=n['label'])\n    return net\n"
                  "app = create_agent()\n"),
    })
    assert _ids(scan(tmp_path)) == ["act", "plan"]


def test_a_text_splitter_does_not_make_a_project_langchain(tmp_path):
    _write(tmp_path, {"t.py": (
        "import openai\nfrom langchain_text_splitters import RecursiveCharacterTextSplitter\n"
        "def summarise(t):\n"
        "    return openai.chat.completions.create(model='m', messages=[{'role': 'user', 'content': 'Sum up ' + t}])\n"
    )})
    assert scan(tmp_path)["framework"] == "custom"
