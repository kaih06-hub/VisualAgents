"""Shapes of agent systems found missing or wrong in a held-out check against README diagrams, each written as a
made-up project: the rules have to carry over to code nobody has looked at."""

from visualagents.collector import is_test_file
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


# -- what is a test ---------------------------------------------------------------------------------------------

def test_a_file_named_like_a_test_is_only_a_test_when_it_holds_tests():
    assert is_test_file("tests/helpers.py", "x = 1")
    assert is_test_file("src/test_api.py", "import pytest\ndef test_ok():\n    assert True\n")
    assert is_test_file("src/test_api.py", "def test_ok():\n    assert 1\n")  # collected by a runner, never called
    # an agent that designs or runs tests: a script whose test_* functions it calls itself
    assert not is_test_file("src/test_designer.py", "def write_cases(q):\n    return q\n")
    assert not is_test_file("src/test_runner.py", "def test_report(d):\n    return d\nif __name__ == '__main__':\n"
                                                  "    test_report(1)\n")


def test_agents_restated_by_the_test_suite_stay_off_the_map(tmp_path):
    _write(tmp_path, {
        "app/graph.py": ("from langgraph.graph import StateGraph, START, END\n"
                         "def plan(s):\n    return s\ndef write(s):\n    return s\n"
                         "g = StateGraph(dict)\ng.add_node('plan', plan)\ng.add_node('write', write)\n"
                         "g.add_edge(START, 'plan')\ng.add_edge('plan', 'write')\ng.add_edge('write', END)\n"),
        "tests/test_graph.py": ("from langgraph.graph import StateGraph\nclass StubAgent:\n    pass\n"
                                "def test_flow():\n    stub = StubAgent(name='fake_agent', fail=True)\n"
                                "    g = StateGraph(dict)\n    g.add_node('review', stub)\n"),
    })
    assert _ids(scan(tmp_path)) == ["plan", "write"]


# -- model calls without a framework ----------------------------------------------------------------------------

def test_the_pre_1_0_openai_sdk_is_a_model_call(tmp_path):
    _write(tmp_path, {"solver.py": ("import openai\ndef solve(q):\n"
                                    "    prompt = f'Solve the following problem step by step: {q}'\n"
                                    "    return openai.ChatCompletion.create(model='m', messages=[{'role': 'user', "
                                    "'content': prompt}])\n")})
    assert _ids(scan(tmp_path)) == ["solve"]


def test_scripts_with_their_own_helper_of_the_same_name_are_different_agents(tmp_path):
    """Each script's top-level code writes its prompts and sends them through its own api_call(): the script is the
    agent, and one script's helper is not the other's."""
    helper = ("from openai import OpenAI\nclient = OpenAI()\n"
              "def api_call(msg):\n    return client.chat.completions.create(model='m', messages=msg)\n")
    _write(tmp_path, {
        "stages/outline.py": helper + "plan = [{'role': 'system', 'content': 'You are a planner who outlines the "
                                      "report section by section.'}]\nout = api_call(plan)\n",
        "stages/draft.py": helper + "msgs = [{'role': 'system', 'content': 'You are a writer who drafts each "
                                    "section from the outline you are given.'}]\nout = api_call(msgs)\n",
        "stages/score.py": ("from openai import OpenAI\nclient = OpenAI()\ndef api_call(request):\n"
                            "    return client.chat.completions.create(**request)\n"
                            "def main():\n    return api_call({'model': 'm', 'messages': []})\n"
                            "if __name__ == '__main__':\n    main()\n"),
    })
    bp = scan(tmp_path)
    assert _ids(bp) == ["draft", "outline", "score"]  # main() is named after its script


def test_one_function_giving_a_helper_several_roles_is_several_agents(tmp_path):
    _write(tmp_path, {
        "llm.py": ("import openai\ndef ask(system_role, user_input):\n"
                   "    return openai.ChatCompletion.create(model='m', messages=[{'role': 'system', 'content': "
                   "system_role}, {'role': 'user', 'content': user_input}])\n"),
        "flow.py": ("from llm import ask\ndef decide(case):\n"
                    "    triager = 'You are a triage nurse who sorts each case into a specialty.'\n"
                    "    reviewer = 'You are a specialist who reviews the case in depth.'\n"
                    "    chair = 'You are the chair who merges the reviews into one decision.'\n"
                    "    area = ask(system_role=triager, user_input=case)\n"
                    "    notes = [ask(system_role=reviewer, user_input=case + area) for _ in range(3)]\n"
                    "    return ask(system_role=chair, user_input=str(notes))\n"),
    })
    bp = scan(tmp_path)
    assert _ids(bp) == ["chair", "reviewer", "triager"]
    assert {("triager", "reviewer"), ("reviewer", "chair")} <= _edges(bp)


def test_a_leftover_helper_nothing_calls_is_not_an_agent(tmp_path):
    _write(tmp_path, {
        "llm.py": ("import openai\n"
                   "def chat(messages):\n    return openai.ChatCompletion.create(model='m', messages=messages)\n"
                   "def old_completion(text):\n    return openai.Completion.create(model='m', prompt=text)\n"),
        "agent.py": ("from llm import chat\ndef summarise(doc):\n"
                     "    return chat([{'role': 'system', 'content': 'You are an editor who summarises documents.'},"
                     " {'role': 'user', 'content': doc}])\n"),
    })
    assert _ids(scan(tmp_path)) == ["summarise"]


def test_one_script_per_dataset_is_one_agent(tmp_path):
    body = ("import openai\ndef fetch(item):\n    prompt = f'Write a function that solves this task: {item}'\n"
            "    return openai.ChatCompletion.create(model='m', messages=[{'role': 'user', 'content': prompt}])\n"
            "if __name__ == '__main__':\n    fetch('x')\n")
    _write(tmp_path, {"src/coder_math.py": body, "src/coder_code.py": body})
    bp = scan(tmp_path)
    assert _ids(bp) == ["coder"]
    assert {"coder_math", "coder_code"} <= set(bp["agents"][0]["aliases"])


# -- agents as instances, messages and graphs -------------------------------------------------------------------

def test_named_instances_of_one_agent_class_are_the_agents(tmp_path):
    _write(tmp_path, {
        "core/base.py": ("import openai\nclass Speaker:\n    def __init__(self, name):\n        self.name = name\n"
                         "        self.memory = []\n"
                         "    def query(self, msgs):\n        return openai.ChatCompletion.create(model='m', "
                         "messages=msgs)\n"
                         "    def ask(self):\n        return self.query(self.memory)\n"
                         "    def remember(self, text):\n        self.memory.append(text)\n"),
        "core/panel.py": ("from base import Speaker\nSIDES = ['Proposer', 'Opponent', 'Referee']\n"
                          "class Panel:\n    def __init__(self):\n"
                          "        self.people = [Speaker(name=n) for n in SIDES]\n"
                          "        self.pro = self.people[0]\n        self.con = self.people[1]\n"
                          "        self.ref = self.people[2]\n"
                          "    def run(self, rounds):\n        for _ in range(rounds):\n"
                          "            self.pro.remember(self.pro.ask())\n            self.con.ask()\n"
                          "            self.ref.ask()\n"
                          "        tie = Speaker(name='Tiebreaker')\n        return tie.ask()\n"),
    })
    bp = scan(tmp_path)
    assert _ids(bp) == ["opponent", "proposer", "referee", "tiebreaker"]
    assert {("proposer", "opponent"), ("opponent", "referee"), ("referee", "proposer"),
            ("referee", "tiebreaker")} <= _edges(bp)


def test_roles_hand_on_work_through_the_actions_they_watch(tmp_path):
    _write(tmp_path, {
        "team/roles.py": ("class Role:\n    def _init_actions(self, a):\n        pass\n"
                          "    def _watch(self, a):\n        pass\n"
                          "class Analyst(Role):\n    def __init__(self):\n        self._init_actions([WriteSpec])\n"
                          "        self._watch([UserAsk])\n"
                          "class Builder(Role):\n    def __init__(self):\n        self._init_actions([WriteCode])\n"
                          "        self._watch([WriteSpec])\n"
                          "class Checker(Role):\n    def __init__(self):\n        self._init_actions([Review])\n"
                          "        self._watch([WriteCode, WriteSpec])\n"
                          "team = [Analyst(), Builder(), Checker()]\n"),
    })
    bp = scan(tmp_path)
    assert {("analyst", "builder"), ("builder", "checker"), ("analyst", "checker")} <= _edges(bp)
    assert bp["entry"] == ["analyst"]


def test_a_graph_node_wrapping_an_agent_by_name_is_that_agent(tmp_path):
    _write(tmp_path, {
        "agents/base.py": "class BaseAgent:\n    def __init__(self, agent_name):\n        self.agent_name = agent_name\n",
        "agents/drafting.py": ("from .base import BaseAgent\nclass DraftingAgent(BaseAgent):\n"
                               "    def __init__(self):\n        super().__init__(agent_name='drafting_agent')\n"),
        "agents/checking.py": ("from .base import BaseAgent\nclass CheckingAgent(BaseAgent):\n"
                               "    def __init__(self):\n        super().__init__(agent_name='checking_agent')\n"),
        "flow.py": ("from langgraph.graph import StateGraph\n"
                    "def wrap(agent, name):\n    return lambda s: agent(s)\n"
                    "class Flow:\n    def build(self, agents):\n        g = StateGraph(dict)\n"
                    "        g.add_node('Draft', wrap(agents['drafting_agent'], 'drafting_agent'))\n"
                    "        g.add_node('Check', wrap(agents['checking_agent'], 'checking_agent'))\n"
                    "        g.add_edge('Draft', 'Check')\n        return g\n"),
    })
    bp = scan(tmp_path)
    assert _ids(bp) == ["check", "draft"]
    assert "DraftingAgent" in next(a for a in bp["agents"] if a["id"] == "draft")["aliases"]


def test_a_node_running_a_compiled_subgraph_hands_work_to_it(tmp_path):
    _write(tmp_path, {"graph.py": (
        "from langgraph.graph import StateGraph, END\n"
        "def a(s):\n    return s\ndef b(s):\n    return s\ndef outline(s):\n    return s\ndef polish(s):\n    return s\n"
        "inner = StateGraph(dict)\ninner.add_node(outline)\ninner.add_node(polish)\n"
        "inner.set_entry_point('outline')\ninner.add_edge('outline', 'polish')\ninner.add_edge('polish', END)\n"
        "outer = StateGraph(dict)\nouter.add_node(a)\nouter.add_node('write', inner.compile())\nouter.add_node(b)\n"
        "outer.set_entry_point('a')\nouter.add_edge('a', 'write')\nouter.add_edge('write', 'b')\n"
        "outer.add_edge('b', END)\napp = outer.compile()\n")})
    bp = scan(tmp_path)
    assert {("write", "outline"), ("polish", "write"), ("a", "write"), ("write", "b")} <= _edges(bp)
    assert bp["entry"] == ["a"]


def test_shell_scripts_running_scripts_in_order_are_a_pipeline(tmp_path):
    llm = ("import openai\ndef {fn}(task):\n    prompt = f'You are a {role}. Work on this task carefully: {{task}}'\n"
           "    return openai.ChatCompletion.create(model='m', messages=[{{'role': 'user', 'content': prompt}}])\n"
           "if __name__ == '__main__':\n    {fn}('x')\n")
    _write(tmp_path, {
        "src/writer_easy.py": llm.format(fn="draft", role="writer"),
        "src/writer_hard.py": llm.format(fn="draft", role="writer"),
        "src/critic_easy.py": llm.format(fn="critique", role="critic"),
        "src/critic_hard.py": llm.format(fn="critique", role="critic"),
        "src/grader_easy.py": ("from writer_easy import draft\nimport json\n"
                               "def grade(items):\n    return [draft(i) for i in items if not json.loads(i)]\n"
                               "if __name__ == '__main__':\n    grade([])\n"),
        "scripts/run.sh": ("# run every stage\npython writer_[easy/hard].py\npython critic_[easy/hard].py \\\n"
                           "    --model m\npython grader_[easy/hard].py\n"),
    })
    facts = {}
    bp = scan(tmp_path, facts=facts)
    assert _ids(bp) == ["critic", "grader", "writer"]
    assert {("writer", "critic"), ("critic", "grader"), ("grader", "writer")} <= _edges(bp)
    assert facts["script_pipelines"] == [{"script": "scripts/run.sh", "runs": ["writer", "critic", "grader"]}]
