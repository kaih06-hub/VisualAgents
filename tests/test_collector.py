"""What the LLM gets to read: an outline of every file, agent definitions first, big files cut, the budget kept."""

import re
from pathlib import Path

from visualagents.collector import SourceFile, collect, pack

HELPER = "def call_llm(prompt, tools):\n    '''Send the prompt to the llm with tools.'''\n    return client.invoke(prompt)\n"


def _write(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _hedge_fund(tmp_path):
    """ai-hedge-fund's shape: keyword-dense helpers outrank five small persona agents on one shared base class."""
    files = {
        "fund/llm/client.py": HELPER * 400,  # ~45K chars, many hits for "llm", "prompt", "tool"
        "fund/llm/cache.py": HELPER * 120,
        "fund/signals/llm_agent.py": "class LLMAgent(AlphaModel):\n    def get_system_prompt(self):\n        raise NotImplementedError\n",
        "fund/tests/test_agents.py": HELPER * 50 + "class FakeAgent(LLMAgent):\n    pass\n",
    }
    for name in ("buffett", "munger", "graham", "lynch", "druckenmiller"):
        cls = name.capitalize()
        files[f"fund/signals/{name}.py"] = (
            f"from fund.signals.llm_agent import LLMAgent\n\n\nclass {cls}(LLMAgent):\n"
            f"    def get_system_prompt(self):\n        return 'You are {cls}, evaluating a single company as an investor.'\n")
    return _write(tmp_path, files)


def test_every_sibling_agent_class_reaches_the_llm_ahead_of_busy_helpers(tmp_path):
    files = collect(_hedge_fund(tmp_path))
    agents = {f.rel for f in files if f.defines_agent}
    assert {"fund/signals/munger.py", "fund/signals/llm_agent.py"} <= agents  # by inheritance, not by name
    assert "fund/tests/test_agents.py" not in agents

    text, included = pack(files, budget_chars=30_000)
    for name in ("buffett", "munger", "graham", "lynch", "druckenmiller"):
        assert f"fund/signals/{name}.py" in included  # in full, not just in the outline
    assert len(text) <= 30_000


def test_the_outline_shows_every_file_with_line_numbers_even_when_its_code_does_not_fit(tmp_path):
    files = collect(_hedge_fund(tmp_path))
    text, included = pack(files, budget_chars=12_000)
    outline = re.search(r"<repo_outline>\n(.*?)</repo_outline>", text, re.S).group(1)
    assert "fund/signals/druckenmiller.py\n    4 class Druckenmiller(LLMAgent): get_system_prompt\n" in outline
    assert '    6 prompt: "You are Druckenmiller, evaluating a single company as an investor"' in outline
    assert "fund/llm/client.py" in outline
    assert "test_agents" not in text  # tests take no budget, not even in the outline
    assert len(text) <= 12_000


def test_a_file_that_does_not_fit_is_skipped_and_smaller_ones_after_it_still_get_in():
    files = [  # ranked: agent.py, then a mid-sized helper too big for what is left, then a small main.py
        SourceFile(Path("agent.py"), "agent.py", "researcher = Agent(role='Researcher')\n" * 80, score=10),
        SourceFile(Path("workflow.py"), "workflow.py", HELPER * 60, score=9),
        SourceFile(Path("main.py"), "main.py", "from agent import researcher\nprint(researcher)\n", score=1),
    ]
    text, included = pack(files, budget_chars=9_000)
    assert included == ["agent.py", "main.py"]  # the old packer stopped at workflow.py and never got to main.py
    assert len(text) <= 9_000


def test_a_big_file_is_cut_to_its_definitions_prompts_and_model_calls_with_its_real_line_numbers(tmp_path):
    filler = "".join(f"x{i} = compute({i})  # bookkeeping\n" for i in range(3000))
    _write(tmp_path, {"app/engine.py": (
        "import openai\n" + filler +
        "class Analyst:\n"
        "    def run(self, data):\n"
        "        system = 'You are Analyst, a data analysis agent that writes and runs code.'\n"
        "        return client.chat.completions.create(model='m', messages=[{'role': 'system', 'content': system}])\n"
        + filler)})
    files = collect(tmp_path)
    text, included = pack(files, budget_chars=20_000)
    assert included == ["app/engine.py (excerpt)"]
    assert '<file path="app/engine.py" excerpt="selected lines">' in text
    assert "3002| class Analyst:" in text and "3005|         return client.chat.completions.create" in text
    assert "   ..." in text and len(text) <= 20_000


def test_one_line_json_or_minified_code_cannot_blow_the_budget(tmp_path):
    _write(tmp_path, {
        "skills/skills.json": '{"agent": "' + "tool " * 40_000 + '"}',  # a single 200K-character line
        "agents.py": "from crewai import Agent\nwriter = Agent(role='Writer', goal='write')\n",
    })
    text, included = pack(collect(tmp_path), budget_chars=20_000)
    assert "agents.py" in included
    assert len(text) <= 20_000


def test_files_static_analysis_points_at_are_sent_first(tmp_path):
    _write(tmp_path, {
        "graph.py": "def researcher(state):\n    return state\n" * 300,  # LangGraph nodes: plain functions
        "noise.py": HELPER * 300,
    })
    files = collect(tmp_path)
    assert pack(files, budget_chars=16_000)[1][0] == "noise.py (excerpt)"  # keyword density alone
    assert pack(files, budget_chars=16_000, prefer={"graph.py"})[1][0].startswith("graph.py")
