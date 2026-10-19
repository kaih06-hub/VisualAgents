import itertools
from pathlib import Path

from visualagents import blueprint as bpm
from visualagents.tracker import Tracker

DEMO_BP = Path(__file__).resolve().parents[1] / "visualagents" / "demo" / "blueprint.json"


def make_tracker(bp=None):
    clock = itertools.count(1000.0, 0.5)
    return Tracker(bp or bpm.load(DEMO_BP), clock=lambda: next(clock))


def feed(t, text):
    for line in text.strip("\n").splitlines():
        t.ingest_line(line)


def path_of(t):
    return [p["agent"] for p in t.snapshot()["path"]]


def test_crewai_style_log_is_attributed_and_loops_counted():
    t = make_tracker()
    t.run_started("demo")
    feed(t, """
🚀 Crew: research_crew
# Agent: Planner
## Thought: split the question
## Using tool: outline_builder
## Final Answer: a plan
Handing off to Researcher
# Agent: Researcher
## Using tool: web_search
## Tool Output: 8 results
Passing results to Writer
# Agent: Writer
Drafting section 1/3…
# Agent: Critic
## Final Answer: REVISE
Sending the draft back to Writer
# Agent: Writer
# Agent: Critic
## Final Answer: APPROVED
""")
    t.run_finished(0)
    s = t.snapshot()
    assert path_of(t) == ["planner", "researcher", "writer", "critic", "writer", "critic"]
    assert s["edge_counts"]["writer->critic"] == 2
    assert s["edge_counts"]["critic->writer"] == 1
    assert s["edge_counts"]["critic->__end__"] == 1
    assert s["agents"]["researcher"]["tools"] == 1
    assert all(a["status"] == "done" for a in s["agents"].values())
    assert s["run"]["status"] == "finished"
    assert s["step_index"] == 3


def test_thought_continuation_lines_keep_their_kind():
    t = make_tracker()
    feed(t, """
# Agent: Planner
## Thought: first line of a thought
second line of the same thought
""")
    kinds = [e["kind"] for e in t.events if e["source"] == "log"]
    assert kinds[-2:] == ["thought", "thought"]


def test_tool_name_switches_to_owning_agent():
    t = make_tracker()
    feed(t, """
# Agent: Planner
Action: fact_check
""")
    assert t.active == "critic"


def test_autogen_format():
    bp = bpm.normalize({"agents": [{"name": "user_proxy"}, {"name": "coder"}]})
    t = make_tracker(bp)
    feed(t, """
user_proxy (to coder):
Write a function
coder (to user_proxy):
Here is the code
""")
    assert path_of(t) == ["user_proxy", "coder"]


def test_unknown_crewai_agent_is_discovered():
    bp = bpm.normalize({"agents": [{"name": "Writer"}]})
    t = make_tracker(bp)
    feed(t, "# Agent: Senior Data Analyst\n## Thought: crunching numbers")
    s = t.snapshot()
    assert s["active"] == "senior_data_analyst"
    assert s["extra_agents"][0]["dynamic"] is True


def test_role_text_matches_agent_via_containment():
    bp = bpm.normalize({"agents": [{"id": "researcher", "name": "researcher", "role": "{topic} Senior Data Researcher"}]})
    t = make_tracker(bp)
    feed(t, "# Agent: AI LLMs Senior Data Researcher")
    assert t.active == "researcher"


def test_structured_nested_agents_return_to_parent():
    bp = bpm.normalize({"agents": [{"name": "manager"}, {"name": "worker"}]})
    t = make_tracker(bp)
    t.ingest_structured({"type": "agent_start", "agent": "manager"})
    t.ingest_structured({"type": "agent_start", "agent": "worker"})
    assert t.snapshot()["agents"]["manager"]["status"] == "waiting"
    t.ingest_structured({"type": "tool_start", "agent": "worker", "tool": "search", "text": "q"})
    t.ingest_structured({"type": "agent_end", "agent": "worker"})
    assert t.active == "manager"
    assert path_of(t) == ["manager", "worker", "manager"]
    assert t.snapshot()["agents"]["worker"]["tools"] == 1


def test_structured_events_override_earlier_log_guesses():
    bp = bpm.normalize({"agents": [{"name": "supervisor"}, {"name": "researcher"}]})
    t = make_tracker(bp)
    feed(t, "[supervisor] Routing to the researcher.")
    assert t.active == "researcher"  # heuristic guess
    t.ingest_structured({"type": "agent_start", "agent": "supervisor"})
    feed(t, "Handing off to researcher")  # logs no longer switch agents
    assert t.active == "supervisor"
    assert path_of(t) == ["supervisor"]
    t.ingest_structured({"type": "agent_end", "agent": "supervisor"})
    t.ingest_structured({"type": "agent_start", "agent": "researcher"})
    assert path_of(t) == ["supervisor", "researcher"]


def test_soft_structured_events_ignore_unknown_names():
    t = make_tracker()
    assert t.ingest_structured({"type": "agent_start", "agent": "RunnableSequence", "soft": True}) == []
    assert t.snapshot()["extra_agents"] == []


def test_marker_lines_are_parsed():
    t = make_tracker()
    t.ingest_line('@@visualagents {"type": "agent_start", "agent": "writer"}')
    assert t.active == "writer"
    assert t.precise


def test_narration_sets_activity():
    t = make_tracker()
    t.apply_narration("researcher", "Comparing battery benchmarks")
    s = t.snapshot()
    assert s["active"] == "researcher"
    assert s["agents"]["researcher"]["activity"] == "Comparing battery benchmarks"


def test_failed_run_marks_active_agent_as_error():
    t = make_tracker()
    feed(t, "# Agent: Writer\nTraceback (most recent call last):")
    t.run_finished(1)
    s = t.snapshot()
    assert s["agents"]["writer"]["status"] == "error"
    assert s["agents"]["writer"]["errors"] == 1
    assert s["run"]["status"] == "failed"
