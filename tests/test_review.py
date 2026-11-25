"""The LLM pass checks the static map and returns changes: what it may change, and what it has to prove."""

from pathlib import Path

from visualagents import blueprint as bpm
from visualagents import llm, review
from visualagents.collector import SourceFile


def _file(rel, text):
    return SourceFile(Path("/project") / rel, rel, text, 1.0)


RUNNER = _file("app/run.py", (
    "import asyncio\n"
    "ROLES = {\n"
    '    "scout": "You are the scout. Find the facts.",\n'
    '    "critic": "You are the critic. Find the holes.",\n'
    "}\n"
    "async def run_all(task):\n"
    "    results = await asyncio.gather(*(ask(role, task) for role in ROLES))\n"
    "    return await judge(results)\n"
))
GRAPH = _file("app/graph.py", (
    "g = StateGraph(State)\n"
    "g.add_node('plan', plan)\n"
    "g.add_node('wait', wait_for_human)\n"
    "g.add_edge('plan', 'wait')\n"
    "planner = PlannerAgent()\n"
))
FILES = [RUNNER, GRAPH]


def _static(agents, edges=()):
    return bpm.normalize({
        "project": "p", "framework": "custom",
        "agents": [{"name": n, "file": f, "line": ln, "keywords": [n.lower()]} for n, f, ln in agents],
        "edges": [{"from": a, "to": b} for a, b in edges],
    })


def _changes(**kw):
    out = {k: [] for k in ("read", "update", "remove", "add", "add_edges", "remove_edges", "entry", "exits", "steps")}
    out.update({"framework": "", "summary": ""}, **kw)
    return out


def _add(name, file, line, quote):
    return {"id": name, "name": name, "role": "", "description": "", "tools": [], "aliases": [], "keywords": [],
            "file": file, "line": line, "evidence": quote}


def _arrow(a, b, file, line, quote):
    return {"from": a, "to": b, "label": "", "conditional": False, "file": file, "line": line, "evidence": quote}


def test_an_added_agent_needs_a_quote_that_is_really_in_the_code():
    static = _static([("judge", "app/run.py", 8)])
    bp, notes = review.apply_changes(static, _changes(add=[
        _add("scout", "app/run.py", 3, '"scout": "You are the scout. Find the facts.",'),
        _add("critic", "app/run.py", 90, '"critic": "You are the critic.'),  # wrong line: moved to where it is
        _add("historian", "app/run.py", 4, '"historian": "You are the historian."'),  # not in the code
        _add("ghost", "app/nowhere.py", 1, "class Ghost:"),  # no such file
        _add("empty", "app/run.py", 2, ""),  # no quote
    ]), set(), FILES)
    by_id = {a["id"]: a for a in bp["agents"]}
    assert set(by_id) == {"judge", "scout", "critic"}
    assert (by_id["scout"]["file"], by_id["scout"]["line"]) == ("app/run.py", 3)
    assert by_id["critic"]["line"] == 4
    assert sum("Left out" in n for n in notes) == 3


def test_a_quote_copied_with_its_line_number_still_counts():
    bp, _ = review.apply_changes(_static([]), _changes(add=[
        _add("scout", "run.py", 3, '   3|     "scout": "You are the scout. Find the facts.",'),
    ]), set(), FILES)  # a path given without its folder is found too, when only one file has that name
    assert [a["id"] for a in bp["agents"]] == ["scout"]


def test_a_statement_spread_over_lines_matches_when_quoted_on_one():
    runtime = _file("rt.py", (
        "def build(settings):\n"
        "\n"
        "    critic_runner = runner_for(\n"
        "        settings.critic_provider, settings.critic_model,\n"
        "    )\n"
    ))
    quote = "critic_runner = runner_for(settings.critic_provider, settings.critic_model)"  # no trailing comma
    bp, _ = review.apply_changes(_static([]), _changes(add=[
        _add("critic", "rt.py", 3, quote),
        _add("short", "rt.py", 5, " ) "),  # too little to point at anything
    ]), set(), [runtime])
    assert [(a["id"], a["line"]) for a in bp["agents"]] == [("critic", 3)]
    assert review.find_evidence({"file": "rt.py", "line": 0, "evidence": "settings.critic_model\n    )"},
                                [runtime]) == ("rt.py", 4)


def test_an_added_arrow_needs_a_quote_too():
    static = _static([("scout", "app/run.py", 3), ("critic", "app/run.py", 4), ("judge", "app/run.py", 8)])
    gather = "results = await asyncio.gather(*(ask(role, task) for role in ROLES))"
    bp, notes = review.apply_changes(static, _changes(add_edges=[
        _arrow("scout", "judge", "app/run.py", 7, gather),
        _arrow("critic", "judge", "app/run.py", 7, gather),
        _arrow("judge", "scout", "app/run.py", 8, "return send_back(results)"),  # made up
        _arrow("scout", "nobody", "app/run.py", 7, gather),  # no such block
    ]), set(), FILES)
    assert {(e["from"], e["to"]) for e in bp["edges"]} == {("scout", "judge"), ("critic", "judge")}
    assert any("judge -> scout" in n for n in notes)


def test_graph_nodes_are_not_removed_as_plumbing_but_can_be_as_an_alternative():
    static = _static([("plan", "app/graph.py", 2), ("wait", "app/graph.py", 3)], [("plan", "wait")])
    fixed = {"plan", "wait"}
    bp, notes = review.apply_changes(static, _changes(
        remove=[{"id": "wait", "reason": "plumbing", "merge_into": "", "why": "no model call"}],
        remove_edges=[{"from": "plan", "to": "wait", "why": ""}],
    ), fixed, FILES)
    assert [a["id"] for a in bp["agents"]] == ["plan", "wait"]
    assert [(e["from"], e["to"]) for e in bp["edges"]] == [("plan", "wait")]
    assert any(n.startswith("Kept wait") for n in notes)

    bp, _ = review.apply_changes(static, _changes(
        remove=[{"id": "wait", "reason": "not_part_of_run", "merge_into": "", "why": "an example graph"}],
    ), fixed, FILES)
    assert [a["id"] for a in bp["agents"]] == ["plan"]


def test_a_duplicate_is_merged_into_the_block_that_stays():
    static = _static([("plan", "app/graph.py", 2), ("PlannerAgent", "app/graph.py", 5), ("wait", "app/graph.py", 3)],
                     [("planner_agent", "wait")])
    bp, _ = review.apply_changes(static, _changes(remove=[
        {"id": "planner_agent", "reason": "duplicate", "merge_into": "plan", "why": "the node runs it"},
        {"id": "plan", "reason": "duplicate", "merge_into": "planner_agent", "why": "the other way round"},
    ]), {"plan", "wait"}, FILES)
    plan = next(a for a in bp["agents"] if a["id"] == "plan")
    assert [a["id"] for a in bp["agents"]] == ["plan", "wait"]  # the graph node stays, not the class it runs
    assert "PlannerAgent" in plan["aliases"]  # so log lines naming the class still find it
    assert [(e["from"], e["to"]) for e in bp["edges"]] == [("plan", "wait")]  # its arrows move with it


def test_merges_that_go_round_in_a_circle_end():
    static = _static([("a", "x.py", 1), ("b", "x.py", 2), ("c", "x.py", 3)])
    bp, _ = review.apply_changes(static, _changes(remove=[
        {"id": "a", "reason": "duplicate", "merge_into": "b", "why": ""},
        {"id": "b", "reason": "duplicate", "merge_into": "a", "why": ""},
    ]), set(), FILES)
    assert [a["id"] for a in bp["agents"]] == ["c"]


def test_the_model_cannot_empty_the_map():
    static = _static([("a", "x.py", 1), ("b", "x.py", 2)])
    bp, notes = review.apply_changes(static, _changes(remove=[
        {"id": "a", "reason": "plumbing", "merge_into": "", "why": ""},
        {"id": "b", "reason": "plumbing", "merge_into": "", "why": ""},
    ]), set(), FILES)
    assert [a["id"] for a in bp["agents"]] == ["a", "b"]
    assert any("no agents" in n for n in notes)


def test_names_and_descriptions_change_but_ids_and_static_names_stay():
    static = _static([("ask_llm_scout", "app/run.py", 3)])
    bp, _ = review.apply_changes(static, _changes(
        update=[{"id": "ask_llm_scout", "name": "Scout", "role": "finds facts", "description": "d", "tools": ["web"],
                 "aliases": ["scout"], "keywords": ["facts"]}],
        entry=["Scout"], steps=[{"order": 1, "title": "Scout", "agents": ["scout"], "description": ""}],
    ), set(), FILES)
    a = bp["agents"][0]
    assert (a["id"], a["name"], a["role"], a["tools"]) == ("ask_llm_scout", "Scout", "finds facts", ["web"])
    assert {"ask_llm_scout", "scout"} <= set(a["aliases"])
    assert a["keywords"] == ["facts", "ask_llm_scout"]
    assert bp["entry"] == ["ask_llm_scout"] and bp["steps"][0]["agents"] == ["ask_llm_scout"]


def test_one_more_round_reads_only_the_projects_own_files(monkeypatch):
    prompts = []

    def complete_json(**kw):
        prompts.append(kw["prompt"])
        if len(prompts) == 1:
            return _changes(read=[{"path": "app/run.py", "start": 2, "end": 5},
                                  {"path": "../../.ssh/id_rsa", "start": 1, "end": 9},
                                  {"path": ".env", "start": 1, "end": 9}])
        return _changes(add=[_add("scout", "app/run.py", 3, '"scout": "You are the scout.')])

    monkeypatch.setattr(llm, "complete_json", complete_json)
    bp = review.check_and_add(_static([("judge", "app/run.py", 8)]), FILES, {}, provider="anthropic", model="m",
                              max_chars=60_000, log=lambda m: None)
    assert len(prompts) == 2
    asked = prompts[1].split("<requested_code>")[1]
    assert '<file path="app/run.py" lines="2-5">' in asked and "ssh" not in asked and ".env" not in asked
    assert "last chance" in prompts[1]
    assert {a["id"] for a in bp["agents"]} == {"judge", "scout"}


def test_a_failed_second_round_keeps_the_first_answer(monkeypatch):
    calls = []

    def complete_json(**kw):
        calls.append(1)
        if len(calls) == 2:
            raise llm.LLMError("busy")
        return _changes(read=[{"path": "app/run.py", "start": 1, "end": 3}],
                        update=[{"id": "judge", "name": "Judge", "role": "decides", "description": "", "tools": [],
                                 "aliases": [], "keywords": []}])

    monkeypatch.setattr(llm, "complete_json", complete_json)
    bp = review.check_and_add(_static([("judge", "app/run.py", 8)]), FILES, {}, provider="anthropic", model="m",
                              max_chars=60_000, log=lambda m: None)
    assert len(calls) == 2 and bp["agents"][0]["role"] == "decides"


def test_the_outline_shows_tables_of_role_names_even_in_a_long_file():
    """Roles run by one generic runner exist only as a table: the model has to see it to add them."""
    from visualagents.collector import outline

    text = "".join(f"def helper_{i}():\n    pass\n" for i in range(60))
    text += 'AGENT_ROLES = ("explorer", "developer", "reviewer")\nTEAM: dict = {"lead": 1, "critic": 2}\n'
    text += 'TIMEOUTS = ("fast", "slow")\nWORKERS = 4\n'  # not role names: other names, or not a table of names
    shape = outline([_file("config/constants.py", text)], 50_000)
    assert "121 AGENT_ROLES = 'explorer', 'developer', 'reviewer'" in shape
    assert "122 TEAM = 'lead', 'critic'" in shape
    assert "TIMEOUTS" not in shape and "WORKERS" not in shape
    assert "def helper_59" not in shape  # the rest of the long file is still cut


def test_the_schema_is_strict_for_structured_outputs():
    """Structured outputs need every property required and no extra ones, at every level."""
    def walk(s):
        if s.get("type") == "object":
            assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
        if s.get("type") == "array":
            walk(s["items"])
    walk(review.REVIEW_SCHEMA)
