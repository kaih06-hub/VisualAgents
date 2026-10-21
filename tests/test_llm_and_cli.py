import json
import sys
import types
from pathlib import Path

import pytest

from visualagents import blueprint as bpm
from visualagents import cli, llm, scanner

ROOT = Path(__file__).resolve().parents[1]


class _FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def _fake_anthropic(monkeypatch, payload, captured, stop_reason="end_turn"):
    import anthropic

    message = types.SimpleNamespace(
        stop_reason=stop_reason,
        content=[types.SimpleNamespace(type="thinking", thinking=""),
                 types.SimpleNamespace(type="text", text=json.dumps(payload))],
    )

    class FakeClient:
        def __init__(self, *a, **kw):
            self.messages = types.SimpleNamespace(stream=lambda **kw: (captured.update(kw, _beta=False), _FakeStream(message))[1])
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(
                stream=lambda **kw: (captured.update(kw, _beta=True), _FakeStream(message))[1]))

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)


def test_scan_uses_structured_output_and_merges_static_aliases(monkeypatch, tmp_path):
    payload = {
        "project": "demo", "framework": "custom", "summary": "Four agents.",
        "agents": [
            {"id": "planner", "name": "Planner", "role": "plans", "description": "d", "tools": ["outline_builder"],
             "file": "research_crew.py", "line": 37, "aliases": [], "keywords": ["plan"]},
            {"id": "writer", "name": "Writer", "role": "writes", "description": "d", "tools": [], "file": "", "line": 0,
             "aliases": [], "keywords": []},
        ],
        "edges": [{"from": "planner", "to": "writer", "label": "plan", "conditional": False}],
        "entry": ["planner"], "exits": ["writer"],
        "steps": [{"order": 1, "title": "Plan", "agents": ["planner"], "description": ""}],
    }
    captured = {}
    _fake_anthropic(monkeypatch, payload, captured)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    bp = scanner.scan_project(ROOT / "visualagents" / "demo", mode="llm", log=lambda m: None)

    assert captured["model"] == "claude-opus-5-5"
    assert captured["_beta"] is True
    assert captured["fallbacks"] == "default"
    assert captured["betas"] == ["server-side-fallback-2026-07-01"]
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["output_config"]["effort"] == "high"
    assert "<static_analysis_hints>" in captured["messages"][0]["content"]
    assert "<repo_outline>" in captured["messages"][0]["content"]  # every file's shape, not just the ones in full
    assert bp["generated_by"] == "anthropic:claude-opus-5-5"
    assert [a["id"] for a in bp["agents"]] == ["planner", "writer"]
    assert "Writer" in bp["agents"][1]["aliases"] or bp["agents"][1]["name"] == "Writer"


def _agent(name, file="", line=0):
    return {"id": bpm.slugify(name), "name": name, "role": "", "description": "", "tools": [], "file": file,
            "line": line, "aliases": [], "keywords": []}


def test_structural_facts_reach_the_llm_and_its_map_is_kept_as_drawn(monkeypatch, tmp_path):
    (tmp_path / "team").mkdir()
    (tmp_path / "team" / "graph.py").write_text(
        "from langgraph.graph import StateGraph\n"
        "g = StateGraph(dict)\n"
        + "".join(f"g.add_node('{n}', {n})\n" for n in ("planner", "researcher", "writer", "reviewer"))
    )
    (tmp_path / "team" / "autogen_version").mkdir()
    (tmp_path / "team" / "autogen_version" / "chat.py").write_text(
        "from autogen import ConversableAgent\n"
        + "".join(f"{n.lower()} = ConversableAgent(name='{n}', system_message='x')\n"
                  for n in ("Coordinator", "Researcher", "Writer", "Reviewer"))
    )
    payload = {
        "project": "team", "framework": "langgraph", "summary": "A research team; an AutoGen version is not drawn.",
        "agents": [_agent("Planner", "team/graph.py", 3), _agent("Researcher", "team/graph.py", 4),
                   _agent("Coordinator", "team/autogen_version/chat.py", 2)],
        "edges": [{"from": "planner", "to": "researcher", "label": "", "conditional": False}],
        "entry": ["planner"], "exits": ["researcher"], "steps": [],
    }
    captured = {}
    _fake_anthropic(monkeypatch, payload, captured)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    bp = scanner.scan_project(tmp_path, mode="llm", log=lambda m: None)

    hints = captured["messages"][0]["content"].split("<static_analysis_hints>")[1].split("</static_analysis_hints>")[0]
    folders = {i["framework"]: i["folder"] for i in json.loads(hints)["facts"]["ports"][0]["implementations"]}
    assert folders == {"langgraph": "team", "autogen": "team/autogen_version"}
    # the model weighs the facts; nothing it drew is deleted behind its back
    assert [a["id"] for a in bp["agents"]] == ["planner", "researcher", "coordinator"]


def test_the_scan_prompt_teaches_structures_not_particular_projects():
    """The guidance is meant to carry over to projects nobody has looked at: patterns, with made-up examples -
    never the names, classes or folders of the projects it was checked against."""
    prompt = llm.SCAN_SYSTEM.lower()
    for structure in ("pipeline:", "router or supervisor:", "turn-taking:", "critic loop:", "planner and executors:",
                      "fan-out:", "nested team:", "several independent systems:", "alternatives", "opt-in parts:",
                      "apps built with the project"):
        assert structure in prompt
    for particular in ("storm", "wikiwriter", "topicexpert", "taskweaver", "code_interpreter", "cli-only", "owl",
                       "community_usecase", "gpt-researcher", "ag2", "chiefeditor", "lambda", "voyager", "xagent",
                       "hedge", "agentlaboratory", "open_deep_research"):
        assert particular not in prompt, particular


def test_every_independent_system_reaches_the_llm(monkeypatch, tmp_path):
    """Two pipelines in one package (STORM and Co-STORM): both show up in the hints and in the source sent in full,
    and the prompt asks for every system - not only the one with more code."""
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "curation.py").write_text(
        "import dspy\n"
        "class WikiWriter(dspy.Module):\n"
        "    def __init__(self):\n"
        "        self.ask = dspy.ChainOfThought('topic -> question')\n"
        "class TopicExpert(dspy.Module):\n"
        "    def __init__(self):\n"
        "        self.answer = dspy.Predict('question -> answer')\n"
        "w, e = WikiWriter(), TopicExpert()\n"
    )
    (tmp_path / "chat").mkdir()
    for i in range(6):  # a bigger second system
        (tmp_path / "chat" / f"agent_{i}.py").write_text(
            "from interface import Agent\n" f"class Expert{i}(Agent):\n" "    def act(self):\n"
            + "        self.turn += 1\n" * 40)
    payload = {"project": "storm", "framework": "dspy", "summary": "s", "agents": [_agent("WikiWriter")],
               "edges": [], "entry": [], "exits": [], "steps": []}
    captured = {}
    _fake_anthropic(monkeypatch, payload, captured)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    scanner.scan_project(tmp_path, mode="llm", max_chars=20_000, log=lambda m: None)

    prompt = captured["messages"][0]["content"]
    hints = json.loads(prompt.split("<static_analysis_hints>")[1].split("</static_analysis_hints>")[0])
    assert {"WikiWriter", "TopicExpert", "Expert0", "Expert5"} <= {a["name"] for a in hints["agents"]}
    assert '<file path="wiki/curation.py">' in prompt
    assert "several independent systems" in captured["system"].lower()


def test_refusal_and_truncation_raise(monkeypatch):
    for reason in ("refusal", "max_tokens"):
        _fake_anthropic(monkeypatch, {}, {}, stop_reason=reason)
        with pytest.raises(llm.LLMError):
            llm.complete_json(provider="anthropic", model="claude-opus-5-5", system="s", prompt="p", schema={})


def test_auto_mode_without_key_falls_back_to_static(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "detect_provider", lambda preferred="auto": None)
    bp = scanner.scan_project(ROOT / "visualagents" / "demo", mode="auto", log=lambda m: None)
    assert bp["generated_by"] == "static"


def test_secrets_are_redacted():
    from visualagents.collector import redact

    out = redact('API_KEY = "sk-ant-abcdefghijklmnopqrstuvwxyz"\ntoken="ghp_abcdefghijklmnopqrstuvwxyz123456"')
    assert "abcdefghijklmnop" not in out
    assert "<redacted>" in out


def test_secrets_named_like_keys_are_redacted_but_code_is_kept():
    from visualagents.collector import redact

    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3MiOiJtIn0.c2lnbmF0dXJlLXZhbHVl"
    toml = f'[qdrant]\nKEY = "{jwt}"\nPASSWORD = "hunter2!x"\nMAX_NEW_TOKENS = 512\nMODEL = "Qwen/Qwen3-30B"\n'
    out = redact(toml, config=True)
    assert jwt not in out and "hunter2!x" not in out
    assert "MAX_NEW_TOKENS = 512" in out and 'MODEL = "Qwen/Qwen3-30B"' in out
    assert "hush-hush-value" not in redact('{"api_key": "hush-hush-value"}', config=True)
    assert "hush-hush-value" not in redact("db_password: hush-hush-value\n", config=True)
    code = "tokenizer = load_tokenizer(name)\ntoken = get_token()\n"
    assert redact(code) == code  # unquoted values in code are left alone


def test_gitignored_files_never_reach_the_llm(tmp_path):
    import subprocess

    from visualagents.collector import collect, pack

    (tmp_path / ".gitignore").write_text("config.toml\ndata/\n")
    (tmp_path / "agents.py").write_text("from crewai import Agent\nresearcher = Agent(role='Researcher')\n")
    (tmp_path / "config.toml").write_text('[llm]\nagent_model = "gpt-5"\nendpoint = "https://private.example"\n')
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "parsed.json").write_text('{"agent": "private document text"}')
    (tmp_path / "eslint.config.mjs").write_text("export default [] // agent config\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    files = collect(tmp_path)
    assert [f.rel for f in files] == ["agents.py"]
    assert "private" not in pack(files)[0]


def test_cli_scan_and_blueprint(tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "app.py").write_text("from agents import Agent\na = Agent(name='Alpha', instructions='x')\n")
    assert cli.main(["scan", str(project), "--static"]) == 0
    saved = bpm.load(project / ".visualagents" / "blueprint.json")
    assert saved["agents"][0]["name"] == "Alpha"
    assert (project / ".visualagents" / ".gitignore").read_text().strip() == "runs/"
    assert cli.main(["blueprint", str(project), "--json"]) == 0
    assert '"Alpha"' in capsys.readouterr().out


def test_run_argument_splitting(tmp_path):
    assert cli._split_run_argv(["run", "--smart", str(tmp_path), "python", "main.py", "--q", "x"]) == (
        ["run", "--smart", str(tmp_path)], ["python", "main.py", "--q", "x"])
    assert cli._split_run_argv(["run", "--port", "9000", "--", "node", "agent.js"]) == (
        ["run", "--port", "9000"], ["node", "agent.js"])
    assert cli._split_run_argv(["run", "python", "x.py"]) == (["run"], ["python", "x.py"])


def _fake_google(monkeypatch, payload, captured, finish="STOP"):
    from google import genai

    resp = types.SimpleNamespace(
        text=json.dumps(payload),
        prompt_feedback=None,
        candidates=[types.SimpleNamespace(finish_reason=finish)],
    )

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured.update(model=model, contents=contents, config=config)
            return resp

    class FakeClient:
        def __init__(self, *a, **kw):
            captured["client_kwargs"] = kw
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)


def _clear_provider_env(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL", "GEMINI_API_KEY",
              "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "ANTHROPIC_PROFILE", "VISUALAGENTS_PROVIDER",
              "VISUALAGENTS_MODEL", "VISUALAGENTS_NARRATE_MODEL"):
        monkeypatch.setenv(k, "")  # empty = unset for VisualAgents, and monkeypatch restores it afterwards


def test_google_provider_sends_json_schema(monkeypatch):
    pytest.importorskip("google.genai")
    captured = {}
    _fake_google(monkeypatch, {"agent": "writer", "activity": "drafting"}, captured)
    out = llm.complete_json(provider="google", model="gemini-flash-lite-latest", system="sys", prompt="p",
                            schema=llm.NARRATE_SCHEMA, effort="low", max_tokens=4000)
    assert out == {"agent": "writer", "activity": "drafting"}
    cfg = captured["config"]
    assert captured["model"] == "gemini-flash-lite-latest"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == llm.NARRATE_SCHEMA
    assert cfg.system_instruction == "sys"
    assert 503 in captured["client_kwargs"]["http_options"].retry_options.http_status_codes


def test_google_truncation_and_safety_raise(monkeypatch):
    pytest.importorskip("google.genai")
    for finish in ("FinishReason.MAX_TOKENS", "FinishReason.SAFETY"):
        _fake_google(monkeypatch, {}, {}, finish=finish)
        with pytest.raises(llm.LLMError):
            llm.complete_json(provider="google", model="m", system="s", prompt="p", schema={})


def test_provider_and_model_selection(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(llm.Path, "home", lambda: Path("/nonexistent-home"))
    assert llm.detect_provider("auto") is None
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert llm.detect_provider("auto") == "google"
    assert llm.default_model("google") == "gemini-pro-latest"
    assert llm.default_model("google", "narrate") == "gemini-flash-lite-latest"
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert llm.detect_provider("auto") == "openai"          # first provider with a key
    monkeypatch.setenv("VISUALAGENTS_PROVIDER", "google")   # explicit default wins
    monkeypatch.setenv("VISUALAGENTS_MODEL", "gemini-flash-latest")
    assert llm.detect_provider("auto") == "google"
    assert llm.default_model("google") == "gemini-flash-latest"
    assert llm.detect_provider("anthropic") == "anthropic"  # --provider wins over everything
    with pytest.raises(llm.LLMError):
        llm.detect_provider("mistral")


def test_dotenv_supplies_keys_and_model_choice(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    (tmp_path / ".env").write_text("GEMINI_API_KEY='abc'\nexport VISUALAGENTS_MODEL=gemini-flash-latest\nUNRELATED=1\n")
    monkeypatch.chdir(tmp_path)
    llm.load_dotenv_keys(tmp_path)
    import os

    assert os.environ["GEMINI_API_KEY"] == "abc"
    assert os.environ["VISUALAGENTS_MODEL"] == "gemini-flash-latest"
    assert "UNRELATED" not in os.environ


def test_run_options_may_follow_the_folder(tmp_path):
    assert cli._split_run_argv(["run", str(tmp_path), "--exit", "--port", "0", "--static"]) == (
        ["run", str(tmp_path), "--exit", "--port", "0", "--static"], [])
