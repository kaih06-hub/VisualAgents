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
    assert bp["generated_by"] == "anthropic:claude-opus-5-5"
    assert [a["id"] for a in bp["agents"]] == ["planner", "writer"]
    assert "Writer" in bp["agents"][1]["aliases"] or bp["agents"][1]["name"] == "Writer"


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
