"""The guided start (bare `visualagents`): project -> provider -> model -> key -> map -> watch."""

import argparse
import json
import os
from pathlib import Path

import pytest

from visualagents import home, llm, tui, wizard


class Script:
    """Answers wizard prompts in order: select() answers are label substrings, ask()/secret() are text."""

    def __init__(self, monkeypatch, selects, texts=(), secrets=()):
        self.selects, self.texts, self.secrets = list(selects), list(texts), list(secrets)
        self.seen = []
        monkeypatch.setattr(tui, "select", self.select)
        monkeypatch.setattr(tui, "ask", lambda prompt, default="", **kw: self.texts.pop(0) if self.texts else default)
        monkeypatch.setattr(tui, "secret", lambda prompt: self.secrets.pop(0))
        monkeypatch.setattr(tui, "confirm", lambda prompt, default=True: True)

    def select(self, title, options, default=0):
        labels = [label for label, _ in options]
        self.seen.append(labels)
        want = self.selects.pop(0)
        return next(i for i, label in enumerate(labels) if want in label)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep the repo's own .env (a real key) out of the wizard's sight
    (tmp_path / "README.md").write_text("# demo\n\n```bash\npython main.py\n```\n")
    (tmp_path / "main.py").write_text(
        'def build():\n    from agents import Agent\n    return Agent(name="Writer", instructions="write")\n'
        'if __name__ == "__main__":\n    print("# Agent: Writer")\n    print("Final Answer: done")\n')
    return tmp_path


def _args(**kw):
    return argparse.Namespace(port=0, no_browser=True, exit=True, **kw)


def test_offline_path_maps_and_starts_the_project(monkeypatch, project):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VISUALAGENTS_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    s = Script(monkeypatch, ["Another folder", "Skip", "Start it now"], texts=[str(project)])
    assert wizard.run(_args()) == 0
    bp = json.loads((project / ".visualagents" / "blueprint.json").read_text())
    assert [a["name"] for a in bp["agents"]] == ["Writer"]
    assert json.loads((project / ".visualagents" / "config.json").read_text())["command"] == ["python", "main.py"]
    assert home.settings()["recent"][0] == str(project.resolve())
    runs = list((project / ".visualagents" / "runs").glob("*.jsonl"))
    assert runs and "Final Answer: done" in runs[0].read_text()


def test_pasted_key_is_checked_saved_and_reused(monkeypatch, project):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "VISUALAGENTS_PROVIDER", "VISUALAGENTS_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "list_models", lambda provider: ["gemini-flash-latest", "gemini-pro-latest"])
    scanned = {}

    def fake_scan(path, **kw):
        scanned.update(kw)
        from visualagents.static_scan import scan

        return scan(path)

    monkeypatch.setattr("visualagents.scanner.scan_project", fake_scan)
    Script(monkeypatch, ["Another folder", "Google Gemini", "gemini-flash-latest", "Just show the map"],
           texts=[str(project)], secrets=["AIza-test-key-1234"])
    assert wizard.run(_args()) == 0
    assert scanned["provider"] == "google" and scanned["model"] == "gemini-flash-latest"
    keys = json.loads((home.home() / "keys.json").read_text())
    assert keys == {"GEMINI_API_KEY": "AIza-test-key-1234"}
    assert oct(os.stat(home.home() / "keys.json").st_mode & 0o777) == "0o600"
    assert home.settings()["provider"] == "google"

    # second time: the project is "last used", Gemini is pre-selected and the saved key is offered
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Script(monkeypatch, [project.name, "Google Gemini", "gemini-flash-latest", "Use the key I already have",
                             "Just show the map"])
    assert wizard.run(_args()) == 0
    assert "Use the key I already have  ••••1234" in s.seen[3]


def test_google_choices_are_aliases_that_follow_googles_current_models():
    """A pinned version (gemini-2.5-flash) goes stale and is eventually retired for new keys; the -latest aliases
    always point at a model Google serves."""
    google = [m for m, _ in wizard.MODEL_CHOICES["google"]]
    assert google == ["gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"]
    assert all(m.endswith("-latest") for m in google)


def test_a_pinned_model_saved_earlier_is_offered_first(monkeypatch):
    s = Script(monkeypatch, ["gemini-3.8-flash"])
    assert wizard._ask_model("google", "gemini-3.8-flash") == "gemini-3.8-flash"
    assert s.seen[0][:4] == ["gemini-3.8-flash", "gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"]


def test_bad_key_is_not_saved(monkeypatch, project):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "has_credentials", lambda p: False)

    def reject(provider):
        raise llm.LLMError("could not list anthropic models: 401 invalid x-api-key")

    monkeypatch.setattr(llm, "list_models", reject)
    Script(monkeypatch, ["Another folder", "Anthropic", "claude-haiku-5-5", "Just show the map"],
           texts=[str(project)], secrets=["bad1", "bad2", "bad3"])
    assert wizard.run(_args()) == 0
    assert not (home.home() / "keys.json").exists()
    assert (project / ".visualagents" / "blueprint.json").exists()  # fell back to the offline map
