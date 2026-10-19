import pytest


@pytest.fixture(autouse=True)
def _isolated_visualagents_home(tmp_path_factory, monkeypatch):
    """Never read or write the developer's real ~/.visualagents (saved keys, daemon, hooks) in tests."""
    monkeypatch.setenv("VISUALAGENTS_HOME", str(tmp_path_factory.mktemp("va-home")))
    monkeypatch.setenv("VISUALAGENTS_NO_BROWSER", "1")
