# Contributing to VisualAgents

Thanks for helping! The most valuable contributions right now:

- **"It didn't map my agents correctly"** — open an issue with the repo (or a minimal example) and what you
  expected. New static detectors live in `visualagents/static_scan.py`.
- **New console formats** — paste a few lines of your framework's output in an issue, or add a pattern to
  `visualagents/patterns.py` with a test in `tests/test_tracker.py`.
- **Framework hooks** — runtime instrumentation lives in `visualagents/_autoinstrument/visualagents_runtime.py`
  (stdlib only, must never break the host app).
- **Start-command detection** — `visualagents/detect.py`.

## Setup

```bash
git clone https://github.com/kaih06-hub/VisualAgents.git && cd VisualAgents
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,all]"
pytest
visualagents demo
```

The dashboard is plain HTML/CSS/JS in `visualagents/web/` — edit and reload, no build step.

## Pull requests

- Keep them focused; add a test for behaviour changes (`pytest` must pass on Linux and macOS).
- Never commit API keys or `.env` files. Tests must not call real LLM APIs (see the fake server in
  `tests/test_openai_hook.py`).
