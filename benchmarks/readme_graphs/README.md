# README-graph benchmark

Ten agent projects whose READMEs draw (or spell out) their agent graph, used as a held-out check of
`visualagents scan`. `score.py` holds the ground truth taken from each README; `run.sh` clones the projects at
the checked commits, scans them and prints the scores. None of these projects is named in the scanner's rules
or prompt - fix what a miss shows about code in general, never the project.

Metrics, per project:

- **node recall** (matching rate) - README components found on the map.
- **node precision, strict** (correct rate) - map blocks that are a README component.
- **node precision, lenient** - also counting blocks that exist in today's code but not in the README drawing
  (a later refiner step, a renamed agent, a pre-processing script).
- **edge recall / precision** - README arrows found, and map arrows that are README arrows. An arrow drawn
  directly in the README that the map routes through one code-only step still counts as found.

- **node recall, code** - node recall leaving out README components the code no longer has (`readme_only`).

Results of the first run (2026-10-21, static mode) are in `results-2026-10-21.json`.

A second held-out set of ten more projects lives in `ground_truth_set2.py`; `run_set2.sh` clones and scores it
(`score.py <out> set2`). Results before/after the fixes it led to are in `results-set2-2026-10-21.json`.

A third set of ten (`ground_truth_set3.py`, `run_set3.sh`, `score.py <out> set3`) had its ground truth written from
the READMEs before any scan. Sets 1-3 before/after the fixes it led to are in `results-sets123-2026-10-21.json`.

A fourth set of ten (`ground_truth_set4.py`, `run_set4.sh`, `score.py <out> set4`), also written before any scan, was
used only to measure: main against dev-refinement, in `results-set4-2026-10-21.json`.

A fifth set of ten (`ground_truth_set5.py`, `run_set5.sh`, `score.py <out> set5`) covers PydanticAI, Agno, DSPy and
smolagents projects. Sets 1-5 before/after the fixes it led to are in `results-sets12345-2026-10-21.json`.

A sixth set of ten (`ground_truth_set6.py`, `run_set6.sh`, `score.py <out> set6`), written from the README diagrams before
any scan, covers TradingAgents, gpt-researcher's multi-agent team, Voyager, Agent Laboratory, STORM, a LlamaIndex
concierge, an ADK loan desk, TaskWeaver, Enterprise Deep Research and XAgent. Sets 1-6 on main and dev-refinement are in
`results-set6-2026-10-22.json`; the write-up is `REPORT-set6-2026-10-22.md`.

A seventh set of ten (`ground_truth_set7.py`, `run_set7.sh`, `score.py <out> set7`), written from the README diagrams
before any scan, covers deer-flow 1.x, MetaGPT, ChatDev 1.0, OWL, DeepCode, RD-Agent, an ADK due-diligence team, a
CrewAI Flow, ChatEval and AgentClinic. Main against dev-refinement, with sets 1-6 re-scored, is in
`results-set7-2026-10-24.json`; the write-up is `REPORT-set7-2026-10-24.md`.

An eighth set of ten (`ground_truth_set8.py`, `run_set8.sh`, `score.py <out> set8`), written from the README diagrams
before any scan, covers ai-hedge-fund, Gemini's fullstack LangGraph quickstart, Open Deep Research, AgentVerse,
Paper2Poster, AI-Researcher, Mobile-Agent-v2, MapCoder, AutoKaggle and OctoTools. Before/after on dev-refinement, with
sets 1-7 re-scored, is in `results-set8-2026-10-25.json`; the write-up is `REPORT-set8-2026-10-25.md`.

A ninth set of ten (`ground_truth_set9.py`, `run_set9.sh`, `score.py <out> set9`), written from the README diagrams
before any scan, covers GPT Newspaper, LangChain's agents-from-scratch and memory-agent, Contoso Creative Writer,
Reflexion, LLMCompiler, AutoSurvey, Self-Refine, MACRec and The AI Scientist. Before/after on dev-refinement, with sets
1-8 re-scored, is in `results-set9-2026-10-26.json`; the write-up is `REPORT-set9-2026-10-26.md`. The dev-refinement
rules those three sets were tuned with were not merged: on the held-out sets below they scored the same as `main`.

## Held-out sets 10, 11 and 12: how the scanner does on projects it was not tuned on

Sets 1-9 were each scored, then the scanner was changed until they scored well, so their numbers say little about a
project nobody looked at. Sets 10, 11 (2026-11-13) and 12 (2026-11-23) were written to measure that. Each ground
truth was written from the README diagram and the names in the source before any version of the scanner ran on the
projects. Set 12's nine projects were also picked by a fixed rule (see its docstring), not by hand.

| First run on unseen projects (static mode) | agents found | blocks that are README agents | blocks that are real agents | arrows found | arrows right |
|---|---|---|---|---|---|
| set 10, `main` | 0.64 | 0.44 | 0.78 | 0.24 | 0.32 |
| set 10, `dev-refinement` (three more tuning rounds) | 0.67 | 0.44 | 0.77 | 0.25 | 0.31 |
| set 11, `main` | 0.68 | 0.53 | 0.79 | 0.45 | 0.48 |
| set 11, `dev-refinement` | 0.68 | 0.54 | 0.79 | 0.45 | 0.48 |
| set 11, `scan-generalization` before it saw set 11 | 0.70 | 0.49 | 0.71 | 0.50 | 0.54 |
| set 12, `main` | 0.60 | 0.61 | 0.70 | 0.27 | 0.27 |
| set 12, `dev-refinement` | 0.60 | 0.61 | 0.70 | 0.27 | 0.27 |
| set 12, `scan-generalization` (0051346) | 0.73 | 0.56 | 0.74 | 0.42 | 0.39 |

So on unseen projects `main` finds about two thirds of the README's agents, not the nine in ten sets 1-6 show, and
rules tuned on one set of ten barely carry to the next: three more tuning rounds (`dev-refinement`) changed nothing
on sets 11 and 12. After its first run each of sets 10 and 11 was used to find general gaps (nodes added in a loop,
files saved with a byte-order mark, model calls reached through a wrapper); the branch built that way scores 0.82 on
set 10 and 0.95 on set 11, but those are no longer unseen. Set 12 is the honest number for it: 0.73 of the agents
(main 0.60) and 0.42 of the arrows (main 0.27), at slightly lower strict precision, because more plain model callers
now join the map.

Where set 12 still fails, by kind of code rather than by project:
- agents that are role names in a settings table handed to one generic runner (ALTA, Clouds-Coder): the code has
  one runner function, not one place per agent;
- arrows of a runner that starts agents in parallel and passes their results on (BetterTradingAgents 0 of 16
  arrows, Adaptive-AI 5 of 10);
- agents deployed as separate apps that talk over the network (race-condition, A2A): no arrows;
- helper services that call a model and are run by no agent's code join the map as agents (Adaptive-AI).

`ground_truth_set10.py` / `ground_truth_set11.py` / `ground_truth_set12.py` hold the projects and pinned commits;
`score.py <out> set10|set11|set12` scores a folder of blueprints against them.
