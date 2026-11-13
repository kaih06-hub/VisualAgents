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

## Held-out sets 10 and 11 (2026-11-13): how the scanner does on projects it was not tuned on

Sets 1-9 were each scored, then the scanner was changed until they scored well, so their numbers say little about a
project nobody looked at. Sets 10 and 11 were written to measure that. Each ground truth was written from the README
diagram and the names in the source before any version of the scanner ran on the projects.

| First run on unseen projects (static mode) | agents found | blocks that are README agents | blocks that are real agents | arrows found | arrows right |
|---|---|---|---|---|---|
| set 10, `main` | 0.64 | 0.44 | 0.78 | 0.24 | 0.32 |
| set 10, `dev-refinement` (three more tuning rounds) | 0.67 | 0.44 | 0.77 | 0.25 | 0.31 |
| set 11, `main` | 0.68 | 0.53 | 0.79 | 0.45 | 0.48 |
| set 11, `dev-refinement` | 0.68 | 0.54 | 0.79 | 0.45 | 0.48 |
| set 11, `scan-generalization` before it saw set 11 | 0.70 | 0.49 | 0.71 | 0.50 | 0.54 |

So on unseen projects the scanner finds about two thirds of the README's agents, not the nine in ten sets 1-6 show,
and rules tuned on one set of ten barely carry to the next. After its first run each set was used to find general
gaps (nodes added in a loop, files saved with a byte-order mark, model calls reached through a wrapper), and the
branch now scores 0.82 on set 10 and 0.95 on set 11 for agents found - but those are no longer unseen. The next
honest number needs a twelfth set, scored once before anything is changed.

`ground_truth_set10.py` / `ground_truth_set11.py` hold the projects and pinned commits; `score.py <out> set10|set11`
scores a folder of blueprints against them.
