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

Results of the first run (2026-10-21, static mode) are in `results-2026-10-21.json`.
