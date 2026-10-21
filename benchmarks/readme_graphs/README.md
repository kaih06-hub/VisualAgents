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
