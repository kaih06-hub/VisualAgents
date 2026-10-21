#!/bin/bash
# Re-run the third README-graph set: clone the ten projects at the checked commits, scan each statically, score.
# usage: benchmarks/readme_graphs/run_set3.sh [workdir]   (needs git and an installed visualagents)
set -e
here="$(cd "$(dirname "$0")" && pwd)"; work="${1:-/tmp/va-readme-graphs-3}"; mkdir -p "$work/repos" "$work/out"
python3 - "$here" <<'PY' > "$work/list.txt"
import sys; sys.path.insert(0, sys.argv[1])
from ground_truth_set3 import GT3
for name, g in GT3.items():
    print(name, g["url"], g["sha"], g.get("path", "."))
PY
while read -r name url sha sub; do
  repo="$work/repos/$(basename "$url")"
  [ -d "$repo" ] || git clone -q "$url" "$repo"
  git -C "$repo" checkout -q "$sha" 2>/dev/null || true
  visualagents scan "$repo/$sub" --static --json -o "$work/out/$name.json" >/dev/null
done < "$work/list.txt"
python3 "$here/score.py" "$work/out" set3
