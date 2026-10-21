#!/bin/bash
# Re-run the README-graph benchmark: clone the ten projects at the checked commits, scan each statically, score.
# usage: benchmarks/readme_graphs/run.sh [workdir]   (needs git and an installed visualagents)
set -e
here="$(cd "$(dirname "$0")" && pwd)"; work="${1:-/tmp/va-readme-graphs}"; mkdir -p "$work/repos" "$work/out"
while read -r name url sha; do
  [ -d "$work/repos/$name" ] || git clone -q "$url" "$work/repos/$name"
  git -C "$work/repos/$name" checkout -q "$sha" 2>/dev/null || true
  visualagents scan "$work/repos/$name" --static --json -o "$work/out/$name.json" >/dev/null
done <<'LIST'
AI-Data-Analysis-MultiAgent https://github.com/starpig1129/AI-Data-Analysis-MultiAgent 9d53a6c
local-deep-researcher https://github.com/langchain-ai/local-deep-researcher 9655564
langgraph-email-automation https://github.com/kaymen99/langgraph-email-automation 9ea0f54
Upwork-AI-jobs-applier https://github.com/kaymen99/Upwork-AI-jobs-applier 074dc0d
openai-cs-agents-demo https://github.com/openai/openai-cs-agents-demo bd7bfca
AutoAgents https://github.com/Link-AGI/AutoAgents 223ad99
MedAgents https://github.com/gersteinlab/MedAgents aaeff04
Multi-Agents-Debate https://github.com/Skytliang/Multi-Agents-Debate e58d146
Paper2Code https://github.com/going-doer/Paper2Code ba91699
AgentCoder https://github.com/huangd1999/AgentCoder d8538d6
LIST
python3 "$here/score.py" "$work/out" set1
