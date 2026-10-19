"""A simulated multi-agent research crew for `visualagents demo`.

It needs no API key: the agents "think" from a script and print CrewAI-style chain-of-thought,
exactly the kind of console output VisualAgents maps onto the blueprint. The tiny framework
below is real enough that `visualagents scan` finds the same four agents statically.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

SPEED = 1.0


class Agent:
    def __init__(self, name: str, role: str, goal: str, tools: list | None = None):
        self.name, self.role, self.goal, self.tools = name, role, goal, tools or []


class Workflow:
    def __init__(self):
        self.edges: list[tuple[str, str]] = []

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.append((src, dst))


def web_search(query: str) -> str: ...
def read_paper(url: str) -> str: ...
def outline_builder(question: str) -> str: ...
def fact_check(claim: str) -> str: ...


planner = Agent(name="Planner", role="Research strategist", goal="Break the question into a research plan",
                tools=[outline_builder])
researcher = Agent(name="Researcher", role="Senior research analyst", goal="Find and verify sources for each sub-question",
                   tools=[web_search, read_paper])
writer = Agent(name="Writer", role="Technical writer", goal="Turn the findings into a clear, cited report")
critic = Agent(name="Critic", role="Reviewer", goal="Check the draft for unsupported claims and gaps",
               tools=[fact_check])

flow = Workflow()
flow.add_edge("planner", "researcher")
flow.add_edge("researcher", "writer")
flow.add_edge("writer", "critic")
flow.add_edge("critic", "writer")


def say(text: str = "", pause: float = 0.35) -> None:
    print(text, flush=True)
    time.sleep(pause / SPEED * random.uniform(0.7, 1.3))


def stream(prefix: str, text: str, pause: float = 0.9) -> None:
    """Print a thought word-by-word-ish, the way a streaming LLM would."""
    sys.stdout.write(prefix)
    for word in text.split(" "):
        sys.stdout.write(word + " ")
        sys.stdout.flush()
        time.sleep(0.035 / SPEED)
    sys.stdout.write("\n")
    sys.stdout.flush()
    time.sleep(pause / SPEED)


def agent_header(agent: Agent) -> None:
    say("")
    say(f"# Agent: {agent.name}", 0.6)


def main() -> None:
    global SPEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("question", nargs="?", default="How are solid-state batteries changing electric vehicles?")
    args = ap.parse_args()
    SPEED = max(0.1, args.speed)

    say(f"🚀 Crew: research_crew · question: {args.question!r}", 1.0)

    agent_header(planner)
    say("## Task: Break the question into a research plan", 0.5)
    stream("## Thought: ", "The question spans chemistry, manufacturing and market adoption. I should split it so each part can be researched independently.")
    say("## Using tool: outline_builder")
    say('## Tool Input: {"question": "solid-state batteries in EVs"}', 0.8)
    say("## Tool Output: 3 sub-questions: energy density vs Li-ion, manufacturing hurdles, which automakers ship first", 0.8)
    stream("## Final Answer: ", "Plan → (1) energy density and charging, (2) manufacturing & cost, (3) adoption timeline by automaker.")
    say("Handing off to Researcher", 0.8)

    agent_header(researcher)
    say("## Task: Find and verify sources for each sub-question", 0.5)
    stream("## Thought: ", "Start with sub-question 1. I need recent, quantitative numbers on energy density, ideally from peer-reviewed work or manufacturer data sheets.")
    say("## Using tool: web_search")
    say('## Tool Input: {"query": "solid-state battery energy density Wh/kg 2025 prototype"}', 1.0)
    say("## Tool Output: 8 results — QuantumScape QSE-5 (~844 Wh/L), Toyota sulfide roadmap, Nature Energy review…", 0.9)
    stream("## Thought: ", "The Nature Energy review summarises lab vs pilot-line numbers. Reading it gives me a credible baseline.")
    say("## Using tool: read_paper")
    say('## Tool Input: {"url": "https://doi.org/10.1038/s41560-0xx"}', 1.2)
    say("## Tool Output: pilot cells reach 350–400 Wh/kg; 10–80% charge in ~15 min; dendrite suppression still limits cycle life", 0.9)
    stream("## Thought: ", "Sub-question 1 covered. Now manufacturing and cost — dry-room processing and solid electrolyte yield are the usual bottlenecks.")
    say("## Using tool: web_search")
    say('## Tool Input: {"query": "solid-state battery manufacturing cost per kWh yield 2025"}', 1.0)
    say("## Tool Output: analysts estimate 2–3× Li-ion cost today; sulfide electrolytes need moisture-free lines", 0.9)
    say("## Using tool: web_search")
    say('## Tool Input: {"query": "automaker solid-state battery production timeline"}', 1.0)
    say("## Tool Output: Toyota targets 2027–28; Nissan pilot line 2028; Mercedes & Factorial road tests underway", 0.9)
    stream("## Final Answer: ", "Findings: 350–400 Wh/kg in pilot cells, ~15 min fast-charge, 2–3× cost today, first production cars 2027–28. 6 sources.")
    say("Passing results to Writer", 0.8)

    agent_header(writer)
    say("## Task: Turn the findings into a clear, cited report", 0.5)
    stream("## Thought: ", "Readers care about range, charging time and when they can buy one. I'll lead with those and keep chemistry details brief.")
    stream("## Thought: ", "Structure: summary, three sections matching the plan, and a short outlook.", 0.6)
    say("Drafting section 1/3: energy density & charging…", 1.2)
    say("Drafting section 2/3: manufacturing & cost…", 1.2)
    say("Drafting section 3/3: who ships first…", 1.2)
    stream("## Final Answer: ", "Draft v1 ready — 640 words, 6 citations.")
    say("Handing off to Critic", 0.8)

    agent_header(critic)
    say("## Task: Check the draft for unsupported claims and gaps", 0.5)
    stream("## Thought: ", "Check every number against a source. The '50% more range' line in the summary has no citation.")
    say("## Using tool: fact_check")
    say('## Tool Input: {"claim": "solid-state packs give 50% more range"}', 1.0)
    say("## Tool Output: unsupported — sources show +20–40% depending on pack design", 0.9)
    stream("## Final Answer: ", "REVISE: correct the range claim to 20–40% and cite it; add one sentence on cycle-life risk.")
    say("Sending the draft back to Writer", 0.8)

    agent_header(writer)
    stream("## Thought: ", "Fix the range figure, add the citation, and mention dendrite-related cycle-life limits in section 1.")
    say("Revising summary and section 1…", 1.4)
    stream("## Final Answer: ", "Draft v2 ready — range claim corrected (20–40%), cycle-life caveat added.")
    say("Handing off to Critic", 0.8)

    agent_header(critic)
    stream("## Thought: ", "All quantitative claims now trace to a source and the risks are stated. Good to ship.")
    stream("## Final Answer: ", "APPROVED ✅")
    say("")
    say("✅ Crew finished: report delivered (640 words, 7 citations)", 0.2)


if __name__ == "__main__":
    main()
