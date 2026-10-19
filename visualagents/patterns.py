"""Log-line patterns used to attribute chain-of-thought output to agents.

Add a pattern here to teach VisualAgents a new framework's console format. Agent-switch
patterns capture the agent in a group named ``agent``; kind patterns classify the line.
"""

from __future__ import annotations

import re
import unicodedata

MARKER = "@@visualagents "

ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
BOX = "│┃|╭╮╰╯─━┌┐└┘├┤┬┴┼║═╔╗╚╝▏▕"
BOX_EDGES = re.compile(f"^[\\s{re.escape(BOX)}]+|[\\s{re.escape(BOX)}]+$")


def clean(raw: str) -> str:
    """Strip ANSI codes, carriage-return overwrites and box-drawing borders (rich panels)."""
    line = ANSI.sub("", raw.rstrip("\n"))
    if "\r" in line:
        parts = [p for p in line.split("\r") if p.strip()]
        line = parts[-1] if parts else ""
    line = BOX_EDGES.sub("", line)
    return line.replace("\t", "    ")


def strip_symbols(text: str) -> str:
    """Drop leading emoji / pictographs so '🔧 Using Tool: x' reads as 'Using Tool: x'."""
    i = 0
    while i < len(text) and (text[i].isspace() or text[i] == "️" or unicodedata.category(text[i]) in {"So", "Sk", "Cf", "Cn"}):
        i += 1
    return text[i:]


# (pattern, strength, kind). strength "strong": an unknown agent name creates a runtime agent;
# "weak": only applies when the name resolves to a known agent.
SWITCH_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # CrewAI: "# Agent: Senior Researcher", rich panel "Agent: X", legacy "Working Agent: X"
    (re.compile(r"^(?:#+\s*)?Agent(?:\s+Name)?\s*:\s*(?P<agent>.{2,80}?)\s*$"), "strong", "status"),
    (re.compile(r"Working Agent:\s*(?P<agent>.{2,80}?)\s*$"), "strong", "status"),
    # AutoGen 0.2: "researcher (to writer):"
    (re.compile(r"^(?P<agent>[\w\-. ]{1,40}?)\s+\(to\s+(?P<to>[\w\-. ]{1,40})\)\s*:?\s*$"), "strong", "message"),
    # AutoGen 0.4 console: "---------- TextMessage (researcher) ----------" / "---------- researcher ----------"
    (re.compile(r"^-{4,}\s*(?:\w*(?:Message|Event)\s*)?\(?(?P<agent>[\w\-.]+)\)?\s*-{4,}\s*$"), "strong", "message"),
    # LangGraph: "Update from node researcher:", message pretty_print "Name: researcher", stream dicts
    (re.compile(r"^Update from (?:node|subgraph)\s+(?P<agent>[\w\-.]+)"), "strong", "status"),
    (re.compile(r"^Name:\s*(?P<agent>[\w\-.]+)\s*$"), "weak", "status"),
    (re.compile(r"""^\{\s*['"](?P<agent>[\w\-]+)['"]\s*:\s*[\{\[]"""), "weak", "status"),
    # OpenAI Agents SDK / Swarm style
    (re.compile(r"(?:Handoff|Handed off|Handing off|Transferr(?:ed|ing))\s+(?:control\s+)?to\s*:?\s*['\"`]?(?P<agent>[\w\- ]{2,40}?)['\"`]?(?:\s+agent)?\s*(?:[.,:!]|$)", re.I), "weak", "handoff"),
    (re.compile(r"\btransfer_to_(?P<agent>\w+)"), "weak", "handoff"),
    # Generic narration
    (re.compile(r"(?:Delegating|Passing|Routing|Sending|Switching|Returning|Forwarding)\s+(?:[\w\-]+\s+){0,3}?to\s+(?:the\s+)?['\"`]?(?P<agent>[\w\- ]{2,40}?)['\"`]?(?:\s+agent)?\s*(?:[.,:!…]|$)", re.I), "weak", "handoff"),
    (re.compile(r"(?:Next agent|Next node|next|goto|Selected agent|Speaker)\s*[:=]\s*['\"`]?(?P<agent>[\w\- ]{2,40}?)['\"`]?\s*$", re.I), "weak", "handoff"),
    (re.compile(r"(?:Entering|Starting|Running|Invoking|Activating|Calling)\s+(?:the\s+)?(?:agent|node)\s*[:\-]?\s*['\"`]?(?P<agent>[\w\- ]{2,40}?)['\"`]?\s*(?:[.,:!…]|\.\.\.|$)", re.I), "weak", "status"),
    (re.compile(r"^(?:=+|-{3,}|\*{3,}|#+|>+)\s*\[?(?P<agent>[A-Za-z][\w\- ]{1,40}?)\]?(?:\s+(?:agent|node))?\s*(?:=+|-{3,}|\*{3,}|#*)?\s*$", re.I), "weak", "status"),
    (re.compile(r"^\[(?P<agent>[A-Za-z][\w\- .]{1,40})\]"), "weak", "log"),
    (re.compile(r"^(?P<agent>[A-Za-z][\w\- ]{1,30}?)\s*(?:>|»|:|\||->)\s+\S"), "weak", "log"),
]

KIND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(Traceback \(most recent call last\)|^\w*(?:Error|Exception)\b\s*[:(]|^ERROR\b|^\[ERROR\]|\bFAILED\b)"), "error"),
    (re.compile(r"^#*\s*(?:Thought|Thinking|Reasoning|Plan)\s*:\s*(?P<text>.*)$", re.I), "thought"),
    (re.compile(r"^#*\s*(?:Using Tool|Tool|Action|Calling tool|Tool call|Calling)\s*:\s*`?(?P<tool>[\w\-.]+)`?(?P<text>.*)$", re.I), "tool"),
    (re.compile(r"^Invoking:\s*`(?P<tool>[^`]+)`(?P<text>.*)$"), "tool"),
    (re.compile(r"^\*{3,}\s*Suggested tool call.*?:\s*(?P<tool>[\w\-.]+)", re.I), "tool"),
    (re.compile(r"^#*\s*(?:Action Input|Tool Input|Tool Args|Arguments)\s*:\s*(?P<text>.*)$", re.I), "tool_input"),
    (re.compile(r"^#*\s*(?:Observation|Tool Output|Tool Result|Tool Response|Result)\s*:\s*(?P<text>.*)$", re.I), "observation"),
    (re.compile(r"^#*\s*(?:Final Answer|Final Output|Final Response|Answer|Output|Response)\s*:\s*(?P<text>.*)$", re.I), "output"),
    (re.compile(r"^#*\s*(?:Task|Assigned Task|Goal)\s*:\s*(?P<text>.*)$", re.I), "task"),
    (re.compile(r"^>?\s*Entering new (?P<text>[\w ]+) chain", re.I), "status"),
    (re.compile(r"^>?\s*Finished chain", re.I), "status"),
]

# Kinds whose following un-prefixed lines are a continuation (multi-line thoughts / answers).
CONTINUES = {"thought", "output", "observation", "tool_input", "message", "task"}
# Kinds that describe what an agent is doing, in order of how informative they are.
ACTIVITY_KINDS = {"thought": 5, "tool": 5, "task": 4, "output": 5, "handoff": 3, "message": 3, "narration": 6,
                  "observation": 2, "tool_input": 1, "log": 1}
