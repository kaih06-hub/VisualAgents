"""A LangGraph supervisor team that runs fully offline (scripted chat model, no API key).

    pip install langgraph
    visualagents run examples/langgraph_supervisor -- python app.py

VisualAgents hooks LangGraph's callbacks automatically, so every node, LLM reply and tool
call is attributed exactly - no log parsing needed.
"""

from __future__ import annotations

import time
from typing import Literal, TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    task: str
    notes: list[str]
    next: str
    turn: int


@tool
def search_docs(query: str) -> str:
    """Search the internal documentation."""
    time.sleep(1.2)
    return f"3 pages about {query}: rate limits, retry headers, token bucket design"


@tool
def write_file(path: str) -> str:
    """Write a file to the workspace."""
    time.sleep(1.0)
    return f"wrote {path} (84 lines)"


@tool
def run_tests(target: str) -> str:
    """Run the unit tests."""
    time.sleep(1.5)
    return "11 passed, 1 failed: test_burst_limit" if "v1" in target else "12 passed"


supervisor_llm = FakeListChatModel(responses=[
    "The task needs background first. Routing to the researcher.",
    "Research is done. The coder can implement the token bucket now.",
    "Code is written. Sending it to the reviewer.",
    "The reviewer found a failing burst test. Back to the coder for a fix.",
    "Fix is in. One more review pass.",
    "All checks pass. Finishing.",
])
researcher_llm = FakeListChatModel(responses=[
    "I should look up how our API exposes rate limits before proposing a design.",
])
coder_llm = FakeListChatModel(responses=[
    "I'll implement a token bucket with a refill rate from the config and write tests.",
    "test_burst_limit fails because the bucket starts empty. Initialise it full.",
])
reviewer_llm = FakeListChatModel(responses=[
    "Run the suite against v1 and check edge cases around bursts.",
    "Re-running the suite on v2 to confirm the fix.",
])

PLAN = ["researcher", "coder", "reviewer", "coder", "reviewer", "finish"]


def supervisor(state: State) -> dict:
    turn = state.get("turn", 0)
    print(f"[supervisor] turn {turn + 1}: checking progress", flush=True)
    reply = supervisor_llm.invoke(f"Notes so far: {state.get('notes', [])}")
    print(f"[supervisor] {reply.content}", flush=True)
    time.sleep(0.8)
    return {"next": PLAN[turn], "turn": turn + 1}


def route(state: State) -> Literal["researcher", "coder", "reviewer", "finish"]:
    return state["next"]  # type: ignore[return-value]


def researcher(state: State) -> dict:
    thought = researcher_llm.invoke(state["task"])
    time.sleep(0.6)
    found = search_docs.invoke({"query": "rate limiting"})
    return {"notes": state.get("notes", []) + [f"research: {found}"]}


def coder(state: State) -> dict:
    coder_llm.invoke(state["task"])
    time.sleep(0.6)
    version = "v1" if not any("code" in n for n in state.get("notes", [])) else "v2"
    out = write_file.invoke({"path": f"ratelimit_{version}.py"})
    return {"notes": state.get("notes", []) + [f"code {version}: {out}"]}


def reviewer(state: State) -> dict:
    reviewer_llm.invoke("review")
    time.sleep(0.5)
    latest = [n for n in state.get("notes", []) if n.startswith("code")][-1]
    result = run_tests.invoke({"target": latest})
    return {"notes": state.get("notes", []) + [f"review: {result}"]}


builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)
builder.add_node("reviewer", reviewer)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor", route, {"researcher": "researcher", "coder": "coder", "reviewer": "reviewer", "finish": END}
)
for worker in ("researcher", "coder", "reviewer"):
    builder.add_edge(worker, "supervisor")
graph = builder.compile()

if __name__ == "__main__":
    final = graph.invoke({"task": "Add rate limiting to the public API", "notes": [], "next": "", "turn": 0})
    print("\nDone:", *final["notes"], sep="\n  ")
