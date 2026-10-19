"""A hand-rolled multi-agent loop (no framework) instrumented with VisualAgents events.

    visualagents run examples/manual_sdk -- python app.py

`visualagents run` makes `visualagents_runtime` importable, so this works even if the
visualagents package isn't installed in the environment running the app.
"""

import time

try:
    from visualagents import agent, emit, handoff
except ImportError:
    from visualagents_runtime import agent, emit, handoff


def triage(question: str) -> str:
    with agent("triage", task=question):
        emit("thought", "triage", "This is a billing question about a duplicate charge.")
        time.sleep(1.0)
    handoff("triage", "billing", "duplicate charge")
    return "billing"


def billing(question: str) -> str:
    with agent("billing"):
        emit("thought", "billing", "Look up the customer's last two invoices.")
        emit("tool_start", "billing", '{"customer": "c_123"}', tool="lookup_invoices")
        time.sleep(1.2)
        emit("tool_end", "billing", "2 invoices, both for $49 on Oct 3", tool="lookup_invoices")
        emit("thought", "billing", "It is a duplicate. Refunds need the refunds agent.")
        time.sleep(0.8)
    handoff("billing", "refunds", "refund one $49 charge")
    return "refunds"


def refunds(question: str) -> str:
    with agent("refunds"):
        emit("tool_start", "refunds", '{"invoice": "in_456", "amount": 49}', tool="issue_refund")
        time.sleep(1.0)
        emit("tool_end", "refunds", "refund re_789 created", tool="issue_refund")
        emit("output", "refunds", "Refunded the duplicate $49 charge (re_789). It will appear in 5-10 days.")
    return "done"


if __name__ == "__main__":
    q = "I was charged twice this month"
    nxt = triage(q)
    nxt = billing(q) if nxt == "billing" else nxt
    refunds(q)
