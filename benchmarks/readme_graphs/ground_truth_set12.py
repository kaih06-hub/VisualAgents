"""Ground truth for the twelfth README-graph set: nine projects that appear in none of sets 1-11.

A test set only, written on 2026-11-23 from each README's mermaid diagram and the names the source gives each box,
BEFORE any version of the scanner ran on these projects, and not edited after the first scan. Do not tune against
it - add a thirteenth set instead.

How the nine were picked, so the choice could not lean towards what the scanner handles: GitHub repository search on
13 agent topics (multi-agent, langgraph, crewai, autogen, agentic-ai, ai-agents, llm-agents, openai-agents-sdk,
smolagents, ...), Python, 5+ stars, updated since 2026-07-01; minus every repository of sets 1-11 (536 left). They
were put in a fixed pseudo-random order (sha1 of "set12:" + lowercase owner/name) and taken in that order when the
README holds a mermaid diagram that draws three or more agents and the Python code implements them with model calls.
Every one of the 536 with a mermaid diagram was checked; nine passed. Passed over, with the reason:
- platforms, frameworks, SDKs, skill packs and templates whose diagram is infrastructure (about 40);
- risk-aware-trading-agents: the drawn macro / event / technical / news agents are deterministic rules;
- cyber-ai-team: its agents are rows in a database, so the code holds no fixed structure;
- jaegis-RAVERSE: the diagram draws MCP tools, not agents;
- redamon: the diagram draws one agent and its tools;
- trading-arena: its code is a copy of TauricResearch/TradingAgents, which set 1 tuned the scanner on, so it is
  not unseen (and its README draws a different system from that code).

Edges are the arrows the README draws between agent boxes. A box the README draws that prompts no model (a code
gate, an orchestrator that only gathers results) is kept as a node but marked readme_only, as in set 11. Arrows
drawn as an undirected line (race-condition) are taken in the order they are written.
"""

GT12 = {
    "SOC": {
        "url": "https://github.com/gapilongo/SOC", "sha": "5dded810",
        "framework": "langgraph (nodes added in a loop over a tuple of names)",
        "source": "README second mermaid 'flowchart LR': gate -> triage -> correlation -> analysis -> decide -> "
                  "escalate / plan response -> await approval -> execute -> close; safety net reopens closed alerts",
        "nodes": {"Gate": r"^gate$", "Triage": r"^triage", "Correlation": r"^correlation", "Analysis": r"^analysis$",
                  "Decide": r"^decide$", "Escalate": r"^escalate$", "Plan": r"plan_response",
                  "AwaitApproval": r"await_approval", "Execute": r"execute_actions|^execute$", "Close": r"^close$",
                  "SafetyNet": r"safety_net"},
        "edges": [("Gate", "Triage"), ("Gate", "Close"), ("Triage", "Close"), ("Triage", "Correlation"),
                  ("Correlation", "Analysis"), ("Correlation", "Decide"), ("Analysis", "Decide"), ("Decide", "Close"),
                  ("Decide", "Escalate"), ("Decide", "Plan"), ("Escalate", "Plan"), ("Escalate", "Close"),
                  ("Escalate", "Analysis"), ("Plan", "AwaitApproval"), ("Plan", "Execute"),
                  ("AwaitApproval", "Execute"), ("Execute", "Close"), ("Execute", "Escalate"), ("Close", "SafetyNet"),
                  ("SafetyNet", "Escalate")],
        "readme_only": {"SafetyNet"},  # a background check that reopens cases, not a graph node
        "real_extra": r"await_analyst",  # the README's "escalate: an analyst decides" is two nodes in the code
    },
    "ALTA": {
        "url": "https://github.com/kyky2347/ALTA", "sha": "68c31c4b",
        "framework": "own runtime (role models picked per step)",
        "source": "README second mermaid 'flowchart TB': thesis assessor + independent challenger -> moderator -> "
                  "code-based ranking -> expression agent -> independent auditor",
        "nodes": {"ThesisAssessor": r"thesis_?assessor", "Challenger": r"disconfirming|challeng",
                  "Moderator": r"moderator", "Ranking": r"ranking|^rank", "Expression": r"expression_?agent|^expression$",
                  "Auditor": r"expression_?auditor|auditor"},
        "edges": [("ThesisAssessor", "Moderator"), ("Challenger", "Moderator"), ("Moderator", "Ranking"),
                  ("Ranking", "Expression"), ("Expression", "Auditor")],
        "readme_only": {"Ranking"},  # "code-based ranking": no model
        "real_extra": r"scout|position_reviewer|deliberat|trader_mind|research",
    },
    "AEGIS-SWARM": {
        "url": "https://github.com/CODERUDRA-X/AEGIS-SWARM", "sha": "d4139b7d",
        "framework": "plain model calls (one function per agent)",
        "source": "README mermaid 'graph TD': scout agent -> risk agent -> critic -> deterministic safety gate -> "
                  "commander agent; the gate sends the critic's reasoning back to risk",
        "nodes": {"Scout": r"scout|analyze_crowd_frame", "Risk": r"^risk|evaluate_risk",
                  "Critic": r"critic|challenge_risk", "Gate": r"safety_gate", "Commander": r"commander|action_plan"},
        "edges": [("Scout", "Risk"), ("Risk", "Critic"), ("Critic", "Gate"), ("Gate", "Risk"), ("Gate", "Commander")],
        "readme_only": {"Gate"},  # "pure Python, no LLM"
        "real_extra": r"$^",
    },
    "BetterTradingAgents": {
        "url": "https://github.com/kingabzpro/BetterTradingAgents", "sha": "288f4d33",
        "framework": "openai-agents",
        "source": "README mermaid 'flowchart LR': researchers in parallel (market, technical, fundamental, news, "
                  "sentiment, forecast) -> bull vs bear debate + judge -> portfolio manager -> risk gate",
        "nodes": {"Market": r"market", "Technical": r"technical", "Fundamental": r"fundamental", "News": r"news",
                  "Sentiment": r"sentiment", "Forecast": r"forecast", "Bull": r"bull", "Bear": r"bear",
                  "Judge": r"judge", "Manager": r"portfolio_?manager|^manager", "RiskGate": r"risk_?gate|^risk$"},
        "edges": [*[(r, side) for r in ("Market", "Technical", "Fundamental", "News", "Sentiment", "Forecast")
                    for side in ("Bull", "Bear")],
                  ("Bull", "Judge"), ("Bear", "Judge"), ("Judge", "Manager"), ("Manager", "RiskGate")],
        "readme_only": {"RiskGate"},  # sizing rules in code
        "real_extra": r"reflection|chat",
    },
    "EpochGo": {
        "url": "https://github.com/AryanDinakaran/EpochGo", "sha": "157657a9",
        "framework": "own agent classes run by a crew class",
        "source": "README mermaid 'flowchart TD': agent 1 data preprocessor -> agent 2 model builder -> agent 3 "
                  "model serving",
        "nodes": {"Preprocessor": r"preprocess", "ModelBuilder": r"model_?build", "Deployer": r"deploy|serv"},
        "edges": [("Preprocessor", "ModelBuilder"), ("ModelBuilder", "Deployer")],
        "real_extra": r"crew",
    },
    "Adaptive-AI": {
        "url": "https://github.com/P-mohith230/Adaptive-AI", "sha": "b3f9cc1d",
        "framework": "own BaseAgent classes (asyncio.gather)",
        "source": "README mermaid 'flowchart LR': orchestrator -> workflow / research / sentiment / trend agents; "
                  "research -> ROI agent; all -> strategy synthesis",
        "nodes": {"Orchestrator": r"orchestrator", "Workflow": r"workflow", "Research": r"research",
                  "Sentiment": r"sentiment", "Trend": r"trend", "ROI": r"^roi", "Strategy": r"strategy"},
        "edges": [*[("Orchestrator", x) for x in ("Workflow", "Research", "Sentiment", "Trend")], ("Research", "ROI"),
                  *[(x, "Strategy") for x in ("Workflow", "Research", "Sentiment", "Trend", "ROI")]],
        "readme_only": {"Orchestrator"},  # gathers the agents and fetches data; prompts no model itself
        "real_extra": r"engine",
    },
    "interviewpilot-ai": {
        "url": "https://github.com/oguzhandurmaz/interviewpilot-ai", "sha": "b7866692",
        "framework": "langgraph",
        "source": "README mermaid 'graph TD': build context -> load resources -> research & summarization -> index "
                  "-> retrieve -> generate question -> wait -> evaluate -> supervisor -> next question / study note "
                  "/ summary",
        "nodes": {"BuildContext": r"build_context", "LoadResources": r"load_resources",
                  "Research": r"research_summary", "Index": r"index_documents", "Retrieve": r"retrieve_context",
                  "Generate": r"generate_question", "Wait": r"wait_for_answer", "Evaluate": r"evaluate_answer",
                  "Supervisor": r"supervisor", "Note": r"followup_actions|create_note|study_note",
                  "Summary": r"^summary"},
        "edges": [("BuildContext", "LoadResources"), ("LoadResources", "Research"), ("Research", "Index"),
                  ("Index", "Retrieve"), ("Retrieve", "Generate"), ("Generate", "Wait"), ("Wait", "Evaluate"),
                  ("Evaluate", "Supervisor"), ("Supervisor", "Generate"), ("Supervisor", "Note"),
                  ("Note", "Generate"), ("Supervisor", "Summary")],
        "real_extra": r"interview_router|planner",
    },
    "race-condition": {
        "url": "https://github.com/GoogleCloudPlatform/race-condition", "sha": "26efc1b4",
        "framework": "google-adk (separate agent apps talking over A2A)",
        "source": "README mermaid 'graph TD': planner agent --- simulator agent --- runner agents (a Go gateway in "
                  "front); second diagram: pre-race -> race engine -> post-race inside the simulator",
        "nodes": {"Planner": r"^planner(_agent)?$", "Simulator": r"^simulator(_agent)?$|simulation_pipeline",
                  "Runner": r"^runners?(_agent)?$"},
        "edges": [("Planner", "Simulator"), ("Simulator", "Runner")],
        # variants of the three apps kept beside them, and the simulator's own sub-agents
        "real_extra": r"planner_with|simulator_with|runner_autopilot|pre_race|race_engine|post_race|^tick$",
    },
    "Clouds-Coder": {
        "url": "https://github.com/FonaTech/Clouds-Coder", "sha": "6f187a3f",
        "framework": "own runtime (one 8 MB Python file plus a split copy)",
        "source": "README second mermaid 'flowchart TB': in a private session the manager hands work to explorer, "
                  "developer and reviewer",
        "nodes": {"Manager": r"manager", "Explorer": r"explorer", "Developer": r"developer", "Reviewer": r"reviewer"},
        "edges": [("Manager", "Explorer"), ("Manager", "Developer"), ("Manager", "Reviewer")],
        "real_extra": r"$^",
    },
}
