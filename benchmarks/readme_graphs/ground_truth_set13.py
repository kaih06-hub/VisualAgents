"""Ground truth for the thirteenth README-graph set: nine projects that appear in none of sets 1-12.

A test set only, written on 2026-11-25 from each README's mermaid diagram and the names the source gives each box,
BEFORE any version of the scanner ran on these projects, and not edited after the first scan. Do not tune against
it - add a fourteenth set instead.

How the nine were picked, so the choice could not lean towards what the scanner handles: GitHub repository search on
17 agent topics (set 12's 13 - multi-agent, multi-agent-system(s), langgraph, crewai, autogen, agentic-ai, ai-agents,
llm-agents, agentic-workflow, openai-agents-sdk, smolagents, agent-orchestration - plus pydantic-ai, agno, google-adk,
ai-agent), Python, 5+ stars, updated since 2026-07-01, up to 300 per topic; minus every repository of sets 1-12 and
every one of the 536 that set 12 examined (1,852 left). They were put in a fixed pseudo-random order (sha1 of
"set13:" + lowercase owner/name) and taken in that order when the README holds a mermaid diagram that draws three or
more agents and the Python code implements them with model calls. The first 400 READMEs were read (70 with a mermaid
diagram); nine passed by position 203. Passed over, with the reason:
- cto-agent-team, investment-council: the agents are prompt files run by Claude Code; no model calls in Python;
- OpenFusion: its tiers are model settings for Claude Code sub-agents; no model calls in Python;
- kaji: each workflow step is handed to the claude / codex command-line tools;
- krystal-loop-protocol: no model calls in its Python code;
- temporal-ai-hitl-adk-langgraph: the diagram draws the Temporal / FastAPI infrastructure, the agents only named
  inside one box per framework;
- frameworks, platforms, infrastructure, skill packs, method documents and single-agent tools (24 more).

Edges are the arrows the README draws between agent boxes; an arrow through a decision diamond or a router shape is
taken from the box before it to each box after it. A box the README draws that the code has no block for (a tool, a
check inside middleware, plain code called from an agent) is kept as a node but marked readme_only, as in sets 11
and 12. A box the code splits into several blocks (an input step, the agent and a merge step) is in `multi`, so
each of them counts as that box.
"""

GT13 = {
    "DeepReason": {
        "url": "https://github.com/IceSugar93/DeepReason", "sha": "abfdf308",
        "framework": "langgraph",
        "source": "README mermaid 'flowchart LR': hybrid retrieval -> planner -> multi-hop retrieval / generator -> "
                  "critic -> finalize or reviser -> verify -> critic / finalize",
        "nodes": {"Retrieve": r"^retrieve(_node)?$|hybrid", "Planner": r"^plan(ner)?(_node)?$|call_planner",
                  "MultiHop": r"multi_hop", "Generator": r"generat", "Critic": r"^critic(_node)?$|call_critic$",
                  "Reviser": r"revis", "Verify": r"^verify(_node)?$|critic_verify", "Finalize": r"finali[sz]"},
        "edges": [("Retrieve", "Planner"), ("Planner", "MultiHop"), ("Planner", "Generator"), ("MultiHop", "Generator"),
                  ("Generator", "Critic"), ("Critic", "Finalize"), ("Critic", "Reviser"), ("Reviser", "Verify"),
                  ("Verify", "Finalize"), ("Verify", "Critic")],
        "real_extra": r"hyde|engine",
    },
    "QuantLab": {
        "url": "https://github.com/QuantSingularity/Agentic-AI-for-Quantitative-Research-and-Alpha-Discovery",
        "sha": "408a6ca7",
        "framework": "langgraph (each node runs an agent object)",
        "source": "README mermaid 'flowchart LR': planner -> literature / hypothesis -> data eng -> model -> backtest "
                  "-> evaluate -> report; a reflective memory store beside them",
        "nodes": {"Planner": r"planner", "Literature": r"literature", "Hypothesis": r"hypothes",
                  "DataEng": r"^data(_agent|_eng\w*)?$|data_engineering", "Model": r"^model(_agent)?$",
                  "Backtest": r"backtest", "Evaluate": r"^evaluat|^eval(_agent)?$", "Report": r"^report"},
        "edges": [("Planner", "Literature"), ("Planner", "Hypothesis"), ("Literature", "Hypothesis"),
                  ("Hypothesis", "DataEng"), ("DataEng", "Model"), ("Model", "Backtest"), ("Backtest", "Evaluate"),
                  ("Evaluate", "Report")],
        "real_extra": r"reflect|memory",  # the reflective memory is drawn as a store; the code has a reflection agent
    },
    "FloatTrip": {
        "url": "https://github.com/shouzhuoshouzhuo/FloatTrip", "sha": "8c8972f5",
        "framework": "langgraph (a chat graph, a planning graph and a revision graph)",
        "source": "README mermaid 'flowchart TD': ReAct main agent -> confirm planning brief -> weather and POI -> "
                  "candidates -> optimizer -> quality check -> save, or revision prep -> planner <-> reviewer -> meals",
        "nodes": {"MainAgent": r"dialogue_agent|react|main_agent", "Brief": r"brief|requirements_and_draft|collect_required",
                  "WeatherPOI": r"weather|attraction_search|poi", "Candidates": r"candidate", "Optimizer": r"optimi[sz]",
                  "Quality": r"quality", "Save": r"^finali[sz]e|save_trip|save_itinerary",
                  "RevisionPrep": r"revision_(prepare|search|dates)", "Planner": r"^planner(_node)?$",
                  "Reviewer": r"^reviewer(_node)?$", "Meals": r"meal"},
        "multi": {"Brief", "WeatherPOI", "RevisionPrep", "Meals"},
        "edges": [("MainAgent", "Brief"), ("Brief", "WeatherPOI"), ("WeatherPOI", "Candidates"),
                  ("Candidates", "Optimizer"), ("Optimizer", "Quality"), ("Quality", "Save"), ("Quality", "Candidates"),
                  ("MainAgent", "RevisionPrep"), ("RevisionPrep", "Planner"), ("Planner", "Reviewer"),
                  ("Reviewer", "Planner"), ("Reviewer", "Meals"), ("Meals", "Save")],
        "real_extra": r"revision_concern|memory|deepseek|doubao|semantic",
    },
    "zkzkAgent": {
        "url": "https://github.com/zkzkGamal/zkzkAgent", "sha": "3df38aef",
        "framework": "langgraph",
        "source": "README mermaid 'flowchart TD': classify -> conversation / execute / plan; plan -> execute; "
                  "execute -> confirmation (dangerous tools) or tools -> execute",
        "nodes": {"Classify": r"classif", "Conversation": r"conversation", "Execute": r"^execut",
                  "Plan": r"^plan(_node)?$|planner", "Tools": r"^tools?(_node)?$|tool_node",
                  "Confirm": r"confirm|human"},
        "edges": [("Classify", "Conversation"), ("Classify", "Execute"), ("Classify", "Plan"), ("Plan", "Execute"),
                  ("Execute", "Confirm"), ("Execute", "Tools"), ("Tools", "Execute")],
        "readme_only": {"Confirm"},  # asked inside the execute step, not a node of the graph
        "real_extra": r"handover|stream",
    },
    "accesspilot": {
        "url": "https://github.com/joinarun/accesspilot", "sha": "467a0d98",
        "framework": "google-adk (a workflow of function nodes and LlmAgent specialists)",
        "source": "README mermaid 'flowchart TD': intake -> load -> detect -> run -> routes -> scan -> normalize -> "
                  "audit -> map -> investigate -> guard -> plan -> repair -> patch -> verify -> regression -> "
                  "preview -> PR draft -> approval -> publish -> report, with the retry and report-only branches",
        "nodes": {"Intake": r"intake", "Load": r"repository_loader|^load", "Detect": r"framework_detector|^detect",
                  "Run": r"local_app_runner|^run(_app)?$", "Routes": r"route_discovery|^routes?$",
                  "Scan": r"accessibility_scan|^scan$", "Normalize": r"normali[sz]",
                  "Audit": r"auditor|^audit", "Map": r"source_mapping|^map$", "Investigate": r"investigat",
                  "Guard": r"repairability", "Plan": r"planner|^plan$", "Repair": r"repairer|^repair$",
                  "Patch": r"patch_validation|^patch$", "Verify": r"verifier|verification_scan|^verify",
                  "Regression": r"regression", "Preview": r"preview", "PRDraft": r"pr_writer|pr_draft",
                  "Approval": r"approval", "Publish": r"pull_request|publish", "Report": r"final_report|^report$"},
        "multi": {"Audit", "Investigate", "Plan", "Repair", "Verify", "PRDraft"},
        "edges": [("Intake", "Load"), ("Load", "Detect"), ("Detect", "Run"), ("Run", "Routes"), ("Routes", "Scan"),
                  ("Scan", "Normalize"), ("Normalize", "Audit"), ("Audit", "Map"), ("Map", "Investigate"),
                  ("Run", "Report"), ("Investigate", "Guard"), ("Guard", "Plan"), ("Plan", "Repair"),
                  ("Repair", "Patch"), ("Patch", "Verify"), ("Verify", "Regression"), ("Guard", "Report"),
                  ("Regression", "Plan"), ("Regression", "Preview"), ("Preview", "PRDraft"), ("PRDraft", "Approval"),
                  ("Approval", "Plan"), ("Approval", "Report"), ("Approval", "Publish"), ("Publish", "Report")],
        "real_extra": r"revision_guard|accesspilot_workflow|root_agent|function_node",
    },
    "advanced-multi-agent": {
        "url": "https://github.com/Aj2280/advanced-multi-agent", "sha": "cae3ef48",
        "framework": "own agent classes over one OpenAI-compatible router",
        "source": "README mermaid 'graph TD': meta-orchestrator -> agent pool (debate / pipeline pattern) -> "
                  "researcher, coder, analyst; search and sandbox tools, message bus, memory and monitoring beside them",
        "nodes": {"Orchestrator": r"orchestrat|meta_agent|^meta", "Researcher": r"research", "Coder": r"coder",
                  "Analyst": r"analyst"},
        "multi": {"Orchestrator"},  # drawn once; the code has a MetaAgent and the Orchestrator it drives
        "edges": [("Orchestrator", "Researcher"), ("Orchestrator", "Coder"), ("Orchestrator", "Analyst")],
        "real_extra": r"writer|debate|pipeline|competitive|router|judge|critic|improv",
    },
    "Market-Rover": {
        "url": "https://github.com/SankarGaneshb/Market-Rover", "sha": "ee3e3c1f",
        "framework": "crewai",
        "source": "README mermaid 'graph TD': portfolio manager -> market strategist (forensic check) / technical "
                  "analyst; strategist -> sentiment analyzer; sentiment + technical -> shadow analyst; all -> report "
                  "generator; a data visualizer beside them",
        "nodes": {"PortfolioManager": r"portfolio_?manager", "Strategist": r"strategist|news_scraper|market_impact",
                  "Forensic": r"forensic|integrity", "Sentiment": r"sentiment", "Technical": r"technical|market_context",
                  "Report": r"report_?(generator|writer)|intelligence_report", "Shadow": r"shadow",
                  "Visualizer": r"visuali[sz]"},
        "edges": [("PortfolioManager", "Strategist"), ("Strategist", "Forensic"), ("Strategist", "Sentiment"),
                  ("PortfolioManager", "Technical"), ("Strategist", "Report"), ("Sentiment", "Shadow"),
                  ("Technical", "Shadow"), ("Shadow", "Report"), ("Technical", "Report"), ("Sentiment", "Report")],
        "readme_only": {"Forensic"},  # a tool the strategist calls (check_accounting_fraud), not an agent
        "real_extra": r"timing|sre|support|all_agent|bull|bear|duel|brand|owner",
    },
    "ai-investment-agent": {
        "url": "https://github.com/rgoerwit/ai-investment-agent", "sha": "308d1a35",
        "framework": "langgraph",
        "source": "README mermaid 'graph TB': parallel dispatch to market / sentiment / news / junior fundamentals / "
                  "foreign language / legal / value trap / forensic auditor; fundamentals barrier -> senior -> "
                  "validator; analyst sync -> fast-fail or bull + bear rounds -> research manager -> valuation / "
                  "APAC / consultant or skip -> post-research barrier -> trader -> risk analysts -> portfolio "
                  "manager -> charts",
        "nodes": {"Dispatch": r"dispatch", "Macro": r"macro", "Market": r"^market(_analyst)?$",
                  "Sentiment": r"^sentiment", "News": r"^news", "Junior": r"junior", "Foreign": r"foreign",
                  "Legal": r"legal", "Trap": r"trap", "Auditor": r"^(forensic_)?auditor$",
                  "FundSync": r"fundamentals_sync|fundamentals_barrier",
                  "Senior": r"^(senior_)?fundamentals(_analyst)?$|senior", "Validator": r"validator",
                  "Sync": r"^sync_check$|analyst_sync", "FastFail": r"fast_?fail",
                  "R1": r"(bull|bear)_researcher_r1", "DebateSync": r"debate_sync_r1",
                  "R2": r"(bull|bear)_researcher_r2", "FinalSync": r"debate_sync_final|sync_final",
                  "ResearchManager": r"research_manager", "Valuation": r"valuation", "APAC": r"apac",
                  "Consultant": r"^(external_)?consultant$", "Skip": r"consultant_skip",
                  "PostSync": r"post_research", "Trader": r"^trader$", "Risk": r"risky|^safe|neutral",
                  "PortfolioManager": r"portfolio_manager", "Charts": r"chart"},
        "multi": {"R1", "R2", "Risk"},  # each box is a bull and a bear (or three risk) analyst nodes
        "edges": [*[("Dispatch", x) for x in ("Market", "Sentiment", "News", "Junior", "Foreign", "Legal", "Trap",
                                              "Auditor")],
                  ("Macro", "News"), ("Junior", "FundSync"), ("Foreign", "FundSync"), ("Legal", "FundSync"),
                  ("FundSync", "Senior"), ("Senior", "Validator"),
                  *[(x, "Sync") for x in ("Market", "Sentiment", "News", "Trap", "Auditor", "Validator")],
                  ("Sync", "FastFail"), ("Sync", "R1"), ("R1", "DebateSync"), ("DebateSync", "R2"),
                  ("DebateSync", "FinalSync"), ("R2", "FinalSync"), ("FinalSync", "ResearchManager"),
                  ("ResearchManager", "Valuation"), ("ResearchManager", "APAC"), ("ResearchManager", "Consultant"),
                  ("ResearchManager", "Skip"), ("APAC", "Consultant"), ("APAC", "Skip"), ("APAC", "PostSync"),
                  ("Valuation", "PostSync"), ("Consultant", "PostSync"), ("Skip", "PostSync"), ("PostSync", "Trader"),
                  ("Trader", "Risk"), ("Risk", "PortfolioManager"), ("FastFail", "Charts"),
                  ("PortfolioManager", "Charts")],
        "real_extra": r"tools?$|tool_node|article|writer|retry",
    },
    "tracord_release": {
        "url": "https://github.com/zapsyq/tracord_release", "sha": "300fda80",
        "framework": "langchain create_agent supervisor with sub-agents as tools (langgraph inside)",
        "source": "README mermaid 'flowchart LR': supervisor intent routing -> travel / operation / chat sub-agents; "
                  "travel -> slot filling -> DFS planner, or guide Q&A over RAG; operation -> bookkeeping / city / "
                  "bills; results -> harness check -> back to the supervisor on failure",
        "nodes": {"Supervisor": r"supervisor|tracord_agent", "Travel": r"travel", "Operation": r"operation",
                  "Chat": r"^chat(_agent)?$", "SlotFill": r"slot", "DFS": r"planner_engine|dfs",
                  "RAGQA": r"^rag|rag_service|guide", "Bookkeeping": r"bill|city", "Harness": r"harness"},
        "multi": {"Bookkeeping"},  # the bill workflow's nodes and the operation tools
        "edges": [("Supervisor", "Travel"), ("Supervisor", "Operation"), ("Supervisor", "Chat"),
                  ("Travel", "SlotFill"), ("SlotFill", "DFS"), ("Travel", "RAGQA"), ("Operation", "Bookkeeping"),
                  ("DFS", "Harness"), ("RAGQA", "Harness"), ("Bookkeeping", "Harness"), ("Harness", "Supervisor")],
        "readme_only": {"DFS", "Harness"},  # plain search code, and a check inside the supervisor's middleware
        "real_extra": r"model|factory|embedding|vector",
    },
}
