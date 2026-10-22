"""Sixth held-out set: ten more agent projects whose READMEs draw (or spell out) their agent graph.

Written from each README's diagram before any project was scanned (same format as GT .. GT5). `real_extra` names
components that exist in today's code but are not drawn, decided from the code, never from scan output.
None of these projects appears in sets 1-5.
"""

GT6 = {
    "TradingAgents": {
        "url": "https://github.com/TauricResearch/TradingAgents", "sha": "e8f5503",
        "framework": "langgraph",
        "source": "README assets/schema.png (analyst team -> bull/bear discussion -> trader -> risk team -> manager)",
        "nodes": {"Market": r"^market", "Social": r"^social|sentiment", "News": r"^news", "Fundamentals": r"^fundamental",
                  "Bull": r"^bull", "Bear": r"^bear", "Trader": r"^trader",
                  "Aggressive": r"^aggressive|^risky", "Neutral": r"^neutral", "Conservative": r"^conservative|^safe",
                  "Manager": r"^portfolio_manager|^risk_judge|^risk_manager"},
        "edges": [*[(a, "Bull") for a in ("Market", "Social", "News", "Fundamentals")],
                  *[(a, "Bear") for a in ("Market", "Social", "News", "Fundamentals")],
                  ("Bull", "Bear"), ("Bear", "Bull"), ("Bull", "Trader"), ("Bear", "Trader"),
                  ("Trader", "Aggressive"), ("Trader", "Neutral"), ("Trader", "Conservative"),
                  ("Aggressive", "Manager"), ("Neutral", "Manager"), ("Conservative", "Manager")],
        # the research manager judges the bull/bear debate in code; memory/reflection and tool nodes are plumbing
        "real_extra": r"research_manager|memory|reflect|tools|wrap_up|msg_clear|signal|rating|screen",
    },
    "gpt-researcher-multi-agents": {
        "url": "https://github.com/assafelovic/gpt-researcher", "sha": "3582a68", "path": "multi_agents",
        "framework": "langgraph",
        "source": "multi_agents/README.md 'Steps' (spelled out from its architecture diagram)",
        "nodes": {"ChiefEditor": r"chief|orchestrator", "Browser": r"^browser|initial_research",
                  "Editor": r"^editor|^planner|plan_research", "Researcher": r"^research(er)?$|depth_research",
                  "Reviewer": r"^reviewer", "Revisor": r"^revis", "Writer": r"^writer", "Publisher": r"^publisher",
                  "Human": r"^human"},
        "edges": [("Browser", "Editor"), ("Editor", "Researcher"), ("Researcher", "Reviewer"),
                  ("Reviewer", "Revisor"), ("Revisor", "Reviewer"), ("Researcher", "Writer"),
                  ("Writer", "Publisher")],
        "real_extra": r"fact_?check|visuali[sz]er|plan_review|draft_review|fact_review|ag2",
    },
    "Voyager": {
        "url": "https://github.com/MineDojo/Voyager", "sha": "55e45a8",
        "framework": "langchain",
        "source": "README images/pull.png (curriculum -> iterative prompting <- skill library; self-verification)",
        "nodes": {"Curriculum": r"curriculum", "Action": r"^action", "Critic": r"critic|verif",
                  "Skill": r"^skill"},
        "edges": [("Curriculum", "Action"), ("Skill", "Action"), ("Action", "Critic"), ("Critic", "Action"),
                  ("Critic", "Skill"), ("Critic", "Curriculum")],
        "real_extra": r"voyager|qa|question",
    },
    "AgentLaboratory": {
        "url": "https://github.com/SamuelSchmidgall/AgentLaboratory", "sha": "d9017d9",
        "framework": "none",
        "source": "README media/AgentLabWF.png (phases, instructor/assistant roles, mle-solver, paper-solver)",
        "nodes": {"PhDStudent": r"phd", "Postdoc": r"postdoc", "SWEngineer": r"sw_?engineer|software",
                  "MLEngineer": r"ml_?engineer", "MLESolver": r"mle_?solver", "PaperSolver": r"paper_?solver",
                  "Professor": r"professor", "Reviewers": r"reviewer"},
        "edges": [("PhDStudent", "Postdoc"), ("Postdoc", "PhDStudent"), ("Postdoc", "SWEngineer"),
                  ("SWEngineer", "MLEngineer"), ("MLEngineer", "SWEngineer"), ("MLEngineer", "MLESolver"),
                  ("MLESolver", "PaperSolver"), ("PaperSolver", "Reviewers"), ("Reviewers", "PhDStudent"),
                  ("Reviewers", "Postdoc")],
        "real_extra": r"laboratory|workflow|agent_?rxiv|hf_?data|arxiv|inference|semantic",
    },
    "storm": {
        "url": "https://github.com/stanford-oval/storm", "sha": "fb951af",
        "framework": "dspy",
        "source": "README two_stages.jpg + the four STORM modules it names",
        "nodes": {"KnowledgeCuration": r"knowledge_?curation|curation|research",
                  "OutlineGeneration": r"outline", "ArticleGeneration": r"article_?gen|write_?section|article$",
                  "ArticlePolishing": r"polish"},
        "edges": [("KnowledgeCuration", "OutlineGeneration"), ("KnowledgeCuration", "ArticleGeneration"),
                  ("OutlineGeneration", "ArticleGeneration"), ("ArticleGeneration", "ArticlePolishing")],
        # Co-STORM (experts, moderator) and the persona / conversation simulators exist in code, not in this drawing
        "real_extra": r"co_?storm|expert|moderator|persona|conv|question|answer|wiki_?writer|topic|mind_?map|"
                      r"simulat|discourse|perspective|grounded|warm_?start|utterance|knowledge_?base|insert|"
                      r"report|lm_?config|runner",
    },
    "multi-agent-concierge": {
        "url": "https://github.com/run-llama/multi-agent-concierge", "sha": "0774165",
        "framework": "llamaindex",
        "source": "README architecture.png + 'What we built' (orchestrator routes to four task agents and back)",
        "nodes": {"Orchestrator": r"orchestrat|concierge", "StockLookup": r"stock", "Authentication": r"auth",
                  "AccountBalance": r"balance", "TransferMoney": r"transfer"},
        "edges": [*[("Orchestrator", x) for x in ("StockLookup", "Authentication", "AccountBalance", "TransferMoney")],
                  *[(x, "Orchestrator") for x in ("StockLookup", "Authentication", "AccountBalance", "TransferMoney")]],
        # the workflow.png steps of the LlamaIndex workflow itself
        "real_extra": r"setup|speak_with_sub_agent|handle_tool|aggregate|external_step|workflow",
    },
    "adk-small-business-loan": {
        "url": "https://github.com/google/adk-samples", "sha": "be63596",
        "path": "contrib/python/small-business-loan-agent",
        "framework": "google-adk",
        "source": "README mermaid sequenceDiagram",
        "nodes": {"Orchestrator": r"orchestrat|small_?business_?loan|root", "DocumentExtraction": r"document_?extract",
                  "Underwriting": r"underwrit", "Pricing": r"pricing", "LoanDecision": r"loan_?decision",
                  "Judge": r"judge"},
        "edges": [("Orchestrator", "DocumentExtraction"), ("Orchestrator", "Underwriting"),
                  ("Orchestrator", "Pricing"), ("Orchestrator", "LoanDecision"), ("Orchestrator", "Judge"),
                  ("DocumentExtraction", "Orchestrator"), ("Underwriting", "Orchestrator"),
                  ("Pricing", "Orchestrator"), ("LoanDecision", "Orchestrator"), ("Judge", "Orchestrator")],
        "real_extra": r"$^",
    },
    "TaskWeaver": {
        "url": "https://github.com/microsoft/TaskWeaver", "sha": "d44ddef",
        "framework": "none",
        "source": "README .asset/taskweaver_arch.png (planner <-> code interpreter)",
        "nodes": {"Planner": r"^planner", "CodeInterpreter": r"code_?interpreter|code_?generator"},
        "edges": [("Planner", "CodeInterpreter"), ("CodeInterpreter", "Planner")],
        # optional roles shipped as extensions; the README drawing shows the two core roles
        "real_extra": r"web_?search|web_?explorer|image_?reader|document_?retriever|recepta|echo|plugin_?select|"
                      r"compress|verif",
    },
    "enterprise-deep-research": {
        "url": "https://github.com/SalesforceAIResearch/enterprise-deep-research", "sha": "59f8f2a",
        "framework": "langgraph",
        "source": "README assets/edr_ppl.png",
        "nodes": {"MasterResearchAgent": r"master|multi_agents_network", "TodoManager": r"todo",
                  "SpecializedAgents": r"search_agent|specialized|^search$", "ResearchReport": r"generate_report|^report",
                  "Reflection": r"reflect"},
        "edges": [("MasterResearchAgent", "TodoManager"), ("TodoManager", "SpecializedAgents"),
                  ("SpecializedAgents", "ResearchReport"), ("ResearchReport", "Reflection"),
                  ("Reflection", "MasterResearchAgent")],
        "real_extra": r"visuali[sz]|answer|refine_query|finalize|coding|math|nl2sql|file_?analy|steer|router",
    },
    "XAgent": {
        "url": "https://github.com/OpenBMB/XAgent", "sha": "0dea799",
        "framework": "none",
        "source": "README assets/readme/workflow.png (outer plan loop, inner dispatch/tool/reflection loop)",
        "nodes": {"PlanAgent": r"plan", "Dispatcher": r"dispatch", "ToolAgent": r"tool_?agent",
                  "Reflection": r"reflect"},
        "edges": [("PlanAgent", "Dispatcher"), ("Dispatcher", "ToolAgent"), ("ToolAgent", "Reflection"),
                  ("Reflection", "PlanAgent")],
        "real_extra": r"summar|xagent|ask_?human|tool_?server|recorder",
    },
}
