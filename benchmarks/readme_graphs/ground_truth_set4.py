"""Fourth held-out set: ten more agent projects whose READMEs draw their agent graph.

Written from each README's diagram before any project was scanned (same format as GT / GT2 / GT3). `real_extra` names
components that exist in today's code but are not drawn, decided from the code, never from scan output.
"""

GT4 = {
    "DeepLiterature": {
        "url": "https://github.com/ScienceOne-AI/DeepLiterature", "sha": "904c4cb",
        "framework": "custom",
        "source": "README mermaid flowchart; roles mapped to code by the README's own role descriptions",
        "nodes": {"Planner": r"^(re)?orchestrator(_?agent)?$|planner", "Optimizer": r"optimizer",
                  "Searcher": r"search_?executor|^searcher$", "Discriminator": r"sufficiency_?validator|selector|discriminat",
                  "URLParser": r"url_?pars|crawl", "CodeGenerator": r"^code(_?agent)?$|code_?gen",
                  "CodeRunner": r"code_?executor|code_?runner", "Writer": r"assis?t?ant(_?agent)?$|writer",
                  "Publisher": r"publish"},
        "multi": {"Planner", "Discriminator"},
        "readme_only": {"URLParser", "Publisher"},  # no such component in today's code
        "edges": [("Planner", "Optimizer"), ("Optimizer", "Planner"), ("Planner", "Searcher"),
                  ("Searcher", "Discriminator"), ("Discriminator", "URLParser"), ("URLParser", "Planner"),
                  ("Planner", "CodeGenerator"), ("CodeGenerator", "CodeRunner"), ("CodeRunner", "Planner"),
                  ("Discriminator", "Planner"), ("CodeRunner", "Writer"), ("Planner", "Writer"),
                  ("Searcher", "Writer"), ("URLParser", "Writer"), ("Writer", "Publisher")],
        "real_extra": r"$^",
    },
    "adk-code-review-assistant": {
        "url": "https://github.com/ayoisio/adk-code-review-assistant", "sha": "515541d",
        "framework": "google-adk",
        "source": "README mermaid architecture graph",
        "nodes": {"Router": r"^code_?review_?assistant$|router", "Analyzer": r"code_?analyzer",
                  "StyleChecker": r"style_?checker", "TestRunner": r"^test_?runner$",
                  "Synthesizer": r"feedback_?synthesizer", "Fixer": r"code_?fixer", "FixTester": r"fix_?test_?runner",
                  "Validator": r"fix_?validator", "FixSynth": r"fix_?synthesizer"},
        "edges": [("Router", "Analyzer"), ("Analyzer", "StyleChecker"), ("StyleChecker", "TestRunner"),
                  ("TestRunner", "Synthesizer"), ("Fixer", "FixTester"), ("FixTester", "Validator"),
                  ("Validator", "Fixer"), ("Validator", "FixSynth")],
        # the Sequential / Loop agents the drawing shows as boxes around the steps
        "real_extra": r"code_?review_?pipeline|fix_?attempt_?loop|code_?fix_?pipeline",
    },
    "Multi-Agent-Research-Assistant-Langgraph": {
        "url": "https://github.com/Phoenix1454/Multi-Agent-Research-Assistant-Langgraph", "sha": "507e4f3",
        "framework": "langgraph",
        "source": "README mermaid flowchart",
        "nodes": {"Supervisor": r"^supervisor", "Researcher": r"^research(er)?(_node)?$", "Writer": r"^writ(er|e)(_node)?$",
                  "Critiquer": r"^critiq?u(er|e)(_node)?$"},
        "edges": [("Supervisor", "Researcher"), ("Supervisor", "Writer"), ("Researcher", "Supervisor"),
                  ("Writer", "Critiquer"), ("Critiquer", "Supervisor")],
        "real_extra": r"$^",
    },
    "Multi-Agent-Outreach-System": {
        "url": "https://github.com/EmirhanHasirci11/Multi-Agent-Outreach-System", "sha": "9e9f530",
        "framework": "openai-agents",
        "source": "README mermaid flowchart",
        "nodes": {"SalesManager": r"sales_?manager", "Professional": r"professional|sales_?agent_?1",
                  "Engaging": r"engaging|sales_?agent_?2", "Busy": r"busy|sales_?agent_?3",
                  "EmailManager": r"email(er)?_?(manager|agent)", "SubjectWriter": r"subject_?writer",
                  "HtmlConverter": r"html_?converter"},
        "edges": [("SalesManager", "Professional"), ("SalesManager", "Engaging"), ("SalesManager", "Busy"),
                  ("Professional", "SalesManager"), ("Engaging", "SalesManager"), ("Busy", "SalesManager"),
                  ("SalesManager", "EmailManager"), ("SubjectWriter", "HtmlConverter")],
        "real_extra": r"$^",
    },
    "Digital-Research-Assistant": {
        "url": "https://github.com/FaiazS/Digital-Research-Assistant", "sha": "45e788d",
        "framework": "openai-agents",
        "source": "README mermaid flowchart (the two search boxes are the search agent)",
        "nodes": {"Planner": r"planner", "Search": r"web_?searcher|search_?agent", "Writer": r"writer",
                  "Email": r"email", "ResearchManager": r"research_?manager"},
        "edges": [("Planner", "Search"), ("Search", "Writer"), ("Writer", "Email")],
        "real_extra": r"$^",
    },
    "FinSight": {
        "url": "https://github.com/kkkano/FinSight", "sha": "4a1c055",
        "framework": "langgraph",
        "source": "README mermaid architecture graph, LangGraph pipeline part",
        "nodes": {"Understand": r"^understand_request$", "Frame": r"intent_?contract|^frame$",
                  "Policy": r"^policy_gate$", "Planner": r"^planner(_stub)?$", "Execute": r"^execute_plan(_stub)?$",
                  "Synthesize": r"^synthesize$", "Render": r"^(chat_)?render(er|_stub)?$"},
        "readme_only": {"Frame"},  # folded into understand_request in today's code
        "edges": [("Understand", "Frame"), ("Frame", "Policy"), ("Policy", "Planner"), ("Planner", "Execute"),
                  ("Execute", "Synthesize"), ("Synthesize", "Render")],
        # the graph's other nodes, and the research agents / insight scorers the drawing shows as one box
        "real_extra": r"build_initial_state|reset_turn_state|prepare_context|trim_history|summarize_history|"
                      r"normalize_ui_context|decide_output_mode|chat_respond|resolve_subject|clarify|parse_operation|"
                      r"alert_extractor|alert_action|confirmation_gate|research_debate|(technical|fundamental|price|"
                      r"macro|risk|deep_?search|news|conversation)_?agent|scorer",
    },
    "StartupPilot-AI": {
        "url": "https://github.com/Sanvith6/StartupPilot-AI", "sha": "51d04d2",
        "framework": "langgraph",
        "source": "README mermaid 10-node StateGraph",
        "nodes": {"Research": r"^research(_agent|_node)?$|senior_research_analyst",
                  "MarketAnalysis": r"^market_analysis(_agent)?$|senior_market_analyst",
                  "CompetitorAnalysis": r"competitor_analysis|competitive_intelligence",
                  "SWOT": r"swot", "BusinessStrategy": r"business_strategy|senior_business_consultant",
                  "HumanApproval": r"human_approval", "AutoGenDebate": r"autogen_(discussion|debate)",
                  "ArchitectureCost": r"architecture_cost|architecture_design|cost_estimation|cloud_architect|"
                                      r"cloud_financial", "Report": r"report_generation|report_writing|technical_writer",
                  "Memory": r"memory_storage|long_term_memory"},
        "multi": {"ArchitectureCost"},
        "edges": [("Research", "MarketAnalysis"), ("MarketAnalysis", "CompetitorAnalysis"),
                  ("CompetitorAnalysis", "SWOT"), ("SWOT", "BusinessStrategy"), ("BusinessStrategy", "HumanApproval"),
                  ("HumanApproval", "AutoGenDebate"), ("HumanApproval", "BusinessStrategy"),
                  ("AutoGenDebate", "ArchitectureCost"), ("ArchitectureCost", "Report"), ("Report", "Memory")],
        # the debate's own agents inside the AutoGen node
        "real_extra": r"business_consultant|financial_analyst|discussion_moderator",
    },
    "FinanceAgents": {
        "url": "https://github.com/patrickjchen/FinanceAgents", "sha": "3c58a91",
        "framework": "custom (four framework ports)",
        "source": "README mermaid architecture flowchart",
        "nodes": {"Router": r"router", "QueryClassification": r"query_classif|classify",
                  "Reddit": r"reddit", "Finance": r"^finance(_?agent)?$", "Yahoo": r"yahoo", "SEC": r"^sec(_?agent)?$",
                  "General": r"general", "RAG": r"^rag(_?agent)?$",
                  "LLMHelpers": r"improve_agent_response|comprehensive_summary|llm_helpers"},
        "multi": {"Router"},  # one router per framework port
        "edges": [("Router", "QueryClassification"), ("Router", "Reddit"), ("Router", "Finance"), ("Router", "Yahoo"),
                  ("Router", "SEC"), ("Router", "General"), ("Finance", "RAG")],
        "real_extra": r"$^",
    },
    "ai_fantasy_rpg": {
        "url": "https://github.com/alipala/ai_fantasy_rpg", "sha": "71ccf0b",
        "framework": "crewai",
        "source": "README mermaid graph (agent boxes; the rest are game data)",
        "nodes": {"GameMaster": r"game_?master", "WorldBuilder": r"world_?builder",
                  "InventoryManager": r"inventory", "SafetyChecker": r"safety"},
        "edges": [("GameMaster", "WorldBuilder"), ("GameMaster", "InventoryManager"), ("GameMaster", "SafetyChecker")],
        "real_extra": r"$^",
    },
    "react-agent": {
        "url": "https://github.com/langchain-ai/react-agent", "sha": "087d90d",
        "framework": "langgraph",
        "source": "README 'Graph view in LangGraph studio UI' screenshot (static/studio_ui.png)",
        "nodes": {"CallModel": r"^call_model$", "Tools": r"^tools$"},
        "edges": [("CallModel", "Tools"), ("Tools", "CallModel")],
        "real_extra": r"$^",
    },
}
