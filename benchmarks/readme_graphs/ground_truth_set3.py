"""Third held-out set: ten more agent projects whose READMEs draw their agent graph.

Written from each README's diagram before any project was scanned. Same format as `score.GT` / `GT2`:
nodes are README components (regexes over the map's agent ids, names and aliases), edges are README arrows, `path`
is the folder scanned, `multi` lets several map blocks stand for one README box, and `real_extra` names components
that exist in today's code but are not drawn (decided from the code, not from scan output).
"""

GT3 = {
    "JARVIS": {
        "url": "https://github.com/microsoft/JARVIS", "sha": "7624cf3", "path": "hugginggpt/server",
        "framework": "custom (LLM over raw HTTP)",
        "source": "README overview figure (hugginggpt/assets/overview.jpg): four stages",
        "nodes": {"TaskPlanning": r"^(parse_task|task_planning)$", "ModelSelection": r"^(choose_model|model_selection)$",
                  "TaskExecution": r"^(run_task|task_execution)$",
                  "ResponseGeneration": r"^(response_results|response_generation)$"},
        "edges": [("TaskPlanning", "ModelSelection"), ("ModelSelection", "TaskExecution"),
                  ("TaskExecution", "ResponseGeneration")],
        "real_extra": r"^chitchat$",  # the small-talk reply when no task is planned
    },
    "babyagi_archive": {
        "url": "https://github.com/yoheinakajima/babyagi_archive", "sha": "e8726e3",
        "framework": "custom (OpenAI SDK)",
        "source": "README 'How It Works' diagram: four agents around three loop steps (agent order taken from the steps)",
        "nodes": {"Execution": r"^execution_agent$", "Context": r"^context_agent$",
                  "TaskCreation": r"^task_creation_agent$", "Prioritization": r"^prioritization_agent$"},
        "edges": [("Execution", "Context"), ("Context", "TaskCreation"), ("TaskCreation", "Prioritization"),
                  ("Prioritization", "Execution")],
        "real_extra": r"$^",  # classic/, extensions/ and babycoder/ are other systems, not this one
    },
    "company-researcher": {
        "url": "https://github.com/langchain-ai/company-researcher", "sha": "d62bcbf",
        "framework": "langgraph",
        "source": "README overview image: Planning -> Web Research -> notes -> Output Schema",
        "nodes": {"Planning": r"^generate_queries$", "WebResearch": r"^research_company$",
                  "Extraction": r"^gather_notes_extract_schema$"},
        "edges": [("Planning", "WebResearch"), ("WebResearch", "Extraction")],
        "real_extra": r"^reflection$",
    },
    "rag-research-agent-template": {
        "url": "https://github.com/langchain-ai/rag-research-agent-template", "sha": "39f3ce6",
        "framework": "langgraph",
        "source": "README 'Graph view in LangGraph studio UI' screenshot (static/studio_ui.png)",
        "nodes": {"AnalyzeRoute": r"^analyze_and_route_query$", "CreatePlan": r"^create_research_plan$",
                  "ConductResearch": r"^conduct_research$", "GenerateQueries": r"^generate_queries$",
                  "RetrieveDocuments": r"^retrieve_documents$", "AskMoreInfo": r"^ask_for_more_info$",
                  "Respond": r"^respond$", "RespondGeneral": r"^respond_to_general_query$"},
        "edges": [("AnalyzeRoute", "CreatePlan"), ("AnalyzeRoute", "AskMoreInfo"), ("AnalyzeRoute", "RespondGeneral"),
                  ("CreatePlan", "ConductResearch"), ("GenerateQueries", "RetrieveDocuments"),
                  ("ConductResearch", "Respond")],
        "real_extra": r"^index_docs$",  # the index graph the README describes in text but does not draw
    },
    "deep-searcher": {
        "url": "https://github.com/zilliztech/deep-searcher", "sha": "ddc36db",
        "framework": "custom",
        "source": "README architecture image (assets/pic/deep-searcher-arch.png), online-serving part",
        "nodes": {"SubQuery": r"sub_quer", "CollectionRouter": r"collection_router",
                  "SemanticSearch": r"search_chunks|semantic_search", "Reflection": r"gap_quer|reflect",
                  "Summarize": r"summar|final_report"},
        "edges": [("SubQuery", "CollectionRouter"), ("CollectionRouter", "SemanticSearch"),
                  ("SemanticSearch", "Reflection"), ("Reflection", "SubQuery"), ("Reflection", "Summarize")],
        "real_extra": r"chain_?of_?rag|naive_?rag|rag_?router",  # alternative agents the drawing leaves out
    },
    "SciAgentsDiscovery": {
        "url": "https://github.com/lamm-mit/SciAgentsDiscovery", "sha": "c5c3045",
        "framework": "autogen",
        "source": "README figure 1 (b: the sequential chain, c: the automated group)",
        "nodes": {"Ontologist": r"^ontologist$", "Scientist1": r"^scientist(_1)?$",
                  "Scientist2": r"^(scientist_2|(hypothesis|outcome|mechanism|design_principles|unexpected_properties"
                                r"|comparison|novelty)(_agent)?)$",
                  "Critic": r"^critic(_agent)?$", "Planner": r"^planner$", "Assistant": r"^assistant$",
                  "Human": r"^(user|human|user_proxy)$"},
        "multi": {"Scientist2"},  # the code splits Scientist 2's seven aspects into seven agents
        "edges": [("Ontologist", "Scientist1"), ("Scientist1", "Scientist2"), ("Scientist2", "Critic")],
        "real_extra": r"^novelty_(assistant|admin)$",
    },
    "LieGraph": {
        "url": "https://github.com/leslieo2/LieGraph", "sha": "94abe36",
        "framework": "langgraph",
        "source": "README mermaid graph",
        "nodes": {"HostSetup": r"^host_setup$", "StageSwitch": r"^host_stage_switch$",
                  "PlayerSpeech": r"^player_speech", "PlayerVote": r"^player_vote",
                  "CheckVotes": r"^check_votes_and_transition$", "HostResult": r"^host_result$"},
        "multi": {"PlayerSpeech", "PlayerVote"},  # one node per player
        "edges": [("HostSetup", "StageSwitch"), ("StageSwitch", "PlayerSpeech"), ("PlayerSpeech", "StageSwitch"),
                  ("StageSwitch", "PlayerVote"), ("PlayerVote", "CheckVotes"), ("CheckVotes", "HostResult"),
                  ("HostResult", "StageSwitch")],
        "real_extra": r"$^",
    },
    "cognivault": {
        "url": "https://github.com/aucontraire/cognivault", "sha": "8c4fa56",
        "framework": "langgraph",
        "source": "README mermaid graph",
        "nodes": {"Refiner": r"refiner", "Critic": r"critic", "Historian": r"historian", "Synthesis": r"synthesis"},
        "edges": [("Refiner", "Critic"), ("Refiner", "Historian"), ("Critic", "Synthesis"),
                  ("Historian", "Synthesis")],
        "real_extra": r"topic",  # store/topic_manager.py asks the model to name topics
    },
    "medical-multi-agent-system": {
        "url": "https://github.com/bcefghj/medical-multi-agent-system", "sha": "09936ed", "path": "python",
        "framework": "langgraph",
        "source": "README mermaid 'Clinical Decision Pipeline'",
        "nodes": {"Intake": r"intake", "Diagnosis": r"diagnos", "Treatment": r"treatment", "Coding": r"coding",
                  "Audit": r"audit"},
        "edges": [("Intake", "Diagnosis"), ("Diagnosis", "Treatment"), ("Diagnosis", "Intake"),
                  ("Treatment", "Coding"), ("Coding", "Audit")],
        "real_extra": r"$^",
    },
    "stratoseer": {
        "url": "https://github.com/MStanislavov/stratoseer", "sha": "6869ddf",
        "framework": "langgraph",
        "source": "README mermaid graphs: daily, weekly and cover-letter pipelines (union)",
        "nodes": {"GoalExtractor": r"^goal_extractor$", "WebScrapers": r"^web_scrapers?$",
                  "URLFilter": r"^url_filter_report$", "DataFormatter": r"^data_formatter$",
                  "SafeDegrade": r"^safe_degrade$", "CEO": r"^ceo(_agent)?$", "CFO": r"^cfo(_agent)?$",
                  "AuditWriter": r"^audit_writer$", "CoverLetter": r"^cover_letter(_agent)?$"},
        "edges": [("GoalExtractor", "WebScrapers"), ("WebScrapers", "URLFilter"), ("URLFilter", "DataFormatter"),
                  ("URLFilter", "SafeDegrade"), ("DataFormatter", "AuditWriter"), ("SafeDegrade", "AuditWriter"),
                  ("DataFormatter", "CEO"), ("SafeDegrade", "CEO"), ("CEO", "CFO"), ("CFO", "AuditWriter"),
                  ("CoverLetter", "AuditWriter")],
        "real_extra": r"^job_expiry_check$|profile",  # a code-only check node; profile extraction
    },
}
