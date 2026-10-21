"""Second held-out set: ten more agent projects whose READMEs draw or spell out their agent graph.

Same format as `score.GT`. `path` is the folder that was scanned when the project keeps the system in a subfolder.
`readme_only` names README components the code no longer has (the drawing is older than the code); they count as
missed for `node_recall` and are left out of `node_recall_code`.
"""

_SALES = ["get_new_leads", "check_for_remaining_leads", "fetch_linkedin_profile_data", "review_company_website",
          "collect_company_information", "analyze_blog_content", "analyze_recent_news",
          "analyze_social_media_content", "generate_digital_presence_report", "generate_full_lead_research_report",
          "score_lead", "generate_custom_outreach_report", "create_outreach_materials", "generate_interview_script",
          "generate_personalized_email", "await_reports_creation", "save_reports_to_google_docs", "update_crm"]

GT2 = {
    "data-enrichment": {
        "url": "https://github.com/langchain-ai/data-enrichment", "sha": "71a71c0",
        "framework": "langgraph",
        "source": "README 'Graph view in LangGraph studio UI' screenshot (static/studio.png)",
        # the screenshot's call_checker is today's `reflect` (routed by route_after_checker)
        "nodes": {"CallModel": r"^call(_agent)?_model$", "Tools": r"^tools$", "Checker": r"^(call_checker|reflect)$",
                  "Correction": r"correction"},
        "readme_only": {"Correction"},
        "edges": [("CallModel", "Tools"), ("Tools", "CallModel"), ("CallModel", "Checker"), ("Checker", "CallModel"),
                  ("CallModel", "Correction"), ("Correction", "CallModel")],
        "real_extra": r"$^",
    },
    "langgraph-reflection": {
        "url": "https://github.com/langchain-ai/langgraph-reflection", "sha": "888d007",
        "framework": "langgraph",
        "source": "README graph image (langgraph-reflection.png)",
        "nodes": {"Graph": r"^graph$", "Reflection": r"^reflection$"},
        "edges": [("Graph", "Reflection"), ("Reflection", "Graph")],
        # the examples' own main agent / judge graphs that plug into graph / reflection
        "real_extra": r"call_model|judge_response|try_running",
    },
    "sales-outreach-automation-langgraph": {
        "url": "https://github.com/kaymen99/sales-outreach-automation-langgraph", "sha": "2e2761c",
        "framework": "langgraph",
        "source": "README workflow image (workflow.png)",
        "nodes": {n: rf"^{n.lower()}$" for n in _SALES},
        "edges": [("get_new_leads", "check_for_remaining_leads"),
                  ("check_for_remaining_leads", "fetch_linkedin_profile_data"),
                  ("fetch_linkedin_profile_data", "review_company_website"),
                  ("review_company_website", "collect_company_information"),
                  *[("collect_company_information", x) for x in
                    ("analyze_blog_content", "analyze_recent_news", "analyze_social_media_content")],
                  *[(x, "generate_digital_presence_report") for x in
                    ("analyze_blog_content", "analyze_recent_news", "analyze_social_media_content")],
                  ("generate_digital_presence_report", "generate_full_lead_research_report"),
                  ("generate_full_lead_research_report", "score_lead"),
                  ("score_lead", "generate_custom_outreach_report"), ("score_lead", "save_reports_to_google_docs"),
                  ("generate_custom_outreach_report", "create_outreach_materials"),
                  ("create_outreach_materials", "generate_interview_script"),
                  ("create_outreach_materials", "generate_personalized_email"),
                  ("generate_interview_script", "await_reports_creation"),
                  ("generate_personalized_email", "await_reports_creation"),
                  ("await_reports_creation", "save_reports_to_google_docs"),
                  ("save_reports_to_google_docs", "update_crm"), ("update_crm", "check_for_remaining_leads")],
        "real_extra": r"$^",
    },
    "Controllable-RAG-Agent": {
        "url": "https://github.com/NirDiamant/Controllable-RAG-Agent", "sha": "929b25f",
        "framework": "langgraph",
        "source": "README 'Solution Schema' image (graphs/final_graph_schema.jpeg)",
        "nodes": {"Anonymize": r"^anonymize_question$", "Planner": r"^planner$", "DeAnonymize": r"^de_anonymize_plan$",
                  "BreakDown": r"^break_down_plan(_to_retrieve_or_answer)?$", "TaskHandler": r"^task_handler$",
                  "RetrieveChunks": r"^retrieve(_book)?_chunks$", "RetrieveQuotes": r"^retrieve_book_quotes$",
                  "RetrieveSummaries": r"^retrieve_summaries$", "KeepRelevant": r"^keep_only_relevant_content$",
                  "Answer": r"^answer$", "Replan": r"^replan$", "FinalAnswer": r"^get_final_answer$"},
        "edges": [("Anonymize", "Planner"), ("Planner", "DeAnonymize"), ("DeAnonymize", "BreakDown"),
                  ("BreakDown", "TaskHandler"),
                  *[("TaskHandler", x) for x in ("RetrieveChunks", "RetrieveQuotes", "RetrieveSummaries", "Answer")],
                  *[(x, "KeepRelevant") for x in ("RetrieveChunks", "RetrieveQuotes", "RetrieveSummaries")],
                  ("KeepRelevant", "Replan"), ("Answer", "Replan"), ("Replan", "BreakDown"), ("Replan", "FinalAnswer")],
        # the per-store retrieval / answer sub-workflows the README shows as separate subgraph images
        "real_extra": r"context_per_question|rewrite_question|answer_question_from_context",
    },
    "retrieval-agent-template": {
        "url": "https://github.com/langchain-ai/retrieval-agent-template", "sha": "6a441ef",
        "framework": "langgraph",
        "source": "README 'Graph view in LangGraph studio UI' screenshot (static/studio_ui.png)",
        "nodes": {"GenerateQuery": r"^generate_query$", "Retrieve": r"^retrieve$", "Respond": r"^respond$"},
        "edges": [("GenerateQuery", "Retrieve"), ("Retrieve", "Respond")],
        "real_extra": r"index_docs",  # the separate indexing graph the README describes in text
    },
    "financial_research_agent": {
        "url": "https://github.com/openai/openai-agents-python", "sha": "64cd5df",
        "path": "examples/financial_research_agent",
        "framework": "openai-agents",
        "source": "examples/financial_research_agent/README 'The flow is' (planning, search, sub-analysts, writing, verification)",
        "nodes": {"Planner": r"planner", "Search": r"search", "Fundamentals": r"fundamental|financials_agent",
                  "Risk": r"risk", "Writer": r"writer", "Verifier": r"verif"},
        "edges": [("Planner", "Search"), ("Search", "Writer"), ("Writer", "Fundamentals"), ("Writer", "Risk"),
                  ("Writer", "Verifier")],
        "real_extra": r"$^",
    },
    "Agent-E": {
        "url": "https://github.com/EmergenceAI/Agent-E", "sha": "f218c3c",
        "framework": "autogen",
        "source": "README system-architecture image (docs/images/agent-e-system-architecture.png) + planner_agent / browser_nav_agent config",
        "nodes": {"Human": r"^user", "Planner": r"planner", "BrowserNav": r"browser_nav(igation)?_agent"},
        "edges": [("Human", "Planner"), ("Planner", "Human"), ("Planner", "BrowserNav"), ("BrowserNav", "Planner")],
        "real_extra": r"executor",
    },
    "translation-agent": {
        "url": "https://github.com/andrewyng/translation-agent", "sha": "e0fc605",
        "framework": "custom (raw OpenAI SDK)",
        "source": "README 'main steps' (translate -> reflect -> improve)",
        "nodes": {"Translate": r"initial_translation", "Reflect": r"reflect", "Improve": r"improve"},
        "edges": [("Translate", "Reflect"), ("Reflect", "Improve")],
        "real_extra": r"$^",
    },
    "AlphaCodium": {
        "url": "https://github.com/Codium-ai/AlphaCodium", "sha": "eb7577d",
        "framework": "custom (LiteLLM)",
        "source": "README proposed-flow image (pics/proposed_flow.png)",
        "nodes": {"Reflection": r"(?<!validate_)self_reflect", "PublicTestsReasoning": r"public_tests_reasoning|reflect_on_tests",
                  "PossibleSolutions": r"possible_solutions", "RankSolutions": r"choose_best|rank",
                  "AITests": r"generate_ai_test", "InitialCode": r"initial_code|initial_solve", "IteratePublic": r"evaluate_public",
                  "IterateAI": r"evaluate_all_ai|evaluate_ai"},
        "readme_only": {"PublicTestsReasoning"},  # done inside the self-reflection stage in today's code
        "edges": [("Reflection", "PublicTestsReasoning"), ("PublicTestsReasoning", "PossibleSolutions"),
                  ("PossibleSolutions", "RankSolutions"), ("RankSolutions", "AITests"), ("AITests", "InitialCode"),
                  ("InitialCode", "IteratePublic"), ("IteratePublic", "IterateAI")],
        "edge_alias": {("Reflection", "PossibleSolutions"): ("PublicTestsReasoning", "PossibleSolutions")},
        "real_extra": r"baseline|fix|validate|analy[sz]e|run_tests",
    },
    "WikiChat": {
        "url": "https://github.com/stanford-oval/WikiChat", "sha": "803683b",
        "framework": "langgraph",
        "source": "README pipeline image (public/img/pipeline.svg): query, retrieve, summarize & filter, generate, extract claims, fact-check, draft, refine",
        "nodes": {"Query": r"^query", "Retrieve": r"^search_stage$", "SummarizeFilter": r"filter_information|summari",
                  "Generate": r"^generate", "ExtractClaims": r"claim", "FactCheck": r"fact_check",
                  "Draft": r"^draft", "Refine": r"^refine"},
        "readme_only": {"FactCheck"},  # today's llm_claim_search_stage + filter stage do extraction and checking
        "edges": [("Query", "Retrieve"), ("Retrieve", "SummarizeFilter"), ("SummarizeFilter", "Draft"),
                  ("Generate", "ExtractClaims"), ("ExtractClaims", "FactCheck"), ("FactCheck", "Draft"),
                  ("Draft", "Refine")],
        "real_extra": r"shift_references",
    },
}
