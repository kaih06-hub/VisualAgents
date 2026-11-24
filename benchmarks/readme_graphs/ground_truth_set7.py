"""Seventh held-out set: ten more agent projects whose READMEs draw (or spell out) their agent graph.

Written from each README's diagram before any project was scanned (same format as GT .. GT6). `real_extra` names
components that exist in today's code but are not drawn, and `readme_only` names drawn components the code no longer
has (or never had as agents) - both decided from the code, never from scan output.
None of these projects appears in sets 1-6.
"""

GT7 = {
    "deer-flow": {
        "url": "https://github.com/bytedance/deer-flow", "sha": "2ab2876",  # branch main-1.x (2.x dropped the diagram)
        "framework": "langgraph",
        "source": "README 'Architecture' assets/architecture.png (LangGraph drawing) + the component list under it",
        "nodes": {"Coordinator": r"^coordinator", "Planner": r"^planner", "HumanFeedback": r"human_feedback",
                  "ResearchTeam": r"research_team", "Researcher": r"^researcher", "Coder": r"^coder",
                  "Reporter": r"^reporter"},
        "edges": [("Coordinator", "Planner"), ("Planner", "HumanFeedback"), ("HumanFeedback", "Planner"),
                  ("Planner", "Reporter"), ("HumanFeedback", "Reporter"), ("HumanFeedback", "ResearchTeam"),
                  ("ResearchTeam", "Planner"), ("ResearchTeam", "Researcher"), ("Researcher", "ResearchTeam"),
                  ("ResearchTeam", "Coder"), ("Coder", "ResearchTeam")],
        # background investigation and the analyst step joined the graph after the drawing; podcast / ppt / prose /
        # prompt-enhancer are separate small graphs in src/
        "real_extra": r"background|analyst|podcast|ppt|prose|prompt_?enhancer|script_?writer|tts|audio|"
                      r"slide|outline|continue|improve|shorter|longer|fix|zap|rag|crawl|search",
    },
    "MetaGPT": {
        "url": "https://github.com/FoundationAgents/MetaGPT", "sha": "11cdf46",
        "framework": "none",
        "source": "README 'Software Company as Multi-Agent System' docs/resources/software_company_cd.jpeg + SOP text "
                  "(product manager -> architect -> project manager -> engineer, QA)",
        "nodes": {"Boss": r"^boss", "ProductManager": r"product_?manager", "Architect": r"^architect",
                  "ProjectManager": r"project_?manager", "Engineer": r"^engineer$", "QA": r"^qa"},
        "readme_only": {"Boss"},  # the boss is the user's one-line requirement, not a role in code
        "edges": [("Boss", "ProductManager"), ("ProductManager", "Architect"), ("Architect", "ProjectManager"),
                  ("ProjectManager", "Engineer"), ("Engineer", "QA")],
        "real_extra": r"research|search|teacher|sales|customer|invoice|tutorial|assistant|interpreter|data_?analyst|"
                      r"team_?leader|swe|engineer_?2|role_?zero|debat|werewolf|minecraft|android|stanford|"
                      r"software_?company|team$|ex_?engineer|reflect|aflow|spo|sela|mle|optimizer|di$|"
                      r"writer|poet|skill|planner|critic|student|moderator|villager|seer|witch|guard",
    },
    "ChatDev": {
        "url": "https://github.com/OpenBMB/ChatDev", "sha": "31fd994",  # branch chatdev1.0 (main is ChatDev 2.0)
        "framework": "none",
        "source": "README misc/company.png (designing -> coding -> testing -> documenting) + the roles the README lists; "
                  "who-talks-to-whom per phase from the chat chain it describes",
        "nodes": {"CEO": r"chief_executive|^ceo", "CPO": r"chief_product|^cpo", "CTO": r"chief_technology|^cto",
                  "Programmer": r"programmer", "Reviewer": r"reviewer", "Tester": r"test_?engineer|tester",
                  "Designer": r"designer|chief_creative|^cco"},
        "edges": [("CEO", "CPO"), ("CEO", "CTO"), ("CTO", "Programmer"), ("Programmer", "Reviewer"),
                  ("Reviewer", "Programmer"), ("Programmer", "Tester"), ("Tester", "Programmer")],
        "real_extra": r"counselor|chief_human|phase|chat_?chain|role_?play|art|incremental|human|visuali[sz]|"
                      r"demand|language|coding|code_?review|code_?complete|test|environment|manual|reflect",
    },
    "owl": {
        "url": "https://github.com/camel-ai/owl", "sha": "005e5c4",
        "framework": "camel",
        "source": "README assets/owl_architecture.png (AI user agent <-> assistant agent -> web/search/coding/document "
                  "agents -> tools pool)",
        "nodes": {"User": r"^user|user_agent", "Assistant": r"^assistant|assistant_agent",
                  "Web": r"^web|browser", "Search": r"^search", "Coding": r"^cod(e|ing)", "Document": r"^document"},
        # in this repo the four actor agents are camel toolkits handed to the assistant, not agents
        "readme_only": {"Web", "Search", "Coding", "Document"},
        "edges": [("User", "Assistant"), ("Assistant", "User"), ("Assistant", "Web"), ("Assistant", "Search"),
                  ("Assistant", "Coding"), ("Assistant", "Document")],
        "real_extra": r"gaia|role_?play|owl|task_?specif|planner|critic|webapp|society|run",
    },
    "DeepCode": {
        "url": "https://github.com/HKUDS/DeepCode", "sha": "49805ef",
        "framework": "none",
        "source": "README assets/readme/framework2.png (blueprint generation -> code generation -> verification)",
        "nodes": {"Segmentation": r"segment", "Concept": r"concept", "Algorithm": r"algorithm",
                  "Planning": r"planner|planning", "CodeRAG": r"reference|code_?rag|code_?index|github_?download",
                  "CodeMem": r"memory", "CodeGen": r"code_?implementation|structure_?generator|code_?gen",
                  "Analysis": r"^analysis|static_?analy", "Modification": r"modif",
                  "Refinement": r"refine|sandbox"},
        # one planner replaced the concept/algorithm pair; phase 3's agents are not in today's code
        "readme_only": {"Concept", "Algorithm", "Analysis", "Modification", "Refinement"},
        "edges": [("Segmentation", "Concept"), ("Segmentation", "Algorithm"), ("Concept", "Planning"),
                  ("Algorithm", "Planning"), ("Planning", "CodeGen"), ("CodeRAG", "CodeGen"),
                  ("CodeMem", "CodeGen"), ("CodeGen", "CodeMem"), ("CodeGen", "Analysis"),
                  ("Analysis", "Modification"), ("Modification", "Refinement")],
        "edge_alias": {("Segmentation", "Planning"): ("Segmentation", "Concept")},
        "real_extra": r"requirement|chat_?planning|orchestrat|workspace|pdf|document_?(conversion|preprocess)|"
                      r"plan_?review|sub_?agent|spawn|team|harness|runner|intelligence|acquisition|synthesi",
    },
    "RD-Agent": {
        "url": "https://github.com/microsoft/RD-Agent", "sha": "484776c",
        "framework": "none",
        "source": "README docs/_static/Framework-RDAgent.png (research: idea -> hypothesis -> experiment; development: "
                  "implementation in workspace -> feedback -> research)",
        "nodes": {"Research": r"hypothesis_?gen|exp_?gen|proposal|idea|research",
                  "Experiment": r"hypothesis_?2_?exp|hypothesis2experiment|to_?experiment",
                  "Development": r"develop|coder|costeer", "Implementation": r"runner|implement|workspace",
                  "Feedback": r"feedback|summari[sz]"},
        "multi": {"Research", "Development", "Feedback", "Experiment", "Implementation"},
        "edges": [("Research", "Experiment"), ("Experiment", "Development"), ("Development", "Implementation"),
                  ("Implementation", "Feedback"), ("Feedback", "Research")],
        "real_extra": r"loop|scen|kaggle|qlib|factor|model|data_?science|trace|knowledge|evaluat|interact|"
                      r"selector|planner|merge|ensemble|pipeline|log|ui|app|utils|test",
    },
    "vc-due-diligence-team": {
        "url": "https://github.com/Shubhamsaboo/awesome-llm-apps", "sha": "6c16b23",
        "path": "advanced_ai_agents/multi_agent_apps/agent_teams/ai_vc_due_diligence_agent_team",
        "framework": "google-adk",
        "source": "README 'Pipeline Architecture' text diagram (stages 1-7 of DueDiligencePipeline)",
        "nodes": {"CompanyResearch": r"company_?research", "MarketAnalysis": r"market_?analysis",
                  "FinancialModeling": r"financial_?model", "RiskAssessment": r"risk", "InvestorMemo": r"memo",
                  "Report": r"report", "Infographic": r"infographic"},
        "edges": [("CompanyResearch", "MarketAnalysis"), ("MarketAnalysis", "FinancialModeling"),
                  ("FinancialModeling", "RiskAssessment"), ("RiskAssessment", "InvestorMemo"),
                  ("InvestorMemo", "Report"), ("Report", "Infographic")],
        # the pipeline container and the root analyst that calls it
        "real_extra": r"due_?diligence|pipeline|analyst$|root",
    },
    "crewai-email-auto-responder-flow": {
        "url": "https://github.com/crewAIInc/crewAI-examples", "sha": "da94a91",
        "path": "flows/email_auto_responder_flow",
        "framework": "crewai",
        "source": "README Email_Flow.png (pull emails -> email crew -> back)",
        "nodes": {"PullEmails": r"fetch_new_emails|pull_?emails|check_?email",
                  "EmailCrew": r"generate_draft|email_filter_crew|email_crew|email_(filter|action|response)"},
        "multi": {"EmailCrew"},
        "edges": [("PullEmails", "EmailCrew"), ("EmailCrew", "PullEmails")],
        "real_extra": r"email_auto_responder|flow|wait_next_run|create_draft|followup|hr_",
    },
    "ChatEval": {
        "url": "https://github.com/thunlp/ChatEval", "sha": "56b320c",
        "framework": "none",
        "source": "README imgs/better_compare.png (referee roles debate in turn, then judge); roles named in the "
                  "llm_eval task the README runs",
        "nodes": {"GeneralPublic": r"general_public|public", "Critic": r"critic"},
        "edges": [("GeneralPublic", "Critic"), ("Critic", "GeneralPublic")],
        "real_extra": r"llm_eval|conversation|agentverse|referee|debat|evaluat",
    },
    "AgentClinic": {
        "url": "https://github.com/SamuelSchmidgall/AgentClinic", "sha": "b6fbe22",
        "framework": "none",
        "source": "README media/mainfigure.png 'Running the AgentClinic' (doctor <-> patient, doctor <-> measurement, "
                  "moderator judges the diagnosis)",
        "nodes": {"Doctor": r"doctor", "Patient": r"patient", "Measurement": r"measurement|meas_",
                  "Moderator": r"moderator|compare_results"},
        "edges": [("Doctor", "Patient"), ("Patient", "Doctor"), ("Doctor", "Measurement"), ("Measurement", "Doctor"),
                  ("Doctor", "Moderator")],
        "real_extra": r"scenario|gen_(medqa|mimic|nejm)|case|query_model",
    },
}
