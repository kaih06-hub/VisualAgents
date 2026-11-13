"""Ground truth for the tenth README-graph set: ten projects that appear in none of sets 1-9.

Unlike sets 1-9 this set is a test set only. It was written from the README diagrams / agent lists and the names the
source uses for each box before any scan, and no scanner rule was written or changed because of what it shows:
it measures how the scanner does on projects it has never seen. Do not tune against it - add an eleventh set instead.

Six were sampled from GitHub topic searches (multi-agent, langgraph, smolagents, ...) among projects whose README
holds a mermaid diagram of their agents; four are well-known projects whose README draws or lists the agents.
"""

GT10 = {
    "FinnewsHunter": {
        "url": "https://github.com/DemonDamon/FinnewsHunter", "sha": "dfe6bbcb",
        "framework": "agenticx (own orchestrator)",
        "source": "README mermaid 'graph TD': investment manager -> debate orchestrator <-> bull / bear / search analyst",
        "nodes": {"Manager": r"investment_?manager", "Orchestrator": r"orchestrator", "Bull": r"bull",
                  "Bear": r"bear", "Searcher": r"search_?analyst"},
        "edges": [("Manager", "Orchestrator"), ("Orchestrator", "Bull"), ("Bull", "Orchestrator"),
                  ("Orchestrator", "Searcher"), ("Searcher", "Orchestrator"), ("Orchestrator", "Bear"),
                  ("Bear", "Orchestrator"), ("Orchestrator", "Manager")],
        # agents of the news pipeline, outside the drawn debate
        "real_extra": r"data_?collector|quick_?analyst|news_?analyst|quantitative|knowledge_?extractor",
    },
    "Paper-Agent": {
        "url": "https://github.com/Tswoen/Paper-Agent", "sha": "af2fc633",
        "framework": "langgraph + own agent classes",
        "source": "README mermaid 'flowchart LR': SearchAgent -> ReadAgent -> AnalyseAgent -> WritingOutlineAgent -> WritingAgent",
        "nodes": {"Search": r"search", "Read": r"read", "Analyse": r"analy", "Outline": r"outline", "Writing": r"writ"},
        "edges": [("Search", "Read"), ("Read", "Analyse"), ("Analyse", "Outline"), ("Outline", "Writing")],
        "real_extra": r"reply",
    },
    "AutoCVE": {
        "url": "https://github.com/larlarua/AutoCVE", "sha": "b26025f2",
        "framework": "own BaseAgent classes",
        "source": "README mermaid 'flowchart LR': orchestrator -> recon -> scan -> triage -> verification; recon -> finding -> verification -> merge",
        "nodes": {"Orchestrator": r"orchestrator", "Recon": r"recon", "Scan": r"^scan(_?agent)?$", "Triage": r"triage",
                  "Finding": r"^finding(_?agent)?$", "Verification": r"verif", "Merge": r"merge|finali[sz]e"},
        "edges": [("Orchestrator", "Recon"), ("Recon", "Scan"), ("Scan", "Triage"), ("Recon", "Finding"),
                  ("Triage", "Verification"), ("Finding", "Verification"), ("Verification", "Merge")],
        "readme_only": {"Merge"},  # a merge step of the runtime, not an agent class
        "real_extra": r"analysis|finding_|candidate|synthesi[sz]er",
    },
    "SmolSQLAgents": {
        "url": "https://github.com/montraydavis/SmolSQLAgents", "sha": "af79b302",
        "framework": "smolagents",
        "source": "README mermaid 'graph TB': pipeline -> entity recognition -> business context -> NL2SQL; core -> indexer; batch manager -> indexer",
        "nodes": {"Pipeline": r"pipeline", "EntityRecognition": r"entity", "BusinessContext": r"business",
                  "NL2SQL": r"nl2sql", "Core": r"core|documentation", "Indexer": r"indexer", "Batch": r"batch"},
        "edges": [("Pipeline", "EntityRecognition"), ("EntityRecognition", "BusinessContext"),
                  ("BusinessContext", "NL2SQL"), ("Core", "Indexer"), ("Batch", "Indexer")],
        "readme_only": {"Batch"},  # BatchIndexingManager queues work for the indexer; it prompts no model
        "real_extra": r"$^",
    },
    "GustoBot": {
        "url": "https://github.com/skygazer42/GustoBot", "sha": "09fadecd",
        "framework": "langgraph (nested graphs)",
        "source": "README mermaid 'graph TB': router -> general / additional / research plan / kb / image / file; "
                  "GraphRAG subgraph guardrails -> planner -> tool selection -> tools -> summarize -> final answer; "
                  "KB subgraph guardrails -> kb router -> local / external search -> finalize",
        "nodes": {"Router": r"analyze_and_route_query", "General": r"respond_to_general_query",
                  "Additional": r"get_additional_info", "Research": r"create_research_plan", "KB": r"create_kb_query",
                  "Image": r"create_image_query", "File": r"create_file_query", "Guardrails": r"^guardrails$",
                  "Planner": r"^planner$", "ToolSelection": r"tool_selection", "Cypher": r"^cypher_query$",
                  "Predefined": r"predefined_cypher", "GraphRAG": r"graphrag|customer_tools",
                  "Text2SQL": r"^text2sql_query$", "Summarize": r"^summarize$", "FinalAnswer": r"^final_answer$",
                  "KBRouter": r"kb_router", "LocalSearch": r"local_search", "ExternalSearch": r"external_search",
                  "Finalize": r"^finalize$"},
        "multi": {"Guardrails"},  # both subgraphs name their first node "guardrails"
        "edges": [*[("Router", x) for x in ("General", "Additional", "Research", "KB", "Image", "File")],
                  ("Research", "Guardrails"), ("Guardrails", "Planner"), ("Planner", "ToolSelection"),
                  *[("ToolSelection", x) for x in ("Cypher", "Predefined", "GraphRAG", "Text2SQL")],
                  *[(x, "Summarize") for x in ("Cypher", "Predefined", "GraphRAG", "Text2SQL")],
                  ("Summarize", "FinalAnswer"), ("KB", "Guardrails"), ("Guardrails", "KBRouter"),
                  ("KBRouter", "LocalSearch"), ("KBRouter", "ExternalSearch"), ("LocalSearch", "ExternalSearch"),
                  ("LocalSearch", "Finalize"), ("ExternalSearch", "Finalize")],
        # the text2sql, text2cypher and chart sub-workflows the drawing folds into single boxes
        "real_extra": r"retrieve_schema|analyze_query|generate_sql|validate_sql|execute_sql|visuali[sz]ation|"
                      r"format_answer|chart|cypher|map_reduce",
    },
    "graph_websearch_agent": {
        "url": "https://github.com/john-adeojo/graph_websearch_agent", "sha": "fbeaf583",
        "framework": "langgraph",
        "source": "README image schema_diagram/LanGraph.png: planner -> web search tool -> researcher -> scraping tool -> "
                  "reporter -> reviewer -> planner / researcher / reporter / final report",
        "nodes": {"Planner": r"^planner", "WebSearch": r"serper", "Researcher": r"selector|researcher",
                  "Scraper": r"scraper", "Reporter": r"^reporter", "Reviewer": r"reviewer",
                  "FinalReport": r"final_report"},
        "edges": [("Planner", "WebSearch"), ("WebSearch", "Researcher"), ("Researcher", "Scraper"),
                  ("Scraper", "Reporter"), ("Reporter", "Reviewer"), ("Reviewer", "Planner"),
                  ("Reviewer", "Researcher"), ("Reviewer", "Reporter"), ("Reviewer", "FinalReport")],
        "real_extra": r"^router$|^end$",  # the code routes the reviewer's verdict through a router node
    },
    "gpt-pilot": {
        "url": "https://github.com/Pythagora-io/gpt-pilot", "sha": "9b763fda",
        "framework": "own BaseAgent classes",
        "source": "README 'How GPT Pilot works' numbered agent list: product owner, specification writer, architect, "
                  "tech lead, developer, code monkey, reviewer, troubleshooter, debugger, technical writer",
        "nodes": {"SpecWriter": r"spec_?writer", "Architect": r"architect", "TechLead": r"tech_?lead",
                  "Developer": r"^developer", "CodeMonkey": r"code_?monkey", "Reviewer": r"review",
                  "Troubleshooter": r"troubleshooter", "Debugger": r"bug_?hunter|debugger",
                  "TechnicalWriter": r"tech(nical)?_?writer", "ProductOwner": r"product_?owner"},
        "edges": [("SpecWriter", "Architect"), ("Architect", "TechLead"), ("TechLead", "Developer"),
                  ("Developer", "CodeMonkey"), ("CodeMonkey", "Reviewer"), ("Reviewer", "CodeMonkey")],
        # "does nothing" per the README; the review is a step inside CodeMonkey, not a class of its own
        "readme_only": {"ProductOwner", "Reviewer"},
        "real_extra": r"orchestrator|executor|error_?handler|external_?doc|frontend|^git|human_?input|importer|"
                      r"legacy_?handler|problem_?solver|task_?completer|wizard",
    },
    "MDAgents": {
        "url": "https://github.com/mitmedialab/MDAgents", "sha": "3adbd760",
        "framework": "plain Python (own Agent class)",
        "source": "README imgs/animation.gif: complexity check -> low (single clinician) / moderate (multi-disciplinary "
                  "team) / high (integrated care team)",
        "nodes": {"Complexity": r"difficulty|complexity", "Basic": r"basic", "Intermediate": r"intermediate|mdt",
                  "Advanced": r"advanced|ict"},
        "edges": [("Complexity", "Basic"), ("Complexity", "Intermediate"), ("Complexity", "Advanced")],
        # the roles played inside each branch
        "real_extra": r"recruit|moderator|decision|summar|expert|group|medical|agent",
    },
    "agentic-context-engine": {
        "url": "https://github.com/kayba-ai/agentic-context-engine", "sha": "3a319830",
        "framework": "plain Python",
        "source": "README mermaid 'flowchart LR': agent <-> environment -> reflector -> skill manager -> skillbook -> agent",
        "nodes": {"Agent": r"^agent$|generator", "Reflector": r"reflector", "SkillManager": r"skill_?manager|curator"},
        "edges": [("Agent", "Reflector"), ("Reflector", "SkillManager"), ("SkillManager", "Agent")],
        "real_extra": r"dedup|recursive|insight|integration|browser|claude|langchain",
    },
    "Curie": {
        "url": "https://github.com/Just-Curieous/Curie", "sha": "db1b1f56",
        "framework": "langgraph",
        "source": "README docs/static/img/curie-overview.png: architect <-> technicians, over the experimental rigor engine",
        "nodes": {"Architect": r"supervisor|architect", "Technicians": r"worker|technician",
                  "Rigor": r"valid|verif|patch|^analyzer|concluder"},
        "multi": {"Technicians", "Rigor"},
        "edges": [("Architect", "Technicians"), ("Technicians", "Architect")],
        # every hand-off passes through the scheduler node, which the drawing leaves out
        "real_extra": r"scheduler|clarif|user_input|data_analyzer",
    },
}
