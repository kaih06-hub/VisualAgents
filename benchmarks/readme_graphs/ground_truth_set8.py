"""Eighth held-out set: ten more agent projects whose READMEs draw their agent graph.

Written from each README's diagram before any project was scanned (same format as GT .. GT7). Node regexes were
chosen from the names the code uses for each drawn component (read from the source, not from scan output).
`real_extra` names components that exist in today's code but are not drawn, and `readme_only` names drawn
components the code doesn't have as agents - both decided from the code, never from scan output.
None of these projects appears in sets 1-7.
"""

GT8 = {
    "ai-hedge-fund": {
        "url": "https://github.com/virattt/ai-hedge-fund", "sha": "29b32f7",
        "framework": "none",
        "source": "README 'AI Hedge Fund - Multi-Strategy Investment System' image: investment analysis -> portfolio "
                  "construction -> risk checks -> simulated execution -> backtests & reports",
        "nodes": {"InvestmentAnalysis": r"buffett|munger|graham|lynch|druckenmiller|pead|analyst|llm_agent|alpha",
                  "PortfolioConstruction": r"portfolio|construct|blend",
                  "RiskChecks": r"risk|limit",
                  "SimulatedExecution": r"execut|broker|order",
                  "Backtests": r"backtest|report|research|scorecard"},
        "multi": {"InvestmentAnalysis"},
        # since the 2.x rewrite only the analysts ask a model; the other four boxes are deterministic functions
        # (blend_signals, apply_limits, execute_decision, the backtester), not agents
        "readme_only": {"PortfolioConstruction", "RiskChecks", "SimulatedExecution", "Backtests"},
        "edges": [("InvestmentAnalysis", "PortfolioConstruction"), ("PortfolioConstruction", "RiskChecks"),
                  ("RiskChecks", "SimulatedExecution"), ("SimulatedExecution", "Backtests")],
        "real_extra": r"$^",
    },
    "gemini-fullstack-langgraph": {
        "url": "https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart", "sha": "e34e569",
        "path": "backend",
        "framework": "langgraph",
        "source": "README agent.png: generate_queries -> web_search (parallel) -> reflection -> (more research) "
                  "web_search | (sufficient) answer generation",
        "nodes": {"GenerateQueries": r"generate_quer", "WebSearch": r"web_(search|research)",
                  "Reflection": r"^reflection", "Answer": r"finali[sz]e|answer"},
        "edges": [("GenerateQueries", "WebSearch"), ("WebSearch", "Reflection"), ("Reflection", "WebSearch"),
                  ("Reflection", "Answer")],
        "real_extra": r"$^",
    },
    "open_deep_research": {
        "url": "https://github.com/langchain-ai/open_deep_research", "sha": "1b7d2e8",
        "framework": "langgraph",
        "source": "README full_diagram (scope: clarify -> brief; research: supervisor <-> sub-agents; write: report) "
                  "+ the LangGraph screenshot under it",
        "nodes": {"Clarify": r"clarif", "Brief": r"research_brief|^brief",
                  "Supervisor": r"^(research_)?supervisor$", "SubAgents": r"^researcher(_subgraph)?$|sub_?agent",
                  "Report": r"final_report"},
        "multi": {"Supervisor", "SubAgents"},
        "edges": [("Clarify", "Brief"), ("Brief", "Supervisor"), ("Supervisor", "SubAgents"),
                  ("SubAgents", "Supervisor"), ("Supervisor", "Report")],
        # tool-executor nodes and compression are real graph nodes the drawing folds in; src/legacy keeps the older
        # plan-and-execute and multi-agent graphs
        "real_extra": r"tools$|compress|report_plan|human_feedback|section|quer|search|final_sections|compile|"
                      r"gather|research_agent|initiali[sz]e|web|planner|legacy|introduction|conclusion",
    },
    "AgentVerse": {
        "url": "https://github.com/OpenBMB/AgentVerse", "sha": "f90c4bd",
        "framework": "none",
        "source": "README Figure 1: expert recruitment -> collaborative decision-making -> action execution -> "
                  "evaluation -> (reward feedback) expert recruitment",
        "nodes": {"Recruitment": r"role_?assign|recruit",
                  "DecisionMaking": r"solver|critic|planner|summari[sz]er|reviewer|manager|decision",
                  "Execution": r"executor|execution",
                  "Evaluation": r"evaluat"},
        "multi": {"DecisionMaking"},
        "edges": [("Recruitment", "DecisionMaking"), ("DecisionMaking", "Execution"), ("Execution", "Evaluation"),
                  ("Evaluation", "Recruitment")],
        # the simulation side (classroom, prisoner's dilemma, Pokemon town, DB diagnosis ...) ships in the same repo
        "real_extra": r"student|professor|teach|suspect|police|prisoner|alice|bob|memory|cpu|dba|steven|^may|maxie|"
                      r"joseph|birch|archie|code_|unit_test|conversation|reflection|tool|player|simulation",
    },
    "Paper2Poster": {
        "url": "https://github.com/Paper2Poster/Paper2Poster", "sha": "746e204",
        "framework": "camel",
        "source": "README assets/posteragent.png: parser -> planner (asset matching, layout) -> painter <-> commenter",
        "nodes": {"Parser": r"pars", "Planner": r"plan|outline|layout|filter|h1_|h2_",
                  "Commenter": r"critic|comment", "Painter": r"paint|actor|content|bullet|fill|style|theme|overflow"},
        "multi": {"Planner", "Commenter", "Painter"},
        "edges": [("Parser", "Planner"), ("Planner", "Painter"), ("Painter", "Commenter"), ("Commenter", "Painter")],
        # the one-shot LLM baseline, the evaluation suite and the vendored camel package
        "real_extra": r"direct|eval|judge|qa|quiz|vlm|camel|role_?play|chat_?agent|critic_?agent|embodied|task",
    },
    "AI-Researcher": {
        "url": "https://github.com/HKUDS/AI-Researcher", "sha": "f9a6f84",
        "framework": "none",
        "source": "README AI-Researcher-Framework.png: prepare agent -> resource analyst <-> idea generator; code agent "
                  "<-> advisor agent; paper writing agents",
        "nodes": {"Prepare": r"prepare", "ResourceAnalyst": r"survey|resource",
                  "IdeaGenerator": r"idea", "CodeAgent": r"^ml|machine_learning|coding_plan|code_agent",
                  "Advisor": r"judge|review|advisor|exp_analy|experiment_analy",
                  "PaperWriter": r"writ|compos|section|abstract|introduction|conclusion|methodology|related_work"},
        "multi": {"ResourceAnalyst", "CodeAgent", "Advisor", "PaperWriter"},
        "edges": [("Prepare", "ResourceAnalyst"), ("ResourceAnalyst", "IdeaGenerator"),
                  ("IdeaGenerator", "ResourceAnalyst"), ("CodeAgent", "Advisor"), ("Advisor", "CodeAgent")],
        "real_extra": r"meta_?chain|tool|github|file_surfer|web|browser|search|benchmark|system_triage|triage|"
                      r"repo|plan_agent|agent_module",
    },
    "Mobile-Agent-v2": {
        "url": "https://github.com/X-PLUG/MobileAgent", "sha": "11cea57", "path": "Mobile-Agent-v2",
        "framework": "none",
        "source": "README assets/role.jpg: planning agent -> decision agent -> reflection agent -> (update) planning; "
                  "decision agent updates the memory unit",
        "nodes": {"Planning": r"plan|process", "Decision": r"action|decision|operat",
                  "Reflection": r"reflect", "Memory": r"memory"},
        "edges": [("Planning", "Decision"), ("Decision", "Reflection"), ("Reflection", "Planning"),
                  ("Decision", "Memory")],
        "real_extra": r"$^",
    },
    "MapCoder": {
        "url": "https://github.com/md-ashraful-pramanik/MapCoder", "sha": "c87e6d9", "path": "src",
        "framework": "none",
        "source": "README images/MapCoder-Overview.png: retrieval agent -> planning agent -> coding agent -> "
                  "debugging agent (-> back to planning when sample I/O still fails)",
        "nodes": {"Retrieval": r"retriev|kb|exemplar|recall", "Planning": r"^(?!self).*plan",
                  "Coding": r"final_code|code_gen|coding", "Debugging": r"debug|improv"},
        "edges": [("Retrieval", "Planning"), ("Planning", "Coding"), ("Coding", "Debugging"),
                  ("Debugging", "Planning")],
        # the other prompting strategies (direct, CoT, analogical, self-planning) are baselines in the same package
        "real_extra": r"direct|cot|analogical|self_?planning|base|gemini|openai|gpt|model|prompting",
    },
    "AutoKaggle": {
        "url": "https://github.com/multimodal-art-projection/AutoKaggle", "sha": "2640d75", "path": "multi_agents",
        "framework": "none",
        "source": "README mdPICs/kaggle_main.png: reader (background) -> planner -> developer -> reviewer -> "
                  "summarizer; reviewer fails back to planner (x3)",
        "nodes": {"Reader": r"reader", "Planner": r"planner", "Developer": r"developer",
                  "Reviewer": r"reviewer", "Summarizer": r"summari[sz]er"},
        "edges": [("Reader", "Planner"), ("Planner", "Developer"), ("Developer", "Reviewer"),
                  ("Reviewer", "Summarizer"), ("Reviewer", "Planner")],
        "real_extra": r"$^",
    },
    "octotools": {
        "url": "https://github.com/octotools/octotools", "sha": "9bee083",
        "framework": "none",
        "source": "README framework_overall.png: query analyzer -> action predictor -> command generator -> command "
                  "executor -> context verifier -> (not done) action predictor | (done) solution summarizer",
        "nodes": {"QueryAnalyzer": r"analy[sz]e_query|query_analy", "ActionPredictor": r"next_step|action_predict",
                  "CommandGenerator": r"tool_command|command_gen", "CommandExecutor": r"execute_tool|command_exec",
                  "ContextVerifier": r"verif", "SolutionSummarizer": r"final_output|summari"},
        "edges": [("QueryAnalyzer", "ActionPredictor"), ("ActionPredictor", "CommandGenerator"),
                  ("CommandGenerator", "CommandExecutor"), ("CommandExecutor", "ContextVerifier"),
                  ("ContextVerifier", "ActionPredictor"), ("ContextVerifier", "SolutionSummarizer")],
        # the planner / executor classes that hold the six steps, and the tools (several ask a model)
        "real_extra": r"^planner$|^executor$|tool|engine|initiali[sz]er|memory|solver|caption|generalist|search|"
                      r"detect|ocr|python|wiki|arxiv|pubmed|google|url|patch|object|image|text|code|direct|base",
    },
}
