"""Ground truth for the fourteenth README-graph set: projects built on CrewAI, Google ADK, PydanticAI, Strands and
Microsoft Agent Framework that appear in none of sets 1-13.

A test set only, written on 2026-11-25 from each README's mermaid diagram and the names the source gives each box,
BEFORE any version of the scanner ran on these projects, and not edited after the first scan. It measures the
`framework-support` branch, whose rules were written from those frameworks' own sample repositories and their
source code; none of these projects was looked at while writing them. Do not tune against it - add a fifteenth set.

How they were picked (selection.md has the pools and every rejection): per framework, GitHub topic search (and, for
Strands and Microsoft Agent Framework, code search for their multi-agent imports), minus sets 1-13 and the
frameworks' own organisations, in a fixed pseudo-random order (sha1 of "set14:" + lowercase owner/name), taken in
that order when the README's mermaid diagram draws three or more agents and the Python code defines two or more
agents with that framework. Three per framework; Strands has two, the pools read held no third.

Edges are the arrows the README draws between agent boxes; an arrow through a decision diamond or a router shape
is taken from the box before it to each box after it. A box the code has no block for is marked readme_only.
"""

GT14 = {
    # -- Strands --------------------------------------------------------------------------------------------------
    "traceforge": {
        "url": "https://github.com/ElijahUmana/traceforge", "sha": "5fdd7cb8",
        "framework": "strands",
        "source": "README mermaid 'flowchart LR': Researcher -> Analyst -> Writer (each reporting to a provenance hook)",
        "nodes": {"Researcher": r"research", "Analyst": r"analy[sz]", "Writer": r"writ"},
        "edges": [("Researcher", "Analyst"), ("Analyst", "Writer")],
    },
    "sample-agent-interoperability": {
        "url": "https://github.com/aws-samples/sample-agent-interoperability", "sha": "20717f0f",
        "framework": "strands",
        "source": "README mermaid 'graph TD': four agents deployed as separate AgentCore runtimes (ADK on Bedrock, ADK "
                  "on Gemini, Strands on Bedrock, Strands on Gemini) behind a Lambda router; no arrows between agents",
        "nodes": {"AdkBedrock": r"^bedrock_agent$|adk\w*bedrock|bedrock\w*adk",
                  "AdkGemini": r"^gemini_agent$|adk\w*gemini|gemini\w*adk",
                  "StrandsBedrock": r"strands\w*bedrock|bedrock\w*strands",
                  "StrandsGemini": r"strands\w*gemini|gemini\w*strands"},
        "edges": [],
    },
    # -- Microsoft Agent Framework --------------------------------------------------------------------------------
    "PrivyDoc": {
        "url": "https://github.com/ShivamGoyal03/PrivyDoc", "sha": "a54d6378",
        "framework": "microsoft-agent-framework",
        "source": "README mermaid 'flowchart TD': text extraction -> section extraction agent -> entity recognition "
                  "agent -> summarize + sentiment agent -> results",
        "nodes": {"SectionExtraction": r"extractor|section", "EntityRecognition": r"^ner$|entit",
                  "SummarizeSentiment": r"analy[sz]er|summar|sentiment"},
        "edges": [("SectionExtraction", "EntityRecognition"), ("EntityRecognition", "SummarizeSentiment")],
    },
    "friendly-fishstick": {
        "url": "https://github.com/ABHISHEK-DBZ/friendly-fishstick", "sha": "3885b5d7",
        "framework": "microsoft-agent-framework",
        "source": "README mermaid 'graph TD': vision agent -> research agent -> advisory agent -> memory agent",
        "nodes": {"Vision": r"vision", "Research": r"research", "Advisory": r"advis", "Memory": r"memory"},
        "edges": [("Vision", "Research"), ("Research", "Advisory"), ("Advisory", "Memory")],
        "real_extra": r"weather|soil|evaluat|mcp",
    },
    "AI_Exam_Assistant": {
        "url": "https://github.com/Nambu89/AI_Exam_Assistant", "sha": "ca494b83",
        "framework": "microsoft-agent-framework",
        "source": "README mermaid 'flowchart TD': coordinator (router) -> tutor / exam / analytics agents; exam -> "
                  "question generator -> validator",
        "nodes": {"Coordinator": r"coordinat|router", "Tutor": r"tutor", "Exam": r"^exam|examiner",
                  "QuestionGenerator": r"question", "Validator": r"validat", "Analytics": r"analytic"},
        "edges": [("Coordinator", "Tutor"), ("Coordinator", "Exam"), ("Coordinator", "Analytics"),
                  ("Exam", "QuestionGenerator"), ("QuestionGenerator", "Validator")],
        "real_extra": r"triage|planner|handoff",
    },
    # -- PydanticAI -----------------------------------------------------------------------------------------------
    "ouroboros": {
        "url": "https://github.com/Tanush1912/ouroboros", "sha": "900cacde",
        "framework": "pydantic-ai",
        "source": "README mermaid 'flowchart TD' of the run graph: plan -> (reproduce) -> implement -> validate -> "
                  "perf / ui validation -> open PR -> review loop -> merge, retries back to implement, escalation to "
                  "a human checkpoint and a post-mortem",
        "nodes": {"Plan": r"^plan(_node|ner)?$|run_planner|planner_agent", "Reproduce": r"reproduce",
                  "Implement": r"^implement(_node|er)?$|run_implementer|implementer_agent",
                  "Validate": r"^validate(_node)?$", "HumanCheckpoint": r"human", "PerfValidate": r"perf",
                  "UiValidate": r"^ui_", "OpenPr": r"open_pr", "Review": r"review", "Merge": r"merge",
                  "PostMortem": r"post_?mortem"},
        "edges": [("Plan", "Reproduce"), ("Plan", "Implement"), ("Reproduce", "Implement"), ("Implement", "Validate"),
                  ("Validate", "Implement"), ("Validate", "HumanCheckpoint"), ("Validate", "PerfValidate"),
                  ("PerfValidate", "UiValidate"), ("UiValidate", "OpenPr"), ("OpenPr", "Review"),
                  ("Review", "Implement"), ("Review", "Merge"), ("HumanCheckpoint", "PostMortem")],
        "real_extra": r"feedback|mutation|cleaner",
    },
    "IRAS": {
        "url": "https://github.com/krishnashakula/IRAS", "sha": "68479857",
        "framework": "pydantic-ai",
        "source": "README mermaid 'graph TB' (LangGraph state machine of PydanticAI agents): ingestion -> triage -> "
                  "context gathering -> RCA (loops back while unsure) -> generate plan -> approval -> remediation / "
                  "escalation -> post-mortem",
        "nodes": {"Ingestion": r"ingest", "Triage": r"triage", "Context": r"context",
                  "Rca": r"^rca|root_cause", "GeneratePlan": r"generate_plan|remediation_plan|planning",
                  "Approval": r"approv", "Remediation": r"apply_remediation|^remediation(_node)?$",
                  "Escalation": r"escalat", "PostMortem": r"post_?mortem"},
        "edges": [("Ingestion", "Triage"), ("Triage", "Context"), ("Context", "Rca"), ("Rca", "Context"),
                  ("Rca", "GeneratePlan"), ("Rca", "Escalation"), ("GeneratePlan", "Approval"),
                  ("Approval", "Remediation"), ("Approval", "Escalation"), ("Remediation", "PostMortem"),
                  ("Escalation", "PostMortem")],
    },
    "foodmind.ai": {
        "url": "https://github.com/vsokoltsov/foodmind.ai", "sha": "667b1804",
        "framework": "pydantic-ai",
        "source": "README mermaid 'flowchart TD': query rewriter -> (router) -> structured planner -> plan executor, "
                  "or straight to one of four specialists (food search, nutrition analysis, product comparison, "
                  "food recommendation); the executor runs specialists and the answer synthesizer",
        "nodes": {"Rewriter": r"rewrit", "Planner": r"planner|^plan(ning)?(_agent)?$", "Executor": r"executor|execute_plan",
                  "FoodSearch": r"food_search|search_agent", "Nutrition": r"nutrition", "Comparison": r"compar",
                  "Recommendation": r"recommend", "Synthesizer": r"synthes"},
        "edges": [("Rewriter", "Planner"), ("Rewriter", "FoodSearch"), ("Rewriter", "Nutrition"),
                  ("Rewriter", "Comparison"), ("Rewriter", "Recommendation"), ("Planner", "Executor"),
                  ("Executor", "FoodSearch"), ("Executor", "Nutrition"), ("Executor", "Comparison"),
                  ("Executor", "Recommendation"), ("Executor", "Synthesizer")],
        "real_extra": r"orchestrat|summar",
    },
    # -- CrewAI ---------------------------------------------------------------------------------------------------
    "bizinsight-agent": {
        "url": "https://github.com/sAjAd-2006/bizinsight-agent", "sha": "3ba8a1b2",
        "framework": "crewai",
        "source": "README mermaid 'graph TD': orchestrator agent -> data analyst / insight generator / report writer",
        "nodes": {"Orchestrator": r"orchestrat|manager", "DataAnalyst": r"data_analy|^analyst",
                  "InsightGenerator": r"insight", "ReportWriter": r"report|writer"},
        "edges": [("Orchestrator", "DataAnalyst"), ("Orchestrator", "InsightGenerator"), ("Orchestrator", "ReportWriter")],
    },
    "lifeops-ai-v2": {
        "url": "https://github.com/thenishantraj/lifeops-ai-v2", "sha": "d5d04f85",
        "framework": "crewai",
        "source": "README mermaid 'graph TD': the CrewAI layer runs a health, a finance, a study and a coordinator agent; "
                  "no arrows between agents",
        "nodes": {"Health": r"health|wellness", "Finance": r"financ", "Study": r"study|learning",
                  "Coordinator": r"coordinat|reflection"},
        "edges": [],
    },
    "et_ai_hackathon": {
        "url": "https://github.com/suryanmandal/et_ai_hackathon", "sha": "08bd0ecc",
        "framework": "crewai",
        "source": "README mermaid 'graph TB': a CrewAI pipeline of a sensor, a source-attribution and a compliance agent "
                  "invoked by the API; no arrows between agents",
        "nodes": {"Sensor": r"sensor", "Attribution": r"attribution", "Compliance": r"complian"},
        "edges": [],
    },
    # -- Google ADK -----------------------------------------------------------------------------------------------
    "career-copilot": {
        "url": "https://github.com/sahanfernad/career-copilot", "sha": "229355e9",
        "framework": "google-adk",
        "source": "README mermaid 'graph TD' of the workflow: security checkpoint -> coordinator node -> resume / roadmap "
                  "branches (prepare input -> agent -> save result -> back) or the interview cycle; exit, default and "
                  "security error end in the final output",
        "nodes": {"SecurityCheck": r"security_check", "SecurityError": r"security_error",
                  "Coordinator": r"^orch|coordinat", "PrepareResume": r"prep\w*resume",
                  "ResumeAgent": r"resume_(job_match_)?agent|resume_job_match", "SaveResume": r"save_resume",
                  "PrepareRoadmap": r"prep\w*roadmap", "RoadmapAgent": r"roadmap_agent|learning_roadmap",
                  "SaveRoadmap": r"save_roadmap", "InterviewCycle": r"interview", "FinalExit": r"exit",
                  "FinalDefault": r"default", "Final": r"^final(_output)?(_node)?$"},
        "edges": [("SecurityCheck", "SecurityError"), ("SecurityCheck", "Coordinator"), ("Coordinator", "PrepareResume"),
                  ("PrepareResume", "ResumeAgent"), ("ResumeAgent", "SaveResume"), ("SaveResume", "Coordinator"),
                  ("Coordinator", "PrepareRoadmap"), ("PrepareRoadmap", "RoadmapAgent"), ("RoadmapAgent", "SaveRoadmap"),
                  ("SaveRoadmap", "Coordinator"), ("Coordinator", "InterviewCycle"), ("InterviewCycle", "Coordinator"),
                  ("Coordinator", "FinalExit"), ("Coordinator", "FinalDefault"), ("FinalExit", "Final"),
                  ("FinalDefault", "Final"), ("SecurityError", "Final")],
    },
    "gemini-tales-v2": {
        "url": "https://github.com/vero-code/gemini-tales-v2", "sha": "8f468ebf",
        "framework": "google-adk",
        "source": "README mermaid 'graph TD': Puck, the live narrator, requests a pipeline from the orchestrator, which "
                  "calls a researcher, a judge and a storysmith over A2A",
        "nodes": {"Puck": r"puck", "Orchestrator": r"orchestrat|pipeline", "Researcher": r"research",
                  "Judge": r"judge", "Storysmith": r"storysmith|content_builder|story"},
        "edges": [("Puck", "Orchestrator"), ("Orchestrator", "Researcher"), ("Orchestrator", "Judge"),
                  ("Orchestrator", "Storysmith")],
        "real_extra": r"escalation|avatar|frontend|illustrat|achievement|award|proxy",
    },
    "SupportMind-AI": {
        "url": "https://github.com/arish096/SupportMind-AI", "sha": "859bca8b",
        "framework": "google-adk",
        "source": "README mermaid 'flowchart TD': customer agent -> memory agent -> (history, MCP context) -> "
                  "resolution agent -> review agent -> support ticket",
        "nodes": {"Customer": r"customer", "Memory": r"memory", "Resolution": r"resolution", "Review": r"review"},
        "edges": [("Customer", "Memory"), ("Memory", "Resolution"), ("Resolution", "Review")],
        "real_extra": r"coordinat|supportmind",
    },
}
