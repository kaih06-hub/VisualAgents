"""Ground truth for the eleventh README-graph set: eight projects that appear in none of sets 1-10.

This is a test set only. It was written on 2026-11-13 from each README's mermaid diagram and the names the source
gives each box (graph node names, agent class names), BEFORE any version of the scanner was run on these projects,
and it was not edited after the first scan. Do not tune against it - add a twelfth set instead.

All eight were sampled from GitHub topic searches among small projects whose README holds a mermaid diagram of their
agents. A ninth candidate (agent-me) was dropped before scanning: its "role" classes are deterministic and call no
model, so it is not an agent system in this benchmark's sense.

Edges are the arrows the README draws between agent boxes, even where the code wires them differently (two projects
draw a star or a chain the code does not follow exactly); that costs every scanner version the same.
"""


def _n(word):
    """The box's own name, alone or with an agent / node suffix (`parking`, `ParkingAgent`, `_data_node`)."""
    return rf"^_?({word})(_?(agent|node))?$"


GT11 = {
    "Beacon": {
        "url": "https://github.com/123-qw-as/Beacon", "sha": "36b0b7de",
        "framework": "langgraph",
        "source": "README mermaid 'graph LR': analyst <-> blueprint critic -> modeler <-> model critic -> coder <-> "
                  "code consistency -> sensitivity -> figure pipeline -> writer <-> paper critic -> table assembler "
                  "-> evaluation -> human review -> LaTeX / PDF",
        "nodes": {"Analyst": _n("analyst"), "BlueprintCritic": r"blueprint_critic", "Modeler": _n("modeler"),
                  "ModelCritic": _n("model_critic"), "Coder": _n("coder"), "CodeConsistency": r"model_code_consistency",
                  "Sensitivity": _n("sensitivity"), "FigurePipeline": _n("figure_pipeline"), "Writer": _n("writer"),
                  "PaperCritic": r"paper_critic", "TableAssembler": r"table_assembler", "Evaluation": _n("evaluation"),
                  "HumanReview": r"human_review", "Latex": _n("latex")},
        "edges": [("Analyst", "BlueprintCritic"), ("BlueprintCritic", "Analyst"), ("BlueprintCritic", "Modeler"),
                  ("Modeler", "ModelCritic"), ("ModelCritic", "Modeler"), ("ModelCritic", "Coder"),
                  ("Coder", "CodeConsistency"), ("CodeConsistency", "Coder"), ("CodeConsistency", "Sensitivity"),
                  ("Sensitivity", "FigurePipeline"), ("FigurePipeline", "Writer"), ("Writer", "PaperCritic"),
                  ("PaperCritic", "Writer"), ("PaperCritic", "TableAssembler"), ("TableAssembler", "Evaluation"),
                  ("Evaluation", "HumanReview"), ("HumanReview", "Latex")],
        # sub-steps of the drawn boxes that the code runs as graph nodes of their own
        "real_extra": r"^modeler_|^coder_|^sensitivity_|figure_critic|figure_analy|writer_section|advance_stage|"
                      r"finali[sz]er",
    },
    "NavPark": {
        "url": "https://github.com/925June/NavPark", "sha": "6b2a7f17",
        "framework": "langgraph + own agent classes",
        "source": "README mermaid 'flowchart TD': SupervisorAgent -> Parking / Reservation / Pricing / Support agent "
                  "-> JudgeAgent -> ResponseAgent",
        "nodes": {"Supervisor": r"supervisor", "Parking": _n("parking"), "Reservation": _n("reservation"),
                  "Pricing": _n("pricing"), "Support": _n("support"), "Judge": _n("judge"),
                  "Response": _n("response")},
        "edges": [*[("Supervisor", x) for x in ("Parking", "Reservation", "Pricing", "Support")],
                  *[(x, "Judge") for x in ("Parking", "Reservation", "Pricing", "Support")], ("Judge", "Response")],
        "real_extra": r"guardrail",  # drawn after the response agent; a content check, not a model agent
    },
    "financial-agent-platform": {
        "url": "https://github.com/Ashleyzm/financial-agent-platform", "sha": "3cf9ad62",
        "framework": "langgraph (nodes added in a loop over an enum)",
        "source": "README mermaid 'flowchart LR': Supervisor -> Data / Research / Prediction / Risk / Report agent",
        "nodes": {"Supervisor": r"supervisor", "Data": _n("data"), "Research": _n("research"),
                  "Prediction": r"prediction", "Risk": _n("risk"), "Report": _n("report")},
        "edges": [("Supervisor", x) for x in ("Data", "Research", "Prediction", "Risk", "Report")],
        "real_extra": r"$^",
    },
    "multi-agent-investment": {
        "url": "https://github.com/flash131307/multi-agent-investment", "sha": "93ef02cb",
        "framework": "own BaseAgent classes (asyncio.gather)",
        "source": "README mermaid 'graph TB': Orchestrator -> Technical / Sentiment / Fundamental agent -> Decision "
                  "Hub -> LLM reasoning",
        "nodes": {"Orchestrator": r"orchestrator|^run_agents$|^runner$", "Technical": r"technical",
                  "Sentiment": r"sentiment", "Fundamental": r"fundamental", "DecisionHub": r"decision_?hub",
                  "Reasoning": r"reasoning"},
        "edges": [*[("Orchestrator", x) for x in ("Technical", "Sentiment", "Fundamental")],
                  *[(x, "DecisionHub") for x in ("Technical", "Sentiment", "Fundamental")],
                  ("DecisionHub", "Reasoning")],
        # the orchestrator gathers the three agents and the hub fuses their signals with arithmetic: neither prompts a model
        "readme_only": {"Orchestrator", "DecisionHub"},
        # the steps inside the fundamental (plan-and-solve) and sentiment (funnel) agents
        "real_extra": r"planner|executor|synthesi[sz]er|profiler|classifier|aggregator",
    },
    "langgraph-agentops-studio": {
        "url": "https://github.com/hyf020908/langgraph-agentops-studio", "sha": "2a12d809",
        "framework": "langgraph (nodes built by factories, two subgraphs)",
        "source": "README mermaid 'flowchart TD': initialize_task -> planner_agent -> supervisor <-> research_pipeline; "
                  "planner -> specialist collaboration -> supervisor; supervisor -> analyst_agent <-> reviewer_agent "
                  "-> executor_agent -> supervisor; human_review",
        "nodes": {"Initialize": r"initialize_task", "Planner": r"planner", "Supervisor": r"supervisor",
                  "Collaboration": r"^collaboration$", "Research": r"^research_pipeline$", "Analyst": r"analyst",
                  "Reviewer": r"reviewer", "HumanReview": r"human_review", "Executor": r"executor"},
        "edges": [("Initialize", "Planner"), ("Planner", "Supervisor"), ("Planner", "Collaboration"),
                  ("Collaboration", "Supervisor"), ("Supervisor", "Research"), ("Research", "Supervisor"),
                  ("Supervisor", "Analyst"), ("Supervisor", "HumanReview"), ("Analyst", "Reviewer"),
                  ("Reviewer", "Analyst"), ("Reviewer", "HumanReview"), ("Reviewer", "Executor"),
                  ("HumanReview", "Executor"), ("HumanReview", "Analyst"), ("Executor", "Supervisor")],
        # the nodes inside the two subgraphs the README draws as one box each
        "real_extra": r"dispatch_specialists|^specialist$|finish_collaboration|research_briefing|research_tools|"
                      r"parse_sources|parser_tools|rank_evidence|ranking_tools|collect_research|compact_research",
    },
    "AutoViralAI": {
        "url": "https://github.com/kgarbacinski/AutoViralAI", "sha": "96338f3f",
        "framework": "langgraph (two graphs)",
        "source": "README mermaid 'graph TB': goal check -> research viral content -> extract patterns -> generate "
                  "post variants -> ranking -> human approval -> publish -> schedule metrics check; collect metrics "
                  "-> analyze performance -> update pattern scores -> adjust strategy",
        "nodes": {"GoalCheck": r"goal_check", "Research": r"research_viral", "Extract": r"extract_patterns",
                  "Generate": r"generate_post", "Rank": r"rank_and_select|ranking", "HumanApproval": r"approval",
                  "Publish": r"publish_post", "Schedule": r"schedule_metrics", "Collect": r"collect_metrics",
                  "Analyze": r"analyze_performance", "Update": r"update_knowledge_base|update_pattern",
                  "Adjust": r"adjust_strategy"},
        "edges": [("GoalCheck", "Research"), ("Research", "Extract"), ("Extract", "Generate"), ("Generate", "Rank"),
                  ("Rank", "HumanApproval"), ("HumanApproval", "Publish"), ("HumanApproval", "Generate"),
                  ("Publish", "Schedule"), ("Collect", "Analyze"), ("Analyze", "Update"), ("Update", "Adjust")],
        "readme_only": {"HumanApproval"},  # an interrupt for the Telegram reply, not a node of the graph
        "real_extra": r"$^",
    },
    "story-shot-agent": {
        "url": "https://github.com/neopen/story-shot-agent", "sha": "d7729315",
        "framework": "langgraph behind an own orchestrator class + agent classes",
        "source": "README mermaid 'flowchart TD' (Chinese labels): script parser -> shot segmenter -> video splitter -> "
                  "prompt converter -> quality auditor -> continuity guard -> asset generator; control nodes loop "
                  "check -> error handler -> human intervention -> output",
        "nodes": {"ScriptParser": r"parse_script|script_?parser", "ShotSegmenter": r"segment_shot|shot_?segmenter",
                  "VideoSplitter": r"split_video|video_?splitter", "PromptConverter": r"convert_prompt|prompt_?converter",
                  "QualityAuditor": r"audit_quality|quality_?audit", "Continuity": r"continuity",
                  "AssetGenerator": r"generate_assets|asset_?generator", "LoopCheck": r"loop_check",
                  "ErrorHandler": r"error_handler", "Human": r"human_intervention", "Output": r"generate_output"},
        "edges": [("ScriptParser", "ShotSegmenter"), ("ShotSegmenter", "VideoSplitter"),
                  ("VideoSplitter", "PromptConverter"), ("PromptConverter", "QualityAuditor"),
                  ("QualityAuditor", "Continuity"), ("Continuity", "AssetGenerator"), ("LoopCheck", "ErrorHandler"),
                  ("ErrorHandler", "Human"), ("Human", "Output")],
        "real_extra": r"route_model",  # a graph node of the code the README does not draw
    },
    "final-review": {
        "url": "https://github.com/lucianwhy/final-review", "sha": "ab12ef65",
        "framework": "langgraph (nodes added in a loop over a dict)",
        "source": "README mermaid 'flowchart TD' (Chinese labels): intent route -> exam info / retrieval -> generate -> "
                  "verify -> (retry retrieval | answer | wait for answers -> grade -> return)",
        "nodes": {"Route": _n("route"), "ExamProfile": r"exam_profile", "Retrieve": _n("retrieve"),
                  "Generate": _n("generate"), "Verify": _n("verify"), "WaitAnswers": r"wait_answers",
                  "Evaluate": _n("evaluate"), "Finalize": _n("finalize")},
        "edges": [("Route", "ExamProfile"), ("Route", "Retrieve"), ("ExamProfile", "Retrieve"),
                  ("Retrieve", "Generate"), ("Generate", "Verify"), ("Verify", "Retrieve"), ("Verify", "Finalize"),
                  ("Verify", "WaitAnswers"), ("WaitAnswers", "Evaluate"), ("Evaluate", "Finalize")],
        # other nodes of the same graph, and the nodes of the older graph kept under langgraph-agent/
        "real_extra": r"quiz_config|note_parse|note_config|note_generate|^refuse$|load_skill|clarify_exam|"
                      r"convert_markdown|clean_markdown|extract_knowledge|generate_questions|generate_explanations|"
                      r"format_output",
    },
}
