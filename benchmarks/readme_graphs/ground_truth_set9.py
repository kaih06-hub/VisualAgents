"""Ninth held-out set: ten more agent projects whose READMEs draw their agent graph.

Written from each README's diagram before any project was scanned (same format as GT .. GT8). Node regexes were
chosen from the names the code uses for each drawn component (read from the source, not from scan output).
`real_extra` names components that exist in today's code but are not drawn, and `readme_only` names drawn
components the code doesn't have as agents (no model call of their own) - both decided from the code, never from
scan output. None of these projects appears in sets 1-8.

Data boxes the drawings put between agents (self-refine's "Output to Refine", Reflexion's trajectory / experience
memories) are folded into the arrow between the agents on either side of them.
"""

GT9 = {
    "gpt-newspaper": {
        "url": "https://github.com/assafelovic/gpt-newspaper", "sha": "1c447b5",
        "framework": "langgraph",
        "source": "README docs/images/newsroom.svg: topics -> one reporter per topic [assignment editor -> search -> "
                  "curator -> writer -> critic -> art director -> designer; curator -> search 'dig further back'; "
                  "critic -> writer 'revise'] -> editor-in-chief -> publisher",
        "nodes": {"Assignment": r"^assignment", "Search": r"^search", "Curator": r"^curator",
                  "Writer": r"^writer", "Critic": r"^criti(c|que)", "ArtDirector": r"illustrat|art_?director",
                  "Designer": r"^designer", "Reporter": r"^reporter", "Editor": r"^editor", "Publisher": r"^publisher"},
        "edges": [("Assignment", "Search"), ("Search", "Curator"), ("Curator", "Search"), ("Curator", "Writer"),
                  ("Writer", "Critic"), ("Critic", "Writer"), ("Critic", "ArtDirector"), ("ArtDirector", "Designer"),
                  ("Reporter", "Assignment"), ("Reporter", "Editor"), ("Editor", "Publisher")],
        "real_extra": r"$^",
    },
    "agents-from-scratch": {
        "url": "https://github.com/langchain-ai/agents-from-scratch", "sha": "603fc7a",
        "framework": "langgraph",
        "source": "README Studio screenshot of email_assistant: triage_router -> response_agent [llm_call <-> "
                  "environment]",
        "nodes": {"TriageRouter": r"^triage_router$", "ResponseAgent": r"^response_agent$",
                  "LLMCall": r"^llm_call$", "Environment": r"^environment$|^tool_node$"},
        "multi": {"TriageRouter", "ResponseAgent", "LLMCall"},
        "edges": [("TriageRouter", "ResponseAgent"), ("ResponseAgent", "LLMCall"), ("LLMCall", "Environment"),
                  ("Environment", "LLMCall")],
        # the same graph is rebuilt with human-in-the-loop and memory in sibling modules; langgraph_101 and the cron
        # ingest graph are other entries in langgraph.json; the eval folder grades triage with a model
        "real_extra": r"interrupt|mark_as_read|call_llm|run_tool|ingest|memory|eval|judge|criteria|grade|triage",
    },
    "contoso-creative-writer": {
        "url": "https://github.com/Azure-Samples/contoso-creative-writer", "sha": "cf6387c",
        "framework": "none (prompty)",
        "source": "README architecture image Creative_writing_aca.png: researcher agent -> writer agent -> editor "
                  "agent; text adds a product agent whose results the writer combines",
        "nodes": {"Researcher": r"research", "Product": r"product|marketing", "Writer": r"^writer?$|write",
                  "Editor": r"^editor?$|edit"},
        "edges": [("Researcher", "Writer"), ("Product", "Writer"), ("Writer", "Editor")],
        # evaluate/ grades the article with prompty evaluators (friendliness, relevance ...)
        "real_extra": r"evaluat|friendl|relevan|fluen|coheren|grounded|designer|orchestrat|create",
    },
    "reflexion": {
        "url": "https://github.com/noahshinn/reflexion", "sha": "218cf0e",
        "framework": "none",
        "source": "README figures/reflexion_rl.png: actor (LM) -> trajectory -> evaluator (LM) -> self-reflection "
                  "(LM) -> experience -> actor",
        "nodes": {"Actor": r"actor|func_impl|implement|react_?agent|cot_?agent|prompt_agent|^agent|action",
                  "Evaluator": r"evaluat|internal_tests|test_gen|execut|judge",
                  "SelfReflection": r"reflect"},
        "multi": {"Actor", "Evaluator", "SelfReflection"},
        "edges": [("Actor", "Evaluator"), ("Evaluator", "SelfReflection"), ("SelfReflection", "Actor")],
        # programming_runs keeps other strategies beside reflexion (simple, immediate refinement, UCS search)
        "real_extra": r"simple|immediate|ucs|refine|trial|alfworld|webshop|hotpot|generat|model|gpt|llm|chat",
    },
    "LLMCompiler": {
        "url": "https://github.com/SqueezeAILab/LLMCompiler", "sha": "a00c9d3",
        "framework": "langchain",
        "source": "README figs/thumbnail.png: LLM planner (DAG of tasks) -> task fetching unit <-> executor "
                  "(function calling units over tools)",
        "nodes": {"Planner": r"planner|plan", "TaskFetchingUnit": r"task_?fetch|fetching",
                  "Executor": r"^executor|function_calling|tool_?executor"},
        # the task fetching unit is a scheduler and the executor runs tools: neither asks a model itself
        "readme_only": {"TaskFetchingUnit", "Executor"},
        "edges": [("Planner", "TaskFetchingUnit"), ("TaskFetchingUnit", "Executor"),
                  ("Executor", "TaskFetchingUnit")],
        # the joiner (`join`) decides finish / replan; src/react is the ReAct baseline the paper compares against
        "real_extra": r"join|llm_?compiler|compiler_?agent|react|agent_executor|math|llm_chain|structured_chat",
    },
    "AutoSurvey": {
        "url": "https://github.com/AutoSurveys/AutoSurvey", "sha": "5e8f389",
        "framework": "none",
        "source": "README figs/overview.png: stage 1 retrieval & outline generation -> stage 2 subsection drafting "
                  "-> stage 3 integration & refinement -> stage 4 evaluation & iteration (pick the best survey)",
        "nodes": {"Outline": r"outline", "Drafting": r"subsection_?writer|write_subsection|draft|^writer$",
                  "Refinement": r"refin|integrat|lce", "Evaluation": r"judge|evaluat|criteria"},
        "multi": {"Outline", "Drafting", "Evaluation"},
        "edges": [("Outline", "Drafting"), ("Drafting", "Refinement"), ("Refinement", "Evaluation")],
        "real_extra": r"database|embed|citation|nli|relevan|mindmap",
    },
    "self-refine": {
        "url": "https://github.com/madaan/self-refine", "sha": "9a206d4",
        "framework": "none",
        "source": "README mermaid flowchart: generator -> output to refine; critics -> feedback -> output; output -> "
                  "refiner -> refined output -> (stopping criteria not met) output to refine",
        "nodes": {"Generator": r"init", "Critic": r"feedback|critic", "Refiner": r"iterat|refine"},
        "multi": {"Generator", "Critic", "Refiner"},
        "edges": [("Generator", "Critic"), ("Critic", "Refiner"), ("Refiner", "Critic")],
        # per-task evaluators and the acronym MCTS variant are code the drawing leaves out
        "real_extra": r"eval|measure|mcts|readab|make_challenging|gpt4|count|pie|prompt",
    },
    "MACRec": {
        "url": "https://github.com/wzf2000/MACRec", "sha": "cbd9ea0",
        "framework": "none",
        "source": "README assets/MAC-workflow.png: interpreter -> manager; manager <-> user analyst, item analyst, "
                  "searcher, reflector",
        "nodes": {"Manager": r"manager", "UserAnalyst": r"analyst", "ItemAnalyst": r"item_?analyst",
                  "Searcher": r"search", "Reflector": r"reflect", "Interpreter": r"interpret"},
        # one Analyst class answers both UserInfo and ItemInfo
        "readme_only": {"ItemAnalyst"},
        "edges": [("Interpreter", "Manager"), ("Manager", "UserAnalyst"), ("UserAnalyst", "Manager"),
                  ("Manager", "ItemAnalyst"), ("ItemAnalyst", "Manager"), ("Manager", "Searcher"),
                  ("Searcher", "Manager"), ("Manager", "Reflector"), ("Reflector", "Manager")],
        # systems/ wires the agents into several systems (react, reflection, analyse, chat, collaboration); tasks/
        # holds generation / RLHF tasks
        "real_extra": r"system|collaborat|react|chat|analyse|task|rlhf|reward|summari|llm|opensource|openai",
    },
    "memory-agent": {
        "url": "https://github.com/langchain-ai/memory-agent", "sha": "6593de0",
        "framework": "langgraph",
        "source": "README static/memory_graph.png (Studio): call_model <-> store_memory",
        "nodes": {"CallModel": r"^call_model$", "StoreMemory": r"^store_memory$"},
        "edges": [("CallModel", "StoreMemory"), ("StoreMemory", "CallModel")],
        "real_extra": r"$^",
    },
    "AI-Scientist": {
        "url": "https://github.com/SakanaAI/AI-Scientist", "sha": "1de1dbc",
        "framework": "none (aider)",
        "source": "README docs/anim-ai-scientist.gif: idea generation [LLM idea/plan -> novelty check -> scoring] -> "
                  "experiment iteration [code via LLM & aider -> experiments <-> update plan -> plots] -> paper "
                  "write-up [text via LLM & aider -> manuscript] -> LLM paper reviewing",
        "nodes": {"IdeaGeneration": r"generate_(next_)?ideas?|idea_gen|^ideas?$",
                  "NoveltyCheck": r"novelty",
                  "Experiments": r"experiment|coder",
                  "WriteUp": r"writeup|write_up|latex|citation",
                  "Review": r"review"},
        "multi": {"Experiments", "WriteUp", "Review"},
        "edges": [("IdeaGeneration", "NoveltyCheck"), ("NoveltyCheck", "Experiments"), ("Experiments", "WriteUp"),
                  ("WriteUp", "Review")],
        # perform_improvement revises the paper from the review; templates/ hold the experiment code
        "real_extra": r"improve|meta_review|plot|template|batch_responses|create_client",
    },
}
