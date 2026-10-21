"""Score VisualAgents blueprints against ground-truth graphs taken from each repo's README.

Ground truth nodes carry regexes that match predicted agent ids/names/aliases. A predicted agent that matches
no GT node is classified as a "code-real extra" (exists in today's code but not in the README diagram) when it
matches the repo's `real_extra` regex, otherwise as wrong (noise, duplicate, test fixture, ...).
"""
import json
import re
import sys
from pathlib import Path

GT = {
    "AI-Data-Analysis-MultiAgent": {
        "url": "https://github.com/starpig1129/AI-Data-Analysis-MultiAgent",
        "framework": "langgraph",
        "source": "README 'System Architecture' image (docs/Architecture.png)",
        "nodes": {
            "Hypothesis": r"^hypothesis$", "HumanChoice": r"^human_choice$", "Process": r"^process$",
            "Code": r"^coder?$", "Search": r"^search$", "Visualization": r"^visualization$", "Report": r"^report$",
            "QualityReview": r"^quality_review$", "NoteTaker": r"^note_taker$", "HumanReview": r"^human_review$",
        },
        "edges": [("Hypothesis", "HumanChoice"), ("HumanChoice", "Hypothesis"), ("HumanChoice", "Process"),
                  *[("Process", x) for x in ("Search", "Code", "Visualization", "Report")],
                  *[(x, "QualityReview") for x in ("Search", "Code", "Visualization", "Report")],
                  *[("QualityReview", x) for x in ("Search", "Code", "Visualization", "Report")],
                  ("QualityReview", "NoteTaker"), ("NoteTaker", "Process"), ("Process", "HumanReview"),
                  ("HumanReview", "Process")],
        "real_extra": r"^refiner$",
    },
    "local-deep-researcher": {
        "url": "https://github.com/langchain-ai/local-deep-researcher",
        "framework": "langgraph",
        "source": "README 'How it works' loop (+ graph screenshot)",
        "nodes": {"GenerateQuery": r"generate_query", "WebResearch": r"web_research",
                  "Summarize": r"summarize", "Reflect": r"reflect", "Finalize": r"finalize"},
        "edges": [("GenerateQuery", "WebResearch"), ("WebResearch", "Summarize"), ("Summarize", "Reflect"),
                  ("Reflect", "WebResearch"), ("Reflect", "Finalize")],
        "real_extra": r"$^",
    },
    "langgraph-email-automation": {
        "url": "https://github.com/kaymen99/langgraph-email-automation",
        "framework": "langgraph",
        "source": "README mermaid flowchart",
        "nodes": {"MonitorInbox": r"load_inbox", "InboxEmpty": r"inbox_empty", "Categorize": r"categorize",
                  "Writer": r"^email_writer$", "SynthesizeQueries": r"rag_queries", "RAG": r"retrieve_from_rag",
                  "ProofReader": r"proofreader", "SendEmail": r"send_email"},
        "edges": [("MonitorInbox", "InboxEmpty"), ("InboxEmpty", "Categorize"), ("Categorize", "InboxEmpty"),
                  ("Categorize", "Writer"), ("Categorize", "SynthesizeQueries"), ("SynthesizeQueries", "RAG"),
                  ("RAG", "Writer"), ("Writer", "ProofReader"), ("ProofReader", "SendEmail"),
                  ("ProofReader", "Writer"), ("SendEmail", "InboxEmpty")],
        "real_extra": r"skip_unrelated",
    },
    "Upwork-AI-jobs-applier": {
        "url": "https://github.com/kaymen99/Upwork-AI-jobs-applier",
        "framework": "langgraph",
        "source": "README mermaid graph",
        "nodes": {"Scrape": r"^scrape", "Score": r"^score", "CheckMatches": r"check_for_job_matches",
                  "GenerateContent": r"^(generate_jobs_applications|create_job_application_content)$",
                  "CoverLetter": r"cover_letter", "Interview": r"interview", "Save": r"^save"},
        "multi": {"GenerateContent"},
        "edges": [("Scrape", "Score"), ("Score", "CheckMatches"), ("CheckMatches", "GenerateContent"),
                  ("GenerateContent", "CoverLetter"), ("GenerateContent", "Interview"), ("CoverLetter", "Save"),
                  ("Interview", "Save"), ("Save", "CheckMatches")],
        # the code loops save -> generate_jobs_applications through the same `need_to_process_matches` check
        "edge_alias": {("Save", "GenerateContent"): ("Save", "CheckMatches")},
        "real_extra": r"gather_relevant|finalize_job",
    },
    "openai-cs-agents-demo": {
        "url": "https://github.com/openai/openai-cs-agents-demo",
        "framework": "openai-agents",
        "source": "README screenshot 'Agent View' (agents + guardrails); handoffs from the SDK customer_service example it implements",
        "nodes": {"Triage": r"triage", "FAQ": r"faq", "SeatBooking": r"seat", "FlightStatus": r"flight_(status|information)",
                  "Cancellation": r"cancel", "RelevanceGuardrail": r"relevance", "JailbreakGuardrail": r"jailbreak"},
        "edges": [*[("Triage", x) for x in ("FAQ", "SeatBooking", "FlightStatus", "Cancellation")],
                  *[(x, "Triage") for x in ("FAQ", "SeatBooking", "FlightStatus", "Cancellation")]],
        "real_extra": r"refund",
    },
    "AutoAgents": {
        "url": "https://github.com/Link-AGI/AutoAgents",
        "framework": "custom (MetaGPT-style roles)",
        "source": "README execution-process image (docs/resources/process.jpg)",
        "nodes": {"Planner": r"^manager$", "AgentObserver": r"observer_agents", "PlanObserver": r"observer_plans",
                  "ActionObserver": r"action_observer", "ExecutionAgents": r"^(custom_role|group)$"},
        "multi": {"ExecutionAgents"},
        "edges": [("Planner", "AgentObserver"), ("AgentObserver", "Planner"), ("Planner", "PlanObserver"),
                  ("PlanObserver", "Planner"), ("Planner", "ExecutionAgents"), ("ActionObserver", "ExecutionAgents"),
                  ("ExecutionAgents", "ActionObserver")],
        "real_extra": r"$^",
    },
    "MedAgents": {
        "url": "https://github.com/gersteinlab/MedAgents",
        "framework": "custom (raw OpenAI SDK)",
        "source": "README overview image (pics/overview.png): recruitment -> analyses -> report -> discussion -> decision",
        "nodes": {"Recruiter": r"classifier", "DomainExperts": r"analy[sz]er|voter",
                  "Summarizer": r"synthesi[sz]er", "DecisionMaker": r"decision|final|answer"},
        "multi": {"Recruiter", "DomainExperts"},
        "edges": [("Recruiter", "DomainExperts"), ("DomainExperts", "Summarizer"), ("Summarizer", "DomainExperts"),
                  ("Summarizer", "DecisionMaker")],
        "real_extra": r"$^",
    },
    "Multi-Agents-Debate": {
        "url": "https://github.com/Skytliang/Multi-Agents-Debate",
        "framework": "custom (raw OpenAI SDK)",
        "source": "README framework image (imgs/framework.png) + caption",
        "nodes": {"Affirmative": r"affirmative", "Negative": r"negative", "Judge": r"moderator|judge"},
        "multi": {"Judge"},
        "edges": [("Affirmative", "Negative"), ("Negative", "Affirmative"), ("Affirmative", "Judge"),
                  ("Negative", "Judge")],
        "real_extra": r"baseline",
    },
    "Paper2Code": {
        "url": "https://github.com/going-doer/Paper2Code",
        "framework": "custom (raw OpenAI SDK / vLLM)",
        "source": "README overview image (assets/papercoder_overview.png): Planning -> Analyzing -> Coding",
        "nodes": {"Planning": r"^1_planning$", "Analyzing": r"^2_analyzing$", "Coding": r"^3_coding$"},
        "edges": [("Planning", "Analyzing"), ("Analyzing", "Coding")],
        "real_extra": r"_llm$|debugging|extract_config|rag_config|pdf_process|coding_sh|^eval$",
    },
    "AgentCoder": {
        "url": "https://github.com/huangd1999/AgentCoder",
        "framework": "custom (raw OpenAI SDK)",
        "source": "README description of the three agents and their feedback loop (+ images/agentcoder_pipeline.pdf)",
        "nodes": {"Programmer": r"programmer", "TestDesigner": r"test_designer", "TestExecutor": r"test_executor"},
        "edges": [("Programmer", "TestExecutor"), ("TestDesigner", "TestExecutor"), ("TestExecutor", "Programmer")],
        # run order programmer -> test designer -> test executor is also how the scripts hand off (files on disk)
        "edge_alias": {("Programmer", "TestDesigner"): ("Programmer", "TestExecutor")},
        "real_extra": r"$^",
    },
}


def keys(agent):
    out = [agent["id"], agent["name"], *agent.get("aliases", [])]
    return [re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_") for k in out if k]


def score(repo, bp, gt=None):
    g = (gt or GT)[repo]
    multi = g.get("multi", set())
    pred2gt, used, dup, wrong, extra = {}, set(), [], [], []
    for a in bp["agents"]:
        hit = None
        # the agent's id / name decide first; aliases only when neither does
        for ks in (keys(a)[:2], keys(a)):
            hit = next((n for n, rx in g["nodes"].items() if any(re.search(rx, k) for k in ks)), None)
            if hit:
                break
        if hit and (hit not in used or hit in multi):
            pred2gt[a["id"]] = hit
            used.add(hit)
        elif hit:
            dup.append(a["id"])
        elif any(re.search(g["real_extra"], k) for k in keys(a)):
            extra.append(a["id"])
        else:
            wrong.append(a["id"])
    gt_edges = set(g["edges"])
    pe = set()
    for e in bp["edges"]:
        s, t = pred2gt.get(e["from"]), pred2gt.get(e["to"])
        if s and t and s != t:
            pe.add(g.get("edge_alias", {}).get((s, t), (s, t)))
    tp_e = gt_edges & pe
    # an edge the README draws directly may pass through one step the README leaves out (a code-real extra node)
    succ = {}
    for e in bp["edges"]:
        succ.setdefault(e["from"], set()).add(e["to"])
    via_extra = set()
    for s_id, sg in pred2gt.items():
        for x in succ.get(s_id, ()):
            if x in extra:
                for t_id in succ.get(x, ()):
                    tg = pred2gt.get(t_id)
                    pair = g.get("edge_alias", {}).get((sg, tg), (sg, tg))
                    if tg and pair in gt_edges and pair not in tp_e:
                        via_extra.add(pair)
    tp_e = tp_e | via_extra
    n_pred = len(bp["agents"])
    return {
        "repo": repo,
        "framework_detected": bp["framework"],
        "gt_nodes": len(g["nodes"]), "pred_nodes": n_pred,
        "matched_nodes": sorted(used), "missed_nodes": sorted(set(g["nodes"]) - used),
        "duplicates": dup, "wrong": wrong, "code_real_extras": extra,
        "node_recall": len(used) / len(g["nodes"]),
        "node_recall_code": len(used - set(g.get("readme_only", ()))) / len(set(g["nodes"]) - set(g.get("readme_only", ()))),
        "node_precision_strict": (len(pred2gt)) / n_pred if n_pred else 0.0,
        "node_precision_lenient": (len(pred2gt) + len(extra)) / n_pred if n_pred else 0.0,
        "gt_edges": len(gt_edges), "pred_edges": len(bp["edges"]),
        "matched_edges": sorted(tp_e), "matched_via_extra_step": sorted(via_extra), "missed_edges": sorted(gt_edges - tp_e),
        "edge_recall": len(tp_e) / len(gt_edges) if gt_edges else 1.0,
        "edge_precision": len(tp_e - via_extra) / len(pe) if pe else 0.0,
    }


def sets():
    sys.path.insert(0, str(Path(__file__).parent))
    from ground_truth_set2 import GT2
    from ground_truth_set3 import GT3
    from ground_truth_set4 import GT4
    from ground_truth_set5 import GT5
    return {"set1": GT, "set2": GT2, "set3": GT3, "set4": GT4, "set5": GT5}


if __name__ == "__main__":
    d = Path(sys.argv[1])
    gt = sets()[sys.argv[2] if len(sys.argv) > 2 else "set1"]
    rows = [score(r, json.load(open(d / f"{r}.json")), gt) for r in gt]
    json.dump(rows, open(d / "scores.json", "w"), indent=1)
    keys_ = ("node_recall", "node_recall_code", "node_precision_strict", "node_precision_lenient", "edge_recall", "edge_precision")
    print(f"{'repo':32} {'fw':14} " + " ".join(f"{k[5:19]:>14}" for k in keys_))
    for r in rows:
        print(f"{r['repo']:32} {r['framework_detected']:14} " + " ".join(f"{r[k]:14.2f}" for k in keys_))
    print(f"{'MEAN':47} " + " ".join(f"{sum(r[k] for r in rows) / len(rows):14.2f}" for k in keys_))
    for r in rows:
        print(f"\n## {r['repo']}  missed={r['missed_nodes']} dup={r['duplicates']} wrong={r['wrong']} "
              f"extra={r['code_real_extras']}\n   missed_edges={r['missed_edges']}")
