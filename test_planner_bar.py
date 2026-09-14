from pipeline import run_researcher, run_planner

cases = [
    "Give me a full review of my current portfolio.",
    "Do I have any concentration risk in my portfolio right now?",
    "Should I buy Microsoft?",
]

for q in cases:
    research = run_researcher(q)
    plan = run_planner(q, research["gathered_data"])
    print(f"=== {q}")
    if plan.get("clarification_needed"):
        print(f"  CLARIFIES: {plan['clarifying_question']}")
    else:
        angles = [a["angle"] for a in (plan.get("analysis_plan") or [])]
        print(f"  PROCEEDS with {len(angles)} angles: {angles}")
    print()
