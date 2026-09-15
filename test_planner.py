from pipeline import run_researcher, run_planner
import json

for question in ["Should I buy Microsoft?", "Should I invest in a random speculative penny stock ticker XYZQ?"]:
    print(f"=== {question} ===")
    research = run_researcher(question)
    plan = run_planner(question, research["gathered_data"])
    print(json.dumps(plan, indent=2))
    print()
