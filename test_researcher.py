from pipeline import run_researcher
import json

result = run_researcher("Should I buy Microsoft?")
for g in result["gathered_data"]:
    print(f"--- {g['tool']}({g['args']}) ---")
    print(json.dumps(g['result'], indent=2)[:500])
    print()

