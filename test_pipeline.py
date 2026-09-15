from pipeline import run_pipeline

# Simple path
result = run_pipeline("What's AAPL's price?")
print(f"[{result['route']}] {result['type']}")
print(result["content"][:300])
print()

# Complex path, expect clarification
result = run_pipeline("Should I buy Microsoft?")
print(f"[{result['route']}] {result['type']}")
print(result["content"][:300])
print()

# Resume with an answer
if result["type"] == "clarification":
    resumed = run_pipeline(
        "5-year horizon, growth focused, about €1000, and yes I'm aware of the fund exposure",
        pending_state=result["pending_state"]
    )
    print(f"[{resumed['route']}] {resumed['type']}")
    print(resumed["content"][:1500])
