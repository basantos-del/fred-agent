# test_planner_tool_validation_debug.py
from pipeline import run_debate, tool_functions

# Confirms the fix for the invisible-drop bug: a plausible-but-wrong tool name
# (a real field name from evaluate_recommendation's output, not a callable tool)
# must show up in skipped_data_requests instead of vanishing silently.
assert "get_portfolio_context" not in tool_functions, "expected this NOT to be a real registered tool"

gathered_data = []
analysis_plan = [
    {
        "angle": "Portfolio fit",
        "why_it_matters_for_this_question": "test angle",
        "data_still_needed": [
            {"tool": "get_portfolio_context", "args": {}},   # invalid — should be skipped + recorded
            {"tool": "evaluate_recommendation", "args": {"ticker": "MSFT"}}  # valid — should be fetched
        ]
    }
]

debate = run_debate("Should I buy Microsoft?", gathered_data, analysis_plan)

skipped = debate.get("skipped_data_requests", [])
assert any(s["requested_tool"] == "get_portfolio_context" for s in skipped), \
    "FAIL: invalid tool name was not recorded in skipped_data_requests"
print(f"PASS: invalid tool request surfaced — {skipped}")

fetched_tools = [g["tool"] for g in gathered_data]
assert "evaluate_recommendation" in fetched_tools, "FAIL: valid tool request was not fetched"
print(f"PASS: valid tool request still fetched — gathered {fetched_tools}")
