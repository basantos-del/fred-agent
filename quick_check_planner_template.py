# quick_check_planner_template.py — no API call, just checks the template formats cleanly
from pipeline import PLANNER_PROMPT_TEMPLATE
from tools import tool_functions

prompt = PLANNER_PROMPT_TEMPLATE.format(
    question="test question",
    gathered_data="(no data)",
    valid_tools=", ".join(sorted(tool_functions.keys()))
)

assert "get_stock_price" in prompt and "evaluate_recommendation" in prompt, \
    "FAIL: valid_tools list didn't make it into the rendered prompt"
print("PASS: tool list renders into the prompt")
print("\n--- excerpt around the tool list ---")
idx = prompt.find("valid tool names")
print(prompt[idx-20:idx+300])
