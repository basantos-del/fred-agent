from agent_claude import run_agent_turn_claude
from agent import SYSTEM_PROMPT

answer, used_tools = run_agent_turn_claude("What's AAPL's current price?", SYSTEM_PROMPT)
print(answer)
print("Tools used:", used_tools)
