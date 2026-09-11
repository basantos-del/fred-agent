from agent import run_agent_turn, SYSTEM_PROMPT

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Should I consider adding to my AAPL exposure?"}
]

answer = run_agent_turn(messages)
print(answer)
