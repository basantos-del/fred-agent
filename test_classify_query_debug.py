# test_classify_query_debug.py
from tools import call_groq_with_retry, call_claude_with_retry
from agent import client
from agent_claude import claude_client

CLASSIFY_PROMPT_TEMPLATE = """Classify this question about financial analysis/portfolio management as either SIMPLE or COMPLEX.

SIMPLE = a single factual lookup answerable with one tool call (e.g. "what's AAPL's price", "what's my portfolio total").
COMPLEX = requires analysis, judgment, a recommendation, or synthesis across multiple sources (e.g. "should I buy X", "how's my portfolio doing", anything involving a bull/bear case).

Question: {question}

Respond with exactly one word: SIMPLE or COMPLEX"""

def classify_groq(question):
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(question=question)
    response = call_groq_with_retry(client, model="openai/gpt-oss-20b", messages=[{"role": "user", "content": prompt}], temperature=0)
    return response.choices[0].message.content.strip().upper()

def classify_claude(question):
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(question=question)
    response = call_claude_with_retry(claude_client, model="claude-sonnet-4-5", max_tokens=10, messages=[{"role": "user", "content": prompt}], extra_body={"temperature": 0})
    return response.content[0].text.strip().upper()

cases = [
    "What's AAPL's current price?",
    "What's my portfolio total?",
    "Should I buy Microsoft?",
    "Give me a full review of my current portfolio.",
    "Do I have any concentration risk in my portfolio right now?",
    "What's Tesla's P/E ratio?",
    "How is my portfolio doing?",
    "Is Nvidia overvalued right now?",
]

mismatches = 0
for q in cases:
    groq_result = classify_groq(q)
    claude_result = classify_claude(q)
    match = "MATCH" if groq_result == claude_result else "MISMATCH"
    if match == "MISMATCH":
        mismatches += 1
    print(f'[{match}] "{q}"\n  Groq: {groq_result} | Claude: {claude_result}\n')

print(f"\n{len(cases) - mismatches}/{len(cases)} agreed. Review any MISMATCH line above before trusting the Groq swap at SIMPLE/COMPLEX boundaries.")
