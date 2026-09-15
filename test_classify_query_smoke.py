# test_classify_query_smoke.py
# Regression check: calls the REAL classify_query() from pipeline.py directly, not a
# reimplementation. test_classify_query_debug.py compares Groq vs Claude behavior with
# its own standalone helpers (correct model string, correct response accessor) — it never
# actually exercises pipeline.classify_query() itself, which is how a typo'd model name
# ("gpt_oss-20b" instead of "gpt-oss-20b") and a leftover Claude-style response accessor
# (response.content[0].text instead of response.choices[0].message.content) both shipped
# silently in the same function after the Triager Groq swap.
from pipeline import classify_query

cases = [
    ("What's AAPL's current price?", "SIMPLE"),
    ("Should I buy Microsoft?", "COMPLEX"),
]

for question, expected in cases:
    result = classify_query(question)
    status = "PASS" if result == expected else "FAIL"
    print(f'[{status}] "{question}" -> {result} (expected {expected})')
