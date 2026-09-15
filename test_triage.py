from pipeline import classify_query

test_questions = [
    "What's AAPL's price?",
    "Should I buy Microsoft?",
    "What's my total portfolio value?",
    "Do I have concentration risk right now?",
    "What's Tesla's P/E ratio?",
]

for q in test_questions:
    print(f"{classify_query(q):8} — {q}")
