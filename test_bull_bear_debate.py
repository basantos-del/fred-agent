# test_bull_bear_debate.py
from pipeline import run_debate

gathered_data = [
    {"tool": "get_key_ratios", "args": {"ticker": "MSFT"}, "result": {"pe_ratio": 32.1, "roe": 45.3, "gross_margin": 69.8}},
    {"tool": "get_peer_average_ratios", "args": {"ticker": "MSFT"}, "result": {
        "peers": ["ORCL", "CRM", "SAP"],
        "peer_medians": {"pe_ratio": 28.4, "roe": 38.1}
    }}
]

analysis_plan = [
    {"angle": "Valuation vs. peers", "why_it_matters_for_this_question": "P/E is above peer median — needs justifying or flagging as a risk"},
    {"angle": "Profitability quality", "why_it_matters_for_this_question": "ROE and margins are well above peer median — a real quality signal"}
]

question = "Should I buy Microsoft?"
debate = run_debate(question, gathered_data, analysis_plan)

def show(label, text):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    print(text)

show("BULL CASE", debate["bull_case"])
show("BEAR CASE", debate["bear_case"])
show("BULL REBUTTAL (responding to bear)", debate["bull_rebuttal"])
show("BEAR REBUTTAL (responding to bull)", debate["bear_rebuttal"])
show("FINAL SYNTHESIS (this is what Bernardo actually sees)", debate["synthesis"])

# Sanity check: the two sides should genuinely diverge, not just paraphrase each
# other under different headers. Crude word-overlap check — not proof of quality,
# just a tripwire for the debate collapsing into two similar-sounding takes.
assert debate["bull_case"] != debate["bear_case"], "Bull and bear cases are identical — something's wrong"
bull_words = set(debate["bull_case"].lower().split())
bear_words = set(debate["bear_case"].lower().split())
overlap = len(bull_words & bear_words) / max(len(bull_words | bear_words), 1)
print(f"\nWord overlap between bull and bear case: {overlap:.0%}")
assert overlap < 0.6, f"Bull and bear cases overlap {overlap:.0%} — may not be genuinely diverging"

print("\nPASS: bull and bear cases present and appear to genuinely diverge")
