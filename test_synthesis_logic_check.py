from pipeline import run_debate

gathered_data = [
    {"tool": "get_key_ratios", "args": {"ticker": "MSFT"}, "result": {"pe_ratio": 32.1, "roe": 45.3, "gross_margin": 69.8}},
    {"tool": "get_peer_average_ratios", "args": {"ticker": "MSFT"}, "result": {
        "peers": ["ORCL", "CRM", "SAP"],
        "peer_medians": {"pe_ratio": 28.4, "roe": 38.1}
    }}
]

analysis_plan = [
    {"angle": "Valuation vs. peers", "why_it_matters_for_this_question": "P/E is above peer median - needs justifying or flagging as a risk"},
    {"angle": "Profitability quality", "why_it_matters_for_this_question": "ROE and margins are well above peer median - a real quality signal"}
]

question = "Should I buy Microsoft?"
debate = run_debate(question, gathered_data, analysis_plan)


def show(label, text):
    print("\n" + "=" * 20 + " " + label + " " + "=" * 20)
    print(text)


show("BULL REBUTTAL (responding to bear)", debate["bull_rebuttal"])
show("BEAR REBUTTAL (responding to bull)", debate["bear_rebuttal"])
show("FINAL SYNTHESIS", debate["synthesis"])

print("\n" + "=" * 60)
print("MANUAL CHECK")
print("=" * 60)
print("""This is the same fixture as test_bull_bear_debate.py, so watch for whatever
form the ROE-premium-vs-P/E-premium comparison takes this run (last time it was
framed as an invented P/E-per-ROE-point ratio in the bull case, which the bear
rebuttal correctly killed, then the bull rebuttal restated the same cross-unit
comparison without the ratio framing, and the synthesis credited it as an
unrebutted point). After the SYNTHESIS_PROMPT_TEMPLATE fix, check whether the
synthesis either flags any surviving version of that comparison as logically
invalid on its own, or explicitly says it is discounting an unrebutted claim on
those grounds. If it is crediting an X-percent-exceeds-Y-percent-therefore-
undervalued style claim at face value with no caveat, the fix has not taken and
we need to strengthen the prompt language further.""")
